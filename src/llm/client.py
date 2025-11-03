import requests
import socket
import time
import json
import logging
from ..config import Config
import re

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

class LLMClient:
    def __init__(self, provider=None, settings=None):
        """
        Initializes the client with configuration for cloud and local models.
        
        Args:
            provider: Optional provider name to use (e.g., 'google', 'openai', 'anthropic', 'local')
            settings: Optional settings dict for the provider. If not provided, will load from Config.
        """
        # Load runtime LLM provider settings (merge of defaults, saved config and env vars)
        runtime_cfg = Config.get_llm_config()
        self.providers = runtime_cfg.get('providers', Config.LLM_PROVIDERS.copy())

        # If explicit settings were provided for a single provider, merge them on top
        if provider and settings:
            # ensure provider entry exists
            self.providers.setdefault(provider, {})
            # merge provided settings (do not remove other providers)
            self.providers[provider].update(settings)
        # Ensure 'local' exists for fallback safety
        if 'local' not in self.providers:
            self.providers['local'] = Config.LLM_PROVIDERS.get('local', {'api_url': 'http://localhost:11434'})
        
        # Set up local model configuration (from config, with sensible defaults)
        local_cfg = Config.LLM_PROVIDERS.get('local', {})
        configured_local_models = local_cfg.get('models', []) if isinstance(local_cfg.get('models', []), list) else []
        default_local_models = ['codellama', 'llama3', 'deepseek-coder']
        # Keep order: configured first then defaults without duplicates
        seen_models = set()
        self.local_models = []
        for m in configured_local_models + default_local_models:
            base_model = m.split(':')[0]
            if m and base_model not in seen_models:
                seen_models.add(base_model)
                self.local_models.append(m.strip())
        self.current_provider = 'local'
        self.model_name = self.local_models[0]  # Start with codellama
        
        # Initialize state
        self.has_internet = False
        self.models = self.local_models.copy()  # Available models for current provider
        
        # If provider is specified and settings provided, use those
        if provider and settings and provider != 'local':
            self.current_provider = provider
            self.model_name = settings.get('model')
            # Check internet since we're using a cloud provider
            self.has_internet = self._check_internet_connection()
            print(f"Using specified provider: {provider}")
            if not self.has_internet:
                print("Warning: No internet connection detected for cloud provider")
                # Fall back to local if no internet
                self.current_provider = 'local'
                self.model_name = self.local_models[0]
            return
            
        # Get available providers, sorted by priority
        self.available_cloud_providers = self.get_available_providers()
        
        # Set initial provider state
        self.current_provider = 'local' # Default to local
        if self.available_cloud_providers:
            self.has_internet = self._check_internet_connection()
            if self.has_internet:
                # If there's internet, set the initial provider to the highest priority one
                self.current_provider = self.available_cloud_providers[0]
                self.model_name = self.providers[self.current_provider]['model']
                print(f"Starting with cloud provider: {self.current_provider}")
            else:
                print("No internet connection detected, will use local provider if available.")
        else:
            print("No enabled cloud providers configured, using local provider.")

        # If the default is local, set the model name
        if self.current_provider == 'local':
            self.model_name = self.local_models[0] if self.local_models else None
        
        # Check if Ollama is running for local model support
        self._check_ollama_connection()

    def _check_internet_connection(self):
        """
        Check if there is an active internet connection by trying to connect to a reliable host.
        """
        try:
            # Try to connect to Google's DNS server
            socket.create_connection(("8.8.8.8", 53), timeout=3)
            return True
        except OSError:
            return False
            
    def _check_ollama_connection(self):
        """
        Check if Ollama is running locally and ensure models are available.
        """
        try:
            ollama_url = self.providers['local']['api_url']
            response = requests.get(f"{ollama_url}/api/tags", timeout=5)
            if response.status_code == 200:
                print("Ollama is running")
                # Check for at least one model
                for model in self.local_models:
                    try:
                        check_response = requests.post(
                            f"{ollama_url}/api/pull",
                            json={"name": model},
                            timeout=5
                        )
                        if check_response.status_code == 200:
                            print(f"Found local model: {model}")
                            return True
                    except:
                        continue
                        
            print("No local models found, attempting to pull codellama...")
            try:
                # Try to pull codellama
                pull_response = requests.post(
                    f"{ollama_url}/api/pull",
                    json={"name": "codellama"},
                    timeout=300  # 5 minutes timeout for pulling
                )
                if pull_response.status_code == 200:
                    print("Successfully pulled codellama model")
                    return True
            except:
                print("Failed to pull codellama model")
                
        except requests.exceptions.RequestException as e:
            print(f"Ollama is not accessible: {str(e)}")
            print("Please ensure Ollama is installed and running ('ollama serve')")
            
        return False

    def generate_sql(self, prompt):
        """
        Attempts to generate SQL using cloud providers first, with local fallback only as last resort.
        Returns a dictionary containing both the complete response and extracted SQL.
        """
        logging.info(f"\n{'='*50}\nSQL Generation Request\n{'='*50}")
        logging.info(f"Original prompt: {prompt}")
        errors = []
        providers_to_try = []
        
        # Always check internet first
        self.has_internet = self._check_internet_connection()

        # Get all configured cloud providers, sorted by priority
        providers_to_try = []
        if self.has_internet:
            cloud_providers = self.get_available_providers()
            if cloud_providers:
                providers_to_try.extend(cloud_providers)
                logging.info(f"Cloud providers to try, in order: {providers_to_try}")
            else:
                logging.warning("No enabled cloud providers with models configured.")
        else:
            logging.warning("No internet connection, skipping cloud providers.")

        # Add local as a fallback if Ollama is running
        if self._check_ollama_connection():
            if 'local' not in providers_to_try:
                providers_to_try.append('local')
        else:
            logging.warning("Ollama is not running, local provider is unavailable.")
                
        logging.info(f"Final provider order: {providers_to_try}")
        
        # If no providers are available at all, raise an error
        if not providers_to_try:
            raise Exception("No providers available. Check your internet connection, configuration, and ensure Ollama is running for local fallback.")

        # Try each provider in sequence
        for i, provider in enumerate(providers_to_try):
            is_last_provider = i == len(providers_to_try) - 1
            
            try:
                self.current_provider = provider
                self.model_name = (self.providers[provider].get('model') if provider != 'local' 
                                else (self.local_models[0] if self.local_models else 'codellama'))
                
                logging.info(f"Attempting generation with provider: {provider}")

                if provider == 'local':
                    # Try local generation
                    return self._try_local_generation(prompt)
                else:
                    # Try cloud generation
                    result = self._try_cloud_generation(prompt)
                    logging.info(f"Successfully generated SQL with {provider}")
                    return result
                    
            except Exception as e:
                error_msg = f"Provider {provider} failed: {str(e)}"
                errors.append(error_msg)
                logging.error(error_msg)
                
                if not is_last_provider:
                    next_provider = providers_to_try[i+1]
                    logging.info(f"Trying next provider: {next_provider}")
                    continue
                else:
                    # This was the last provider
                    logging.error("All providers failed.")
                    raise Exception(f"All providers failed:\n" + "\n".join(errors))
        
        # This part should not be reached if there is at least one provider
        raise Exception(f"Failed to generate SQL with all providers:\n" + "\n".join(errors))

    def get_available_providers(self):
        """
        Returns a list of available cloud provider names, sorted by priority.
        A provider is available if it's not 'local', is a dictionary, is not explicitly
        disabled, and has a model configured.
        """
        cloud_candidates = []
        for name, cfg in self.providers.items():
            # Skip local provider here; it's handled separately as a fallback
            if name == 'local' or not isinstance(cfg, dict):
                continue

            # Treat explicit false-like enabled values as disabled
            enabled_val = cfg.get('enabled', True)
            if (isinstance(enabled_val, str) and enabled_val.lower() in ('false', '0', 'no')) or enabled_val is False:
                logging.info(f"Provider {name} skipped: Explicitly disabled in config.")
                continue

            model = (cfg.get('model') or '').strip()
            if not model:
                logging.info(f"Provider {name} skipped: No model configured.")
                continue

            # If we reach here, the provider is a candidate.
            priority = cfg.get('priority', 99)
            cloud_candidates.append((priority, name))
        
        # Sort by priority (lower number is higher priority)
        cloud_candidates.sort(key=lambda x: x[0])
        
        available = [name for _, name in cloud_candidates]
        logging.info(f"Available cloud providers, sorted by priority: {available}")
        return available

    def _extract_sql(self, text):
        """Extract likely SQL from mixed text (comments, markdown, explanations).

        Strategy:
        - Remove markdown code fences
        - Find the first DML/DDL keyword and return from there
        - Trim trailing backticks or fences
        """
        if not text:
            return text
        cleaned = text.replace('```sql', '').replace('```', '').strip()
        upper = cleaned.upper()
        keywords = ["SELECT", "INSERT", "UPDATE", "DELETE", "CREATE", "ALTER", "DROP"]
        start_idx = -1
        for kw in keywords:
            i = upper.find(kw)
            if i != -1 and (start_idx == -1 or i < start_idx):
                start_idx = i
        if start_idx != -1:
            return cleaned[start_idx:].strip()
        return cleaned

    def _try_cloud_generation(self, prompt, retries=3, delay=2):
        """
        Attempts to generate SQL using cloud providers (OpenAI/Google/Anthropic).
        Includes retry logic for rate limits.
        """
        config = self.providers[self.current_provider]
        last_error = None
        
        # Validate API configuration
        if not config.get('api_key'):
            raise Exception(f"No API key configured for {self.current_provider}")
        if not config.get('api_url'):
            raise Exception(f"No API URL configured for {self.current_provider}")
            
        print(f"Using {self.current_provider} for generation...")
        
        for attempt in range(retries):
            if attempt > 0:
                print(f"Retrying request (attempt {attempt + 1}/{retries})...")
                time.sleep(delay * (2 ** (attempt - 1)))  # Exponential backoff
                
            try:
                if self.current_provider == 'openai':
                    headers = {
                        "Authorization": f"Bearer {config['api_key']}",
                        "Content-Type": "application/json"
                    }
                    payload = {
                        "model": config['model'],
                        "messages": [
                            {"role": "system", "content": "You are a senior SQL engineer. Respond with a concise SQL solution. It's OK to include brief comments or markdown, but ensure the final SQL is present."},
                            {"role": "user", "content": prompt}
                        ],
                        "temperature": 0.1,
                        "max_tokens": 500,
                        "frequency_penalty": 0,
                        "presence_penalty": 0
                    }
                    
                elif self.current_provider == 'google':
                    api_key = config.get('api_key')
                    if not api_key:
                        raise Exception("Google API key not configured")
                        
                    headers = {
                        "Content-Type": "application/json",
                        "x-goog-api-key": api_key
                    }
                    
                    payload = {
                        "contents": [
                            {
                                "role": "user",
                                "parts": [{"text": "Generate a SQL query for this request. You may include brief comments, but include executable SQL: " + prompt}]
                            }
                        ],
                        "generationConfig": {
                            "temperature": 0.1,
                            "candidateCount": 1,
                            "stopSequences": [],
                            "maxOutputTokens": 500,
                            "topP": 0.8,
                            "topK": 40
                        }
                    }
                    
                    # Use the complete API URL from config
                    api_url = config['api_url']
                    
                elif self.current_provider == 'anthropic':
                    headers = {
                        "x-api-key": config['api_key'],
                        "Content-Type": "application/json"
                    }
                    payload = {
                        "model": config['model'],
                        "prompt": f"\n\nHuman: Generate only an SQL query without any explanations for this request: {prompt}\n\nAssistant:",
                        "max_tokens_to_sample": 150,
                        "temperature": 0.1
                    }

                try:
                    # Use configured API URL directly for all providers
                    # Get provider-specific timeout or use default
                    timeout = config.get('timeout', 30)  # Default 30s timeout for cloud providers
                    
                    response = requests.post(
                        config['api_url'],
                        headers=headers, 
                        json=payload,
                        timeout=timeout
                    )
                    
                    if response.status_code == 429:  # Rate limit error
                        last_error = f"{self.current_provider} rate limit exceeded"
                        logging.warning(f"Rate limit hit for {self.current_provider}")
                        time.sleep(delay * (2 ** attempt))  # Exponential backoff
                        continue
                        continue
                    
                    response.raise_for_status()
                    
                    if response.status_code == 204:  # No content
                        raise Exception("API returned no content")
                        
                    result = response.json()
                    
                    if 'error' in result:
                        error_msg = result['error'].get('message', 'Unknown error')
                        raise Exception(f"API error: {error_msg}")
                        
                except requests.exceptions.RequestException as e:
                    if attempt < retries - 1:
                        last_error = f"Request failed: {str(e)}"
                        continue
                    raise Exception(f"API request failed: {str(e)}")
                except json.JSONDecodeError as e:
                    raise Exception(f"Invalid JSON response: {str(e)}")
                    
                # Successfully got response
                if not result:
                    raise Exception("Empty response from API")
                
                # Extract response based on provider
                if self.current_provider == 'openai':
                    if 'choices' in result and len(result['choices']) > 0:
                        full_response = result['choices'][0]['message']['content'].strip()
                        sql = self._extract_sql(full_response)
                        return {
                            'full_response': full_response,
                            'sql': sql,
                            'explanation': full_response.replace(sql, '').strip()
                        }
                        
                elif self.current_provider == 'google':
                    if 'error' in result:
                        raise Exception(f"Google API error: {result['error']['message']}")
                        
                    # Extract the response
                    if 'candidates' in result and result['candidates']:
                        candidate = result['candidates'][0]
                        content = candidate.get('content', {})
                        if content and 'parts' in content and content['parts']:
                            full_response = content['parts'][0].get('text', '').strip()
                            if full_response:
                                sql = self._extract_sql(full_response)
                                return {
                                    'full_response': full_response,
                                    'sql': sql,
                                    'explanation': full_response.replace(sql, '').strip()
                                }
                    
                    # Check for errors
                    if 'promptFeedback' in result:
                        feedback = result['promptFeedback']
                        if 'blockReason' in feedback:
                            raise Exception(f"Google API blocked: {feedback['blockReason']}")
                        if 'safetyRatings' in feedback:
                            ratings = feedback['safetyRatings']
                            blocked_categories = [r['category'] for r in ratings if r.get('blocked')]
                            if blocked_categories:
                                raise Exception(f"Content blocked for: {', '.join(blocked_categories)}")
                    
                    # If we get here, something went wrong
                    raise Exception(f"Could not extract valid SQL from Google API response. Got: {result}")
                        
                elif self.current_provider == 'anthropic':
                    if 'completion' in result:
                        full_response = result['completion'].strip()
                        sql = self._extract_sql(full_response)
                        return {
                            'full_response': full_response,
                            'sql': sql,
                            'explanation': full_response.replace(sql, '').strip()
                        }
                        
                raise Exception(f"Invalid response format from {self.current_provider} API")
                
            except requests.exceptions.RequestException as e:
                last_error = str(e)
                if response.status_code != 429:  # If not a rate limit error
                    break  # Don't retry other errors
                    
        # If we've exhausted all retries, raise the last error
        raise Exception(f"Cloud provider failed after {retries} attempts: {last_error}")

    def generate(self, prompt):
        """General purpose text generation wrapper.

        For compatibility with code that expects a `generate` method (e.g. error analysis),
        delegate to the same provider selection but return the full textual response.
        """
        # Use same provider selection logic as generate_sql but return text
        try:
            result = self.generate_sql(prompt)
            if isinstance(result, dict):
                return result.get('full_response') or result.get('sql') or ''
            return str(result)
        except Exception as e:
            # If SQL generation fails, at least try local generation attempt to get any textual analysis
            try:
                loc = self._try_local_generation(prompt)
                return loc.get('full_response') if isinstance(loc, dict) else str(loc)
            except Exception:
                # Last resort: return exception message
                return str(e)

    def _try_local_generation(self, prompt):
        """
        Attempts to generate SQL using local models via Ollama.
        """
        last_error = None
        ollama_url = self.providers['local']['api_url'].rstrip('/')

        # First check which models are available and make sure we're connected
        try:
            # Try to connect to Ollama first
            response = requests.get(f"{ollama_url}/api/tags", timeout=5)
            if response.status_code == 200:
                logging.info("Successfully connected to Ollama")
                available_models = [m.get('name') for m in response.json().get('models', []) if m.get('name')]
                available_model_map = {m.split(':')[0]: m for m in available_models}
                if not available_models:
                    logging.warning("No models found in Ollama, will try to pull codellama")
                    self._ensure_model_available('codellama')
                    # Refresh model list after pull
                    response = requests.get(f"{ollama_url}/api/tags", timeout=5)
                    if response.status_code == 200:
                        available_models = [m.get('name') for m in response.json().get('models', []) if m.get('name')]
                        available_model_map = {m.split(':')[0]: m for m in available_models}
                        if not available_models:
                            raise Exception("Still no models available after pulling codellama")
                    logging.info(f"Available Ollama models: {available_models}")
            else:
                raise Exception(f"Failed to get models list: {response.status_code}")
        except Exception as e:
            raise Exception(f"Failed to connect to Ollama or list models: {str(e)}")

        # Try each model in sequence
        for model_to_try in self.local_models:
            base_model_name = model_to_try.split(':')[0]
            
            # Find a matching installed model
            ollama_model_name = None
            if model_to_try in available_models:
                ollama_model_name = model_to_try
            elif base_model_name in available_model_map:
                ollama_model_name = available_model_map[base_model_name]

            if not ollama_model_name:
                logging.warning(f"Model '{model_to_try}' not available in Ollama, attempting to pull...")
                try:
                    self._ensure_model_available(base_model_name)
                    # If successful, update the model name
                    ollama_model_name = base_model_name
                except Exception as e:
                    logging.error(f"Failed to pull model {base_model_name}: {str(e)}")
                    continue
                
            # Try to generate with current model
            try:
                print(f"Trying generation with model: {ollama_model_name}")
                # Get local provider timeout from config
                local_timeout = self.providers['local'].get('timeout', 120)  # Default 2 minutes for local
                logging.info(f"Using {local_timeout}s timeout for local generation")
                
                response = requests.post(
                    f"{ollama_url}/api/generate",
                    json={
                        "model": ollama_model_name,
                        "prompt": "Generate only the SQL query, no explanations or markdown: " + prompt,
                        "stream": False,
                        "temperature": 0.1,
                        "stop": [";", "\n\n"]
                    },
                    timeout=local_timeout
                )
                
                if response.status_code == 200:
                    result = response.json().get('response', '').strip()
                    if result:
                        sql = self._extract_sql(result)
                        print(f"Successfully generated with {ollama_model_name}")
                        # Normalize local response to dict similar to cloud providers
                        return {
                            'full_response': result,
                            'sql': sql,
                            'explanation': result.replace(sql, '').strip() if sql else ''
                        }
                else:
                    print(f"Model {ollama_model_name} failed with status {response.status_code}")
                    
            except Exception as e:
                last_error = e
                print(f"Error with model {ollama_model_name}: {str(e)}")
                continue
                
        # If we get here, all models failed
        error_msg = f"All local models failed. Last error: {str(last_error)}" if last_error else "No successful generation from any local model"
        raise Exception(error_msg)

    def _ensure_model_available(self, model_name):
        """
        Ensures a model is available locally, pulling it if necessary.
        """
        try:
            # Check if model exists
            ollama_url = self.providers['local']['api_url']
            response = requests.get(
                f"{ollama_url}/api/tags",
                timeout=5
            )
            response.raise_for_status()
            
            available_models = response.json()
            model_exists = any(
                model['name'] == model_name 
                for model in available_models.get('models', [])
            )
            
            if not model_exists:
                print(f"Model {model_name} not found locally. Pulling...")
                response = requests.post(
                    f"{ollama_url}/api/pull",
                    json={"name": model_name},
                    timeout=600  # 10 minute timeout for pulling
                )
                response.raise_for_status()
                print(f"Successfully pulled {model_name}")
                
            return True
            
        except Exception as e:
            print(f"Error ensuring model availability: {str(e)}")
            return False

    def get_models(self):
        return self.models

    def classify(self, prompt, timeout=15):
        """Ask the LLM to classify whether the prompt requires SQL or can be answered as text.

        Returns a dict: {'intent': 'sql'|'text'|'unknown', 'explain': str, 'suggested_sql': str|None}
        This method will try cloud providers first (if internet) then local Ollama.
        """
        logging.info("Running intent classification for prompt")
        # Build a short instruction
        instruction = (
            "Classify whether the user's request requires executing SQL against the database to fulfill. "
            "Respond on the first line with either SQL or TEXT (uppercase). Optionally include a suggested SQL query "
            "enclosed in ```sql ... ``` after the first line if it's SQL. Then include a brief one-line explanation.\n\n"
            f"User request:\n{prompt}"
        )

        # Try cloud providers if internet is available
        try:
            self.has_internet = self._check_internet_connection()
            providers = self.get_available_providers() if self.has_internet else []

            # Try cloud first
            for provider in providers:
                try:
                    cfg = self.providers.get(provider, {})
                    if provider == 'openai':
                        headers = {"Authorization": f"Bearer {cfg.get('api_key')}", "Content-Type": "application/json"}
                        payload = {
                            "model": cfg.get('model'),
                            "messages": [
                                {"role": "system", "content": "You are an intent classifier. Reply briefly."},
                                {"role": "user", "content": instruction}
                            ],
                            "temperature": 0.0,
                            "max_tokens": 200,
                        }
                        resp = requests.post(cfg.get('api_url'), headers=headers, json=payload, timeout=timeout)
                        resp.raise_for_status()
                        body = resp.json()
                        if 'choices' in body and len(body['choices']) > 0:
                            text = body['choices'][0]['message']['content'].strip()
                            return self._parse_classify_response(text)

                    # For other cloud providers fallback to generic POST if configured
                    if provider in ('google', 'anthropic'):
                        # Try a generic generate request similar to cloud path in _try_cloud_generation
                        headers = {"Content-Type": "application/json"}
                        if cfg.get('api_key'):
                            headers.update({ 'Authorization': f"Bearer {cfg.get('api_key')}" })
                        payload = { 'prompt': instruction, 'max_tokens': 200, 'temperature': 0.0 }
                        resp = requests.post(cfg.get('api_url'), headers=headers, json=payload, timeout=timeout)
                        resp.raise_for_status()
                        body = resp.json()
                        # Best-effort extraction
                        text = ''
                        if isinstance(body, dict):
                            if 'choices' in body and body['choices']:
                                text = body['choices'][0].get('text') or body['choices'][0].get('message', {}).get('content', '')
                            elif 'completion' in body:
                                text = body.get('completion', '')
                        if text:
                            return self._parse_classify_response(text.strip())
                except Exception as e:
                    logging.debug(f"Provider {provider} classify failed: {e}")
                    continue

            # Local Ollama fallback
            try:
                ollama_url = self.providers['local']['api_url'].rstrip('/')
                resp = requests.post(f"{ollama_url}/api/generate", json={
                    "model": self.model_name,
                    "prompt": instruction,
                    "temperature": 0.0,
                    "stream": False,
                    "max_tokens": 200
                }, timeout=timeout)
                resp.raise_for_status()
                body = resp.json()
                text = ''
                if isinstance(body, dict):
                    text = body.get('response') or body.get('choices', [{}])[0].get('text', '')
                if text:
                    return self._parse_classify_response(text.strip())
            except Exception as e:
                logging.debug(f"Local classify failed: {e}")

        except Exception as e:
            logging.debug(f"Classification top-level error: {e}")

        return {'intent': 'unknown', 'explain': 'Could not classify intent', 'suggested_sql': None}

    def _parse_classify_response(self, text: str):
        """Parse the classifier response text into structured dict."""
        lines = [l.strip() for l in text.splitlines() if l.strip()]
        intent = 'unknown'
        suggested_sql = None
        explain = ''
        if lines:
            first = lines[0].upper()
            if first.startswith('SQL') or first.startswith('YES'):
                intent = 'sql'
            elif first.startswith('TEXT') or first.startswith('NO'):
                intent = 'text'
            else:
                # if the first line contains backtick fenced sql, detect it
                if lines[0].startswith('```sql') or 'SELECT' in lines[0].upper():
                    intent = 'sql'

        # Extract fenced SQL if present
        m = re.search(r"```sql\s*(.*?)\s*```", text, flags=re.I | re.S)
        if m:
            suggested_sql = m.group(1).strip()
        else:
            # Look for first SQL-like statement substring
            s_idx = None
            for kw in ("SELECT", "INSERT", "UPDATE", "DELETE", "CREATE", "ALTER", "DROP"):
                i = text.upper().find(kw)
                if i != -1 and (s_idx is None or i < s_idx):
                    s_idx = i
            if s_idx is not None:
                suggested_sql = text[s_idx:].strip()

        # Explanation is anything after first line and not the fenced SQL
        if len(lines) > 1:
            explain = '\n'.join(lines[1:])
        else:
            explain = ''

        return {'intent': intent, 'explain': explain, 'suggested_sql': suggested_sql}