import requests
import socket
import time
import json
import logging
from ..config import Config

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
        # Always include local provider config for fallback safety
        if settings is None:
            self.providers = Config.LLM_PROVIDERS
        else:
            self.providers = {provider: settings}
            # Merge in local provider so fallback never KeyErrors
            self.providers['local'] = Config.LLM_PROVIDERS.get('local', {
                'api_url': 'http://localhost:11434'
            })
        
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
            
        # Check if any cloud providers are properly configured (enabled + api key), sorted by priority
        self.available_cloud_providers = []
        cloud_candidates = []
        for name, cfg in self.providers.items():
            if name == 'local':
                continue
            if not isinstance(cfg, dict):
                continue
            if not cfg.get('api_key') or not str(cfg.get('api_key')).strip():
                continue
            if cfg.get('enabled') is False:
                continue
            cloud_candidates.append((cfg.get('priority', 99), name))
        cloud_candidates.sort(key=lambda x: x[0])
        self.available_cloud_providers = [name for _, name in cloud_candidates]
        if self.available_cloud_providers:
            # Set current to top priority cloud
            self.current_provider = self.available_cloud_providers[0]
            self.model_name = self.providers[self.current_provider].get('model')
        
        # Only check internet if we have configured cloud providers
        if self.available_cloud_providers:
            self.has_internet = self._check_internet_connection()
            
            # If we have internet and configured providers, start with cloud
            if self.has_internet:
                self.current_provider = self.available_cloud_providers[0]
                self.model_name = self.providers[self.current_provider]['model']
                print(f"Starting with cloud provider: {self.current_provider}")
            else:
                print("No internet connection detected, using local provider")
        else:
            print("No cloud providers configured, using local provider")
        
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
        """
        logging.info(f"\n{'='*50}\nSQL Generation Request\n{'='*50}")
        logging.info(f"Original prompt: {prompt}")
        errors = []
        providers_to_try = []
        
        # Always check internet first
        self.has_internet = self._check_internet_connection()
        
        # Build provider list from config (enabled + api_key), sorted by priority
        if self.has_internet:
            cloud_candidates = []
            for name, cfg in self.providers.items():
                if name == 'local':
                    continue
                if not isinstance(cfg, dict):
                    continue
                if not cfg.get('api_key') or not str(cfg.get('api_key')).strip():
                    continue
                if cfg.get('enabled') is False:
                    continue
                cloud_candidates.append((cfg.get('priority', 99), name))
            cloud_candidates.sort(key=lambda x: x[0])
            for _, name in cloud_candidates:
                providers_to_try.append(name)
                logging.info(f"Added cloud provider to try: {name}")
            
            if providers_to_try:
                logging.info(f"Will try cloud providers in order: {providers_to_try}")
            else:
                logging.warning("No cloud providers properly configured")
        else:
            logging.warning("No internet connection for cloud providers")
        
        # Only add local as last resort
        providers_to_try.append('local')
        logging.info(f"Final provider order: {providers_to_try}")
        
        # Try each provider in sequence
        for i, provider in enumerate(providers_to_try):
            is_last_provider = i == len(providers_to_try) - 1
            
            try:
                self.current_provider = provider
                self.model_name = (self.providers[provider]['model'] if provider != 'local' 
                                else (self.models[0] if self.models else 'codellama'))
                
                if provider == 'local':
                    if not is_last_provider:
                        logging.warning("Skipping local provider - cloud providers still available")
                        continue
                    logging.info("Attempting local generation as last resort...")
                    return self._try_local_generation(prompt)
                    
                # Try cloud provider
                logging.info(f"Attempting generation with cloud provider: {provider}")
                try:
                    result = self._try_cloud_generation(prompt)
                    logging.info(f"Successfully generated SQL with {provider}")
                    return result
                except Exception as cloud_error:
                    error_str = str(cloud_error)
                    if "403" in error_str or "401" in error_str:
                        logging.error(f"❌ Authentication failed for {provider}")
                        if provider == "google":
                            logging.error("   Hint: Check Google API key permissions")
                        elif provider == "openai":
                            logging.error("   Hint: Verify OpenAI API key and credits")
                    elif "429" in error_str:
                        logging.warning(f"⚠️ Rate limit hit for {provider}")
                    else:
                        logging.error(f"❌ Error with {provider}: {error_str}")
                    
                    if not is_last_provider:
                        logging.info(f"Trying next provider...")
                        continue
                    raise  # Re-raise if this is the last provider
                    
            except Exception as e:
                error_msg = f"Provider {provider} failed: {str(e)}"
                print(error_msg)
                errors.append(error_msg)
                continue
        
        # If we get here, all providers failed
        error_details = "\n".join(errors)
        raise Exception(f"Failed to generate SQL with all providers:\n{error_details}")

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
                    response = requests.post(
                        config['api_url'],
                        headers=headers, 
                        json=payload,
                        timeout=10)
                    
                    if response.status_code == 429:  # Rate limit error
                        last_error = "Rate limit exceeded"
                        # Backoff before next retry
                        time.sleep(delay * (2 ** attempt))
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
                        content = result['choices'][0]['message']['content'].strip()
                        return self._extract_sql(content)
                        
                elif self.current_provider == 'google':
                    if 'error' in result:
                        raise Exception(f"Google API error: {result['error']['message']}")
                        
                    # Extract the SQL from the response
                    if 'candidates' in result and result['candidates']:
                        candidate = result['candidates'][0]
                        content = candidate.get('content', {})
                        if content and 'parts' in content and content['parts']:
                            text = content['parts'][0].get('text', '').strip()
                            if text:
                                return self._extract_sql(text)
                    
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
                        return self._extract_sql(result['completion'].strip())
                        
                raise Exception(f"Invalid response format from {self.current_provider} API")
                
            except requests.exceptions.RequestException as e:
                last_error = str(e)
                if response.status_code != 429:  # If not a rate limit error
                    break  # Don't retry other errors
                    
        # If we've exhausted all retries, raise the last error
        raise Exception(f"Cloud provider failed after {retries} attempts: {last_error}")

    def _try_local_generation(self, prompt):
        """
        Attempts to generate SQL using local models via Ollama.
        """
        last_error = None
        ollama_url = self.providers['local']['api_url'].rstrip('/')

        # First check which models are available
        try:
            response = requests.get(f"{ollama_url}/api/tags")
            if response.status_code == 200:
                available_models = [m.get('name') for m in response.json().get('models', []) if m.get('name')]
                available_model_map = {m.split(':')[0]: m for m in available_models}
                if not available_models:
                    raise Exception("No models found in Ollama")
                print(f"Available Ollama models: {available_models}")
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
                print(f"Model '{model_to_try}' not available in Ollama, skipping.")
                continue
                
            # Try to generate with current model
            try:
                print(f"Trying generation with model: {ollama_model_name}")
                response = requests.post(
                    f"{ollama_url}/api/generate",
                    json={
                        "model": ollama_model_name,
                        "prompt": "Generate only the SQL query, no explanations or markdown: " + prompt,
                        "stream": False,
                        "temperature": 0.1,
                        "stop": [";", "\n\n"]
                    },
                    timeout=60 # Increased timeout
                )
                
                if response.status_code == 200:
                    result = response.json().get('response', '').strip()
                    if result:
                        sql = self._extract_sql(result)
                        print(f"Successfully generated with {ollama_model_name}")
                        return sql
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