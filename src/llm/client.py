import requests
import socket
import time
from config import Config

class LLMClient:
    def __init__(self):
        """
        Initializes the client with configuration for cloud and local models.
        """
        self.providers = Config.LLM_PROVIDERS
        self.models = self.providers['local']['models']
        
        # Start with local models configuration
        self.current_provider = 'local'
        self.model_name = self.models[0] if self.models else 'codellama'
        self.fallback_model = self.models[1] if len(self.models) > 1 else self.model_name
        
        # Check if any cloud providers are properly configured
        self.available_cloud_providers = []
        for provider in ['openai', 'google', 'anthropic']:
            if self.providers[provider]['api_key'] and self.providers[provider]['api_key'].strip():
                self.available_cloud_providers.append(provider)
        
        # Only check internet if we have configured cloud providers
        self.has_internet = (
            self._check_internet_connection() 
            if self.available_cloud_providers 
            else False
        )
        
        # If we have internet and configured providers, try to use them
        if self.has_internet and self.available_cloud_providers:
            self.current_provider = self.available_cloud_providers[0]
            self.model_name = self.providers[self.current_provider]['model']
            print(f"Starting with cloud provider: {self.current_provider}")

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

    def generate_sql(self, prompt):
        """
        Attempts to generate SQL using cloud provider first, falls back to local if needed.
        """
        try:
            # If we have internet and a cloud provider configured, try it first
            if self.has_internet and self.current_provider != 'local':
                try:
                    return self._try_cloud_generation(prompt)
                except Exception as e:
                    print(f"Cloud provider {self.current_provider} failed: {str(e)}")
                    # Try other cloud providers before falling back to local
                    for provider in ['openai', 'google', 'anthropic']:
                        if provider != self.current_provider and self.providers[provider]['api_key']:
                            try:
                                self.current_provider = provider
                                self.model_name = self.providers[provider]['model']
                                return self._try_cloud_generation(prompt)
                            except Exception as e:
                                print(f"Cloud provider {provider} failed: {str(e)}")
                                continue
                    
                    # All cloud providers failed, fall back to local
                    print("All cloud providers failed, falling back to local model")
                    self.current_provider = 'local'
                    self.model_name = self.models[0] if self.models else 'codellama'
            
            # Try local generation
            return self._try_local_generation(prompt)
            
        except Exception as e:
            raise Exception(f"Failed to generate SQL: {str(e)}")

    def _try_cloud_generation(self, prompt, retries=3, delay=1):
        """
        Attempts to generate SQL using cloud providers (OpenAI/Google/Anthropic).
        Includes retry logic for rate limits.
        """
        config = self.providers[self.current_provider]
        last_error = None
        
        for attempt in range(retries):
            if attempt > 0:
                print(f"Rate limit hit. Retrying request (attempt {attempt + 1}/{retries})...")
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
                            {"role": "system", "content": "You are an SQL expert. Generate ONLY the SQL query without any markdown formatting, comments, or explanations. Do not include ```sql or ``` tags. Return just the raw SQL query."},
                            {"role": "user", "content": prompt}
                        ],
                        "temperature": 0.1,
                        "max_tokens": 500,
                        "frequency_penalty": 0,
                        "presence_penalty": 0
                    }
                    
                elif self.current_provider == 'google':
                    headers = {
                        "Content-Type": "application/json",
                        "X-goog-api-key": config['api_key']
                    }
                    payload = {
                        "contents": [{
                            "parts": [{
                                "text": f"You are an SQL expert. Generate a SQL query for this request: {prompt}"
                            }]
                        }]
                    }
                    
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

                response = requests.post(config['api_url'], headers=headers, json=payload)
                
                if response.status_code == 429:  # Rate limit error
                    last_error = "Rate limit exceeded"
                    continue  # Try again after delay
                    
                response.raise_for_status()
                result = response.json()
                
                # Extract response based on provider
                if self.current_provider == 'openai':
                    if 'choices' in result and len(result['choices']) > 0:
                        sql = result['choices'][0]['message']['content'].strip()
                        # Remove any markdown formatting or code blocks
                        sql = sql.replace('```sql', '').replace('```', '').strip()
                        return sql
                        
                elif self.current_provider == 'google':
                    if 'candidates' in result:
                        sql = result['candidates'][0]['content']['parts'][0]['text'].strip()
                        return sql.replace('```sql', '').replace('```', '').strip()
                    elif 'promptFeedback' in result:
                        raise Exception(f"Google API feedback: {result['promptFeedback']}")
                        
                elif self.current_provider == 'anthropic':
                    if 'completion' in result:
                        sql = result['completion'].strip()
                        return sql.replace('```sql', '').replace('```', '').strip()
                        
                raise Exception(f"Invalid response format from {self.current_provider} API")
                
            except requests.exceptions.RequestException as e:
                last_error = str(e)
                if response.status_code != 429:  # If not a rate limit error
                    break  # Don't retry other errors
                    
        raise Exception(f"Cloud provider failed after {retries} attempts: {last_error}")
        
        # Format request based on provider
        if self.current_provider == 'openai':
            headers = {
                "Authorization": f"Bearer {config['api_key']}",
                "Content-Type": "application/json"
            }
            payload = {
                "model": config['model'],
                "messages": [
                    {"role": "system", "content": "You are an SQL expert. Generate ONLY the SQL query without any markdown formatting, comments, or explanations. Do not include ```sql or ``` tags. Return just the raw SQL query."},
                    {"role": "user", "content": prompt}
                ],
                "temperature": 0.1,
                "max_tokens": 500,
                "frequency_penalty": 0,
                "presence_penalty": 0
            }
            
        elif self.current_provider == 'google':
            headers = {
                "Content-Type": "application/json",
                "X-goog-api-key": config['api_key']
            }
            
            payload = {
                "contents": [{
                    "parts": [{
                        "text": f"You are an SQL expert. Generate a SQL query for this request: {prompt}"
                    }]
                }]
            }
            
        elif self.current_provider == 'anthropic':
            headers = {
                "x-api-key": config['api_key'],
                "Content-Type": "application/json"
            }
            payload = {
                "model": config['model'],
                "prompt": f"\n\nHuman: {prompt}\n\nAssistant: Here's the SQL query:",
                "max_tokens_to_sample": 150,
                "temperature": 0.1
            }

        response = requests.post(config['api_url'], headers=headers, json=payload)
        response.raise_for_status()
        
        result = response.json()
        
        # Extract response based on provider
        if self.current_provider == 'openai':
            if 'choices' in result and len(result['choices']) > 0:
                sql = result['choices'][0]['message']['content'].strip()
                # Remove any markdown formatting or code blocks
                sql = sql.replace('```sql', '').replace('```', '').strip()
                return sql
                
        elif self.current_provider == 'google':
            if 'candidates' in result:
                return result['candidates'][0]['content']['parts'][0]['text'].strip()
            elif 'promptFeedback' in result:
                raise Exception(f"Google API feedback: {result['promptFeedback']}")
                
        elif self.current_provider == 'anthropic':
            if 'completion' in result:
                return result['completion'].strip()
                
        raise Exception(f"Invalid response format from {self.current_provider} API")

    def _try_local_generation(self, prompt):
        """
        Attempts to generate SQL using local models (Ollama).
        """
        try:
            payload = {
                "model": self.model_name,
                "prompt": prompt,
                "stream": False,
                "temperature": 0.1,
                "top_p": 0.9,
                "num_predict": 200,
                "stop": [";", "\n\n"]
            }
            
            response = requests.post(self.providers['local']['api_url'], json=payload)
            response.raise_for_status()
            
            result = response.json()
            if 'response' in result:
                return result['response'].strip()
            raise Exception("Invalid response format from local API")
                
        except requests.RequestException as e:
            # If the first model fails, try the fallback model
            if len(self.models) > 1 and self.model_name != self.models[1]:
                print(f"Primary model failed, trying fallback model: {self.models[1]}")
                self.model_name = self.models[1]
                return self._try_local_generation(prompt)
            raise Exception(f"Local model failed: {str(e)}")

    def get_models(self):
        return self.models