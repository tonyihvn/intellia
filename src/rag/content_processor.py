from typing import List, Dict, Union, Tuple
import requests
from bs4 import BeautifulSoup
import logging
from ..llm.client import LLMClient
import json
import os
import mimetypes
from pathlib import Path
import re
from urllib.parse import urlparse

class ContentProcessor:
    # Supported file types (add more as needed)
    SUPPORTED_EXTENSIONS = {
        '.txt', '.md', '.json', '.csv', '.yml', '.yaml', 
        '.html', '.htm', '.xml', '.doc', '.docx', '.pdf'
    }
    
    # Supported MIME types for URLs
    SUPPORTED_MIMETYPES = {
        'text/plain', 'text/html', 'text/markdown', 'text/csv',
        'application/json', 'application/xml', 'application/pdf',
        'application/msword', 'application/vnd.openxmlformats-officedocument.wordprocessingml.document'
    }

    def __init__(self):
        """Initialize the content processor with an LLM client."""
        self.llm = LLMClient()

    def validate_url(self, url: str) -> Tuple[bool, str]:
        """Validate URL accessibility and content type.
        
        Args:
            url: The URL to validate
            
        Returns:
            Tuple[bool, str]: (is_valid, error_message)
        """
        try:
            # Basic URL validation
            parsed = urlparse(url)
            if not all([parsed.scheme, parsed.netloc]):
                return False, "Invalid URL format"

            # Check URL accessibility and content type
            response = requests.head(url, allow_redirects=True)
            if response.status_code != 200:
                return False, f"URL not accessible (status {response.status_code})"

            content_type = response.headers.get('content-type', '').split(';')[0].lower()
            if content_type not in self.SUPPORTED_MIMETYPES:
                return False, f"Unsupported content type: {content_type}"

            return True, ""
        except Exception as e:
            return False, f"Error validating URL: {str(e)}"

    def validate_file(self, file_path: Union[str, Path]) -> Tuple[bool, str]:
        """Validate file type and accessibility.
        
        Args:
            file_path: Path to the file
            
        Returns:
            Tuple[bool, str]: (is_valid, error_message)
        """
        try:
            file_path = Path(file_path)
            if not file_path.exists():
                return False, "File does not exist"
            
            if not file_path.is_file():
                return False, "Path is not a file"
            
            extension = file_path.suffix.lower()
            if extension not in self.SUPPORTED_EXTENSIONS:
                return False, f"Unsupported file type: {extension}"
            
            mime_type, _ = mimetypes.guess_type(str(file_path))
            if mime_type and mime_type.split(';')[0] not in self.SUPPORTED_MIMETYPES:
                return False, f"Unsupported MIME type: {mime_type}"
            
            # Try to open the file to verify read access
            try:
                with open(file_path, 'r', encoding='utf-8') as f:
                    f.read(1)  # Try to read 1 byte
                return True, ""
            except Exception as e:
                return False, f"Cannot read file: {str(e)}"
                
        except Exception as e:
            return False, f"Error validating file: {str(e)}"

    def process_url(self, url: str) -> List[Dict[str, str]]:
        """Process a URL and return chunks of analyzed content.
        
        Args:
            url: The URL to process
            
        Returns:
            List of dicts with 'content' and 'metadata' keys
        """
        try:
            response = requests.get(url)
            response.raise_for_status()
            
            soup = BeautifulSoup(response.text, 'html.parser')
            # Extract main content, removing scripts, styles, etc.
            for tag in soup(['script', 'style', 'meta', 'noscript']):
                tag.decompose()
            
            raw_text = soup.get_text(separator='\n', strip=True)
            return self._process_text(raw_text, metadata={'source': url, 'type': 'url'})
            
        except Exception as e:
            logging.error(f"Error processing URL {url}: {str(e)}")
            return []

    def process_file(self, file_path: Union[str, Path]) -> List[Dict[str, str]]:
        """Process a file and return chunks of analyzed content.
        
        Args:
            file_path: Path to the file to process
            
        Returns:
            List of dicts with 'content' and 'metadata' keys
        """
        try:
            file_path = Path(file_path)
            if not file_path.exists():
                raise FileNotFoundError(f"File not found: {file_path}")

            logging.info(f"Starting file processing: {file_path}")
            # Handle different file types
            extension = file_path.suffix.lower()
            content = ""
            
            if extension == '.pdf':
                logging.info("Processing PDF file...")
                try:
                    import PyPDF2
                    with open(file_path, 'rb') as f:
                        pdf_reader = PyPDF2.PdfReader(f)
                        content = []
                        total_pages = len(pdf_reader.pages)
                        logging.info(f"PDF has {total_pages} pages")
                        for page in range(total_pages):
                            page_content = pdf_reader.pages[page].extract_text()
                            if page_content.strip():
                                content.append(page_content)
                            logging.info(f"Processed page {page + 1}/{total_pages}")
                        content = "\n\n".join(content)
                        logging.info(f"Extracted {len(content)} characters of text from PDF")
                except ImportError:
                    error_msg = "PyPDF2 is not installed. Please install it to process PDF files."
                    logging.error(error_msg)
                    raise ImportError(error_msg)
                except Exception as e:
                    error_msg = f"Error processing PDF: {str(e)}"
                    logging.error(error_msg)
                    raise Exception(error_msg)
            else:
                # For text-based files
                logging.info(f"Processing text file with {extension} extension...")
                try:
                    with open(file_path, 'r', encoding='utf-8') as f:
                        content = f.read()
                except UnicodeDecodeError:
                    logging.info("UTF-8 decode failed, trying latin-1 encoding...")
                    # Try different encodings if UTF-8 fails
                    with open(file_path, 'r', encoding='latin-1') as f:
                        content = f.read()

            if not content or not content.strip():
                error_msg = "No content could be extracted from the file"
                logging.error(error_msg)
                raise ValueError(error_msg)

            logging.info(f"Successfully extracted content, length: {len(content)} characters")
            logging.info("Processing content with LLM for rule extraction...")
            
            result = self._process_text(content, metadata={
                'source': file_path.name,
                'type': 'file',
                'path': str(file_path)
            })
            
            logging.info(f"LLM processing complete. Extracted {len(result)} chunks with rules")
            return result

        except Exception as e:
            logging.error(f"Error processing file {file_path}: {str(e)}")
            return []

    def _process_text(self, text: str, metadata: Dict = None) -> List[Dict[str, str]]:
        """Use LLM to analyze and chunk text intelligently.
        
        Args:
            text: The text to process
            metadata: Optional metadata to include with each chunk
            
        Returns:
            List of dicts with 'content' and 'metadata' keys
        """
        # Ask LLM to analyze and chunk the content
        prompt = f"""Analyze the following content and extract business rules, policies, and important domain knowledge.
Focus on identifying:
1. Explicit business rules and policies
2. Implicit rules from process descriptions
3. Constraints and requirements
4. Decision criteria and conditions
5. Relationships between entities
6. Validation rules and data requirements

For each meaningful segment, provide:
- The original content for context
- A list of clear, actionable business rules extracted from that content

Format the output as a JSON array of objects with 'content' and 'rules' properties, where each rule should be specific and self-contained.

Content to analyze:
{text[:8000]}

Example output format:
[
  {{
    "content": "Customer credit limit approval process: For new customers, credit limits up to $10,000 can be approved by sales managers. Limits between $10,001 and $50,000 require regional director approval. Any limit above $50,000 needs CFO approval.",
    "rules": [
      "Rule: Sales managers can approve credit limits up to $10,000 for new customers",
      "Rule: Credit limits between $10,001 and $50,000 require regional director approval",
      "Rule: Credit limits above $50,000 require CFO approval"
    ]
  }}
]

Ensure each rule is:
- Clear and specific
- Actionable
- Self-contained
- Written in a consistent format
"""
        try:
            response = self.llm.generate(prompt)
            # Extract JSON array from response
            match = re.search(r'\[[\s\S]*\]', response)
            if not match:
                logging.error("No valid JSON array found in LLM response")
                return []
                
            chunks = json.loads(match.group())
            results = []
            
            for chunk in chunks:
                if not isinstance(chunk, dict):
                    continue
                    
                content = chunk.get('content', '').strip()
                rules = chunk.get('rules', [])
                
                if content and rules:
                    chunk_metadata = {
                        'rules': rules,
                        **(metadata or {})
                    }
                    results.append({
                        'content': content,
                        'metadata': chunk_metadata
                    })
            
            return results
            
        except Exception as e:
            logging.error(f"Error processing text with LLM: {str(e)}")
            return []