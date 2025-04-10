import os
from typing import List, Dict, Any, Optional, Tuple  # Updated import
from langchain.docstore.document import Document
from langchain_community.embeddings import OpenAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_community.document_loaders import DirectoryLoader, TextLoader, CSVLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
# Configure logging
import logging
import json, bcrypt  # Add bcrypt for password hashing
import re  # Add this line
import streamlit as st

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class KnowledgeAgent:
    """Intelligent knowledge agent with secure order verification"""
    
    def __init__(
        self,
        knowledge_base_dir: str = "knowledge_base/",
        index_path: str = "faiss_index",
        chunk_size: int = 1000,
        chunk_overlap: int = 100
    ):
        self.knowledge_base_dir = knowledge_base_dir
        self.index_path = index_path
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.vector_store = self._initialize_vector_store()
        self.order_data = []
        self.product_data = []
        self._load_structured_data()

    @st.cache_resource  # Cache the FAISS vector store
    def _initialize_vector_store(_self):
        """Reliable vector store initialization from original working code"""
        try:
            if os.path.exists(_self.index_path):
                logger.info("Loading existing FAISS index")
                return FAISS.load_local(
                    _self.index_path,
                    OpenAIEmbeddings(),
                    allow_dangerous_deserialization=True
                )

            logger.info("Creating new FAISS index")
            text_loader = DirectoryLoader(_self.knowledge_base_dir, glob="**/*.txt", loader_cls=TextLoader)
            csv_loader = DirectoryLoader(_self.knowledge_base_dir, glob="**/*.csv", loader_cls=CSVLoader)
            documents = text_loader.load() + csv_loader.load()

            splitter = RecursiveCharacterTextSplitter(
                chunk_size=_self.chunk_size,
                chunk_overlap=_self.chunk_overlap
            )
            chunks = splitter.split_documents(documents)

            vector_store = FAISS.from_documents(chunks, OpenAIEmbeddings())
            vector_store.save_local(_self.index_path)
            return vector_store

        except Exception as e:
            logger.error(f"Vector store init failed: {str(e)}")
            return None

    def _load_structured_data(self):
        """Load and validate structured order/product data from files"""
        self.structured_files = []  # Track files we've processed
        
        for root, _, files in os.walk(self.knowledge_base_dir):
            for file in files:
                file_path = os.path.join(root, file)
                try:
                    content = self._load_file_content(file_path)
                    if not content:
                        continue

                    # Check if file contains order data
                    if self._is_order_file(content):
                        self.structured_files.append(file_path)
                        orders = self._parse_structured_data(content, 'order')
                        self.order_data.extend(self._validate_orders(orders))
                        logger.info(f"Loaded {len(orders)} orders from {file}")

                    # Check if file contains product data    
                    elif self._is_product_file(content):
                        self.structured_files.append(file_path)
                        products = self._parse_structured_data(content, 'product')
                        self.product_data.extend(self._validate_products(products))
                        logger.info(f"Loaded {len(products)} products from {file}")

                except Exception as e:
                    logger.error(f"Failed to process {file}: {str(e)}")

    @st.cache_data  # Cache file content to avoid reloading
    def _load_file_content(_self, path: str) -> str:
        """Secure file content loader with validation"""
        try:
            with open(path, 'r', encoding='utf-8', errors='ignore') as f:
                content = f.read().strip()
                if len(content) < 50:  # Skip small files
                    logger.warning(f"Skipped short file: {path}")
                    return ""
                return content
        except Exception as e:
            logger.error(f"File read error ({path}): {str(e)}")
            return ""

    def _is_order_file(self, content: str) -> bool:
        """Detect order files using strict pattern matching"""
        required_patterns = [
            r'\border\b', r'\b(?:customer|user)\b', 
            r'\bemail\b', r'\bproducts?\b', r'\bstatus\b'
        ]
        return sum(re.search(p, content, re.IGNORECASE) is not None 
                   for p in required_patterns) >= 3

    def _is_product_file(self, content: str) -> bool:
        """Detect product files using strict pattern matching"""
        required_patterns = [
            r'\bproduct\b', r'\b(?:name|title)\b',
            r'\b(?:description|details?)\b', r'\bprice\b'
        ]
        return sum(re.search(p, content, re.IGNORECASE) is not None 
                   for p in required_patterns) >= 3
    
    def _is_csv(self, content: str) -> bool:
        """Improved CSV detection"""
        return bool(re.search(r'^[\w\s]+,[\w\s]+([,\n]|$)', content.strip()))

    @st.cache_data  # Cache parsed structured data
    def _parse_structured_data(_self, content: str, data_type: str) -> List[Dict]:
        """Parse structured data from multiple formats"""
        try:
            # Attempt JSON parsing
            if content.strip().startswith(('{', '[')):
                data = json.loads(content)
                return data if isinstance(data, list) else [data]

            # Attempt CSV parsing
            if _self._is_csv(content):
                return _self._parse_csv(content, data_type)

            # Fallback to key-value parsing
            return _self._parse_key_value(content, data_type)

        except Exception as e:
            logger.error(f"Parse error: {str(e)}")
            return []

    @st.cache_data  # Cache validated orders
    def _validate_orders(_self, orders: List[Dict]) -> List[Dict]:
        """Validate order structure and mask sensitive data"""
        valid_orders = []
        for order in orders:
            try:
                # Validate required fields
                if not all(key in order for key in ['order_id', 'email', 'products']):
                    continue
                
                # Mask email before storage
                order['email'] = _self._mask_email(order.get('email', ''))
                valid_orders.append(order)
                
            except Exception as e:
                logger.warning(f"Invalid order format: {str(e)}")
        return valid_orders

    @st.cache_data  # Cache validated products
    def _validate_products(_self, products: List[Dict]) -> List[Dict]:
        """Validate product structure and sanitize data"""
        valid_products = []
        for product in products:
            try:
                # Validate required fields
                if not all(key in product for key in ['product_id', 'name']):
                    continue
                
                # Sanitize description
                product['description'] = product.get('description', '')[:500]
                valid_products.append(product)
                
            except Exception as e:
                logger.warning(f"Invalid product format: {str(e)}")
        return valid_products

    @st.cache_data  # Cache context retrieval results
    def get_context(_self, query: str, max_docs: int = 10) -> Dict[str, Any]:
        """Working context retrieval from original implementation"""
        if not _self.vector_store:
            return {"text": "Knowledge base unavailable", "sources": []}

        try:
            results = _self.vector_store.similarity_search_with_score(query, k=max_docs)
            return {
                "text": "\n\n".join([doc.page_content for doc, _ in results]),
                "sources": [{
                    "content": doc.page_content,
                    "score": float(score),
                    "metadata": doc.metadata
                } for doc, score in results]
            }
        except Exception as e:
            logger.error(f"Search error: {str(e)}")
            return {"text": "Search failed", "sources": []}
        
    def _parse_key_value(self, content: str, data_type: str) -> List[Dict]:
        """Parse key-value pairs from text content"""
        parsed_data = []
        current_entry = {}
        
        for line in content.split('\n'):
            line = line.strip()
            if not line:
                if current_entry:
                    parsed_data.append(current_entry)
                    current_entry = {}
                continue
                
            if ':' in line:
                key, value = line.split(':', 1)
                current_entry[key.strip()] = value.strip()
            elif '=' in line:
                key, value = line.split('=', 1)
                current_entry[key.strip()] = value.strip()
        
        if current_entry:
            parsed_data.append(current_entry)
            
        return parsed_data

    def _parse_csv(self, content: str, data_type: str) -> List[Dict]:
        """Parse CSV content into structured data"""
        from io import StringIO
        import csv
        
        parsed_data = []
        reader = csv.DictReader(StringIO(content))
        
        for row in reader:
            # Convert numeric fields
            converted_row = {}
            for key, value in row.items():
                if key.lower() in ['price', 'score', 'rating']:
                    try:
                        converted_row[key] = float(value)
                    except ValueError:
                        converted_row[key] = value
                elif value.isdigit():
                    converted_row[key] = int(value)
                else:
                    converted_row[key] = value
            parsed_data.append(converted_row)
            
        return parsed_data