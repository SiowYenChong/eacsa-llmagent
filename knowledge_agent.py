import os
from typing import List, Dict, Any
from langchain_community.embeddings import OpenAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_community.document_loaders import DirectoryLoader, TextLoader, CSVLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter

class KnowledgeAgent:
    """
    Agent responsible for knowledge retrieval from vector database.
    Manages document loading, indexing, and similarity search.
    """
    
    def __init__(self, knowledge_base_dir: str = "knowledge_base/", 
                 index_path: str = "faiss_index"):
        """
        Initialize the knowledge agent.
        
        Args:
            knowledge_base_dir (str): Directory containing knowledge base documents
            index_path (str): Path to save/load the FAISS index
        """
        self.knowledge_base_dir = knowledge_base_dir
        self.index_path = index_path
        self.vector_store = self._initialize_vector_store()
        
    def _initialize_vector_store(self):
        """Initialize or load vector store from documents"""
        try:
            # Check if we have a pre-saved vector store
            if os.path.exists(self.index_path):
                embeddings = OpenAIEmbeddings()
                vector_store = FAISS.load_local(self.index_path, embeddings, allow_dangerous_deserialization=True)
                return vector_store

            # If not, create from scratch
            # Load documents from the knowledge_base directory
            text_loader = DirectoryLoader(self.knowledge_base_dir, glob="**/*.txt", loader_cls=TextLoader)
            text_documents = text_loader.load()

            csv_loader = DirectoryLoader(self.knowledge_base_dir, glob="**/*.csv", loader_cls=CSVLoader)
            csv_documents = csv_loader.load()

            documents = text_documents + csv_documents

            # Split documents into chunks
            text_splitter = RecursiveCharacterTextSplitter(
                chunk_size=1000,
                chunk_overlap=100
            )
            texts = text_splitter.split_documents(documents)

            # Create vector store
            embeddings = OpenAIEmbeddings()
            vector_store = FAISS.from_documents(texts, embeddings)

            # Save for future use
            vector_store.save_local(self.index_path)

            return vector_store
        except Exception as e:
            print(f"Error initializing vector store: {str(e)}")
            return None
    
    def get_context(self, query: str, max_docs: int = 10) -> str:
        """
        Retrieve relevant context for a query from the vector store.
        
        Args:
            query (str): The user query to find relevant information for
            max_docs (int): Maximum number of documents to retrieve
            
        Returns:
            str: Concatenated relevant context
        """
        if self.vector_store is None:
            return "Knowledge base unavailable."

        relevant_docs = self.vector_store.similarity_search(query, k=max_docs)
        context = "\n\n".join([doc.page_content for doc in relevant_docs])
        return context
