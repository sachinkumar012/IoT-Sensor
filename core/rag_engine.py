import os
import json
import chromadb
from typing import List, Dict, Any, Optional
from sentence_transformers import SentenceTransformer
import pandas as pd
from pathlib import Path
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class CustomEmbeddingFunction:
    """Custom embedding function compatible with ChromaDB 0.4.16+"""
    
    def __init__(self, model):
        self.model = model
    
    def __call__(self, input):
        """Required interface method for ChromaDB 0.4.16+"""
        if isinstance(input, str):
            input = [input]
        embeddings = self.model.encode(input)
        return embeddings.tolist()

class RAGEngine:
    """
    Core RAG engine for processing maintenance manuals and building specifications
    """
    
    def __init__(self, persist_directory: str = "./data/chroma_db"):
        self.persist_directory = persist_directory
        self.client = chromadb.PersistentClient(path=persist_directory)
        
        # Initialize sentence transformer model
        self.embedding_model = SentenceTransformer('all-MiniLM-L6-v2')
        
        # Create or get collections with custom embedding function
        self.manuals_collection = self.client.get_or_create_collection(
            name="maintenance_manuals",
            metadata={"description": "Maintenance manuals and procedures"},
            embedding_function=CustomEmbeddingFunction(self.embedding_model)
        )
        
        self.specs_collection = self.client.get_or_create_collection(
            name="building_specs",
            metadata={"description": "Building specifications and technical documents"},
            embedding_function=CustomEmbeddingFunction(self.embedding_model)
        )
        
        logger.info("RAG Engine initialized successfully")
    
    def _get_embedding_function(self):
        """Create a custom embedding function to avoid onnxruntime dependency"""
        return CustomEmbeddingFunction(self.embedding_model)
    
    def chunk_text(self, text: str, chunk_size: int = 512, overlap: int = 50) -> List[str]:
        """
        Split text into overlapping chunks for better retrieval
        """
        words = text.split()
        chunks = []
        
        for i in range(0, len(words), chunk_size - overlap):
            chunk = " ".join(words[i:i + chunk_size])
            if chunk.strip():
                chunks.append(chunk)
        
        return chunks
    
    def add_documents(self, documents: List[Dict[str, Any]], collection_name: str = "manuals"):
        """
        Add documents to the vector database
        """
        collection = self.manuals_collection if collection_name == "manuals" else self.specs_collection
        
        for doc in documents:
            # Chunk the document
            chunks = self.chunk_text(doc['content'])
            
            # Generate embeddings and add to collection
            for i, chunk in enumerate(chunks):
                chunk_id = f"{doc['id']}_{i}"
                
                # Generate embedding
                embedding = self.embedding_model.encode(chunk).tolist()
                
                # Add to collection
                collection.add(
                    embeddings=[embedding],
                    documents=[chunk],
                    metadatas=[{
                        'doc_id': doc['id'],
                        'title': doc['title'],
                        'type': doc['type'],
                        'chunk_index': i,
                        'source': doc.get('source', 'unknown')
                    }],
                    ids=[chunk_id]
                )
        
        logger.info(f"Added {len(documents)} documents to {collection_name} collection")
    
    def search(self, query: str, collection_name: str = "manuals", n_results: int = 5) -> List[Dict[str, Any]]:
        """
        Search for relevant documents based on query
        """
        collection = self.manuals_collection if collection_name == "manuals" else self.specs_collection
        
        # Generate query embedding
        query_embedding = self.embedding_model.encode(query).tolist()
        
        # Search in collection
        results = collection.query(
            query_embeddings=[query_embedding],
            n_results=n_results,
            include=['documents', 'metadatas', 'distances']
        )
        
        # Format results
        formatted_results = []
        for i in range(len(results['documents'][0])):
            formatted_results.append({
                'content': results['documents'][0][i],
                'metadata': results['metadatas'][0][i],
                'distance': results['distances'][0][i],
                'relevance_score': 1 - results['distances'][0][i]  # Convert distance to similarity
            })
        
        return formatted_results
    
    def hybrid_search(self, query: str, n_results: int = 5) -> List[Dict[str, Any]]:
        """
        Search across both collections and merge results
        """
        manuals_results = self.search(query, "manuals", n_results)
        specs_results = self.search(query, "specs", n_results)
        
        # Combine and sort by relevance
        all_results = manuals_results + specs_results
        all_results.sort(key=lambda x: x['relevance_score'], reverse=True)
        
        return all_results[:n_results]
    
    def get_collection_stats(self) -> Dict[str, Any]:
        """
        Get statistics about the collections
        """
        manuals_count = self.manuals_collection.count()
        specs_count = self.specs_collection.count()
        
        return {
            'manuals_count': manuals_count,
            'specs_count': specs_count,
            'total_documents': manuals_count + specs_count
        }
    
    def clear_collections(self):
        """
        Clear all collections (useful for testing)
        """
        self.client.delete_collection("maintenance_manuals")
        self.client.delete_collection("building_specs")
        
        # Recreate collections with custom embedding function
        self.manuals_collection = self.client.create_collection(
            name="maintenance_manuals",
            metadata={"description": "Maintenance manuals and procedures"},
            embedding_function=CustomEmbeddingFunction(self.embedding_model)
        )
        
        self.specs_collection = self.client.create_collection(
            name="building_specs",
            metadata={"description": "Building specifications and technical documents"},
            embedding_function=CustomEmbeddingFunction(self.embedding_model)
        )
        
        logger.info("Collections cleared and recreated")
