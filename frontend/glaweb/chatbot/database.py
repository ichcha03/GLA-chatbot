import faiss
import json
import os
import pickle
import numpy as np
from sentence_transformers import SentenceTransformer
from django.conf import settings

class GLADatabase:
    def __init__(self):
        print("Initializing GLA Database...")
        try:
            self.embedding_model = SentenceTransformer('all-MiniLM-L6-v2')
            print("Sentence transformer model loaded successfully")
        except Exception as e:
            print(f"Error loading sentence transformer: {e}")
            self.embedding_model = None
            
        self.embedding_dim = 384  # Dimension for all-MiniLM-L6-v2
        self.index = faiss.IndexFlatIP(self.embedding_dim)  # Inner product for cosine similarity
        self.documents = []
        self.metadatas = []
        self.ids = []
        self.db_path = "./gla_faiss_db"
        self.index_path = os.path.join(self.db_path, "faiss_index.bin")
        self.metadata_path = os.path.join(self.db_path, "metadata.pkl")
        
        # Create directory if it doesn't exist
        os.makedirs(self.db_path, exist_ok=True)
        
        # Load existing database if available
        self._load_database()
        
    def _load_database(self):
        """Load existing FAISS database if available"""
        try:
            if os.path.exists(self.index_path) and os.path.exists(self.metadata_path):
                # Load FAISS index
                self.index = faiss.read_index(self.index_path)
                
                # Load metadata
                with open(self.metadata_path, 'rb') as f:
                    data = pickle.load(f)
                    self.documents = data['documents']
                    self.metadatas = data['metadatas']
                    self.ids = data['ids']
                
                print(f"Loaded existing database with {len(self.documents)} documents")
        except Exception as e:
            print(f"Error loading database: {e}")
            # Reset to empty state
            self.index = faiss.IndexFlatIP(self.embedding_dim)
            self.documents = []
            self.metadatas = []
            self.ids = []
    
    def _save_database(self):
        """Save FAISS database to disk"""
        try:
            # Save FAISS index
            faiss.write_index(self.index, self.index_path)
            
            # Save metadata
            with open(self.metadata_path, 'wb') as f:
                pickle.dump({
                    'documents': self.documents,
                    'metadatas': self.metadatas,
                    'ids': self.ids
                }, f)
            
            print("Database saved successfully")
        except Exception as e:
            print(f"Error saving database: {e}")
        
    def initialize_database(self):
        """Initialize the database with hardcoded JSON data"""
        # Check if database is already populated
        if len(self.documents) > 0:
            print("Database already initialized")
            return
            
        # Load JSON data
        json_path = os.path.join(os.path.dirname(__file__), '..', 'data', 'gla_data.json')
        with open(json_path, 'r', encoding='utf-8') as file:
            data = json.load(file)
        
        documents = []
        metadatas = []
        ids = []
        
        # Process admissions data
        for key, value in data['admissions'].items():
            if isinstance(value, dict):
                for subkey, subvalue in value.items():
                    doc_text = f"Admissions {key} {subkey}: {subvalue}"
                    documents.append(doc_text)
                    metadatas.append({"category": "admissions", "subcategory": key, "type": subkey})
                    ids.append(f"admissions_{key}_{subkey}")
            else:
                doc_text = f"Admissions {key}: {value}"
                documents.append(doc_text)
                metadatas.append({"category": "admissions", "subcategory": key})
                ids.append(f"admissions_{key}")
        
        # Process academics data
        for key, value in data['academics'].items():
            if isinstance(value, list):
                doc_text = f"Academics {key}: {', '.join(value)}"
            else:
                doc_text = f"Academics {key}: {value}"
            documents.append(doc_text)
            metadatas.append({"category": "academics", "subcategory": key})
            ids.append(f"academics_{key}")
        
        # Process fees data
        for key, value in data['fees'].items():
            if isinstance(value, dict):
                for subkey, subvalue in value.items():
                    doc_text = f"Fees {key} {subkey}: {subvalue}"
                    documents.append(doc_text)
                    metadatas.append({"category": "fees", "subcategory": key, "type": subkey})
                    ids.append(f"fees_{key}_{subkey}")
            elif isinstance(value, list):
                doc_text = f"Fees {key}: {', '.join(value)}"
                documents.append(doc_text)
                metadatas.append({"category": "fees", "subcategory": key})
                ids.append(f"fees_{key}")
            else:
                doc_text = f"Fees {key}: {value}"
                documents.append(doc_text)
                metadatas.append({"category": "fees", "subcategory": key})
                ids.append(f"fees_{key}")
        
        # Process placements data
        for key, value in data['placements'].items():
            if isinstance(value, dict):
                for subkey, subvalue in value.items():
                    doc_text = f"Placements {key} {subkey}: {subvalue}"
                    documents.append(doc_text)
                    metadatas.append({"category": "placements", "subcategory": key, "type": subkey})
                    ids.append(f"placements_{key}_{subkey}")
            elif isinstance(value, list):
                doc_text = f"Placements {key}: {', '.join(value)}"
                documents.append(doc_text)
                metadatas.append({"category": "placements", "subcategory": key})
                ids.append(f"placements_{key}")
            else:
                doc_text = f"Placements {key}: {value}"
                documents.append(doc_text)
                metadatas.append({"category": "placements", "subcategory": key})
                ids.append(f"placements_{key}")
        
        # Process facilities data
        for key, value in data['facilities'].items():
            if isinstance(value, dict):
                for subkey, subvalue in value.items():
                    if isinstance(subvalue, list):
                        doc_text = f"Facilities {key} {subkey}: {', '.join(subvalue)}"
                    else:
                        doc_text = f"Facilities {key} {subkey}: {subvalue}"
                    documents.append(doc_text)
                    metadatas.append({"category": "facilities", "subcategory": key, "type": subkey})
                    ids.append(f"facilities_{key}_{subkey}")
            elif isinstance(value, list):
                doc_text = f"Facilities {key}: {', '.join(value)}"
                documents.append(doc_text)
                metadatas.append({"category": "facilities", "subcategory": key})
                ids.append(f"facilities_{key}")
            else:
                doc_text = f"Facilities {key}: {value}"
                documents.append(doc_text)
                metadatas.append({"category": "facilities", "subcategory": key})
                ids.append(f"facilities_{key}")
        
        # Process campus life data
        for key, value in data['campus_life'].items():
            if isinstance(value, list):
                doc_text = f"Campus life {key}: {', '.join(value)}"
            else:
                doc_text = f"Campus life {key}: {value}"
            documents.append(doc_text)
            metadatas.append({"category": "campus_life", "subcategory": key})
            ids.append(f"campus_life_{key}")
        
        # Process contact data
        for key, value in data['contact'].items():
            doc_text = f"Contact {key}: {value}"
            documents.append(doc_text)
            metadatas.append({"category": "contact", "subcategory": key})
            ids.append(f"contact_{key}")
        
        # Generate embeddings and add to FAISS index
        print("Generating embeddings...")
        embeddings = self.embedding_model.encode(documents)
        
        # Normalize embeddings for cosine similarity
        faiss.normalize_L2(embeddings)
        
        # Add to FAISS index
        self.index.add(embeddings.astype('float32'))
        
        # Store documents and metadata
        self.documents = documents
        self.metadatas = metadatas
        self.ids = ids
        
        # Save to disk
        self._save_database()
        
        print(f"Database initialized with {len(documents)} documents")
    
    def search_similar(self, query, n_results=5):
        """Search for similar documents in the database"""
        try:
            if len(self.documents) == 0:
                print("No documents in database")
                return None
                
            if self.embedding_model is None:
                print("Embedding model not available")
                return None
            
            print(f"Searching for: {query}")
            # Generate embedding for query
            query_embedding = self.embedding_model.encode([query])
            faiss.normalize_L2(query_embedding)
            
            # Search in FAISS index
            scores, indices = self.index.search(query_embedding.astype('float32'), min(n_results, len(self.documents)))
            
            # Format results similar to ChromaDB format
            result_documents = []
            result_metadatas = []
            result_ids = []
            result_distances = []
            
            for i, idx in enumerate(indices[0]):
                if idx != -1:  # Valid index
                    result_documents.append(self.documents[idx])
                    result_metadatas.append(self.metadatas[idx])
                    result_ids.append(self.ids[idx])
                    result_distances.append(float(scores[0][i]))
            
            print(f"Found {len(result_documents)} relevant documents")
            return {
                'documents': [result_documents],
                'metadatas': [result_metadatas],
                'ids': [result_ids],
                'distances': [result_distances]
            }
            
        except Exception as e:
            print(f"Error searching database: {e}")
            return None
    
    def get_all_documents(self):
        """Get all documents from the database"""
        try:
            return {
                'documents': self.documents,
                'metadatas': self.metadatas,
                'ids': self.ids
            }
        except Exception as e:
            print(f"Error getting documents: {e}")
            return None

# Initialize database when module is imported
db = GLADatabase()
