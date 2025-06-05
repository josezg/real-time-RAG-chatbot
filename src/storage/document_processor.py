from pathlib import Path
from typing import Dict, List
from llama_index.core import (
    VectorStoreIndex,
    SimpleDirectoryReader,
    StorageContext,
)
from llama_index.vector_stores.chroma import ChromaVectorStore
import chromadb
from llama_index.core.prompts import PromptTemplate
from llama_index.core.response import Response
import hashlib
import json
import os

class DocumentProcessor:
    def __init__(self, docs_dir: str, persist_dir: str):
        self.docs_dir = Path(docs_dir)
        self.persist_dir = Path(persist_dir)
        self.metadata_file = self.persist_dir / "document_metadata.json"

        # Create directories if they don't exist
        self.docs_dir.mkdir(parents=True, exist_ok=True)
        self.persist_dir.mkdir(parents=True, exist_ok=True)

        # Initialize ChromaDB
        self.db = chromadb.PersistentClient(path=str(self.persist_dir))
        self.chroma_collection = self.db.get_or_create_collection("documents_collection")
        self.vector_store = ChromaVectorStore(chroma_collection=self.chroma_collection)
        self.storage_context = StorageContext.from_defaults(vector_store=self.vector_store)
        
        self.index = None
        self.query_engine = None
        self.document_metadata = self._load_document_metadata()

        # Define the custom prompt template
        self.qa_tmpl_str = (
            "You are a helpful AI assistant. "
            "Use ONLY the following context to answer the question. "
            "Do NOT use any prior knowledge. "
            "If the answer is not in the context, clearly state 'I don't know based on the provided information.'\n\n"
            "Context:\n"
            "---------------------\n"
            "{context_str}\n"
            "---------------------\n"
            "Question: {query_str}\n"
            "Answer: "
        )
        self.qa_tmpl = PromptTemplate(self.qa_tmpl_str)

    def _load_document_metadata(self) -> Dict:
        """Load document metadata from file."""
        if self.metadata_file.exists():
            try:
                with open(self.metadata_file, 'r') as f:
                    return json.load(f)
            except Exception as e:
                print(f"Error loading metadata: {e}")
        return {}

    def _save_document_metadata(self):
        """Save document metadata to file."""
        try:
            with open(self.metadata_file, 'w') as f:
                json.dump(self.document_metadata, f, indent=2)
        except Exception as e:
            print(f"Error saving metadata: {e}")

    def _get_file_hash(self, file_path: Path) -> str:
        """Calculate hash of file content."""
        try:
            with open(file_path, 'rb') as f:
                content = f.read()
                return hashlib.md5(content).hexdigest()
        except Exception as e:
            print(f"Error calculating hash for {file_path}: {e}")
            return ""

    def _get_current_documents_info(self) -> Dict[str, Dict]:
        """Get current state of all documents in the directory."""
        current_docs = {}
        
        if not self.docs_dir.exists():
            return current_docs
            
        for file_path in self.docs_dir.iterdir():
            if file_path.is_file() and file_path.suffix.lower() in ['.txt', '.pdf', '.docx', '.md']:
                try:
                    stat = file_path.stat()
                    file_hash = self._get_file_hash(file_path)
                    
                    current_docs[str(file_path)] = {
                        'hash': file_hash,
                        'modified_time': stat.st_mtime,
                        'size': stat.st_size
                    }
                except Exception as e:
                    print(f"Error processing {file_path}: {e}")
                    
        return current_docs

    def _identify_changes(self, current_docs: Dict[str, Dict]) -> tuple:
        """Identify what documents have changed, been added, or removed."""
        added = []
        modified = []
        removed = []
        
        # Find removed files
        for old_file in self.document_metadata:
            if old_file not in current_docs:
                removed.append(old_file)
        
        # Find added or modified files
        for file_path, file_info in current_docs.items():
            if file_path not in self.document_metadata:
                added.append(file_path)
            elif self.document_metadata[file_path]['hash'] != file_info['hash']:
                modified.append(file_path)
        
        return added, modified, removed

    def _remove_documents_from_index(self, file_paths: List[str]):
        """Remove specific documents from the vector store."""
        try:
            # ChromaDB delete by metadata filter
            for file_path in file_paths:
                file_name = Path(file_path).name
                # Delete documents with matching file_name in metadata
                self.chroma_collection.delete(
                    where={"file_name": file_name}
                )
                print(f"Removed {file_name} from index")
        except Exception as e:
            print(f"Error removing documents from index: {e}")

    def index_documents(self, force_full_reindex: bool = False):
        """Index documents with smart incremental updates."""
        print(f"Checking documents in {self.docs_dir}...")
        
        current_docs = self._get_current_documents_info()
        
        if not current_docs:
            print("No documents found.")
            self.index = None
            self.query_engine = None
            return
        
        if not force_full_reindex:
            added, modified, removed = self._identify_changes(current_docs)
            
            # If no changes, skip reindexing
            if not added and not modified and not removed:
                print("No document changes detected. Skipping reindex.")
                # Still need to ensure index exists
                if self.index is None:
                    self._rebuild_index_from_existing()
                return
            
            print(f"Document changes detected:")
            print(f"  Added: {len(added)} files")
            print(f"  Modified: {len(modified)} files") 
            print(f"  Removed: {len(removed)} files")
            
            # Handle removals
            if removed:
                self._remove_documents_from_index(removed)
            
            # Handle additions and modifications
            if added or modified:
                files_to_process = added + modified
                self._process_specific_files(files_to_process)
        else:
            print("Performing full reindex...")
            self._full_reindex()
        
        # Update metadata
        self.document_metadata = current_docs
        self._save_document_metadata()
        
        # Rebuild query engine
        self._create_query_engine()
        print("Indexing complete.")

    def _process_specific_files(self, file_paths: List[str]):
        """Process only specific files for indexing."""
        try:
            # Remove modified files from index first
            modified_files = [f for f in file_paths if f in self.document_metadata]
            if modified_files:
                self._remove_documents_from_index(modified_files)
            
            # Load and index the specific files
            documents = []
            for file_path in file_paths:
                try:
                    file_docs = SimpleDirectoryReader(
                        input_files=[file_path]
                    ).load_data()
                    documents.extend(file_docs)
                    print(f"Loaded {Path(file_path).name}")
                except Exception as e:
                    print(f"Error loading {file_path}: {e}")
            
            if documents:
                if self.index is None:
                    # Create new index
                    self.index = VectorStoreIndex.from_documents(
                        documents,
                        storage_context=self.storage_context,
                        show_progress=True,
                    )
                else:
                    # Add to existing index
                    for doc in documents:
                        self.index.insert(doc)
                
        except Exception as e:
            print(f"Error processing specific files: {e}")
            # Fallback to full reindex
            self._full_reindex()

    def _full_reindex(self):
        """Perform a complete reindex of all documents."""
        try:
            # Clear existing index
            self.chroma_collection.delete(where={})
            
            # Load all documents
            documents = SimpleDirectoryReader(str(self.docs_dir)).load_data()
            print(f"Found {len(documents)} documents for full reindex.")
            
            if documents:
                self.index = VectorStoreIndex.from_documents(
                    documents,
                    storage_context=self.storage_context,
                    show_progress=True,
                )
            else:
                self.index = None
                
        except Exception as e:
            print(f"Error during full reindex: {e}")
            self.index = None

    def _rebuild_index_from_existing(self):
        """Rebuild index object from existing vector store."""
        try:
            self.index = VectorStoreIndex.from_vector_store(
                self.vector_store,
                storage_context=self.storage_context
            )
            print("Rebuilt index from existing vector store.")
        except Exception as e:
            print(f"Error rebuilding index: {e}")
            # Fallback to full reindex
            self._full_reindex()

    def _create_query_engine(self):
        """Create or recreate the query engine."""
        if self.index:
            self.query_engine = self.index.as_query_engine(
                response_mode="compact",
                text_qa_template=self.qa_tmpl,
                verbose=True
            )
        else:
            self.query_engine = None

    def query(self, user_query: str) -> Dict:
        """Execute a query and return the response with source information."""
        if not self.query_engine:
            return {
                "response_text": "No documents indexed yet. Please ensure documents are in the specified folder and the index is built.",
                "source_info": []
            }
        
        try:
            llama_response: Response = self.query_engine.query(user_query)
            
            source_info = []
            if llama_response.source_nodes:
                for node in llama_response.source_nodes:
                    file_name = node.metadata.get('file_name', 'N/A')
                    page_label = node.metadata.get('page_label', 'N/A')
                    
                    source_info.append({
                        "file_name": file_name,
                        "page_label": page_label,
                    })
            
            return {
                "response_text": llama_response.response,
                "source_info": source_info
            }
            
        except Exception as e:
            print(f"Error during query: {e}")
            return {
                "response_text": f"Error processing query: {str(e)}",
                "source_info": []
            }

    def force_full_reindex(self):
        """Force a complete reindex of all documents."""
        print("Forcing full reindex...")
        self.index_documents(force_full_reindex=True)

    def get_index_stats(self) -> Dict:
        """Get statistics about the current index."""
        try:
            collection_count = self.chroma_collection.count()
            return {
                "total_chunks": collection_count,
                "total_documents": len(self.document_metadata),
                "has_index": self.index is not None,
                "has_query_engine": self.query_engine is not None
            }
        except Exception as e:
            print(f"Error getting index stats: {e}")
            return {
                "total_chunks": 0,
                "total_documents": 0,
                "has_index": False,
                "has_query_engine": False
            }