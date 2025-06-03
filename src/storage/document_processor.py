from pathlib import Path
from llama_index.core import (
    VectorStoreIndex,
    SimpleDirectoryReader,
    StorageContext,
)
from llama_index.vector_stores.chroma import ChromaVectorStore
import chromadb
from llama_index.core.prompts import PromptTemplate

class DocumentProcessor:
    def __init__(self, docs_dir: str, persist_dir: str):
        self.docs_dir = Path(docs_dir)
        self.persist_dir = Path(persist_dir)

        # Create directories if they don't exist
        self.docs_dir.mkdir(parents=True, exist_ok=True)
        self.persist_dir.mkdir(parents=True, exist_ok=True)

        db = chromadb.PersistentClient(path=str(self.persist_dir))
        chroma_collection = db.get_or_create_collection("documents_collection")
        self.vector_store = ChromaVectorStore(chroma_collection=chroma_collection)
        self.storage_context = StorageContext.from_defaults(vector_store=self.vector_store)
        self.index = None
        self.query_engine = None

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

    def index_documents(self):
        print(f"Loading documents from {self.docs_dir}...")
        documents = SimpleDirectoryReader(str(self.docs_dir)).load_data()
        print(f"Found {len(documents)} documents.")

        if documents:
            print("Creating or updating index...")
            self.index = VectorStoreIndex.from_documents(
                documents,
                storage_context=self.storage_context,
                show_progress=True,
            )
            self.query_engine = self.index.as_query_engine(
                response_mode="compact",
                text_qa_template=self.qa_tmpl,
                verbose=True
            )
            print("Indexing complete.")
        else:
            print("No documents to index.")
            self.index = None
            self.query_engine = None

    def query(self, user_query: str):
        if self.query_engine:
            return self.query_engine.query(user_query)
        else:
            return "No documents indexed yet. Please ensure documents are in the specified folder and the index is built."