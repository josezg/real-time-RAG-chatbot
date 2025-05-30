import streamlit as st
import os
import sys
from pathlib import Path
from watchdog.observers import Observer
from watchdog.events import FileSystemEventHandler
import datetime # For displaying file modification time
import pandas as pd # For presenting file info in a table
import threading # Still needed for watchdog thread

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), 'src')))

# Import your custom modules
from models.llm_model import LLMModel
from models.embedding_model import EmbeddingModel
from storage.document_processing import DocumentProcessingPipeline
from llama_index.core import Settings # Crucial import for LlamaIndex global settings

# --- Streamlit App Setup ---
st.set_page_config(page_title="Real-Time RAG Chatbot", layout="wide")
st.title("Real-Time RAG Chatbot")
st.markdown("""
Welcome to the Real-Time RAG Chatbot!
Type your question below and get instant responses powered by Retrieval-Augmented Generation.
""")

# --- Constants & Session State Initialization ---
# Use st.session_state to persist variables across reruns
if "observer" not in st.session_state:
    st.session_state.observer = None
if "document_processor" not in st.session_state:
    st.session_state.document_processor = None
if "root_dir_input" not in st.session_state: # Renamed for clarity: this is the root path
    st.session_state.root_dir_input = ""
if "query_input" not in st.session_state:
    st.session_state.query_input = ""
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []
if "is_setup_complete" not in st.session_state: # New state to track if setup is done
    st.session_state.is_setup_complete = False
if "documents_list" not in st.session_state: # Stores the list of documents for display
    st.session_state.documents_list = []
if "documents_subdir_path" not in st.session_state: # Store actual documents path
    st.session_state.documents_subdir_path = None
# Removed: if "script_run_context" not in st.session_state:
# Removed: st.session_state.script_run_context = None


# --- Model Loading (Cached) ---
@st.cache_resource
def load_llm_model():
    """Loads and caches the LLM model."""
    llm = LLMModel()
    return llm.llm # Return the actual LLM instance

@st.cache_resource
def load_embedding_model():
    """Loads and caches the embedding model."""
    embed_model_instance = EmbeddingModel()
    return embed_model_instance.embed_model # Return the actual embedding model instance

# Load models and set LlamaIndex global settings
llm_instance = load_llm_model()
embed_model_instance = load_embedding_model()

Settings.llm = llm_instance
Settings.embed_model = embed_model_instance
Settings.chunk_size = 256 # Set your desired chunk size here globally

# --- Helper function to get document info ---
def get_documents_info(docs_path: Path):
    """
    Scans the documents directory and returns a list of file info.
    """
    files_info = []
    if docs_path.is_dir():
        for f in docs_path.iterdir():
            if f.is_file(): # Only list actual files, not subdirectories
                try:
                    file_size_bytes = f.stat().st_size
                    # Convert to human-readable format
                    if file_size_bytes < 1024:
                        file_size = f"{file_size_bytes} B"
                    elif file_size_bytes < 1024**2:
                        file_size = f"{file_size_bytes / 1024:.2f} KB"
                    elif file_size_bytes < 1024**3:
                        file_size = f"{file_size_bytes / (1024**2):.2f} MB"
                    else:
                        file_size = f"{file_size_bytes / (1024**3):.2f} GB"

                    last_modified = datetime.datetime.fromtimestamp(f.stat().st_mtime).strftime('%Y-%m-%d %H:%M:%S')
                    files_info.append({
                        "File Name": f.name,
                        "Size": file_size,
                        "Last Modified": last_modified
                    })
                except Exception as e:
                    print(f"Could not get info for {f.name}: {e}")
    return files_info

# --- Watchdog Event Handler ---
class StreamlitDocumentEventHandler(FileSystemEventHandler):
    # Removed: context: ScriptRunContext from __init__ signature
    def __init__(self, processor_instance: DocumentProcessingPipeline, docs_path_for_display: Path):
        self.processor = processor_instance
        self.docs_path_for_display = docs_path_for_display
        # Removed: self.context = context

    def _reindex_documents(self, event_path):
        """
        Re-indexes documents and triggers a Streamlit rerun to update the UI.
        """
        print(f"File system event detected: {event_path}. Re-indexing documents...")
        self.processor.index_documents()
        # Update the documents list in session state
        st.session_state.documents_list = get_documents_info(self.docs_path_for_display)

        # Trigger Streamlit rerun directly
        # Note: Calling st.rerun() from a background thread is generally discouraged
        # by Streamlit as it can lead to less stable behavior compared to using
        # Streamlit's internal script_run_context. However, this is done as per request
        # to remove the script_run_context logic.
        st.rerun()

    def on_modified(self, event):
        if not event.is_directory:
            self._reindex_documents(event.src_path)

    def on_created(self, event):
        if not event.is_directory:
            self._reindex_documents(event.src_path)

    def on_deleted(self, event):
        if not event.is_directory:
            self._reindex_documents(event.src_path)

    def on_moved(self, event):
        if not event.is_directory:
            self._reindex_documents(event.src_path)

# --- Directory Input and Setup Logic ---
st.markdown("### 1. Select Base Directory")
root_directory_input = st.text_input(
    "Enter the full path to your project's base directory (e.g., where 'documents' folder is):",
    value=st.session_state.root_dir_input,
    placeholder="e.g., /home/user/my_project"
)

# Removed: Capture the script run context at the start of the Streamlit script
# Removed: current_script_context = get_script_run_context()
# Removed: if current_script_context and st.session_state.script_run_context is None:
# Removed: st.session_state.script_run_context = current_script_context


# Use a button to trigger directory processing
if st.button("Process Directory") and root_directory_input:
    st.session_state.root_dir_input = root_directory_input # Update session state

    root_path = Path(root_directory_input)
    documents_subdir_path = root_path / "documents" # The required 'documents' subdirectory
    chroma_db_subdir_path = root_path / "chroma_db" # The 'chroma_db' subdirectory

    # 1. Validate the root directory
    if not root_path.is_dir():
        st.error(f"Error: The provided path '{root_directory_input}' is not a valid directory.")
        st.session_state.is_setup_complete = False
        st.session_state.documents_list = [] # Clear document list
    else:
        # 2. Validate the 'documents' subdirectory
        if not documents_subdir_path.is_dir():
            st.error(f"Error: No 'documents' subdirectory found at '{documents_subdir_path}'. "
                     "Please ensure your base directory contains a folder named 'documents' with your files.")
            st.session_state.is_setup_complete = False
            st.session_state.documents_list = [] # Clear document list
        else:
            # Store the documents subdirectory path in session state for the event handler
            st.session_state.documents_subdir_path = documents_subdir_path

            # 3. Check if 'documents' subdirectory has any files
            current_files_info = get_documents_info(documents_subdir_path)
            if not current_files_info: # Check if the list is empty
                st.warning(f"The 'documents' directory at '{documents_subdir_path}' is empty. "
                           "Please add some documents to this folder.")
                st.session_state.is_setup_complete = False
                st.session_state.documents_list = [] # Clear document list
            else:
                st.success(f"Valid 'documents' directory found at: {documents_subdir_path}")

                # Ensure chroma_db directory exists
                chroma_db_subdir_path.mkdir(parents=True, exist_ok=True)
                st.info(f"ChromaDB will be stored in: {chroma_db_subdir_path}")

                # Stop existing observer if any
                if st.session_state.observer and st.session_state.observer.is_alive():
                    st.session_state.observer.stop()
                    st.session_state.observer.join()
                    print("Stopped previous observer.")
                    st.session_state.observer = None # Clear observer from session state

                # Initialize DocumentProcessingPipeline for the new directory
                st.session_state.document_processor = DocumentProcessingPipeline(
                    docs_dir=str(documents_subdir_path), # Pass the actual documents path
                    persist_dir=str(chroma_db_subdir_path)
                )
                with st.spinner("Indexing documents... This may take a moment."):
                    st.session_state.document_processor.index_documents() # Initial indexing

                # Start new observer, passing the documents path
                event_handler = StreamlitDocumentEventHandler(
                    st.session_state.document_processor,
                    documents_subdir_path,
                    # Removed: st.session_state.script_run_context # No longer passing context
                )
                observer = Observer()
                observer.schedule(event_handler, str(documents_subdir_path), recursive=True)
                observer.start()
                st.session_state.observer = observer # Store observer in session state
                st.info(f"Monitoring '{documents_subdir_path}' for document changes...")
                st.session_state.is_setup_complete = True
                st.session_state.documents_list = current_files_info # Populate list after indexing
                st.success("Setup complete! You can now ask questions.")
                # No need for st.rerun() here as the button click already triggers a rerun.
                # The _reindex_documents in the event handler will handle subsequent reruns.


# --- Display Document List ---
st.markdown("### Documents in Folder")
if st.session_state.is_setup_complete and st.session_state.documents_list:
    st.markdown(f"**Monitoring**: `{st.session_state.documents_subdir_path}`")
    df = pd.DataFrame(st.session_state.documents_list)
    st.dataframe(df, use_container_width=True, hide_index=True)
elif st.session_state.is_setup_complete: # Setup is complete but no files
    st.info("The 'documents' folder is currently empty. Please add files to begin indexing.")
else:
    st.info("Select and process a directory above to see the list of documents.")


# --- Chat Interface (Conditional Display) ---
st.markdown("### 2. Ask Your Question")

if not st.session_state.is_setup_complete:
    st.info("Please select a valid base directory and process it to enable the chatbot.")
else:
    # Display chat history
    for i, chat in enumerate(st.session_state.chat_history):
        if i % 2 == 0: # User query
            st.chat_message("user").write(chat)
        else: # Bot response
            st.chat_message("assistant").write(chat)

    user_query_input = st.chat_input("Type your message here...", key="chat_input")

    if user_query_input:
        # Append user query to chat history
        st.session_state.chat_history.append(user_query_input)
        with st.chat_message("user"):
            st.write(user_query_input)

        # Get response
        if st.session_state.document_processor and st.session_state.document_processor.query_engine:
            with st.spinner("Thinking..."):
                response = st.session_state.document_processor.query(user_query_input)
                # Ensure we get the string representation of the response
                response_text = response.response if hasattr(response, 'response') else str(response)
                st.session_state.chat_history.append(response_text)
                with st.chat_message("assistant"):
                    st.write(response_text)
        else:
            error_msg = "Chatbot is not ready. Ensure directory is processed and documents are indexed."
            st.session_state.chat_history.append(error_msg)
            with st.chat_message("assistant"):
                st.error(error_msg)

# --- Clean up observer on app shutdown/rerun (best effort) ---
# This is a bit tricky with Streamlit's stateless nature.
# The observer is a daemon thread, so it might not stop cleanly on simple app refresh.
# It's more reliable when the Streamlit server itself is stopped.
# For robustness, you might need a more sophisticated process management.

# Footer
st.markdown("---")
st.markdown("Developed by Jose. Powered by Streamlit and LlamaIndex.")
