import streamlit as st
import os
import sys
from pathlib import Path
from watchdog.observers import Observer
from watchdog.events import FileSystemEventHandler
import datetime # For displaying file modification time
import pandas as pd # For presenting file info in a table
import threading
import time # Needed for the debounce mechanism and polling

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), 'src')))

# Import your custom modules
from models.llm_model import LLMModel
from models.embedding_model import EmbeddingModel
from storage.document_processor import DocumentProcessor
from llama_index.core import Settings # Crucial import for LlamaIndex global settings

# --- Streamlit App Setup ---
st.set_page_config(page_title="Real-Time RAG Chatbot", layout="wide")
st.title("Real-Time RAG Chatbot")
st.markdown("""
Welcome to the Real-Time RAG Chatbot!
Type your question below and get instant responses powered by Retrieval-Augmented Generation.
""")

# --- Constants & Session State Initialization ---
if "observer" not in st.session_state:
    st.session_state.observer = None
if "document_processor" not in st.session_state:
    st.session_state.document_processor = None
if "root_dir_input" not in st.session_state:
    st.session_state.root_dir_input = ""
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []
if "is_setup_complete" not in st.session_state:
    st.session_state.is_setup_complete = False
if "documents_list" not in st.session_state:
    st.session_state.documents_list = []
if "documents_subdir_path" not in st.session_state:
    st.session_state.documents_subdir_path = None
if "is_reindexing" not in st.session_state:
    st.session_state.is_reindexing = False
if "file_change_detected" not in st.session_state: # Flag set by watchdog thread
    st.session_state.file_change_detected = False
if "last_reindex_time" not in st.session_state:
    st.session_state.last_reindex_time = 0.0 # Use float for time.time()
if "initial_indexing_done" not in st.session_state:
    st.session_state.initial_indexing_done = False


# --- Model Loading (Cached) ---
@st.cache_resource
def load_llm_model():
    """Loads and caches the LLM model."""
    llm = LLMModel()
    return llm.llm

@st.cache_resource
def load_embedding_model():
    """Loads and caches the embedding model."""
    embed_model_instance = EmbeddingModel() # Corrected class name
    return embed_model_instance.embed_model

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
            # Only list actual files and filter by common document extensions
            if f.is_file() and f.suffix.lower() in ['.txt', '.pdf', '.docx', '.md']:
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

# --- Watchdog Event Handler (Modified for Streamlit Communication) ---
class SmartDocumentEventHandler(FileSystemEventHandler):
    def __init__(self, docs_path: Path):
        self.docs_path = docs_path
        self.debounce_time = 0.000001  # Seconds to wait before signaling Streamlit
        self._last_event_time = 0.0 # To track when the last event was handled
        self._lock = threading.Lock() # To protect _last_event_time

    def _trigger_reindex_flag(self, event_path):
        """
        Sets a flag in st.session_state to indicate a file change,
        with a debounce mechanism. This flag is then picked up by the main Streamlit thread.
        """
        current_time = time.time()
        with self._lock:
            # Only set the flag if enough time has passed since the last event
            if (current_time - self._last_event_time) < self.debounce_time:
                print(f"Debouncing event for {event_path}. Last event was too recent. Skipping signal.")
                return
            self._last_event_time = current_time

        print(f"Watchdog: File change detected in {event_path}. Setting file_change_detected flag.")
        st.session_state.file_change_detected = True

    def on_created(self, event):
        if not event.is_directory and Path(event.src_path).suffix.lower() in ['.txt', '.pdf', '.docx', '.md']:
            self._trigger_reindex_flag(event.src_path)

    def on_modified(self, event):
        if not event.is_directory and Path(event.src_path).suffix.lower() in ['.txt', '.pdf', '.docx', '.md']:
            self._trigger_reindex_flag(event.src_path)

    def on_deleted(self, event):
        if not event.is_directory and Path(event.src_path).suffix.lower() in ['.txt', '.pdf', '.docx', '.md']:
            self._trigger_reindex_flag(event.src_path)

    def on_moved(self, event):
        # Trigger if source or destination is a document file
        if not event.is_directory and (Path(event.src_path).suffix.lower() in ['.txt', '.pdf', '.docx', '.md'] or
                                      Path(event.dest_path).suffix.lower() in ['.txt', '.pdf', '.docx', '.md']):
            self._trigger_reindex_flag(event.src_path)

# --- Reindexing Logic (called from main Streamlit thread) ---
def perform_reindexing():
    """Performs the actual reindexing operation and updates UI state."""
    # Ensure all session_state variables are initialized before access
    # This function should only be called after initial setup, where these are guaranteed.
    if st.session_state.document_processor is None or st.session_state.documents_subdir_path is None:
        print("Error: Document processor or subdirectory path not initialized for reindexing.")
        st.error("Error: Please process the directory first before attempting reindexing.")
        return False

    st.session_state.is_reindexing = True
    print("Starting reindexing process in main Streamlit thread...")
    
    try:
        # Display a spinner in the UI while reindexing
        with st.spinner("🔄 Reindexing documents... Please wait."):
            st.session_state.document_processor.index_documents()
        
        # Update documents list and last reindex time
        new_docs_info = get_documents_info(st.session_state.documents_subdir_path)
        st.session_state.documents_list = new_docs_info
        st.session_state.last_reindex_time = time.time()
        
        print("Reindexing completed successfully.")
        return True
        
    except Exception as e:
        print(f"Error during reindexing: {e}")
        st.error(f"Error during reindexing: {e}")
        return False
    finally:
        st.session_state.is_reindexing = False
        st.session_state.file_change_detected = False # Reset the flag after processing


# --- Main UI Logic ---

def check_and_perform_reindexing_on_rerun():
    """
    Checks if reindexing is needed based on state flags and triggers it from the main thread.
    This runs on every Streamlit rerun.
    """
    print(st.session_state.is_setup_complete)
    print(st.session_state.documents_subdir_path)
    if st.session_state.is_setup_complete and st.session_state.documents_subdir_path:
        # Case 1: Initial indexing needs to be done after successful directory setup
        print(st.session_state.initial_indexing_done)
        print(st.session_state.file_change_detected)
        print(st.session_state.is_reindexing)

        if not st.session_state.initial_indexing_done:
            print("Performing initial indexing after setup on rerun...")
            perform_reindexing()
            st.session_state.initial_indexing_done = True
            # No st.rerun() here; the current rerun cycle will complete and display results.

        # Case 2: File change detected by watchdog or manual button click
        
        elif st.session_state.file_change_detected and not st.session_state.is_reindexing:
            print("File change detected and not currently reindexing. Triggering reindex.")
            perform_reindexing()

# Call the checker function at the top of the main UI logic.
check_and_perform_reindexing_on_rerun()


# Display reindexing status prominently
if st.session_state.is_reindexing:
    st.warning("🔄 **Reindexing documents...** Please wait. Queries are temporarily disabled.")

# Directory Input and Setup Logic
st.markdown("### 1. Select Base Directory")
root_directory_input = st.text_input(
    "Enter the full path to your project's base directory (e.g., where 'documents' folder is):",
    value=st.session_state.root_dir_input,
    placeholder="e.g., /home/user/my_project",
    disabled=st.session_state.is_reindexing # Disable input during reindexing
)


if st.button("Process Directory", disabled=st.session_state.is_reindexing):
    st.session_state.root_dir_input = root_directory_input # Update session state

    if not root_directory_input: # Handle empty input case for button click
        st.error("Please enter a directory path.")
    else:
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
                    st.warning(f"The 'documents' directory is empty. Add files and click 'Process Directory' again.")
                    st.session_state.is_setup_complete = False
                    st.session_state.documents_list = [] # Clear document list
                else:
                    st.success(f"Valid 'documents' directory found at: {documents_subdir_path}")
                    
                    # Setup directories
                    chroma_db_subdir_path.mkdir(parents=True, exist_ok=True)
                    st.info(f"ChromaDB will be stored in: {chroma_db_subdir_path}")

                    # Stop existing observer
                    if st.session_state.observer and st.session_state.observer.is_alive():
                        st.session_state.observer.stop()
                        st.session_state.observer.join()
                        print("Stopped previous observer.")
                        st.session_state.observer = None # Clear observer from session state

                    # Initialize DocumentProcessor
                    st.session_state.document_processor = DocumentProcessor(
                        docs_dir=str(documents_subdir_path), # Pass the actual documents path
                        persist_dir=str(chroma_db_subdir_path)
                    )
                    
                    st.session_state.is_setup_complete = True
                    st.session_state.initial_indexing_done = False # Keep False so check_and_perform_reindexing_on_rerun handles it
                                                                    # on the *next* rerun triggered by this button.

                    # Start file watching (SmartDocumentEventHandler)
                    event_handler = SmartDocumentEventHandler(documents_subdir_path)
                    observer = Observer()
                    observer.schedule(event_handler, str(documents_subdir_path), recursive=True)
                    observer.start()
                    st.session_state.observer = observer
                    
                    st.info(f"Monitoring '{documents_subdir_path}' for document changes...")
                    st.success("Setup complete! You can now ask questions.")
                    # Force a rerun so that `check_and_perform_reindexing_on_rerun` runs for initial indexing
                    st.rerun() 

# Display Document List
st.markdown("### Documents in Folder")
if st.session_state.is_setup_complete and st.session_state.documents_list:
    st.markdown(f"**Monitoring**: `{st.session_state.documents_subdir_path}`")
    
    # Provide visual feedback if reindexing is active
    if st.session_state.is_reindexing:
        st.info("📝 Document list is being updated after reindexing...")
    
    df = pd.DataFrame(st.session_state.documents_list)
    st.dataframe(df, use_container_width=True, hide_index=True)
    
    # Show last update time
    if st.session_state.last_reindex_time > 0:
        last_update = datetime.datetime.fromtimestamp(st.session_state.last_reindex_time)
        st.caption(f"Last updated: {last_update.strftime('%Y-%m-%d %H:%M:%S')}")
        
elif st.session_state.is_setup_complete: # Setup is complete but no files
    st.info("The 'documents' folder is currently empty. Please add files to begin indexing.")
else:
    st.info("Select and process a directory above to see the list of documents.")

# Chat Interface
st.markdown("### 2. Ask Your Question")

if not st.session_state.is_setup_complete:
    st.info("Please select a valid base directory and process it to enable the chatbot.")
elif st.session_state.is_reindexing:
    st.info("⏳ Please wait while documents are being reindexed. Chat will be available shortly.")
else:
    # Display chat history
    for chat_entry in st.session_state.chat_history:
        if chat_entry["role"] == "user":
            st.chat_message("user").write(chat_entry["content"])
        else:
            st.chat_message("assistant").write(chat_entry["content"])

    user_query_input = st.chat_input(
        "Type your message here...", 
        key="chat_input",
        # Disable chat input if reindexing is active
        disabled=st.session_state.is_reindexing
    )

    if user_query_input and not st.session_state.is_reindexing:
        # Add user message
        st.session_state.chat_history.append({"role": "user", "content": user_query_input})
        with st.chat_message("user"):
            st.write(user_query_input)

        # Get response
        if st.session_state.document_processor and st.session_state.document_processor.query_engine:
            with st.spinner("Thinking..."):
                rag_result = st.session_state.document_processor.query(user_query_input)
                
                response_text = rag_result["response_text"]
                source_info = rag_result["source_info"]
                
                # Format sources
                sources_display = ""
                if source_info:
                    sources_display = "\n\n**Sources:**\n"
                    unique_sources = set()
                    for source in source_info:
                        source_identifier = f"Document: {source['file_name']}"
                        if source['page_label'] != 'N/A':
                            source_identifier += f", Page: {source['page_label']}"
                        unique_sources.add(source_identifier)
                    
                    for s_id in sorted(list(unique_sources)):
                        sources_display += f"- {s_id}\n"
                
                full_bot_response = response_text + sources_display
                
                # Add bot response
                st.session_state.chat_history.append({"role": "assistant", "content": full_bot_response})
                with st.chat_message("assistant"):
                    st.write(full_bot_response)
        else:
            error_msg = "Chatbot is not ready. Ensure directory is processed and documents are indexed."
            st.session_state.chat_history.append({"role": "assistant", "content": error_msg})
            with st.chat_message("assistant"):
                st.error(error_msg)

# Add manual refresh and clear chat buttons for user control
if st.session_state.is_setup_complete:
    st.markdown("---")
    col1, col2 = st.columns([1, 1])
    with col1:
        # A button to manually trigger re-indexing and UI refresh
        # This button also manually sets file_change_detected and triggers a rerun.
        if st.button("🔄 Force Re-index & Refresh UI", disabled=st.session_state.is_reindexing):
            st.session_state.file_change_detected = True # Manually trigger the flag
            st.rerun() # Force a rerun to pick up the flag
    with col2:
        if st.button("🗑️ Clear Chat History"):
            st.session_state.chat_history = []
            st.rerun()

# --- Automatic rerun for background updates ---
if st.session_state.file_change_detected or (st.session_state.is_setup_complete and not st.session_state.initial_indexing_done):
    # while also allowing Streamlit to process UI updates.
    time.sleep(0.5) 
    st.rerun()

# Footer
st.markdown("---")
st.markdown("Powered by Streamlit and LlamaIndex.")
