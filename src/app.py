import streamlit as st
import os
import sys
from pathlib import Path
import datetime # For displaying file modification time
import pandas as pd # For presenting file info in a table
import time # Needed for timestamps and showing 'recently indexed' status

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), 'src')))

# Import your custom modules
from models.llm_model import LLMModel
from models.embedding_model import EmbeddingModel
from storage.document_processor import DocumentProcessor
from llama_index.core import Settings # Crucial import for LlamaIndex global settings

# --- Streamlit App Setup ---
st.set_page_config(page_title="Real-Time RAG Local Chatbot", layout="wide")
st.title("Real-Time RAG Local Chatbot")
st.markdown("""
Welcome to the Real-Time RAG Local Chatbot!
Type your question below and get instant responses powered by Retrieval-Augmented Generation.
""")

# --- Constants & Session State Initialization ---
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
if "reindex_required" not in st.session_state:
    st.session_state.reindex_required = False
if "last_reindex_time" not in st.session_state:
    st.session_state.last_reindex_time = 0.0 # Use float for time.time()
if "initial_indexing_done" not in st.session_state:
    st.session_state.initial_indexing_done = False
# Snapshot of files at last successful indexing
if "indexed_files_snapshot" not in st.session_state:
    st.session_state.indexed_files_snapshot = []
if "timed_message_info" not in st.session_state:
    st.session_state.timed_message_info = {"message": "", "type": "", "display_until": 0.0}


# --- Model Loading (Cached) ---
@st.cache_resource
def load_llm_model():
    """Loads and caches the LLM model."""
    llm = LLMModel()
    return llm.llm

@st.cache_resource
def load_embedding_model():
    """Loads and caches the embedding model."""
    embed_model_instance = EmbeddingModel()
    return embed_model_instance.embed_model

# Load models and set LlamaIndex global settings
llm_instance = load_llm_model()
embed_model_instance = load_embedding_model()

Settings.llm = llm_instance
Settings.embed_model = embed_model_instance
Settings.chunk_size = 256 # Set your desired chunk size here globally

# --- Helper function to get document info and snapshot ---
def get_documents_info(docs_path: Path):
    """
    Scans the documents directory and returns a list of file info for display.
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

def get_file_snapshot(docs_path: Path):
    """
    Returns a consistent snapshot of files (path, modification_time) for comparison.
    """
    snapshot = []
    if docs_path.is_dir():
        for f in docs_path.iterdir():
            if f.is_file() and f.suffix.lower() in ['.txt', '.pdf', '.docx', '.md']:
                try:
                    snapshot.append((str(f.absolute()), f.stat().st_mtime))
                except Exception as e:
                    print(f"Could not get snapshot for {f.name}: {e}")
    return sorted(snapshot) # Sort for consistent comparison

def has_directory_changed(current_docs_path: Path):
    """
    Compares the current directory state with the last indexed snapshot.
    Returns True if changes are detected, False otherwise.
    """
    if not st.session_state.is_setup_complete or not st.session_state.documents_subdir_path:
        return False # Cannot detect changes if not set up

    current_snapshot = get_file_snapshot(current_docs_path)
    # If the stored snapshot is empty, and current is not, it's a change (e.g., first files added)
    if not st.session_state.indexed_files_snapshot and current_snapshot:
        return True
        
    return current_snapshot != st.session_state.indexed_files_snapshot


# --- Reindexing Logic (called from main Streamlit thread) ---
def perform_reindexing(source_trigger="manual"):
    """Performs the actual reindexing operation and updates UI state."""
    if st.session_state.document_processor is None or st.session_state.documents_subdir_path is None:
        print("Error: Document processor or subdirectory path not initialized for reindexing.")
        # Store message for timed display
        st.session_state.timed_message_info = {
            "message": "Error: Please process the directory first before attempting reindexing.",
            "type": "error",
            "display_until": time.time() + 5 # Display for 5 seconds
        }
        return False

    st.session_state.is_reindexing = True
    print(f"Starting reindexing process in main Streamlit thread (Trigger: {source_trigger})...")
    
    try:
        # Display a spinner in the UI while reindexing
        with st.spinner("🔄 Reindexing documents... Please wait."):
            st.session_state.document_processor.index_documents()
        
        # Update documents list and last reindex time
        new_docs_info = get_documents_info(st.session_state.documents_subdir_path)
        st.session_state.documents_list = new_docs_info
        st.session_state.last_reindex_time = time.time()
        # Update the file snapshot after successful reindexing
        st.session_state.indexed_files_snapshot = get_file_snapshot(st.session_state.documents_subdir_path)
        
        print("Reindexing completed successfully.")
        # Store message for timed display
        st.session_state.timed_message_info = {
            "message": "Documents re-indexed successfully!",
            "type": "success",
            "display_until": time.time() + 5 # Display for 5 seconds
        }
        return True
        
    except Exception as e:
        print(f"Error during reindexing: {e}")
        # Store message for timed display
        st.session_state.timed_message_info = {
            "message": f"Error during reindexing: {e}",
            "type": "error",
            "display_until": time.time() + 10 # Display errors for longer
        }
        return False
    finally:
        st.session_state.is_reindexing = False
        st.session_state.reindex_required = False # Reset flag after successful re-indexing

# --- Helper to display and clear timed messages ---
def display_and_clear_timed_message():
    message_info = st.session_state.timed_message_info
    if message_info["message"] and time.time() < message_info["display_until"]:
        if message_info["type"] == "info":
            st.info(message_info["message"])
        elif message_info["type"] == "warning":
            st.warning(message_info["message"])
        elif message_info["type"] == "error":
            st.error(message_info["message"])
        elif message_info["type"] == "success":
            st.success(message_info["message"])
    elif message_info["message"] and time.time() >= message_info["display_until"]:
        # Clear the message if it has expired
        st.session_state.timed_message_info = {"message": "", "type": "", "display_until": 0.0}

# --- Main UI Logic ---

# Call the message displayer at the very top to ensure it's processed early
display_and_clear_timed_message()

# Check for initial indexing after setup
if st.session_state.is_setup_complete and not st.session_state.initial_indexing_done:
    print("Performing initial indexing after setup...")
    perform_reindexing(source_trigger="initial_setup")
    st.session_state.initial_indexing_done = True
    st.rerun() # Rerun to show updated document list and disable process button correctly

# Conditional Reindexing triggered by explicit button click
# This block runs on every rerun. If reindex_required is True AND not already reindexing,
# it means the manual "Re-index" button was clicked.
if (st.session_state.is_setup_complete and 
    st.session_state.documents_subdir_path and 
    st.session_state.reindex_required and # Only reindex if this flag is set by manual button
    not st.session_state.is_reindexing): # Prevent re-entry if already indexing
    
    print("Manual re-index requested and not currently reindexing. Triggering reindex.")
    # Perform reindexing
    success = perform_reindexing(source_trigger="manual_button")
    if success:
        # If reindexing was successful, trigger a full rerun to update the UI
        # and clear the reindex_required flag (handled by perform_reindexing).
        st.rerun()


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
        # Store message for timed display
        st.session_state.timed_message_info = {
            "message": "Please enter a directory path.",
            "type": "error",
            "display_until": time.time() + 5
        }
        st.rerun() # Rerun to display message
    else:
        root_path = Path(root_directory_input)
        documents_subdir_path = root_path / "documents" # The required 'documents' subdirectory
        chroma_db_subdir_path = root_path / "chroma_db" # The 'chroma_db' subdirectory

        # 1. Validate the root directory
        if not root_path.is_dir():
            # Store message for timed display
            st.session_state.timed_message_info = {
                "message": f"Error: The provided path '{root_directory_input}' is not a valid directory.",
                "type": "error",
                "display_until": time.time() + 5
            }
            st.session_state.is_setup_complete = False
            st.session_state.documents_list = [] # Clear document list
            st.rerun() # Rerun to display message
        else:
            # 2. Validate the 'documents' subdirectory
            if not documents_subdir_path.is_dir():
                # Store message for timed display
                st.session_state.timed_message_info = {
                    "message": f"Error: No 'documents' subdirectory found at '{documents_subdir_path}'. "
                               "Please ensure your base directory contains a folder named 'documents' with your files.",
                    "type": "error",
                    "display_until": time.time() + 10
                }
                st.session_state.is_setup_complete = False
                st.session_state.documents_list = [] # Clear document list
                st.rerun() # Rerun to display message
            else:
                # Store the documents subdirectory path in session state for the event handler
                st.session_state.documents_subdir_path = documents_subdir_path

                # 3. Check if 'documents' subdirectory has any files
                current_files_info = get_documents_info(documents_subdir_path)
                if not current_files_info: # Check if the list is empty
                    # Store message for timed display
                    st.session_state.timed_message_info = {
                        "message": f"The 'documents' directory is empty. Add files and click 'Process Directory' again.",
                        "type": "warning",
                        "display_until": time.time() + 7
                    }
                    st.session_state.is_setup_complete = False
                    st.session_state.documents_list = [] # Clear document list
                    st.rerun() # Rerun to display message
                else:
                    st.success(f"Valid 'documents' directory found at: {documents_subdir_path}")
                    
                    # Setup directories
                    chroma_db_subdir_path.mkdir(parents=True, exist_ok=True)
                    st.info(f"ChromaDB will be stored in: {chroma_db_subdir_path}")

                    # Initialize DocumentProcessor
                    st.session_state.document_processor = DocumentProcessor(
                        docs_dir=str(documents_subdir_path), # Pass the actual documents path
                        persist_dir=str(chroma_db_subdir_path)
                    )
                    
                    st.session_state.is_setup_complete = True
                    st.session_state.initial_indexing_done = False # Will trigger initial indexing on next rerun

                    st.info(f"Monitoring '{documents_subdir_path}' for document changes (manual re-index required).")
                    st.success("Setup complete! You can now ask questions.")
                    st.rerun() # Force a rerun to initiate the initial indexing via the top-level checker

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

# --- Manual Re-index Button with Notifications ---
if st.session_state.is_setup_complete and not st.session_state.is_reindexing:
    col_reindex, col_spacer = st.columns([0.3, 0.7]) # Adjust column width
    with col_reindex:
        if st.button("🔄 Re-index & Refresh", help="Re-index documents if changes are detected"):
            current_path = Path(st.session_state.documents_subdir_path)
            
            # Check for changes before triggering full re-indexing
            if has_directory_changed(current_path):
                st.session_state.reindex_required = True # Flag that a re-index is needed
                # Store message for timed display
                st.session_state.timed_message_info = {
                    "message": "Changes detected. Re-indexing will start shortly...",
                    "type": "info",
                    "display_until": time.time() + 5
                }
                st.rerun() # Trigger a rerun to execute perform_reindexing
            else:
                # No changes detected, check last re-index time
                if time.time() - st.session_state.last_reindex_time < 30: # Within 30 seconds
                    # Store message for timed display
                    st.session_state.timed_message_info = {
                        "message": "Index is already up-to-date and was recently re-indexed.",
                        "type": "warning",
                        "display_until": time.time() + 5
                    }
                else:
                    # Store message for timed display
                    st.session_state.timed_message_info = {
                        "message": "No new file changes detected in the documents folder.",
                        "type": "info",
                        "display_until": time.time() + 5
                    }
                st.session_state.reindex_required = False # Ensure flag is false
                st.rerun() # Rerun to display the message (even if no re-index happens)
        
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

# Add clear chat history button 
if st.session_state.is_setup_complete:
    st.markdown("---")
    col1, col2 = st.columns([1, 1])
    with col1:
        if st.button("🗑️ Clear Chat History"):
            st.session_state.chat_history = []
            st.rerun()

# Footer
st.markdown("---")
st.markdown("Powered by Streamlit and LlamaIndex.")
