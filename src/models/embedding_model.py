from llama_index.embeddings.huggingface import HuggingFaceEmbedding
import os
from dotenv import load_dotenv
load_dotenv()

class EmbeddingModel:
    def __init__(self, model_name=os.getenv('EMBEDDING_MODEL_NAME')):
        """
        Initialize the EmbeddingModel with the specified model name.
        Chunk size will be set globally via LlamaIndex Settings.
        """
        self.embed_model = HuggingFaceEmbedding(model_name=model_name)