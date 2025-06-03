from llama_index.embeddings.huggingface import HuggingFaceEmbedding

class EmbeddingModel:
    def __init__(self, model_name="BAAI/bge-small-en-v1.5"):
        """
        Initialize the EmbeddingModel with the specified model name.
        Chunk size will be set globally via LlamaIndex Settings.
        """
        self.embed_model = HuggingFaceEmbedding(model_name=model_name)