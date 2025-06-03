import torch
from llama_index.llms.huggingface import HuggingFaceLLM
from transformers import BitsAndBytesConfig

class LLMModel:
    def __init__(self):
        self.llm = self._initialize_llm()

    def _initialize_llm(self):
        try:
            bnb_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_compute_dtype=torch.bfloat16,
                bnb_4bit_use_double_quant=True,
            )
            llm = HuggingFaceLLM(
                model_name="microsoft/phi-2",
                tokenizer_name="microsoft/phi-2",
                context_window=2048,
                max_new_tokens=128,
                device_map="auto",
                model_kwargs={"quantization_config": bnb_config, "torch_dtype": torch.bfloat16},
            )
        except ImportError:
            print("bitsandbytes or accelerate not found, loading Phi-2 without quantization.")
            llm = HuggingFaceLLM(
                model_name="microsoft/phi-2",
                tokenizer_name="microsoft/phi-2",
                context_window=2048,
                max_new_tokens=128,
                device_map="auto",
            )
        except Exception as e:
            print(f"Error initializing quantized Phi-2: {e}. Loading without quantization.")
            llm = HuggingFaceLLM(
                model_name="microsoft/phi-2",
                tokenizer_name="microsoft/phi-2",
                context_window=2048,
                max_new_tokens=128,
                device_map="auto",
            )
        return llm