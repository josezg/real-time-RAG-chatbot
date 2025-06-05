import torch
from llama_index.llms.huggingface import HuggingFaceLLM
from transformers import BitsAndBytesConfig
import os
from dotenv import load_dotenv
load_dotenv()

class LLMModel:
    def __init__(self):
        self.llm = self._initialize_llm()

    def _initialize_llm(self):
        llm_model_name = os.getenv('GEN_LLM_MODEL_NAME')
        tokenizer_name = os.getenv('GEN_TOKENIZER_NAME')
        context_window = int(os.getenv('LLM_CONTEXT_WINDOW', 2048))
        max_new_tokens = int(os.getenv('LLM_MAX_NEW_TOKENS', 128))
        try:
            bnb_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_compute_dtype=torch.bfloat16,
                bnb_4bit_use_double_quant=True,
            )
            llm = HuggingFaceLLM(
                model_name=llm_model_name,
                tokenizer_name=tokenizer_name,
                context_window=context_window,
                max_new_tokens=max_new_tokens,
                device_map="auto",
                model_kwargs={"quantization_config": bnb_config, "torch_dtype": torch.bfloat16},
            )
        except ImportError:
            print("bitsandbytes or accelerate not found, loading Phi-2 without quantization.")
            llm = HuggingFaceLLM(
                model_name=llm_model_name,
                tokenizer_name=tokenizer_name,
                context_window=context_window,
                max_new_tokens=max_new_tokens,
                device_map="auto",
            )
        except Exception as e:
            print(f"Error initializing quantized Phi-2: {e}. Loading without quantization.")
            llm = HuggingFaceLLM(
                model_name=llm_model_name,
                tokenizer_name=tokenizer_name,
                context_window=context_window,
                max_new_tokens=max_new_tokens,
                device_map="auto",
            )
        return llm