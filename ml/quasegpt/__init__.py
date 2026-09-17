"""QuaseGPT: own decoder-only Transformer, trained from scratch.

No OpenAI / Anthropic / Gemini / HuggingFace-hosted / Ollama dependency.
PyTorch only.
"""
from .config import QuaseGPTConfig
from .model import QuaseGPT
from .tokenizer import BPETokenizer, load_tokenizer
from .checkpoint import save_checkpoint, load_checkpoint
from .device import choose_device
from .generation import generate, generate_stream, validate_params
from .prompt import PromptFormatter, build_context, conversation_title

__all__ = [
    "QuaseGPTConfig",
    "QuaseGPT",
    "BPETokenizer",
    "load_tokenizer",
    "save_checkpoint",
    "load_checkpoint",
    "choose_device",
    "generate",
    "generate_stream",
    "validate_params",
    "PromptFormatter",
    "build_context",
    "conversation_title",
]

__version__ = "0.1.0"
