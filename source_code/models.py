import os
import sys
import time
from typing import List, Dict, Any, Optional
from .config import CONFIG

# --- Provider Imports (Lazy loaded via clients) ---
import ollama
try:
    from google import genai
except ImportError:
    genai = None

try:
    from groq import Groq
except ImportError:
    Groq = None

import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer

# ---------------------------------------------------------------------------
# Client Management (Lazy Loading)
# ---------------------------------------------------------------------------

_clients = {
    "ollama": None,
    "gemini": None,
    "groq": None
}

def get_ollama_client() -> ollama.Client:
    """Return a persistent Ollama client."""
    if _clients["ollama"] is None:
        _clients["ollama"] = ollama.Client(host=CONFIG["OLLAMA_LOCAL_URL"])
    return _clients["ollama"]

def get_gemini_client():
    """Return a Google GenAI client."""
    if _clients["gemini"] is None:
        if genai is None:
            raise ImportError("google-genai is not installed.")
        _clients["gemini"] = genai.Client(api_key=CONFIG["GEMINI_API_KEY"])
    return _clients["gemini"]

def get_groq_client():
    """Return a Groq client."""
    if _clients["groq"] is None:
        if Groq is None:
            raise ImportError("groq is not installed.")
        _clients["groq"] = Groq(api_key=CONFIG["GROQ_API_KEY"])
    return _clients["groq"]

# ---------------------------------------------------------------------------
# Chat / Generation API
# ---------------------------------------------------------------------------

def chat(
    prompt: Optional[str] = None,
    system_prompt: Optional[str] = None,
    messages: Optional[List[Dict[str, str]]] = None,
    model: Optional[str] = None,
    provider: Optional[str] = None,
    **kwargs
) -> str:
    """
    Unified chat interface. Handles different providers and message formats.
    
    Args:
        prompt: Simple user prompt string.
        system_prompt: Optional system-level instructions.
        messages: OpenAI-style list of message dicts.
        model: Override model name from config.
        provider: Override provider ("gemini", "ollama", "groq").
        **kwargs: Additional parameters like temperature, num_ctx, etc.
    """
    provider = provider or CONFIG["providers"]["chat"]
    model_config = CONFIG["model"]
    model_name = model or model_config["model"]
    
    # Standardize messages
    if messages is None:
        messages = []
        if prompt:
            messages.append({"role": "user", "content": prompt})

    # --- GOOGLE GEMINI ---
    if provider == "gemini":
        client = get_gemini_client()
        # Gemini-genai uses a slightly different structure
        config_args = {}
        if system_prompt:
            config_args["system_instruction"] = system_prompt
        
        # Merge model_config and kwargs
        temperature = kwargs.get("temperature", model_config.get("temperature", 0.3))
        max_tokens = kwargs.get("max_tokens", model_config.get("max_tokens", 4096))
        
        config_args["temperature"] = temperature
        config_args["max_output_tokens"] = max_tokens
        if "top_p" in model_config:
            config_args["top_p"] = model_config["top_p"]

        try:
            final_prompt = prompt or (messages[-1]["content"] if messages else "")
            response = client.models.generate_content(
                model=model_name,
                contents=final_prompt,
                config=config_args
            )
            return response.text
        except Exception as e:
            return f"⚠ Gemini Error: {e}"

    # --- OLLAMA ---
    elif provider == "ollama":
        client = get_ollama_client()
        
        full_messages = []
        if system_prompt:
            full_messages.append({"role": "system", "content": system_prompt})
        full_messages.extend(messages)
        
        options = {
            "temperature": kwargs.get("temperature", model_config.get("temperature", 0.25)),
            "num_ctx": kwargs.get("num_ctx", model_config.get("num_ctx", 8192)),
        }
        if "top_p" in model_config:
            options["top_p"] = model_config["top_p"]
        
        try:
            response = client.chat(
                model=model_name,
                messages=full_messages,
                options=options,
            )
            return response["message"]["content"]
        except Exception as e:
            return f"⚠ Ollama Error: {e}"

    # --- GROQ ---
    elif provider == "groq":
        client = get_groq_client()
        
        full_messages = []
        if system_prompt:
            full_messages.append({"role": "system", "content": system_prompt})
        full_messages.extend(messages)
        
        try:
            completion = client.chat.completions.create(
                model=model,
                messages=full_messages,
                temperature=kwargs.get("temperature", 0.6),
                max_tokens=kwargs.get("max_tokens", 4096),
            )
            return completion.choices[0].message.content
        except Exception as e:
            return f"⚠ Groq Error: {e}"

    else:
        return f"⚠ Unsupported provider: {provider}"

# ---------------------------------------------------------------------------
# Embedding API
# ---------------------------------------------------------------------------

def embed(texts: List[str], model: Optional[str] = None, provider: Optional[str] = None) -> List[List[float]]:
    """Generate vector embeddings for a list of texts."""
    provider = provider or CONFIG["providers"]["embedding"]
    model = model or CONFIG["providers"]["embedding_model"]
    
    if provider == "ollama":
        client = get_ollama_client()
        vectors = []
        try:
            for text in texts:
                res = client.embeddings(
                    model=model,
                    prompt=text,
                    keep_alive="10m"
                )
                vectors.append(res["embedding"])
            return vectors
        except Exception as e:
            print(f"[models.embed] Error: {e}")
            return []
    
    return []

# ---------------------------------------------------------------------------
# Reranking API (Cross-Encoder)
# ---------------------------------------------------------------------------

_rerank_model = None
_rerank_tokenizer = None

def _load_reranker():
    global _rerank_model, _rerank_tokenizer
    if _rerank_model is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
        model_id = CONFIG["rag"]["cross_encoder"]["model"]
        print(f"[models.rerank] Loading {model_id} on {device}...")
        _rerank_tokenizer = AutoTokenizer.from_pretrained(model_id, padding_side="left")
        _rerank_model = AutoModelForSequenceClassification.from_pretrained(
            model_id,
            torch_dtype=torch.float16 if device == "cuda" else torch.float32,
        ).to(device).eval()

def rerank(query: str, documents: List[str], model: Optional[str] = None) -> List[float]:
    """Score document relevance to a query using a Cross-Encoder."""
    _load_reranker()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    # Qwen3-Reranker format
    instruction = "Given a student's academic query, retrieve relevant lecture notes or syllabus passages that answer the query"
    
    pairs = []
    prefix = (
        '<|im_start|>system\n'
        'Judge whether the Document meets the requirements based on the '
        'Query and the Instruct provided. Note that the answer can only '
        'be "yes" or "no".<|im_end|>\n'
        '<|im_start|>user\n'
    )
    suffix = '<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n'
    
    for doc in documents:
        pairs.append(f"{prefix}<Instruct>: {instruction}\n<Query>: {query}\n<Document>: {doc}{suffix}")

    with torch.no_grad():
        inputs = _rerank_tokenizer(
            pairs,
            padding=True,
            truncation=True,
            max_length=8192,
            return_tensors="pt",
        ).to(device)
        
        logits = _rerank_model(**inputs).logits.squeeze(-1)
        if logits.dim() == 0:
            logits = logits.unsqueeze(0)
        
        scores = logits.sigmoid().cpu().tolist()
    
    return scores

# ---------------------------------------------------------------------------
# Vision / VLM API
# ---------------------------------------------------------------------------

def vision(images: Any, prompt: str, model: Optional[str] = None, provider: Optional[str] = None) -> str:
    """
    Analyze one or more images using a Vision-Language Model.
    
    Args:
        images: Single image path, list of paths, or list of image bytes.
        prompt: The text instructions for the model.
    """
    provider = provider or CONFIG["providers"]["vision"]
    model = model or CONFIG["providers"]["vision_model"]
    
    if provider == "ollama":
        client = get_ollama_client()
        try:
            image_payload = []
            if isinstance(images, (str, bytes)):
                images = [images]
            
            for img in images:
                if isinstance(img, str) and os.path.exists(img):
                    with open(img, "rb") as f:
                        image_payload.append(f.read())
                else:
                    image_payload.append(img) # assume bytes
            
            response = client.generate(
                model=model,
                prompt=prompt,
                images=image_payload
            )
            return response["response"]
        except Exception as e:
            return f"⚠ Vision Error: {e}"
            
    elif provider == "huggingface":
        if genai is None: # We'll use InferenceClient directly if we can
             pass
        from huggingface_hub import InferenceClient
        try:
            client = InferenceClient(api_key=CONFIG["HF_TOKEN"])
            
            # Prepare image content
            content = []
            if isinstance(images, (str, bytes)):
                images = [images]
            
            # Note: For HF Inference API v1 Chat Completion, images should be URLs or Base64
            from source_code.utils import pil_to_base64
            from PIL import Image
            import io

            for img in images:
                if isinstance(img, str) and os.path.exists(img):
                    with open(img, "rb") as f:
                        img_bytes = f.read()
                elif isinstance(img, bytes):
                    img_bytes = img
                else:
                    img_bytes = None
                
                if img_bytes:
                    pil_img = Image.open(io.BytesIO(img_bytes))
                    content.append({"type": "image_url", "image_url": {"url": pil_to_base64(pil_img)}})
            
            content.append({"type": "text", "text": prompt})
            
            response = client.chat.completions.create(
                model=CONFIG["model"]["model"], # Or specific HF model if we add it
                messages=[{"role": "user", "content": content}],
                max_tokens=8192,
            )
            return response.choices[0].message.content
        except Exception as e:
            return f"⚠ Vision Error (HF): {e}"
            
    return "⚠ Vision provider not implemented."
