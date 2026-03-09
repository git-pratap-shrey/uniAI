import torch
print(f"CUDA Available: {torch.cuda.is_available()}")
import transformers
print(f"Transformers: {transformers.__version__}")
from rag.cross_encoder import rerank_cross_encoder
import config
print(f"MIN_CROSS_SCORE: {config.MIN_CROSS_SCORE}")
