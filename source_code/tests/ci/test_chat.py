import pytest
import os
from source_code import config

def test_chat_config():
    """Test that the chat configuration is loaded correctly."""
    assert config.MODEL_CHAT is not None
    assert isinstance(config.MODEL_CHAT, str)
    assert "gemini" in config.MODEL_CHAT.lower() or "gpt" in config.MODEL_CHAT.lower() or "llama" in config.MODEL_CHAT.lower()
