import os
import sys
import unittest
from unittest.mock import patch, MagicMock

# Add project root to sys.path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if project_root not in sys.path:
    sys.path.append(project_root)

from source_code import models
from source_code import config

class TestModelsRegistry(unittest.TestCase):

    @patch('source_code.models.get_ollama_client')
    def test_ollama_chat_call(self, mock_get_client):
        # Setup mock
        mock_client = MagicMock()
        mock_get_client.return_value = mock_client
        mock_client.chat.return_value = {"message": {"content": "Ollama response"}}
        
        # Call chat
        res = models.chat("Hello", provider="ollama", model="qwen3")
        
        # Verify
        self.assertEqual(res, "Ollama response")
        mock_client.chat.assert_called_once()
        args, kwargs = mock_client.chat.call_args
        self.assertEqual(kwargs['model'], "qwen3")

    @patch('source_code.models.get_gemini_client')
    def test_gemini_chat_call(self, mock_get_client):
        # Setup mock
        mock_client = MagicMock()
        mock_get_client.return_value = mock_client
        mock_response = MagicMock()
        mock_response.text = "Gemini response"
        mock_client.models.generate_content.return_value = mock_response
        
        # Call chat
        res = models.chat("Hello", provider="gemini", model="gemini-2.0-flash")
        
        # Verify
        self.assertEqual(res, "Gemini response")
        mock_client.models.generate_content.assert_called_once()

    @patch('source_code.models.get_ollama_client')
    def test_embed_call(self, mock_get_client):
        # Setup mock
        mock_client = MagicMock()
        mock_get_client.return_value = mock_client
        mock_client.embeddings.return_value = {"embedding": [0.1, 0.2, 0.3]}
        
        # Call embed
        vectors = models.embed(["test"], provider="ollama")
        
        # Verify
        self.assertEqual(len(vectors), 1)
        self.assertEqual(vectors[0], [0.1, 0.2, 0.3])

if __name__ == '__main__':
    unittest.main()
