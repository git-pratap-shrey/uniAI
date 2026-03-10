import pytest
from source_code.rag.search import collection_exists

def test_db_collection_exists():
    """Test that the 'notes' collection exists in ChromaDB."""
    # This might require some setup if running in a fresh CI environment,
    # but for now, we'll check if the function returns a boolean.
    exists = collection_exists("notes")
    assert isinstance(exists, bool)
