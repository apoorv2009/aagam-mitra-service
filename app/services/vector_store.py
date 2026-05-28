from pinecone import Pinecone

from app.core.config import get_settings

_index = None


def get_index():
    global _index
    if _index is None:
        settings = get_settings()
        pc = Pinecone(api_key=settings.pinecone_api_key)
        _index = pc.Index(settings.pinecone_index_name)
    return _index
