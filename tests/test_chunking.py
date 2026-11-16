from rag_service.chunking import chunk_document, chunk_text
from rag_service.config import Settings
from rag_service.models import Document


def test_windows_cover_the_text_and_overlap() -> None:
    text = " ".join(f"word{i:02d}" for i in range(40))
    chunks = chunk_text(text, size=40, overlap=10)
    assert len(chunks) > 1
    assert text.startswith(chunks[0])
    assert chunks[0][-10:] in text
    rejoined = " ".join(chunks)
    for token in text.split():
        assert token in rejoined


def test_chunk_ids_are_stable() -> None:
    settings = Settings(chunk_size=200, chunk_overlap=20, log_level="WARNING")
    document = Document(document_id="policy", title="Policy", text="alpha " * 80)
    first = chunk_document(document, settings)
    second = chunk_document(document, settings)
    assert [chunk.chunk_id for chunk in first] == [chunk.chunk_id for chunk in second]
    assert first[0].chunk_id.startswith("policy:0:")
