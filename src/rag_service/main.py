"""HTTP entrypoint. Uvicorn loads ``app``; the image does not require GCP to import."""

from rag_service.app import create_app
from rag_service.config import Settings

app = create_app()


def main() -> None:
    import uvicorn

    settings = Settings()
    uvicorn.run(app, host=settings.host, port=settings.port)
