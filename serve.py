from __future__ import annotations

import uvicorn

from server.settings import load_server_settings


def main() -> None:
    settings = load_server_settings()
    uvicorn.run(
        "server.app:app",
        host=settings.host,
        port=settings.port,
        reload=False,
    )


if __name__ == "__main__":
    main()
