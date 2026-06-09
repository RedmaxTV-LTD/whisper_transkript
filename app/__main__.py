"""Точка входа: python -m app"""

import uvicorn

from app.persistent_logs import install_app_console_logging, install_persistent_logging
from app.settings import get_settings


def main() -> None:
    s = get_settings()
    install_persistent_logging(s.logs_dir)
    install_app_console_logging()
    uvicorn.run(
        "app.main:app",
        host=s.listen_host,
        port=s.listen_port,
        factory=False,
        proxy_headers=True,
        forwarded_allow_ips="*",
    )


if __name__ == "__main__":
    main()
