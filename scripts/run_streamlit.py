"""Launch Streamlit reliably on Windows Python 3.14 environments."""
from __future__ import annotations

import asyncio
import sys


def main() -> None:
    if sys.platform == "win32" and hasattr(asyncio, "WindowsSelectorEventLoopPolicy"):
        asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())
    from streamlit.web.cli import main as streamlit_main
    streamlit_main()


if __name__ == "__main__":
    main()
