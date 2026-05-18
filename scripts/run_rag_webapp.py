"""Run the local FastAPI review intelligence web app."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path


PROJECT_DIR = Path(__file__).resolve().parents[1]
if str(PROJECT_DIR) not in sys.path:
    sys.path.insert(0, str(PROJECT_DIR))


def main() -> int:
    parser = argparse.ArgumentParser(description="Run the local RAG review intelligence web app.")
    parser.add_argument("--host", default="127.0.0.1", help="Host interface. Default: 127.0.0.1")
    parser.add_argument("--port", type=int, default=8000, help="Port. Default: 8000")
    parser.add_argument("--reload", action="store_true", help="Enable uvicorn reload for development.")
    args = parser.parse_args()

    try:
        import uvicorn
    except ImportError:
        print("Missing dependency: uvicorn. Install with: python -m pip install -r requirements_webapp.txt")
        return 1

    url = f"http://{args.host}:{args.port}"
    print("[webapp] Starting Vietnamese E-commerce Review Intelligence demo")
    print(f"[webapp] URL: {url}")
    print("[webapp] Press Ctrl+C to stop.")
    uvicorn.run("backend.app:app", host=args.host, port=args.port, reload=args.reload)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
