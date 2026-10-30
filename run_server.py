"""
PromptShield AI — Phase 6 REST API Service Runner
Starts the Uvicorn server hosting the FastAPI backend service.

Usage:
    python run_server.py
    python run_server.py --host 0.0.0.0 --port 8000 --reload
"""

import os
import sys
import logging
import warnings
import argparse

# Suppress noisy library deprecation warnings (PyTorch JIT script, HF unauthenticated notices)
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning)
logging.getLogger("transformers").setLevel(logging.ERROR)
logging.getLogger("huggingface_hub").setLevel(logging.ERROR)
os.environ["TOKENIZERS_PARALLELISM"] = "false"

import uvicorn


def main():
    parser = argparse.ArgumentParser(description="PromptShield AI REST API Server")
    parser.add_argument("--host", default="127.0.0.1", help="Host interface to bind (default: 127.0.0.1)")
    parser.add_argument("--port", type=int, default=8000, help="Port to listen on (default: 8000)")
    parser.add_argument("--reload", action="store_true", help="Enable auto-reload on code changes")
    args = parser.parse_args()

    print("=" * 70)
    print("  PromptShield AI — REST Backend Service")
    print(f"  Status: Active & Listening on http://{args.host}:{args.port}")
    print(f"  Interactive Swagger Docs: http://{args.host}:{args.port}/docs")
    print("  Extension Endpoint: http://localhost:8000/analyze")
    print("=" * 70)
    print("  Keep this terminal open while using the Chrome extension.")
    print("  To stop the server, press CTRL+C.\n")

    uvicorn.run(
        "backend.api.app:app",
        host=args.host,
        port=args.port,
        reload=args.reload,
        log_level="info",
    )


if __name__ == "__main__":
    main()
