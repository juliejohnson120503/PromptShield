"""
PromptShield AI — Phase 6 REST API Service Runner
Starts the Uvicorn server hosting the FastAPI backend service.

Usage:
    python run_server.py
    python run_server.py --host 0.0.0.0 --port 8000 --reload
"""

import argparse
import uvicorn


def main():
    parser = argparse.ArgumentParser(description="PromptShield AI REST API Server")
    parser.add_argument("--host", default="127.0.0.1", help="Host interface to bind (default: 127.0.0.1)")
    parser.add_argument("--port", type=int, default=8000, help="Port to listen on (default: 8000)")
    parser.add_argument("--reload", action="store_true", help="Enable auto-reload on code changes")
    args = parser.parse_args()

    print(f"Starting PromptShield AI REST Service on http://{args.host}:{args.port}")
    print(f"Interactive API Documentation: http://{args.host}:{args.port}/docs")

    uvicorn.run(
        "backend.api.app:app",
        host=args.host,
        port=args.port,
        reload=args.reload,
        log_level="info",
    )


if __name__ == "__main__":
    main()
