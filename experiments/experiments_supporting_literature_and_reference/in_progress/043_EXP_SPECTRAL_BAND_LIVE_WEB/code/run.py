"""
CLI entry point for Spectral Band Graph Viewer.

Usage:
    python run.py --mode web      # Start web server
"""

import argparse
import asyncio


def run_web(args):
    # Windows proactor loop can throw noisy ConnectionResetError when clients
    # close polling requests; selector loop is quieter and more stable here.
    if hasattr(asyncio, "WindowsSelectorEventLoopPolicy"):
        asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())
    
    # Suppress noisy connection reset exceptions (harmless, caused by browser aborting requests)
    import logging
    import sys
    
    class ConnectionResetFilter(logging.Filter):
        def filter(self, record):
            msg = record.getMessage()
            if "ConnectionResetError" in msg or "WinError 10054" in msg:
                return False
            return True
    
    # Also silence asyncio's exception handler for these specific errors
    def silent_exception_handler(loop, context):
        exception = context.get("exception")
        if isinstance(exception, ConnectionResetError):
            return  # Ignore silently
        # Default handling for other exceptions
        loop.default_exception_handler(context)
    
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    loop.set_exception_handler(silent_exception_handler)

    import uvicorn
    from server import app

    print(f"Starting Spectral Band Graph Viewer on http://localhost:{args.port}")
    uvicorn.run(app, host="0.0.0.0", port=args.port, workers=1, loop="asyncio")


def main():
    parser = argparse.ArgumentParser(description="Spectral Band Graph Viewer")
    parser.add_argument("--mode", type=str, default="web", choices=["web"],
                        help="Run mode: web (server)")
    parser.add_argument("--port", type=int, default=8042, help="Web server port")
    args = parser.parse_args()
    run_web(args)


if __name__ == "__main__":
    main()