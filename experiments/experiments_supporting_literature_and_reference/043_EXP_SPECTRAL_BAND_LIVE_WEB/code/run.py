"""
CLI entry point for Spectral Band Graph Viewer.

Usage:
    python run.py --mode web      # Start web server
"""

import argparse
import asyncio
import os
import signal
import sys
import threading


def run_web(args):
    # Windows proactor loop can throw noisy ConnectionResetError when clients
    # close polling requests; selector loop is quieter and more stable here.
    if hasattr(asyncio, "WindowsSelectorEventLoopPolicy"):
        asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())
    
    # Force immediate exit on Ctrl+C (os._exit bypasses cleanup hangs)
    def force_exit(signum, frame):
        print("\nForce exit...")
        os._exit(0)
    
    signal.signal(signal.SIGINT, force_exit)
    if hasattr(signal, "SIGBREAK"):
        signal.signal(signal.SIGBREAK, force_exit)

    import uvicorn
    from server import app

    print(f"Starting Spectral Band Graph Viewer on http://localhost:{args.port}")
    print("Press Ctrl+C to stop")
    
    # Run uvicorn in a daemon thread so main thread can catch signals
    def run_server():
        config = uvicorn.Config(app, host="0.0.0.0", port=args.port, log_level="info")
        server = uvicorn.Server(config)
        server.run()
    
    server_thread = threading.Thread(target=run_server, daemon=True)
    server_thread.start()
    
    # Keep main thread alive to receive signals
    try:
        while server_thread.is_alive():
            server_thread.join(timeout=1.0)
    except KeyboardInterrupt:
        print("\nShutting down...")
        os._exit(0)


def main():
    parser = argparse.ArgumentParser(description="Spectral Band Graph Viewer")
    parser.add_argument("--mode", type=str, default="web", choices=["web"],
                        help="Run mode: web (server)")
    parser.add_argument("--port", type=int, default=8042, help="Web server port")
    args = parser.parse_args()
    run_web(args)


if __name__ == "__main__":
    main()