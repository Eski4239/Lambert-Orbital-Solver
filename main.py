"""Entry point: starts the Orbit Lab web app and opens it in the browser.

    python main.py                 # http://127.0.0.1:8050
    python main.py --port 8060     # another port
    python main.py --no-browser

The original Tkinter GUI is still available: python legacy/gui.py
"""

import argparse
import threading
import webbrowser

from app.server import create_app

HOST = "127.0.0.1"

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Orbit Lab")
    parser.add_argument("--port", type=int, default=8050)
    parser.add_argument("--no-browser", action="store_true")
    args = parser.parse_args()

    app = create_app()
    if not args.no_browser:
        threading.Timer(1.2, lambda: webbrowser.open(f"http://{HOST}:{args.port}")).start()
    app.run(host=HOST, port=args.port, debug=False)
