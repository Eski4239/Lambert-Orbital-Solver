"""Entry point: starts the Orbit Lab web app and opens it in the browser.

    python main.py            # http://127.0.0.1:8050
    python main.py --no-browser

The original Tkinter GUI is still available: python legacy/gui.py
"""

import sys
import threading
import webbrowser

from app.server import create_app

HOST, PORT = "127.0.0.1", 8050

if __name__ == "__main__":
    app = create_app()
    if "--no-browser" not in sys.argv:
        threading.Timer(1.2, lambda: webbrowser.open(f"http://{HOST}:{PORT}")).start()
    app.run(host=HOST, port=PORT, debug=False)
