"""Entry point: launches the Lambert's-problem orbit-determination GUI."""

from gui import LambertGUI

if __name__ == "__main__":
    app = LambertGUI()
    app.mainloop()
