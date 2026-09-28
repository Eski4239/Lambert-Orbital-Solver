"""
Tkinter GUI front end for the Lambert-problem orbit-determination tool.

Two input modes:
- "AER + time": two topocentric observations (UTC datetime, Az, El,
  Range) plus an editable fixed station ECEF position. Each observation's
  GMST is computed independently (time_utils.gmst_degrees) since the two
  epochs are generally hours apart.
- "Position vectors + time": directly supply r1, r2 (ECI, km) and dt (s),
  skipping the coordinate transform.

Both modes funnel into the same core functions used by verify.py --
lamsolbert.lamsolbert and elements.rv_to_elements -- so there is no
duplicated orbital-mechanics logic between the GUI and the verified core.

Layout: the window uses a `grid` layout with weighted rows/columns so it
resizes cleanly at any screen size. The left column (inputs + results) is
wrapped in a scrollable canvas so it stays reachable even when the window
is small; the right column (3D plot) expands to fill remaining space.

Color palette: a small fixed set of colors (see COLOR_* constants below)
is applied via ttk.Style so button/label emphasis is consistent and does
not depend on the OS theme.
"""

import os
import sys
import tkinter as tk
from tkinter import ttk, messagebox
from datetime import datetime, timezone

import numpy as np
import matplotlib
matplotlib.use("TkAgg")
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.figure import Figure

# Allow `python legacy/gui.py` from anywhere: make the project root importable.
_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from core.time_utils import gmst_degrees
from core.frames import aer_to_eci, R_EARTH
from core.lamsolbert import lamsolbert
from core.elements import rv_to_elements, ECC_TOL

MU_EARTH = 398600.4418

DEFAULT_STATION_ECEF = (1344.143, 6068.601, 1429.311)

# ----------------------------------------------------------------------
# Color palette (single source of truth for all widget colors below).
# Kept small and applied via ttk.Style so appearance is consistent
# regardless of the OS's default ttk theme (light/dark).
# ----------------------------------------------------------------------
COLOR_BG = "#f4f6f8"
COLOR_PANEL_BG = "#ffffff"
COLOR_TEXT = "#1f2933"
COLOR_MUTED_TEXT = "#52606d"
COLOR_PRIMARY = "#2563eb"        # Solve button - the primary action
COLOR_PRIMARY_ACTIVE = "#1d4ed8"
COLOR_PRIMARY_TEXT = "#ffffff"
COLOR_SECONDARY = "#e2e8f0"      # neutral controls (radiobuttons area)
COLOR_ACCENT = "#0d9488"         # r1 / obs-1 marker, secondary emphasis
COLOR_ACCENT2 = "#dc2626"        # r2 / obs-2 marker, error emphasis
COLOR_ORBIT = "#f59e0b"          # orbit trace color


def _parse_float(entry_widget, field_name):
    text = entry_widget.get().strip()
    try:
        return float(text)
    except ValueError:
        raise ValueError(f"'{field_name}' must be a number (got: {text!r})")


def _parse_datetime_utc(entry_widget, field_name):
    text = entry_widget.get().strip()
    try:
        # Accept "YYYY-MM-DD HH:MM:SS" or with a "T" separator.
        text_norm = text.replace("T", " ")
        dt = datetime.strptime(text_norm, "%Y-%m-%d %H:%M:%S")
        return dt.replace(tzinfo=timezone.utc)
    except ValueError:
        raise ValueError(
            f"'{field_name}' must be UTC datetime as YYYY-MM-DD HH:MM:SS "
            f"(got: {text!r})"
        )


class LambertGUI(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("Lambert's Problem - Orbit Determination")
        self.configure(bg=COLOR_BG)

        # Responsive default size: a fraction of the screen, with a floor
        # so the window never opens too small to use, and never larger
        # than the screen itself.
        screen_w = self.winfo_screenwidth()
        screen_h = self.winfo_screenheight()
        win_w = max(900, min(int(screen_w * 0.8), 1400))
        win_h = max(600, min(int(screen_h * 0.8), 900))
        self.geometry(f"{win_w}x{win_h}")
        self.minsize(820, 560)

        self.mode = tk.StringVar(value="aer")

        self._configure_style()
        self._build_layout()

    # ------------------------------------------------------------------
    # Style / palette
    # ------------------------------------------------------------------
    def _configure_style(self):
        style = ttk.Style(self)
        # "clam" renders custom colors reliably across platforms, unlike
        # some native themes that ignore ttk background/foreground options.
        try:
            style.theme_use("clam")
        except tk.TclError:
            pass

        style.configure("TFrame", background=COLOR_BG)
        style.configure("Panel.TFrame", background=COLOR_PANEL_BG)
        style.configure("TLabelframe", background=COLOR_BG, foreground=COLOR_TEXT)
        style.configure("TLabelframe.Label", background=COLOR_BG, foreground=COLOR_TEXT,
                         font=("Segoe UI", 9, "bold"))
        style.configure("TLabel", background=COLOR_BG, foreground=COLOR_TEXT)
        # Small muted caption style for inline hints (units, format examples).
        style.configure("Hint.TLabel", background=COLOR_BG, foreground=COLOR_MUTED_TEXT,
                         font=("Segoe UI", 8))
        style.configure("FieldLabel.TLabel", background=COLOR_BG, foreground=COLOR_MUTED_TEXT,
                         font=("Segoe UI", 9))
        style.configure("TRadiobutton", background=COLOR_BG, foreground=COLOR_TEXT)
        style.map("TRadiobutton", background=[("active", COLOR_SECONDARY)])

        # Primary action button (Solve): distinctive accent color, larger
        # padding, bold text -- the one action the user must not miss.
        style.configure(
            "Primary.TButton",
            background=COLOR_PRIMARY,
            foreground=COLOR_PRIMARY_TEXT,
            font=("Segoe UI", 11, "bold"),
            padding=(12, 10),
            borderwidth=0,
        )
        style.map(
            "Primary.TButton",
            background=[("active", COLOR_PRIMARY_ACTIVE), ("pressed", COLOR_PRIMARY_ACTIVE)],
            foreground=[("disabled", COLOR_MUTED_TEXT)],
        )

    # ------------------------------------------------------------------
    # Layout construction
    # ------------------------------------------------------------------
    def _build_layout(self):
        self.columnconfigure(0, weight=0)  # left: inputs/results, fixed-ish
        self.columnconfigure(1, weight=1)  # right: plot, grows with window
        self.rowconfigure(0, weight=1)

        left_container = ttk.Frame(self, padding=(10, 10, 5, 10))
        left_container.grid(row=0, column=0, sticky="nsw")
        left_container.rowconfigure(0, weight=1)
        left_container.columnconfigure(0, weight=1)

        right = ttk.Frame(self, padding=(5, 10, 10, 10))
        right.grid(row=0, column=1, sticky="nsew")
        right.rowconfigure(0, weight=1)
        right.columnconfigure(0, weight=1)

        left_scrollable = self._build_scrollable_left_column(left_container)
        self._build_input_panel(left_scrollable)
        self._build_results_panel(left_scrollable)
        self._build_plot_panel(right)

    def _build_scrollable_left_column(self, parent):
        """
        Wrap the left column's contents in a Canvas + Scrollbar so that on
        small screens the input fields remain reachable (scrollable)
        instead of being clipped or forcing a minimum window size.
        """
        # Widened from the original 340px so labels/hints/values have room
        # to breathe; still non-expanding (grid weight 0 on this column)
        # so the 3D plot column keeps growing with the window and stays
        # the visually dominant panel.
        canvas = tk.Canvas(parent, bg=COLOR_BG, highlightthickness=0, width=380)
        vscroll = ttk.Scrollbar(parent, orient="vertical", command=canvas.yview)
        canvas.configure(yscrollcommand=vscroll.set)

        canvas.grid(row=0, column=0, sticky="nsew")
        vscroll.grid(row=0, column=1, sticky="ns")

        inner = ttk.Frame(canvas)
        inner_id = canvas.create_window((0, 0), window=inner, anchor="nw")

        def _on_inner_configure(_event):
            canvas.configure(scrollregion=canvas.bbox("all"))

        def _on_canvas_configure(event):
            # Keep the inner frame's width matched to the canvas so widgets
            # can stretch to fill the available width as the window grows.
            canvas.itemconfigure(inner_id, width=event.width)

        inner.bind("<Configure>", _on_inner_configure)
        canvas.bind("<Configure>", _on_canvas_configure)

        def _on_mousewheel(event):
            canvas.yview_scroll(int(-1 * (event.delta / 120)), "units")

        # Scoped to only fire while the pointer is over this canvas (bind/
        # unbind on Enter/Leave) rather than bind_all, so scrolling over the
        # 3D plot drives the plot's own zoom handler instead of fighting
        # with this scrollbar.
        def _bind_wheel(_event):
            canvas.bind_all("<MouseWheel>", _on_mousewheel)

        def _unbind_wheel(_event):
            canvas.unbind_all("<MouseWheel>")

        canvas.bind("<Enter>", _bind_wheel)
        canvas.bind("<Leave>", _unbind_wheel)

        inner.columnconfigure(0, weight=1)
        return inner

    def _build_input_panel(self, parent):
        row = 0

        mode_frame = ttk.LabelFrame(parent, text="Input mode", padding=8)
        mode_frame.grid(row=row, column=0, sticky="ew", pady=(0, 8))
        mode_frame.columnconfigure(0, weight=1)
        row += 1

        ttk.Radiobutton(mode_frame, text="AER + time", variable=self.mode,
                         value="aer", command=self._on_mode_change).grid(
            row=0, column=0, sticky="w", pady=2)
        ttk.Label(mode_frame, text="Two topocentric Az/El/Range observations",
                  style="Hint.TLabel").grid(row=1, column=0, sticky="w", padx=(20, 0))
        ttk.Radiobutton(mode_frame, text="Position vectors + time", variable=self.mode,
                         value="vectors", command=self._on_mode_change).grid(
            row=2, column=0, sticky="w", pady=(6, 2))
        ttk.Label(mode_frame, text="Directly supply r1, r2 (ECI km) and dt (s)",
                  style="Hint.TLabel").grid(row=3, column=0, sticky="w", padx=(20, 0))

        self.aer_frame = ttk.LabelFrame(parent, text="AER observations", padding=8)
        self.vec_frame = ttk.LabelFrame(parent, text="Position vectors", padding=8)
        self.aer_frame.columnconfigure(0, weight=1)
        self.vec_frame.columnconfigure(0, weight=1)

        self._build_aer_frame(self.aer_frame)
        self._build_vector_frame(self.vec_frame)

        self._input_frame_row = row
        self.aer_frame.grid(row=row, column=0, sticky="ew", pady=(0, 8))
        row += 1
        # vec_frame grid/remove on mode change, same row as aer_frame

        solve_btn = ttk.Button(parent, text="Solve", command=self._on_solve,
                                style="Primary.TButton")
        solve_btn.grid(row=row, column=0, sticky="ew", pady=(4, 8), ipady=2)
        row += 1

        self._results_start_row = row

    def _build_aer_frame(self, frame):
        station_frame = ttk.LabelFrame(frame, text="Ground station (ECEF, km)", padding=6)
        station_frame.grid(row=0, column=0, sticky="ew", pady=(0, 6))
        for c in range(6):
            station_frame.columnconfigure(c, weight=1 if c % 2 else 0)

        ttk.Label(station_frame, text="Fixed Earth-centered, Earth-fixed position",
                  style="Hint.TLabel").grid(row=0, column=0, columnspan=6, sticky="w",
                                             pady=(0, 4))

        self.station_entries = {}
        for i, axis in enumerate(("X", "Y", "Z")):
            ttk.Label(station_frame, text=f"{axis}:", style="FieldLabel.TLabel").grid(
                row=1, column=2 * i, sticky="e", padx=(0, 2))
            e = ttk.Entry(station_frame, width=10)
            e.insert(0, str(DEFAULT_STATION_ECEF[i]))
            e.grid(row=1, column=2 * i + 1, sticky="ew", padx=(0, 8))
            self.station_entries[axis] = e

        self.obs_entries = {}
        for obs_row, (obs_idx, defaults) in enumerate(zip((1, 2), [
            {"dt": "2023-04-02 00:30:00", "az": "132.67", "el": "32.44", "rng": "16945.450"},
            {"dt": "2023-04-02 03:00:00", "az": "123.08", "el": "50.06", "rng": "37350.340"},
        ]), start=1):
            obs_frame = ttk.LabelFrame(frame, text=f"Observation {obs_idx}", padding=6)
            obs_frame.grid(row=obs_row, column=0, sticky="ew", pady=(0, 6))
            obs_frame.columnconfigure(1, weight=1)

            labels = [
                ("UTC datetime", "e.g. 2023-04-02 00:30:00", "dt"),
                ("Azimuth (deg)", "clockwise from North, 0-360", "az"),
                ("Elevation (deg)", "up from horizon, -90 to 90", "el"),
                ("Range (km)", "slant range to target", "rng"),
            ]
            entries = {}
            grid_row = 0
            for label, hint, key in labels:
                ttk.Label(obs_frame, text=label, style="FieldLabel.TLabel").grid(
                    row=grid_row, column=0, sticky="w", pady=(4, 0), padx=(0, 6))
                e = ttk.Entry(obs_frame)
                e.insert(0, defaults[key])
                e.grid(row=grid_row, column=1, sticky="ew", pady=(4, 0))
                entries[key] = e
                grid_row += 1
                ttk.Label(obs_frame, text=hint, style="Hint.TLabel").grid(
                    row=grid_row, column=0, columnspan=2, sticky="w")
                grid_row += 1
            self.obs_entries[obs_idx] = entries

    def _build_vector_frame(self, frame):
        self.vector_entries = {}
        defaults = {
            "r1x": "15945.34", "r1y": "0.0", "r1z": "0.0",
            "r2x": "12214.83899", "r2y": "10249.46731", "r2z": "0.0",
            "dt": "4560.0",
        }
        frame.columnconfigure(1, weight=1)

        ttk.Label(frame, text="r1 - position at epoch 1 (ECI, km)",
                  style="Hint.TLabel").grid(row=0, column=0, columnspan=2, sticky="w")
        for i, axis in enumerate(("x", "y", "z")):
            ttk.Label(frame, text=f"r1 {axis}", style="FieldLabel.TLabel").grid(
                row=1 + i, column=0, sticky="w", pady=1, padx=(0, 6))
            e = ttk.Entry(frame)
            e.insert(0, defaults[f"r1{axis}"])
            e.grid(row=1 + i, column=1, sticky="ew", pady=1)
            self.vector_entries[f"r1{axis}"] = e

        ttk.Label(frame, text="r2 - position at epoch 2 (ECI, km)",
                  style="Hint.TLabel").grid(row=4, column=0, columnspan=2, sticky="w",
                                             pady=(8, 0))
        for i, axis in enumerate(("x", "y", "z")):
            ttk.Label(frame, text=f"r2 {axis}", style="FieldLabel.TLabel").grid(
                row=5 + i, column=0, sticky="w", pady=1, padx=(0, 6))
            e = ttk.Entry(frame)
            e.insert(0, defaults[f"r2{axis}"])
            e.grid(row=5 + i, column=1, sticky="ew", pady=1)
            self.vector_entries[f"r2{axis}"] = e

        ttk.Label(frame, text="Time of flight, r1 -> r2", style="Hint.TLabel").grid(
            row=8, column=0, columnspan=2, sticky="w", pady=(8, 0))
        ttk.Label(frame, text="dt (s)", style="FieldLabel.TLabel").grid(
            row=9, column=0, sticky="w", pady=1, padx=(0, 6))
        e = ttk.Entry(frame)
        e.insert(0, defaults["dt"])
        e.grid(row=9, column=1, sticky="ew", pady=1)
        self.vector_entries["dt"] = e

    def _build_results_panel(self, parent):
        results_frame = ttk.LabelFrame(parent, text="Orbital elements", padding=8)
        results_frame.grid(row=self._results_start_row, column=0, sticky="nsew", pady=(0, 8))
        results_frame.columnconfigure(0, weight=1)
        results_frame.rowconfigure(0, weight=1)
        parent.rowconfigure(self._results_start_row, weight=1)

        self.results_text = tk.Text(
            results_frame, width=1, height=18, state="disabled", wrap="word",
            font=("Consolas", 10), bg=COLOR_PANEL_BG, fg=COLOR_TEXT,
            relief="flat", borderwidth=0,
        )
        self.results_text.grid(row=0, column=0, sticky="nsew")

    def _build_plot_panel(self, parent):
        plot_frame = ttk.LabelFrame(parent, text="3D orbit view  (drag to rotate, scroll to zoom)",
                                     padding=6)
        plot_frame.grid(row=0, column=0, sticky="nsew")
        plot_frame.rowconfigure(0, weight=1)
        plot_frame.columnconfigure(0, weight=1)

        self.fig = Figure(figsize=(6, 6), facecolor=COLOR_PANEL_BG)
        self.ax3d = self.fig.add_subplot(111, projection="3d")
        self.canvas = FigureCanvasTkAgg(self.fig, master=plot_frame)
        self.canvas.get_tk_widget().grid(row=0, column=0, sticky="nsew")
        self._draw_empty_plot()
        self._setup_plot_interaction()

    def _setup_plot_interaction(self):
        """
        Replace Matplotlib's default 3D mouse handling with a damped
        version. The stock Axes3D drag-to-rotate is 1:1 pixel-to-degree
        with no elevation clamp, so a fast or long drag can spin past the
        pole and flip the view disorientingly; and there is no scroll-to-
        zoom at all by default. Here rotation is scaled down and elevation
        is clamped to (-89, 89) deg to avoid the pole-flip, and scroll
        drives an explicit, gently-stepped zoom on the axis limits.
        """
        # Disable the built-in trackball-rotate button handling so our own
        # handlers are the only thing driving view_init.
        self.ax3d.disable_mouse_rotation()

        self._drag_state = {"button_down": False, "last_xy": None}
        self._view_elev = self.ax3d.elev
        self._view_azim = self.ax3d.azim
        self._zoom_scale = 1.0  # relative to the scale set by the last Solve/reset

        def on_press(event):
            if event.inaxes != self.ax3d or event.button != 1:
                return
            self._drag_state["button_down"] = True
            self._drag_state["last_xy"] = (event.x, event.y)

        def on_release(event):
            self._drag_state["button_down"] = False
            self._drag_state["last_xy"] = None

        def on_motion(event):
            if not self._drag_state["button_down"] or event.inaxes != self.ax3d:
                return
            last_x, last_y = self._drag_state["last_xy"]
            dx = event.x - last_x
            dy = event.y - last_y
            self._drag_state["last_xy"] = (event.x, event.y)

            # Damping factor: default Matplotlib rotation is ~1 deg per
            # pixel, which feels frantic; scale it down substantially.
            sensitivity = 0.25
            self._view_azim = (self._view_azim - dx * sensitivity) % 360
            self._view_elev = float(np.clip(
                self._view_elev + dy * sensitivity, -89.0, 89.0))
            self.ax3d.view_init(elev=self._view_elev, azim=self._view_azim)
            self.canvas.draw_idle()

        def on_scroll(event):
            if event.inaxes != self.ax3d:
                return
            # Gentle, fixed-step zoom per wheel notch rather than a
            # continuous/uncapped factor, so scrolling can't runaway-zoom.
            step = 0.9 if event.button == "up" else (1 / 0.9)
            self._zoom_scale *= step
            self._zoom_scale = float(np.clip(self._zoom_scale, 0.05, 20.0))
            self._rescale_view()

        self.canvas.mpl_connect("button_press_event", on_press)
        self.canvas.mpl_connect("button_release_event", on_release)
        self.canvas.mpl_connect("motion_notify_event", on_motion)
        self.canvas.mpl_connect("scroll_event", on_scroll)

    def _rescale_view(self):
        """Apply the current zoom factor on top of the last base range."""
        base = getattr(self, "_base_range", R_EARTH * 3)
        r = base * self._zoom_scale
        self.ax3d.set_xlim(-r, r)
        self.ax3d.set_ylim(-r, r)
        self.ax3d.set_zlim(-r, r)
        self.ax3d.set_box_aspect([1, 1, 1])
        self.canvas.draw_idle()

    def _on_mode_change(self):
        if self.mode.get() == "aer":
            self.vec_frame.grid_remove()
            self.aer_frame.grid(row=self._input_frame_row, column=0, sticky="ew", pady=(0, 8))
        else:
            self.aer_frame.grid_remove()
            self.vec_frame.grid(row=self._input_frame_row, column=0, sticky="ew", pady=(0, 8))

    # ------------------------------------------------------------------
    # Solve pipeline
    # ------------------------------------------------------------------
    def _on_solve(self):
        try:
            if self.mode.get() == "aer":
                r1, r2, dt = self._gather_aer_inputs()
            else:
                r1, r2, dt = self._gather_vector_inputs()

            if dt <= 0:
                raise ValueError("Time of flight (dt) must be positive.")

            v1, v2 = lamsolbert(r1, r2, dt, mu=MU_EARTH, prograde=True)
            els = rv_to_elements(r1, v1, mu=MU_EARTH)

            self._display_results(r1, r2, v1, v2, els)
            self._plot_orbit(r1, r2, els)

        except ValueError as exc:
            messagebox.showerror("Invalid input", str(exc))
        except RuntimeError as exc:
            messagebox.showerror("Solver error", str(exc))
        except Exception as exc:  # noqa: BLE001 - last-resort friendly error, not a stack trace
            messagebox.showerror("Unexpected error", f"Something went wrong:\n{exc}")

    def _gather_aer_inputs(self):
        station_ecef = [
            _parse_float(self.station_entries[axis], f"Station {axis}")
            for axis in ("X", "Y", "Z")
        ]

        obs_data = []
        for obs_idx in (1, 2):
            entries = self.obs_entries[obs_idx]
            dt_utc = _parse_datetime_utc(entries["dt"], f"Observation {obs_idx} datetime")
            az = _parse_float(entries["az"], f"Observation {obs_idx} azimuth")
            el = _parse_float(entries["el"], f"Observation {obs_idx} elevation")
            rng = _parse_float(entries["rng"], f"Observation {obs_idx} range")

            if rng <= 0:
                raise ValueError(f"Observation {obs_idx} range must be positive.")
            if not (-90 <= el <= 90):
                raise ValueError(f"Observation {obs_idx} elevation must be within [-90, 90] deg.")
            if not (0 <= az < 360):
                raise ValueError(f"Observation {obs_idx} azimuth must be within [0, 360) deg.")

            obs_data.append((dt_utc, az, el, rng))

        dt_seconds = (obs_data[1][0] - obs_data[0][0]).total_seconds()

        gmst1 = gmst_degrees(obs_data[0][0])
        gmst2 = gmst_degrees(obs_data[1][0])

        r1 = aer_to_eci(obs_data[0][1], obs_data[0][2], obs_data[0][3], station_ecef, gmst1)
        r2 = aer_to_eci(obs_data[1][1], obs_data[1][2], obs_data[1][3], station_ecef, gmst2)

        return r1, r2, dt_seconds

    def _gather_vector_inputs(self):
        v = self.vector_entries
        r1 = np.array([
            _parse_float(v["r1x"], "r1 x"),
            _parse_float(v["r1y"], "r1 y"),
            _parse_float(v["r1z"], "r1 z"),
        ])
        r2 = np.array([
            _parse_float(v["r2x"], "r2 x"),
            _parse_float(v["r2y"], "r2 y"),
            _parse_float(v["r2z"], "r2 z"),
        ])
        dt = _parse_float(v["dt"], "dt")
        return r1, r2, dt

    # ------------------------------------------------------------------
    # Display
    # ------------------------------------------------------------------
    def _display_results(self, r1, r2, v1, v2, els):
        def fmt(x, unit=""):
            if x is None:
                return "undefined"
            return f"{x:.6g}{unit}"

        lines = []
        lines.append("--- State vectors ---")
        lines.append(f"r1 = [{r1[0]:.3f}, {r1[1]:.3f}, {r1[2]:.3f}] km")
        lines.append(f"r2 = [{r2[0]:.3f}, {r2[1]:.3f}, {r2[2]:.3f}] km")
        lines.append(f"v1 = [{v1[0]:.6f}, {v1[1]:.6f}, {v1[2]:.6f}] km/s")
        lines.append(f"v2 = [{v2[0]:.6f}, {v2[1]:.6f}, {v2[2]:.6f}] km/s")
        lines.append("")
        lines.append("--- Classical orbital elements (at obs 1 epoch) ---")
        lines.append(f"Semi-major axis a : {fmt(els['a'], ' km')}")
        lines.append(f"Eccentricity e    : {fmt(els['e'])}")
        lines.append(f"Inclination i     : {fmt(els['i'], ' deg')}")
        lines.append(f"RAAN              : {fmt(els['raan'], ' deg')}")
        lines.append(f"Argument of perigee: {fmt(els['argp'], ' deg')}")
        lines.append(f"True anomaly nu   : {fmt(els['nu'], ' deg')}")
        lines.append(f"Ang. momentum h   : {fmt(els['h'], ' km^2/s')}")

        if els["notes"]:
            lines.append("")
            lines.append("Notes:")
            for note in els["notes"]:
                lines.append(f"  - {note}")

        self.results_text.configure(state="normal")
        self.results_text.delete("1.0", "end")
        self.results_text.insert("1.0", "\n".join(lines))
        self.results_text.configure(state="disabled")

    # ------------------------------------------------------------------
    # Plot
    # ------------------------------------------------------------------
    # Extra headroom applied on top of the geometry-fit range so the model
    # (Earth sphere / orbit ellipse) doesn't crowd all the way out to the
    # box edges -- without this the axis panes, ticks, and grid lines sit
    # right up against the surface and are hard to see.
    _AXIS_PADDING_FACTOR = 1.6

    def _apply_equal_3d_scale(self, max_range):
        """
        Lock the 3D axes to a cubic box spanning +/- (max_range *
        padding) on every axis, with visible panes/grid/ticks so the
        axes read clearly around the plotted geometry rather than being
        crowded out by it. Matplotlib's 3D axes default to an unequal box
        aspect even when xlim/ylim/zlim are set identically, which
        distorts the Earth sphere into an ellipsoid -- set_box_aspect
        ([1,1,1]) is what actually fixes that (Matplotlib >= 3.3).
        """
        padded_range = max_range * self._AXIS_PADDING_FACTOR
        self.ax3d.set_xlim(-padded_range, padded_range)
        self.ax3d.set_ylim(-padded_range, padded_range)
        self.ax3d.set_zlim(-padded_range, padded_range)
        self.ax3d.set_box_aspect([1, 1, 1])
        self._style_axes()
        # Record as the zoom baseline: the scroll-wheel zoom handler
        # multiplies this range rather than the arbitrary current limits,
        # and each Solve/reset here establishes a fresh, correctly-fit
        # baseline (already padded) with zoom reset to 1x.
        self._base_range = padded_range
        self._zoom_scale = 1.0

    def _style_axes(self):
        """
        Make the axis panes, grid lines, and ticks clearly visible against
        the plot background, filling the full cubic box rather than
        fading into it -- addresses the axes becoming hard to see once
        the Earth/orbit geometry is drawn close to the box edges.
        """
        pane_color = (0.93, 0.95, 0.97, 1.0)
        grid_color = (0.55, 0.6, 0.65, 0.6)
        for axis in (self.ax3d.xaxis, self.ax3d.yaxis, self.ax3d.zaxis):
            axis.pane.set_facecolor(pane_color)
            axis.pane.set_edgecolor((0.6, 0.65, 0.7, 1.0))
            axis._axinfo["grid"]["color"] = grid_color
            axis._axinfo["grid"]["linewidth"] = 0.7
            axis.line.set_color((0.4, 0.45, 0.5, 1.0))
            axis.set_tick_params(colors=COLOR_MUTED_TEXT, labelsize=8)

    def _draw_empty_plot(self):
        self.ax3d.clear()
        self._draw_earth()
        self._apply_equal_3d_scale(R_EARTH * 3)
        self.ax3d.set_xlabel("X (km)")
        self.ax3d.set_ylabel("Y (km)")
        self.ax3d.set_zlabel("Z (km)")
        self.ax3d.set_title("Orbit (solve to populate)", color=COLOR_TEXT)
        self.canvas.draw()

    def _draw_earth(self):
        """
        Draw a stylized, shaded globe: no external texture/image is used
        (project is numpy/scipy/matplotlib only, no PIL, no guaranteed
        internet access), so realism comes from a higher-resolution mesh,
        simple one-sided directional lighting (Lambertian shading against
        a fixed light direction), an ocean-blue base color with lighter
        polar ice caps by latitude, and faint lat/lon grid lines.
        """
        n_lon, n_lat = 80, 40
        u, v = np.mgrid[0:2 * np.pi:n_lon * 1j, 0:np.pi:n_lat * 1j]
        x = R_EARTH * np.cos(u) * np.sin(v)
        y = R_EARTH * np.sin(u) * np.sin(v)
        z = R_EARTH * np.cos(v)

        # Surface normals on a sphere point radially outward.
        nx, ny, nz = x / R_EARTH, y / R_EARTH, z / R_EARTH

        # Fixed light direction (upper-left-front), normalized.
        light = np.array([-0.5, -0.6, 0.7])
        light = light / np.linalg.norm(light)
        intensity = np.clip(nx * light[0] + ny * light[1] + nz * light[2], 0, 1)
        # Ambient floor so the night side isn't pure black.
        shade = 0.35 + 0.65 * intensity

        # Base ocean color (RGB) with lighter polar ice caps by latitude.
        lat = np.pi / 2 - v  # v=0 at north pole -> lat=+90deg
        ocean = np.array([0.11, 0.42, 0.75])
        ice = np.array([0.92, 0.95, 0.98])
        polar_frac = np.clip((np.abs(lat) - np.radians(60)) / np.radians(20), 0, 1)
        base_rgb = ocean[None, None, :] * (1 - polar_frac[..., None]) + \
            ice[None, None, :] * polar_frac[..., None]

        rgb = base_rgb * shade[..., None]
        rgba = np.dstack([rgb, np.ones_like(shade)])

        self.ax3d.plot_surface(
            x, y, z, facecolors=rgba, linewidth=0, antialiased=True,
            shade=False, rcount=n_lat, ccount=n_lon,
        )

        # Faint lat/lon grid lines for a more globe-like read of curvature.
        grid_color = (1, 1, 1, 0.15)
        for lat_deg in range(-60, 90, 30):
            vv = np.full(100, np.radians(90 - lat_deg))
            uu = np.linspace(0, 2 * np.pi, 100)
            gx = R_EARTH * np.cos(uu) * np.sin(vv)
            gy = R_EARTH * np.sin(uu) * np.sin(vv)
            gz = R_EARTH * np.cos(vv)
            self.ax3d.plot(gx, gy, gz, color=grid_color, linewidth=0.6)
        for lon_deg in range(0, 360, 30):
            uu = np.full(100, np.radians(lon_deg))
            vv = np.linspace(0, np.pi, 100)
            gx = R_EARTH * np.cos(uu) * np.sin(vv)
            gy = R_EARTH * np.sin(uu) * np.sin(vv)
            gz = R_EARTH * np.cos(vv)
            self.ax3d.plot(gx, gy, gz, color=grid_color, linewidth=0.6)

    def _orbit_ellipse_points(self, r1, a, e, e_vec, h_vec, n_points=300):
        """
        Build the ECI points of the orbit ellipse directly from the
        eccentricity vector (periapsis direction) and angular momentum
        vector (orbit normal), instead of going through RAAN/inclination/
        argument-of-perigee as Euler rotation angles.

        This is the fix for a bug where the old RAAN/i/argp-based
        reconstruction (a) always started its true-anomaly sweep at
        periapsis regardless of where r1 actually sits on the orbit, and
        (b) silently substituted 0 deg for RAAN/argp whenever they were
        None (near-equatorial/near-circular orbits), which is not a valid
        stand-in and visibly mis-rotated the ellipse away from r1/r2.

        e_vec and h_vec are always well-defined (unlike raan/argp), so
        this construction works unconditionally:
            p_hat = periapsis direction (e_vec normalized), or, if the
                    orbit is near-circular (|e_vec| ~ 0) and so has no
                    well-defined periapsis direction, the projection of
                    r1 onto the orbital plane -- this both avoids a 0/0
                    division and guarantees the drawn curve still passes
                    through the plotted r1 marker.
            w_hat = orbit-plane normal (h_vec normalized)
            q_hat = w_hat x p_hat (completes the right-handed in-plane basis)
        Any point on the orbit is then r(nu) = r_orbit(nu) * (cos(nu) p_hat
        + sin(nu) q_hat), which by construction reproduces r1 and r2 at
        their true true-anomaly values -- no phase alignment needed.
        """
        w_hat = h_vec / np.linalg.norm(h_vec)

        if e > ECC_TOL:
            p_hat = e_vec / e
        else:
            r1_in_plane = r1 - np.dot(r1, w_hat) * w_hat
            norm = np.linalg.norm(r1_in_plane)
            p_hat = r1_in_plane / norm if norm > 0 else np.array([1.0, 0.0, 0.0])

        q_hat = np.cross(w_hat, p_hat)

        nu_range = np.linspace(0, 2 * np.pi, n_points)
        p = a * (1 - e**2)
        r_orbit = p / (1 + e * np.cos(nu_range))

        pts = (r_orbit * np.cos(nu_range))[None, :] * p_hat[:, None] + \
            (r_orbit * np.sin(nu_range))[None, :] * q_hat[:, None]
        return pts

    def _plot_orbit(self, r1, r2, els):
        self.ax3d.clear()
        self._draw_earth()

        a, e = els["a"], els["e"]

        if np.isfinite(a) and e < 1.0:
            pts = self._orbit_ellipse_points(r1, a, e, els["e_vec"], els["h_vec"])
            self.ax3d.plot(pts[0], pts[1], pts[2], color=COLOR_ORBIT,
                            linewidth=2, label="Transfer orbit")
        else:
            self.ax3d.set_title("Orbit not plotted (hyperbolic/parabolic or invalid a)",
                                 color=COLOR_TEXT)

        self.ax3d.scatter(*r1, color=COLOR_ACCENT, s=50, label="r1 (obs 1)", depthshade=False)
        self.ax3d.scatter(*r2, color=COLOR_ACCENT2, s=50, label="r2 (obs 2)", depthshade=False)
        self.ax3d.legend(loc="upper left", fontsize=8)

        max_range = max(np.linalg.norm(r1), np.linalg.norm(r2), R_EARTH) * 1.2
        self._apply_equal_3d_scale(max_range)
        self.ax3d.set_xlabel("X (km)")
        self.ax3d.set_ylabel("Y (km)")
        self.ax3d.set_zlabel("Z (km)")

        self.canvas.draw()


if __name__ == "__main__":
    app = LambertGUI()
    app.mainloop()
