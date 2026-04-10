"""
BAG Wreck Detection GUI
========================
GUI for scanning NOAA BAG files, detecting wrecks, correcting coordinates,
and visualizing with CARIS-style vertical exaggeration.

Features:
  - Scan all BAG files in a directory (recursive)
  - Real depth anomaly detection (not mock/hardcoded wrecks)
  - Masking gap filtering (gaps ≠ new wrecks)
  - Spatial clustering (126 pings at 3 locations → 3 wrecks, not 126)
  - Cross-file deduplication
  - Coordinate correction from known reference points
  - Vertical exaggeration slider (like CARIS)
  - Color overlay for restored areas (orange/red vs blue/green original)
  - KML/KMZ export
"""

import tkinter as tk
from tkinter import ttk, messagebox, filedialog
import threading
import os
from pathlib import Path
from datetime import datetime
from typing import Optional, List, Dict, Any

import numpy as np

from bag_wreck_detector import (
    BAGPipeline, BAGReader, CoordinateTransformer, RestorationEngine,
    CoordinateCorrection, WreckDetection, BAGInfo,
    HAS_H5PY, HAS_PYPROJ, HAS_SCIPY
)

try:
    import matplotlib
    matplotlib.use('TkAgg')
    from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
    from matplotlib.figure import Figure
    import matplotlib.pyplot as plt
    HAS_MPL = True
except ImportError:
    HAS_MPL = False


class BAGWreckGUI:
    """Main GUI application"""

    DEFAULT_BAG_DIR = r"c:\Temp\Garminjunk\HistoryofCESARSNIFFERBAGFILE\bagfilework\Lake Erie Bag Files"
    DEFAULT_OUTPUT_DIR = r"c:\Temp\Garminjunk\HistoryofCESARSNIFFERBAGFILE\bagfilework\outputs\bag_scan"

    def __init__(self, root):
        self.root = root
        self.root.title("Wreck Hunter 2000 – BAG Wreck Detection")
        self.root.geometry("1400x900")
        self.root.minsize(1100, 700)

        style = ttk.Style()
        style.theme_use('clam')

        # State
        self.pipeline: Optional[BAGPipeline] = None
        self.scan_thread: Optional[threading.Thread] = None
        self.is_scanning = False
        self.detections: List[WreckDetection] = []
        self.last_elevation: Optional[np.ndarray] = None
        self.last_bag_info: Optional[BAGInfo] = None
        self.correction = None  # CoordinateCorrection

        self._build_ui()
        self._check_deps()

    # =======================================================================
    # UI CONSTRUCTION
    # =======================================================================

    def _build_ui(self):
        """Build the complete UI"""
        # Top bar: directory selection
        top = ttk.LabelFrame(self.root, text="Scan Configuration", padding=8)
        top.pack(fill='x', padx=8, pady=(8, 4))

        ttk.Label(top, text="BAG Directory:").grid(row=0, column=0, sticky='w')
        self.dir_var = tk.StringVar(value=self.DEFAULT_BAG_DIR)
        ttk.Entry(top, textvariable=self.dir_var, width=70).grid(
            row=0, column=1, sticky='ew', padx=5)
        ttk.Button(top, text="Browse…", command=self._browse_dir).grid(
            row=0, column=2, padx=5)

        ttk.Label(top, text="Output:").grid(row=1, column=0, sticky='w', pady=(5,0))
        self.out_var = tk.StringVar(value=self.DEFAULT_OUTPUT_DIR)
        ttk.Entry(top, textvariable=self.out_var, width=70).grid(
            row=1, column=1, sticky='ew', padx=5, pady=(5,0))
        ttk.Button(top, text="Browse…", command=self._browse_output).grid(
            row=1, column=2, padx=5, pady=(5,0))

        top.columnconfigure(1, weight=1)

        # Detection parameters
        params = ttk.LabelFrame(self.root, text="Detection Parameters", padding=8)
        params.pack(fill='x', padx=8, pady=4)

        ttk.Label(params, text="Min height above floor (m):").grid(row=0, column=0, sticky='w')
        self.min_height_var = tk.DoubleVar(value=1.0)
        ttk.Spinbox(params, from_=0.1, to=20.0, increment=0.5,
                     textvariable=self.min_height_var, width=8).grid(
            row=0, column=1, padx=5)

        ttk.Label(params, text="Min cluster cells:").grid(row=0, column=2, sticky='w', padx=(20,0))
        self.min_cells_var = tk.IntVar(value=4)
        ttk.Spinbox(params, from_=1, to=100, increment=1,
                     textvariable=self.min_cells_var, width=8).grid(
            row=0, column=3, padx=5)

        ttk.Label(params, text="Dedup radius (m):").grid(row=0, column=4, sticky='w', padx=(20,0))
        self.merge_radius_var = tk.DoubleVar(value=200.0)
        ttk.Spinbox(params, from_=10, to=2000, increment=50,
                     textvariable=self.merge_radius_var, width=8).grid(
            row=0, column=5, padx=5)

        # Coordinate correction
        corr_frame = ttk.LabelFrame(self.root, text="Coordinate Correction (for mis-labeled metadata)", padding=8)
        corr_frame.pack(fill='x', padx=8, pady=4)

        self.use_correction_var = tk.BooleanVar(value=False)
        ttk.Checkbutton(corr_frame, text="Apply correction from known reference point",
                        variable=self.use_correction_var,
                        command=self._toggle_correction).grid(
            row=0, column=0, columnspan=6, sticky='w')

        ttk.Label(corr_frame, text="Ref name:").grid(row=1, column=0, sticky='w')
        self.ref_name_var = tk.StringVar(value="Known Wreck")
        ttk.Entry(corr_frame, textvariable=self.ref_name_var, width=20).grid(
            row=1, column=1, padx=5)

        ttk.Label(corr_frame, text="Known Lat:").grid(row=1, column=2, sticky='w', padx=(10,0))
        self.ref_lat_var = tk.DoubleVar(value=0.0)
        ttk.Entry(corr_frame, textvariable=self.ref_lat_var, width=12).grid(
            row=1, column=3, padx=5)

        ttk.Label(corr_frame, text="Known Lon:").grid(row=1, column=4, sticky='w', padx=(10,0))
        self.ref_lon_var = tk.DoubleVar(value=0.0)
        ttk.Entry(corr_frame, textvariable=self.ref_lon_var, width=12).grid(
            row=1, column=5, padx=5)

        ttk.Label(corr_frame, text="Detected Lat:").grid(row=2, column=2, sticky='w', padx=(10,0))
        self.det_lat_var = tk.DoubleVar(value=0.0)
        ttk.Entry(corr_frame, textvariable=self.det_lat_var, width=12).grid(
            row=2, column=3, padx=5)

        ttk.Label(corr_frame, text="Detected Lon:").grid(row=2, column=4, sticky='w', padx=(10,0))
        self.det_lon_var = tk.DoubleVar(value=0.0)
        ttk.Entry(corr_frame, textvariable=self.det_lon_var, width=12).grid(
            row=2, column=5, padx=5)

        self.offset_label = ttk.Label(corr_frame, text="Offset: not computed", foreground='gray')
        self.offset_label.grid(row=3, column=0, columnspan=6, sticky='w', pady=(5,0))

        # ---- Middle: controls + exaggeration ----
        mid = ttk.Frame(self.root)
        mid.pack(fill='x', padx=8, pady=4)

        # Scan button
        self.btn_scan = ttk.Button(mid, text="▶  Scan All BAG Files",
                                    command=self._start_scan, width=22)
        self.btn_scan.pack(side='left', padx=5)

        self.btn_stop = ttk.Button(mid, text="■  Stop", command=self._stop_scan,
                                    state='disabled', width=10)
        self.btn_stop.pack(side='left', padx=5)

        # Exaggeration slider (CARIS-style)
        ttk.Label(mid, text="Vertical Exaggeration:").pack(side='left', padx=(30, 5))
        self.exag_var = tk.DoubleVar(value=1.0)
        self.exag_scale = ttk.Scale(mid, from_=1.0, to=10.0,
                                     variable=self.exag_var, orient='horizontal',
                                     length=200, command=self._on_exag_changed)
        self.exag_scale.pack(side='left', padx=5)
        self.exag_label = ttk.Label(mid, text="1.0x")
        self.exag_label.pack(side='left')

        # Restoration toggle
        self.restore_var = tk.BooleanVar(value=True)
        ttk.Checkbutton(mid, text="Show Restoration",
                        variable=self.restore_var,
                        command=self._refresh_plot).pack(side='left', padx=(20,5))

        # KML button
        self.btn_kml = ttk.Button(mid, text="Open KML", command=self._open_kml,
                                   state='disabled', width=12)
        self.btn_kml.pack(side='right', padx=5)

        # Remote Sensing launcher
        self.btn_remote = ttk.Button(mid, text="🛰️ Remote Sensing",
                                      command=self._launch_remote_sensing, width=18)
        self.btn_remote.pack(side='right', padx=5)

        # ---- Bottom: results + visualization side by side ----
        bottom = ttk.PanedWindow(self.root, orient='horizontal')
        bottom.pack(fill='both', expand=True, padx=8, pady=(4, 8))

        # Left: results text
        results_frame = ttk.LabelFrame(bottom, text="Results", padding=5)
        bottom.add(results_frame, weight=1)

        self.results_text = tk.Text(results_frame, wrap='word', font=('Consolas', 9),
                                     bg='#1e1e1e', fg='#d4d4d4',
                                     insertbackground='white')
        scroll = ttk.Scrollbar(results_frame, orient='vertical',
                                command=self.results_text.yview)
        self.results_text.configure(yscrollcommand=scroll.set)
        self.results_text.pack(side='left', fill='both', expand=True)
        scroll.pack(side='right', fill='y')

        # Configure text tags for colored output
        self.results_text.tag_configure('header', foreground='#569cd6', font=('Consolas', 10, 'bold'))
        self.results_text.tag_configure('wreck', foreground='#4ec9b0')
        self.results_text.tag_configure('artifact', foreground='#808080')
        self.results_text.tag_configure('error', foreground='#f44747')
        self.results_text.tag_configure('info', foreground='#d4d4d4')
        self.results_text.tag_configure('highlight', foreground='#dcdcaa')
        self.results_text.tag_configure('good', foreground='#b5cea8')

        # Right: visualization canvas
        viz_frame = ttk.LabelFrame(bottom, text="BAG Visualization", padding=5)
        bottom.add(viz_frame, weight=2)

        if HAS_MPL:
            self.fig = Figure(figsize=(7, 5), dpi=100, facecolor='#2d2d2d')
            self.ax = self.fig.add_subplot(111)
            self.ax.set_facecolor('#1e1e1e')
            self.canvas = FigureCanvasTkAgg(self.fig, master=viz_frame)
            self.canvas.get_tk_widget().pack(fill='both', expand=True)
        else:
            ttk.Label(viz_frame, text="matplotlib not available for visualization").pack()

        # Status bar
        self.status_var = tk.StringVar(value="Ready — Select a BAG directory and click Scan")
        ttk.Label(self.root, textvariable=self.status_var, relief='sunken',
                  anchor='w').pack(fill='x', padx=8, pady=(0, 5))

    # =======================================================================
    # DEPENDENCY CHECK
    # =======================================================================

    def _check_deps(self):
        """Check required dependencies and warn"""
        missing = []
        if not HAS_H5PY:
            missing.append("h5py (pip install h5py)")
        if not HAS_PYPROJ:
            missing.append("pyproj (pip install pyproj)")
        if not HAS_SCIPY:
            missing.append("scipy (pip install scipy)")

        if missing:
            msg = "Missing dependencies:\n" + "\n".join(f"  • {m}" for m in missing)
            self._log(msg + "\n\nInstall with: pip install h5py pyproj scipy\n", 'error')

    # =======================================================================
    # ACTIONS
    # =======================================================================

    def _browse_dir(self):
        d = filedialog.askdirectory(initialdir=self.dir_var.get(),
                                     title="Select BAG file directory")
        if d:
            self.dir_var.set(d)

    def _browse_output(self):
        d = filedialog.askdirectory(initialdir=self.out_var.get(),
                                     title="Select output directory")
        if d:
            self.out_var.set(d)

    def _toggle_correction(self):
        """Update offset display when correction is toggled"""
        if self.use_correction_var.get():
            self._compute_offset()
        else:
            self.offset_label.configure(text="Offset: disabled", foreground='gray')

    def _compute_offset(self):
        """Compute coordinate offset from reference point"""
        try:
            known_lat = self.ref_lat_var.get()
            known_lon = self.ref_lon_var.get()
            det_lat = self.det_lat_var.get()
            det_lon = self.det_lon_var.get()

            if known_lat == 0 and known_lon == 0:
                self.offset_label.configure(
                    text="Offset: enter known and detected positions",
                    foreground='orange')
                return

            off_lat = known_lat - det_lat
            off_lon = known_lon - det_lon
            dist_m = ((off_lat * 111000)**2 + (off_lon * 111000 * np.cos(np.radians(known_lat)))**2)**0.5
            self.offset_label.configure(
                text=f"Offset: Δlat={off_lat:+.6f}° Δlon={off_lon:+.6f}° (~{dist_m:.0f}m)",
                foreground='#4ec9b0')
        except Exception as e:
            self.offset_label.configure(text=f"Offset error: {e}", foreground='red')

    def _start_scan(self):
        """Start scanning in background thread"""
        if self.is_scanning:
            return

        bag_dir = self.dir_var.get()
        if not os.path.isdir(bag_dir):
            messagebox.showerror("Error", f"Directory not found:\n{bag_dir}")
            return

        output_dir = self.out_var.get()
        os.makedirs(output_dir, exist_ok=True)

        # Build coordinate correction
        correction = None
        if self.use_correction_var.get():
            try:
                known_lat = self.ref_lat_var.get()
                known_lon = self.ref_lon_var.get()
                det_lat = self.det_lat_var.get()
                det_lon = self.det_lon_var.get()
                correction = CoordinateCorrection(
                    name=self.ref_name_var.get(),
                    known_lat=known_lat,
                    known_lon=known_lon,
                    detected_lat=det_lat,
                    detected_lon=det_lon,
                    offset_lat=known_lat - det_lat,
                    offset_lon=known_lon - det_lon,
                    applied=True
                )
            except Exception as e:
                messagebox.showwarning("Correction Error",
                                        f"Invalid correction values: {e}\nProceeding without correction.")

        self.is_scanning = True
        self.btn_scan.configure(state='disabled')
        self.btn_stop.configure(state='normal')
        self.btn_kml.configure(state='disabled')
        self.results_text.delete('1.0', 'end')
        self.detections = []

        self._log("=" * 60 + "\n", 'header')
        self._log("BAG WRECK DETECTION PIPELINE\n", 'header')
        self._log("=" * 60 + "\n", 'header')
        self._log(f"Directory: {bag_dir}\n", 'info')
        self._log(f"Min height: {self.min_height_var.get()}m\n", 'info')
        self._log(f"Min cluster: {self.min_cells_var.get()} cells\n", 'info')
        self._log(f"Dedup radius: {self.merge_radius_var.get()}m\n", 'info')
        if correction:
            self._log(f"Correction: Δ{correction.offset_lat:+.6f}°, Δ{correction.offset_lon:+.6f}°\n", 'highlight')
        self._log("\n", 'info')

        def progress_cb(current, total, message):
            self.root.after(0, self._update_progress, current, total, message)

        self.pipeline = BAGPipeline(
            min_height_m=self.min_height_var.get(),
            min_cluster_cells=self.min_cells_var.get(),
            merge_radius_m=self.merge_radius_var.get(),
            correction=correction,
            progress_callback=progress_cb
        )

        def scan_worker():
            try:
                report = self.pipeline.run(bag_dir, output_dir)
                self.root.after(0, self._scan_complete, report)
            except Exception as e:
                self.root.after(0, self._scan_error, str(e))

        self.scan_thread = threading.Thread(target=scan_worker, daemon=True)
        self.scan_thread.start()

    def _stop_scan(self):
        """Stop scanning"""
        self.is_scanning = False
        self.status_var.set("Scan stopped")
        self.btn_scan.configure(state='normal')
        self.btn_stop.configure(state='disabled')

    def _update_progress(self, current, total, message):
        """Progress callback from pipeline"""
        if total > 0:
            pct = current / total * 100
            self.status_var.set(f"[{current}/{total}] {pct:.0f}% — {message}")
        else:
            self.status_var.set(message)
        self._log(f"  [{current}/{total}] {message}\n", 'info')

    def _scan_complete(self, report: Dict[str, Any]):
        """Called when scan finishes"""
        self.is_scanning = False
        self.btn_scan.configure(state='normal')
        self.btn_stop.configure(state='disabled')

        if 'error' in report:
            self._log(f"\nERROR: {report['error']}\n", 'error')
            self.status_var.set(f"Error: {report['error']}")
            return

        self.detections = self.pipeline.final_detections
        self.kml_path = report.get('kml_path', '')
        self.kmz_path = report.get('kmz_path', '')

        # Display results
        self._log("\n" + "=" * 60 + "\n", 'header')
        self._log("SCAN RESULTS\n", 'header')
        self._log("=" * 60 + "\n", 'header')
        self._log(f"Files scanned:        {report['files_scanned']}\n", 'good')
        self._log(f"Files w/ detections:  {report['files_with_detections']}\n", 'good')
        self._log(f"Raw anomalies:        {report['total_raw_anomalies']}\n", 'info')
        self._log(f"Masking artifacts:    {report['masking_artifacts_filtered']} (filtered out)\n", 'artifact')
        self._log(f"After dedup:          {report['after_deduplication']} unique locations\n", 'highlight')
        self._log(f"KML: {report['kml_path']}\n", 'info')

        # Per-file breakdown
        self._log("\n--- Per-File Results ---\n", 'header')
        for fname, info in report.get('file_results', {}).items():
            status = info['status']
            dets = info.get('detections', 0)

            if status == 'processed' and dets > 0:
                self._log(f"  ✓ {fname}: {dets} wrecks", 'wreck')
                self._log(f"  (depth: {info['depth_range']}, "
                          f"masking: {info['masking_pct']:.0f}%, "
                          f"artifacts filtered: {info['artifacts_filtered']})\n", 'info')
            elif status == 'processed':
                self._log(f"  · {fname}: no wrecks "
                          f"(masking: {info['masking_pct']:.0f}%)\n", 'artifact')
            elif status == 'skipped':
                self._log(f"  ⊘ {fname}: skipped ({info['reason']})\n", 'artifact')
            elif status == 'error':
                self._log(f"  ✗ {fname}: ERROR - {info['error']}\n", 'error')

        # Wreck list
        if self.detections:
            self._log("\n--- Detected Wrecks ---\n", 'header')
            for i, d in enumerate(self.detections, 1):
                tag = 'wreck' if d.size_feet >= 50 else 'info'
                star = "★" if d.size_feet >= 50 else "·"
                merged = d.metadata.get('merged_from', 1)
                self._log(f"  {star} #{i}: {d.latitude:.6f}, {d.longitude:.6f}\n", tag)
                self._log(f"      Size: {d.size_feet:.0f} ft ({d.size_meters:.0f}m) | "
                          f"Depth: {d.depth_meters:.1f}m | "
                          f"Height: {d.height_above_floor:.1f}m | "
                          f"Confidence: {d.confidence:.0%}\n", 'info')
                self._log(f"      Survey: {d.survey_id} | "
                          f"File: {d.bag_file}", 'info')
                if merged > 1:
                    self._log(f" | Merged from {merged} detections", 'highlight')
                self._log("\n", 'info')

        # Update viz
        self._plot_all_detections()

        # Enable KML button
        if self.kml_path:
            self.btn_kml.configure(state='normal')

        self.status_var.set(
            f"Complete: {report['after_deduplication']} wrecks from "
            f"{report['files_scanned']} files")

    def _scan_error(self, error_msg: str):
        """Called on scan error"""
        self.is_scanning = False
        self.btn_scan.configure(state='normal')
        self.btn_stop.configure(state='disabled')
        self._log(f"\nERROR: {error_msg}\n", 'error')
        self.status_var.set(f"Error: {error_msg}")

    def _open_kml(self):
        """Open the generated KML file"""
        if hasattr(self, 'kmz_path') and os.path.exists(self.kmz_path):
            os.startfile(self.kmz_path)
        elif hasattr(self, 'kml_path') and os.path.exists(self.kml_path):
            os.startfile(self.kml_path)

    def _launch_remote_sensing(self):
        """Open the Remote Sensing panel in a separate window."""
        win = tk.Toplevel(self.root)
        win.title("Wreck Hunter 2000 — Remote Sensing Pipeline")
        win.geometry("1200x700")
        try:
            from erie_remote_gui_tab import create_remote_sensing_tab

            # Build a minimal shim so create_remote_sensing_tab can attach
            class _Shim:
                pass
            shim = _Shim()
            shim.notebook = ttk.Notebook(win)
            shim.notebook.pack(fill='both', expand=True)
            create_remote_sensing_tab(shim)
        except Exception as e:
            messagebox.showerror("Remote Sensing",
                                 f"Could not load Remote Sensing tab:\n{e}")
            win.destroy()

    # =======================================================================
    # VISUALIZATION
    # =======================================================================

    def _plot_all_detections(self):
        """Plot all detections on a map-like scatter plot"""
        if not HAS_MPL or not self.detections:
            return

        self.ax.clear()
        self.ax.set_facecolor('#1e1e1e')

        lats = [d.latitude for d in self.detections]
        lons = [d.longitude for d in self.detections]
        sizes = [max(20, d.size_feet / 5) for d in self.detections]
        colors = ['#ff4444' if d.size_feet >= 50 else '#44aaff' for d in self.detections]

        self.ax.scatter(lons, lats, s=sizes, c=colors, alpha=0.8,
                        edgecolors='white', linewidths=0.5, zorder=5)

        # Label the larger ones
        for d in self.detections:
            if d.size_feet >= 50:
                self.ax.annotate(
                    f"{d.size_feet:.0f}ft",
                    (d.longitude, d.latitude),
                    textcoords="offset points", xytext=(8, 8),
                    fontsize=7, color='white', alpha=0.9
                )

        self.ax.set_xlabel("Longitude", color='white', fontsize=9)
        self.ax.set_ylabel("Latitude", color='white', fontsize=9)
        self.ax.set_title("Wreck Detections", color='white', fontsize=11)
        self.ax.tick_params(colors='white', labelsize=8)
        for spine in self.ax.spines.values():
            spine.set_color('#555')
        self.ax.grid(True, alpha=0.2, color='#555')

        self.fig.tight_layout()
        self.canvas.draw()

    def _plot_bag_elevation(self, elevation: np.ndarray,
                             restoration_mask: np.ndarray,
                             exaggeration: float = 1.0):
        """Plot a single BAG file's elevation with exaggeration and restoration colors"""
        if not HAS_MPL:
            return

        self.ax.clear()
        self.ax.set_facecolor('#1e1e1e')

        engine = RestorationEngine()
        exag_elev = engine.apply_exaggeration(elevation, exaggeration)

        # Build display array
        display = exag_elev.copy()
        display[np.isnan(display)] = 0

        # Color: original = blue/green, restored = orange/red
        if self.restore_var.get() and np.any(restoration_mask):
            rgba = engine.get_color_array(exag_elev, restoration_mask, exaggeration)
            self.ax.imshow(rgba, aspect='auto', origin='lower')
            self.ax.set_title(
                f"Elevation (exag: {exaggeration:.1f}x) — "
                f"Blue=original, Orange=restored",
                color='white', fontsize=10)
        else:
            valid = ~np.isnan(exag_elev)
            show = np.where(valid, exag_elev, np.nanmedian(exag_elev) if np.any(valid) else 0)
            self.ax.imshow(show, cmap='viridis', aspect='auto', origin='lower')
            self.ax.set_title(
                f"Elevation (exag: {exaggeration:.1f}x)",
                color='white', fontsize=10)

        self.ax.tick_params(colors='white', labelsize=8)
        for spine in self.ax.spines.values():
            spine.set_color('#555')

        self.fig.tight_layout()
        self.canvas.draw()

    def _on_exag_changed(self, value):
        """Exaggeration slider changed"""
        exag = float(value)
        self.exag_label.configure(text=f"{exag:.1f}x")
        if self.last_elevation is not None:
            self._refresh_plot()

    def _refresh_plot(self):
        """Refresh the visualization with current settings"""
        if self.last_elevation is not None:
            engine = RestorationEngine()
            restored, mask = engine.restore_masked_areas(
                self.last_elevation, self.last_bag_info
            )
            self._plot_bag_elevation(restored, mask, self.exag_var.get())
        elif self.detections:
            self._plot_all_detections()

    # =======================================================================
    # HELPERS
    # =======================================================================

    def _log(self, text: str, tag: str = 'info'):
        """Append text to results panel with color tag"""
        self.results_text.insert('end', text, tag)
        self.results_text.see('end')


# ============================================================================
# ENTRY POINT
# ============================================================================

def main():
    root = tk.Tk()
    app = BAGWreckGUI(root)
    root.mainloop()


if __name__ == "__main__":
    main()
