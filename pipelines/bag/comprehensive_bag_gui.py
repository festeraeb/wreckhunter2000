#!/usr/bin/env python3
"""
NOAA BAG File Analysis & Wreck Identification System
Comprehensive GUI for redaction breaking, wreck detection, and analysis
"""

import tkinter as tk
from tkinter import ttk, filedialog, messagebox, scrolledtext
import threading
import queue
import os
import sys
import json
import time
import webbrowser
import requests
from pathlib import Path
import sqlite3
import pandas as pd
import numpy as np
from datetime import datetime
import logging
import subprocess

# Import our analysis modules
sys.path.append(str(Path(__file__).parent))
sys.path.append(str(Path(__file__).parent / "development_and_tools"))

try:
    from robust_bag_scanner import RobustBagScanner
    from multi_mode_scanner import MultiModeScanner, ScanMode, ScanStrategy
    from enhanced_wreck_scanner import WreckSignatureDetector, ScanConfig
    from scrub_detection_engine import ScrubDetectionEngine
    from advanced_redaction_breaker import AdvancedRedactionBreaker
    from comprehensive_redaction_recovery import ComprehensiveRedactionRecovery
    from estimate_locations import estimate_location_from_place_name
    import newspaper_com_search
except ImportError as e:
    print(f"Warning: Could not import analysis modules: {e}")

try:
    from erie_remote_gui_tab import create_remote_sensing_tab
    HAS_ERIE_REMOTE_TAB = True
except ImportError:
    HAS_ERIE_REMOTE_TAB = False

class BAGAnalysisGUI:
    """Comprehensive GUI for BAG file analysis and wreck identification"""

    def __init__(self, root):
        self.root = root
        self.root.title("NOAA BAG Wreck Analysis & Redaction Breaker v2.0")
        self.root.geometry("1800x1000")
        self.root.configure(bg='#2b2b2b')

        # Initialize data
        self.current_bag_file = None
        self.analysis_results = {}
        self.scan_thread = None
        self.scan_queue = queue.Queue()

        # Database connection
        self.db_conn = sqlite3.connect('wrecks.db')
        self.db_cursor = self.db_conn.cursor()

        # Setup logging
        self.setup_logging()

        # Create GUI
        self.setup_gui()

        # Start queue processor
        self.root.after(100, self.process_queue)

    def setup_logging(self):
        """Setup logging for the application"""
        logging.basicConfig(
            filename='bag_analysis_gui.log',
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s'
        )
        self.logger = logging.getLogger(__name__)

    def setup_gui(self):
        """Setup the main GUI layout"""
        # Create main container
        main_container = ttk.Frame(self.root)
        main_container.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)

        # Create notebook for tabbed interface
        self.notebook = ttk.Notebook(main_container)
        self.notebook.pack(fill=tk.BOTH, expand=True)

        # Create tabs
        self.create_file_tab()
        self.create_analysis_tab()
        self.create_redaction_tab()
        self.create_database_tab()
        self.create_reports_tab()
        self.create_settings_tab()
        self.create_external_tools_tab()
        if HAS_ERIE_REMOTE_TAB:
            create_remote_sensing_tab(self)

        # Status bar
        self.status_var = tk.StringVar()
        self.status_var.set("Ready")
        status_bar = ttk.Label(main_container, textvariable=self.status_var, relief=tk.SUNKEN)
        status_bar.pack(fill=tk.X, pady=(5,0))

        # Progress bar
        self.progress_var = tk.DoubleVar()
        self.progress_bar = ttk.Progressbar(main_container, variable=self.progress_var, maximum=100)
        self.progress_bar.pack(fill=tk.X, pady=(2,0))

    def create_file_tab(self):
        """Create file selection and basic info tab"""
        file_tab = ttk.Frame(self.notebook)
        self.notebook.add(file_tab, text="📁 File Selection")

        # File selection frame
        file_frame = ttk.LabelFrame(file_tab, text="BAG File Selection", padding=10)
        file_frame.pack(fill=tk.X, padx=10, pady=10)

        ttk.Button(file_frame, text="Select BAG File", command=self.select_bag_file).pack(side=tk.LEFT, padx=5)
        ttk.Button(file_frame, text="Select Directory", command=self.select_directory).pack(side=tk.LEFT, padx=5)

        self.file_path_var = tk.StringVar()
        ttk.Entry(file_frame, textvariable=self.file_path_var, width=80).pack(side=tk.LEFT, padx=5, fill=tk.X, expand=True)

        # File info frame
        info_frame = ttk.LabelFrame(file_tab, text="File Information", padding=10)
        info_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)

        self.file_info_text = scrolledtext.ScrolledText(info_frame, height=15, wrap=tk.WORD)
        self.file_info_text.pack(fill=tk.BOTH, expand=True)

    def create_analysis_tab(self):
        """Create analysis tools tab"""
        analysis_tab = ttk.Frame(self.notebook)
        self.notebook.add(analysis_tab, text="🔍 Analysis Tools")

        # Control frame
        control_frame = ttk.LabelFrame(analysis_tab, text="Analysis Controls", padding=10)
        control_frame.pack(fill=tk.X, padx=10, pady=10)

        # Scan mode selection
        ttk.Label(control_frame, text="Scan Mode:").grid(row=0, column=0, sticky=tk.W, pady=2)
        self.scan_mode_var = tk.StringVar(value="quick")
        scan_modes = ["quick", "full", "pdf_guided", "hybrid", "targeted"]
        ttk.Combobox(control_frame, textvariable=self.scan_mode_var, values=scan_modes, state="readonly").grid(row=0, column=1, padx=5, pady=2)

        # Analysis buttons
        button_frame = ttk.Frame(control_frame)
        button_frame.grid(row=1, column=0, columnspan=2, pady=10)

        ttk.Button(button_frame, text="🚀 Start Analysis", command=self.start_analysis).pack(side=tk.LEFT, padx=5)
        ttk.Button(button_frame, text="⏹️ Stop Analysis", command=self.stop_analysis).pack(side=tk.LEFT, padx=5)
        ttk.Button(button_frame, text="📊 View Results", command=self.view_results).pack(side=tk.LEFT, padx=5)

        # Results frame
        results_frame = ttk.LabelFrame(analysis_tab, text="Analysis Results", padding=10)
        results_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)

        self.analysis_text = scrolledtext.ScrolledText(results_frame, height=20, wrap=tk.WORD)
        self.analysis_text.pack(fill=tk.BOTH, expand=True)

    def create_redaction_tab(self):
        """Create redaction breaking tools tab"""
        redaction_tab = ttk.Frame(self.notebook)
        self.notebook.add(redaction_tab, text="🔓 Redaction Breaker")

        # PDF selection frame
        pdf_frame = ttk.LabelFrame(redaction_tab, text="PDF File Selection", padding=10)
        pdf_frame.pack(fill=tk.X, padx=10, pady=10)

        ttk.Button(pdf_frame, text="Select PDF", command=self.select_pdf_file).pack(side=tk.LEFT, padx=5)
        ttk.Button(pdf_frame, text="Select PDF Directory", command=self.select_pdf_directory).pack(side=tk.LEFT, padx=5)

        self.pdf_path_var = tk.StringVar()
        ttk.Entry(pdf_frame, textvariable=self.pdf_path_var, width=80).pack(side=tk.LEFT, padx=5, fill=tk.X, expand=True)

        # Redaction tools frame
        tools_frame = ttk.LabelFrame(redaction_tab, text="Redaction Breaking Tools", padding=10)
        tools_frame.pack(fill=tk.X, padx=10, pady=10)

        ttk.Button(tools_frame, text="🔍 Analyze Redactions", command=self.analyze_redactions).pack(side=tk.LEFT, padx=5)
        ttk.Button(tools_frame, text="🛠️ Advanced Recovery", command=self.advanced_recovery).pack(side=tk.LEFT, padx=5)
        ttk.Button(tools_frame, text="📄 Extract Hidden Text", command=self.extract_hidden_text).pack(side=tk.LEFT, padx=5)

        # Results frame
        results_frame = ttk.LabelFrame(redaction_tab, text="Redaction Analysis Results", padding=10)
        results_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)

        self.redaction_text = scrolledtext.ScrolledText(results_frame, height=20, wrap=tk.WORD)
        self.redaction_text.pack(fill=tk.BOTH, expand=True)

    def create_database_tab(self):
        """Create database integration tab"""
        db_tab = ttk.Frame(self.notebook)
        self.notebook.add(db_tab, text="🗄️ Database Integration")

        # Search frame
        search_frame = ttk.LabelFrame(db_tab, text="Coordinate Search", padding=10)
        search_frame.pack(fill=tk.X, padx=10, pady=10)

        ttk.Label(search_frame, text="Latitude:").grid(row=0, column=0, sticky=tk.W, pady=2)
        self.search_lat_var = tk.StringVar()
        ttk.Entry(search_frame, textvariable=self.search_lat_var).grid(row=0, column=1, padx=5, pady=2)

        ttk.Label(search_frame, text="Longitude:").grid(row=0, column=2, sticky=tk.W, pady=2)
        self.search_lon_var = tk.StringVar()
        ttk.Entry(search_frame, textvariable=self.search_lon_var).grid(row=0, column=3, padx=5, pady=2)

        ttk.Button(search_frame, text="🔍 Search Database", command=self.search_database).grid(row=0, column=4, padx=5, pady=2)
        ttk.Button(search_frame, text="🌐 Internet Search", command=self.internet_search).grid(row=0, column=5, padx=5, pady=2)

        # Results frame
        results_frame = ttk.LabelFrame(db_tab, text="Search Results", padding=10)
        results_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)

        self.db_text = scrolledtext.ScrolledText(results_frame, height=20, wrap=tk.WORD)
        self.db_text.pack(fill=tk.BOTH, expand=True)

    def create_reports_tab(self):
        """Create reports and export tab"""
        reports_tab = ttk.Frame(self.notebook)
        self.notebook.add(reports_tab, text="📋 Reports & Export")

        # Report generation frame
        report_frame = ttk.LabelFrame(reports_tab, text="Report Generation", padding=10)
        report_frame.pack(fill=tk.X, padx=10, pady=10)

        ttk.Button(report_frame, text="📊 Generate Analysis Report", command=self.generate_analysis_report).pack(side=tk.LEFT, padx=5)
        ttk.Button(report_frame, text="🗺️ Export to KML", command=self.export_kml).pack(side=tk.LEFT, padx=5)
        ttk.Button(report_frame, text="🌐 Export to GeoJSON", command=self.export_geojson).pack(side=tk.LEFT, padx=5)
        ttk.Button(report_frame, text="📄 Create PDF Report", command=self.create_pdf_report).pack(side=tk.LEFT, padx=5)

        # Report content frame
        content_frame = ttk.LabelFrame(reports_tab, text="Report Content", padding=10)
        content_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)

        self.report_text = scrolledtext.ScrolledText(content_frame, height=20, wrap=tk.WORD)
        self.report_text.pack(fill=tk.BOTH, expand=True)

    def create_settings_tab(self):
        """Create settings and configuration tab"""
        settings_tab = ttk.Frame(self.notebook)
        self.notebook.add(settings_tab, text="⚙️ Settings")

        # Analysis settings
        analysis_frame = ttk.LabelFrame(settings_tab, text="Analysis Settings", padding=10)
        analysis_frame.pack(fill=tk.X, padx=10, pady=10)

        ttk.Label(analysis_frame, text="Tile Size (m):").grid(row=0, column=0, sticky=tk.W, pady=2)
        self.tile_size_var = tk.StringVar(value="20")
        ttk.Entry(analysis_frame, textvariable=self.tile_size_var).grid(row=0, column=1, padx=5, pady=2)

        ttk.Label(analysis_frame, text="Overlap Ratio:").grid(row=1, column=0, sticky=tk.W, pady=2)
        self.overlap_var = tk.StringVar(value="0.5")
        ttk.Entry(analysis_frame, textvariable=self.overlap_var).grid(row=1, column=1, padx=5, pady=2)

        ttk.Label(analysis_frame, text="Max Workers:").grid(row=2, column=0, sticky=tk.W, pady=2)
        self.workers_var = tk.StringVar(value="4")
        ttk.Entry(analysis_frame, textvariable=self.workers_var).grid(row=2, column=1, padx=5, pady=2)

        # Database settings
        db_frame = ttk.LabelFrame(settings_tab, text="Database Settings", padding=10)
        db_frame.pack(fill=tk.X, padx=10, pady=10)

        ttk.Button(db_frame, text="🔄 Rebuild Database", command=self.rebuild_database).pack(side=tk.LEFT, padx=5)
        ttk.Button(db_frame, text="📊 Database Stats", command=self.show_db_stats).pack(side=tk.LEFT, padx=5)

        # System info frame
        info_frame = ttk.LabelFrame(settings_tab, text="System Information", padding=10)
        info_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)

        self.system_text = scrolledtext.ScrolledText(info_frame, height=15, wrap=tk.WORD)
        self.system_text.pack(fill=tk.BOTH, expand=True)

        # Show system info
        self.show_system_info()

    def create_external_tools_tab(self):
        """Create a tab for launching external tools"""
        tools_tab = ttk.Frame(self.notebook)
        self.notebook.add(tools_tab, text="🛠️ External Tools")

        tools_frame = ttk.LabelFrame(tools_tab, text="Tool Launcher", padding=10)
        tools_frame.pack(fill=tk.X, padx=10, pady=10)

        # Add the ToolDet button
        ttk.Button(tools_frame, text="Launch ToolDet", command=self.launch_tooldet_tool).pack(side=tk.LEFT, padx=5)

    def launch_tooldet_tool(self):
        """Launches the tooldet GUI in a new window."""
        tooldet_window = tk.Toplevel(self.root)
        tooldet_window.title("ToolDet Runner")
        tooldet_window.geometry("700x200")
        
        try:
            from tooldet_gui import create_tooldet_gui
            tooldet_frame = create_tooldet_gui(tooldet_window)
            tooldet_frame.pack(fill="both", expand=True)
        except ImportError as e:
            messagebox.showerror("Error", f"Could not load ToolDet GUI: {e}")

    # File operations
    def select_bag_file(self):
        """Select a BAG file for analysis"""
        file_path = filedialog.askopenfilename(
            title="Select BAG File",
            filetypes=[("BAG files", "*.bag"), ("All files", "*.*")]
        )
        if file_path:
            self.file_path_var.set(file_path)
            self.current_bag_file = file_path
            self.analyze_bag_file()

    def select_directory(self):
        """Select a directory containing BAG files"""
        dir_path = filedialog.askdirectory(title="Select BAG Directory")
        if dir_path:
            self.file_path_var.set(dir_path)
            # Could implement directory scanning here

    def analyze_bag_file(self):
        """Analyze the selected BAG file and display info"""
        if not self.current_bag_file:
            return

        try:
            # Basic file info
            file_size = os.path.getsize(self.current_bag_file) / (1024 * 1024)  # MB
            file_date = datetime.fromtimestamp(os.path.getmtime(self.current_bag_file))

            info_text = f"""BAG File Analysis
{'='*50}
File: {os.path.basename(self.current_bag_file)}
Path: {self.current_bag_file}
Size: {file_size:.2f} MB
Modified: {file_date.strftime('%Y-%m-%d %H:%M:%S')}

Attempting to read BAG metadata...
"""

            self.file_info_text.delete(1.0, tk.END)
            self.file_info_text.insert(tk.END, info_text)

            # Try to read BAG metadata (simplified)
            try:
                import rasterio
                with rasterio.open(f'NETCDF:{self.current_bag_file}:elevation') as src:
                    bounds = src.bounds
                    crs = src.crs
                    width, height = src.width, src.height

                    metadata_text = f"""
Dataset Info:
- Width: {width} pixels
- Height: {height} pixels
- Bounds: {bounds}
- CRS: {crs}

Ready for analysis!
"""
                    self.file_info_text.insert(tk.END, metadata_text)

            except Exception as e:
                self.file_info_text.insert(tk.END, f"\nCould not read BAG metadata: {e}")

        except Exception as e:
            self.file_info_text.insert(tk.END, f"Error analyzing file: {e}")

    # Analysis operations
    def start_analysis(self):
        """Start the analysis process"""
        if not self.current_bag_file:
            messagebox.showerror("Error", "Please select a BAG file first")
            return

        self.status_var.set("Starting analysis...")
        self.progress_var.set(0)

        # Start analysis in background thread
        self.scan_thread = threading.Thread(target=self.run_analysis)
        self.scan_thread.daemon = True
        self.scan_thread.start()

    def run_analysis(self):
        """Run the analysis in background"""
        try:
            self.scan_queue.put(("status", "Initializing analysis..."))

            # Create scan configuration
            config = ScanConfig(
                tile_size_m=int(self.tile_size_var.get()),
                overlap_ratio=float(self.overlap_var.get()),
                max_workers=int(self.workers_var.get())
            )

            # Initialize scanner
            detector = WreckSignatureDetector(config)

            self.scan_queue.put(("status", "Scanning for wreck signatures..."))

            # Run analysis (simplified for demo)
            results = detector.scan_bag_file(self.current_bag_file)

            self.scan_queue.put(("results", results))
            self.scan_queue.put(("status", "Analysis complete"))
            self.scan_queue.put(("progress", 100))

        except Exception as e:
            self.scan_queue.put(("error", str(e)))

    def stop_analysis(self):
        """Stop the current analysis"""
        self.status_var.set("Analysis stopped")
        self.progress_var.set(0)

    def view_results(self):
        """Display analysis results"""
        if self.analysis_results:
            results_text = json.dumps(self.analysis_results, indent=2)
            self.analysis_text.delete(1.0, tk.END)
            self.analysis_text.insert(tk.END, results_text)
        else:
            messagebox.showinfo("Info", "No analysis results available")

    # Redaction operations
    def select_pdf_file(self):
        """Select a PDF file for redaction analysis"""
        file_path = filedialog.askopenfilename(
            title="Select PDF File",
            filetypes=[("PDF files", "*.pdf"), ("All files", "*.*")]
        )
        if file_path:
            self.pdf_path_var.set(file_path)

    def select_pdf_directory(self):
        """Select a directory containing PDF files"""
        dir_path = filedialog.askdirectory(title="Select PDF Directory")
        if dir_path:
            self.pdf_path_var.set(dir_path)

    def analyze_redactions(self):
        """Analyze redactions in the selected PDF"""
        pdf_path = self.pdf_path_var.get()
        if not pdf_path:
            messagebox.showerror("Error", "Please select a PDF file first")
            return

        try:
            breaker = AdvancedRedactionBreaker()
            results = breaker.analyze_pdf_structure(pdf_path)

            # Display results
            results_text = json.dumps(results, indent=2)
            self.redaction_text.delete(1.0, tk.END)
            self.redaction_text.insert(tk.END, results_text)

        except Exception as e:
            messagebox.showerror("Error", f"Redaction analysis failed: {e}")

    def advanced_recovery(self):
        """Perform advanced redaction recovery"""
        pdf_path = self.pdf_path_var.get()
        if not pdf_path:
            messagebox.showerror("Error", "Please select a PDF file first")
            return

        try:
            recovery = ComprehensiveRedactionRecovery()
            results = recovery.analyze_document(pdf_path)

            results_text = json.dumps(results, indent=2)
            self.redaction_text.delete(1.0, tk.END)
            self.redaction_text.insert(tk.END, results_text)

        except Exception as e:
            messagebox.showerror("Error", f"Advanced recovery failed: {e}")

    def extract_hidden_text(self):
        """Extract hidden text from PDF"""
        pdf_path = self.pdf_path_var.get()
        if not pdf_path:
            messagebox.showerror("Error", "Please select a PDF file first")
            return

        try:
            breaker = AdvancedRedactionBreaker()
            text_blocks = breaker.extract_text_pypdf2(pdf_path)

            self.redaction_text.delete(1.0, tk.END)
            self.redaction_text.insert(tk.END, f"Extracted {len(text_blocks)} text blocks:\n\n")
            for i, block in enumerate(text_blocks[:50]):  # Limit display
                self.redaction_text.insert(tk.END, f"{i+1}: {block}\n")

        except Exception as e:
            messagebox.showerror("Error", f"Text extraction failed: {e}")

    # Database operations
    def search_database(self):
        """Search database for wrecks near coordinates"""
        try:
            lat = float(self.search_lat_var.get())
            lon = float(self.search_lon_var.get())

            # Search for wrecks within 1 degree
            self.db_cursor.execute("""
                SELECT name, latitude, longitude, date, historical_place_names
                FROM features
                WHERE latitude BETWEEN ? AND ?
                AND longitude BETWEEN ? AND ?
                AND latitude IS NOT NULL
                LIMIT 20
            """, (lat-1, lat+1, lon-1, lon+1))

            results = self.db_cursor.fetchall()

            self.db_text.delete(1.0, tk.END)
            if results:
                self.db_text.insert(tk.END, f"Found {len(results)} wrecks near {lat:.4f}, {lon:.4f}:\n\n")
                for name, db_lat, db_lon, date, place in results:
                    distance = ((lat - db_lat)**2 + (lon - db_lon)**2)**0.5
                    self.db_text.insert(tk.END, f"• {name}\n")
                    self.db_text.insert(tk.END, f"  Location: {db_lat:.4f}, {db_lon:.4f} (distance: {distance:.4f}°)\n")
                    self.db_text.insert(tk.END, f"  Date: {date or 'Unknown'}\n")
                    self.db_text.insert(tk.END, f"  Place: {place or 'Unknown'}\n\n")
            else:
                self.db_text.insert(tk.END, f"No wrecks found near {lat:.4f}, {lon:.4f}")

        except ValueError:
            messagebox.showerror("Error", "Please enter valid latitude and longitude")
        except Exception as e:
            messagebox.showerror("Error", f"Database search failed: {e}")

    def internet_search(self):
        """Search internet for wrecks near coordinates"""
        try:
            lat = float(self.search_lat_var.get())
            lon = float(self.search_lon_var.get())

            # Create search query
            search_query = f"shipwreck OR wreck site {lat:.2f} {lon:.2f}"
            search_url = f"https://www.google.com/search?q={search_query.replace(' ', '+')}"

            # Open in browser
            webbrowser.open(search_url)

            self.db_text.delete(1.0, tk.END)
            self.db_text.insert(tk.END, f"Opened internet search for coordinates: {lat:.4f}, {lon:.4f}\n")
            self.db_text.insert(tk.END, f"Search URL: {search_url}\n\n")
            self.db_text.insert(tk.END, "Check your browser for results...")

        except ValueError:
            messagebox.showerror("Error", "Please enter valid latitude and longitude")

    # Report operations
    def generate_analysis_report(self):
        """Generate comprehensive analysis report"""
        report = f"""NOAA BAG Analysis Report
{'='*50}
Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

File Analyzed: {self.current_bag_file or 'None selected'}

Analysis Results:
{json.dumps(self.analysis_results, indent=2) if self.analysis_results else 'No analysis results available'}

Database Summary:
- Total wrecks: {self.get_db_count()}
- With coordinates: {self.get_db_count('latitude IS NOT NULL')}
- With PDFs: {self.get_db_count('newspaper_clip IS NOT NULL')}

System Status: Ready for further analysis
"""

        self.report_text.delete(1.0, tk.END)
        self.report_text.insert(tk.END, report)

    def export_kml(self):
        """Export results to KML"""
        try:
            subprocess.run([sys.executable, "export_to_kml.py"])
            messagebox.showinfo("Success", "KML export completed")
        except Exception as e:
            messagebox.showerror("Error", f"KML export failed: {e}")

    def export_geojson(self):
        """Export results to GeoJSON"""
        try:
            subprocess.run([sys.executable, "prepare_mbtiles.py"])
            messagebox.showinfo("Success", "GeoJSON export completed")
        except Exception as e:
            messagebox.showerror("Error", f"GeoJSON export failed: {e}")

    def create_pdf_report(self):
        """Create PDF report"""
        try:
            subprocess.run([sys.executable, "newspaper_com_search.py", "50"])
            messagebox.showinfo("Success", "PDF research reports created")
        except Exception as e:
            messagebox.showerror("Error", f"PDF report creation failed: {e}")

    # Settings operations
    def rebuild_database(self):
        """Rebuild the wreck database"""
        if messagebox.askyesno("Confirm", "This will rebuild the entire database. Continue?"):
            try:
                self.status_var.set("Rebuilding database...")
                # Re-run data import scripts
                subprocess.run([sys.executable, "populate_wrecks_database.py"])
                subprocess.run([sys.executable, "parse_swayze_excel.py"])
                subprocess.run([sys.executable, "estimate_locations.py"])
                messagebox.showinfo("Success", "Database rebuilt successfully")
                self.status_var.set("Database rebuilt")
            except Exception as e:
                messagebox.showerror("Error", f"Database rebuild failed: {e}")
                self.status_var.set("Database rebuild failed")

    def show_db_stats(self):
        """Show database statistics"""
        try:
            total = self.get_db_count()
            with_coords = self.get_db_count('latitude IS NOT NULL')
            with_pdfs = self.get_db_count('newspaper_clip IS NOT NULL')

            stats = f"""Database Statistics
{'='*30}
Total Records: {total}
With Coordinates: {with_coords} ({with_coords/total*100:.1f}%)
With PDF Reports: {with_pdfs} ({with_pdfs/total*100:.1f}%)
Last Updated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
"""
            messagebox.showinfo("Database Stats", stats)
        except Exception as e:
            messagebox.showerror("Error", f"Could not get stats: {e}")

    def get_db_count(self, condition=None):
        """Get database count with optional condition"""
        try:
            if condition:
                self.db_cursor.execute(f"SELECT COUNT(*) FROM features WHERE {condition}")
            else:
                self.db_cursor.execute("SELECT COUNT(*) FROM features")
            return self.db_cursor.fetchone()[0]
        except:
            return 0

    def show_system_info(self):
        """Show system information"""
        import platform
        import psutil

        try:
            cpu_count = psutil.cpu_count()
            memory = psutil.virtual_memory()
            disk = psutil.disk_usage('/')

            info = f"""System Information
{'='*30}
OS: {platform.system()} {platform.release()}
Python: {sys.version.split()[0]}
CPU Cores: {cpu_count}
Memory: {memory.total // (1024**3)} GB total, {memory.available // (1024**3)} GB available
Disk: {disk.total // (1024**3)} GB total, {disk.free // (1024**3)} GB free

Analysis Tools Status:
• BAG Scanner: {'Available' if 'RobustBagScanner' in globals() else 'Not Available'}
• Redaction Breaker: {'Available' if 'AdvancedRedactionBreaker' in globals() else 'Not Available'}
• Database: Connected ({self.get_db_count()} records)
"""
            self.system_text.delete(1.0, tk.END)
            self.system_text.insert(tk.END, info)
        except:
            self.system_text.insert(tk.END, "Could not retrieve system information")

    def process_queue(self):
        """Process messages from background threads"""
        try:
            while True:
                message = self.scan_queue.get_nowait()
                msg_type, data = message

                if msg_type == "status":
                    self.status_var.set(data)
                elif msg_type == "progress":
                    self.progress_var.set(data)
                elif msg_type == "results":
                    self.analysis_results = data
                    self.view_results()
                elif msg_type == "error":
                    messagebox.showerror("Analysis Error", data)
                    self.status_var.set("Analysis failed")

        except queue.Empty:
            pass

        # Schedule next check
        self.root.after(100, self.process_queue)

    def __del__(self):
        """Cleanup database connection"""
        if hasattr(self, 'db_conn'):
            self.db_conn.close()


def main():
    """Main application entry point"""
    root = tk.Tk()
    app = BAGAnalysisGUI(root)
    root.mainloop()


if __name__ == "__main__":
    main()