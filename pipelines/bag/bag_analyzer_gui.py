import subprocess
import sys
import tkinter as tk
from tkinter import ttk, filedialog, messagebox
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent


def create_bag_analyzer_tab(parent):
    frame = ttk.Frame(parent)

    ttk.Label(frame, text="BAG File Analyzer Tool", font=("Arial", 14, "bold")).pack(pady=10)

    # File selection row
    row = ttk.Frame(frame)
    row.pack(fill="x", padx=20, pady=5)
    ttk.Label(row, text="BAG File:").pack(side="left")
    bag_var = tk.StringVar()
    ttk.Entry(row, textvariable=bag_var, width=55).pack(side="left", padx=5)

    def select_bag_file():
        path = filedialog.askopenfilename(
            title="Select BAG File",
            filetypes=[("BAG files", "*.bag"), ("All files", "*.*")]
        )
        if path:
            bag_var.set(path)

    def run_anomaly_detection():
        bag_path = bag_var.get().strip()
        gui_script = Path(__file__).parent / "bag_wreck_gui.py"
        if not gui_script.exists():
            messagebox.showerror("Error", f"Could not find bag_wreck_gui.py at:\n{gui_script}")
            return
        cmd = [sys.executable, str(gui_script)]
        if bag_path:
            cmd.append(bag_path)
        subprocess.Popen(cmd,
                         creationflags=subprocess.CREATE_NEW_CONSOLE if sys.platform == "win32" else 0)
        messagebox.showinfo("Launched", "BAG Wreck Detector opened in a new window.")

    ttk.Button(row, text="Browse…", command=select_bag_file).pack(side="left")
    btn_row = ttk.Frame(frame)
    btn_row.pack(pady=10)
    ttk.Button(btn_row, text="🔍 Run Anomaly Detection",
               command=run_anomaly_detection).pack(side="left", padx=5)
    ttk.Label(frame, text="Launches the BAG Wreck Detector with the selected file.",
              foreground="gray").pack()

    return frame


if __name__ == "__main__":
    root = tk.Tk()
    root.title("BAG File Analyzer")
    root.geometry("700x200")
    create_bag_analyzer_tab(root).pack(fill="both", expand=True)
    root.mainloop()