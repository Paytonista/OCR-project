import os
import threading
import tkinter as tk
from tkinter import filedialog, messagebox, scrolledtext, ttk

import cv2
from PIL import Image, ImageTk

import main as ocr

PREPROCESSING_OPTIONS = [
    "Inverted",
    "Grayscaled",
    "Binarized",
    "Denoised",
    "Dilation & Erosion",
]


class OCRApp:
    def __init__(self, root):
        self.root = root
        self.root.title("OCR Image Processor")
        self.root.geometry("960x660")
        self.root.minsize(700, 500)

        self.image_path = None
        self.results = {}
        self.photo = None  # prevent GC

        self._build_ui()

    # ------------------------------------------------------------------
    def _build_ui(self):
        # ---- Top bar ----
        top = tk.Frame(self.root, padx=10, pady=8)
        top.pack(fill=tk.X)

        tk.Button(top, text="Browse...", command=self._browse).pack(side=tk.LEFT)

        self.path_var = tk.StringVar(value="No file selected")
        tk.Label(top, textvariable=self.path_var, anchor="w", fg="gray").pack(
            side=tk.LEFT, padx=8, fill=tk.X, expand=True
        )

        self.deskew_var = tk.BooleanVar(value=False)
        tk.Checkbutton(top, text="Apply deskew", variable=self.deskew_var).pack(
            side=tk.LEFT, padx=8
        )

        self.process_btn = tk.Button(
            top,
            text="Process",
            command=self._process,
            state=tk.DISABLED,
            bg="#2563eb",
            fg="white",
            padx=14,
            relief=tk.FLAT,
        )
        self.process_btn.pack(side=tk.LEFT)

        # ---- Main split pane ----
        pane = tk.PanedWindow(self.root, orient=tk.HORIZONTAL, sashwidth=6, sashrelief=tk.RAISED)
        pane.pack(fill=tk.BOTH, expand=True, padx=10, pady=(0, 4))

        # Left: image preview
        left = tk.LabelFrame(pane, text="Preview", padx=4, pady=4)
        self.img_label = tk.Label(left, text="No image loaded", fg="gray")
        self.img_label.pack(fill=tk.BOTH, expand=True)
        pane.add(left, minsize=300)

        # Right: controls + OCR result
        right = tk.Frame(pane, padx=6)

        ctrl = tk.Frame(right)
        ctrl.pack(fill=tk.X, pady=(4, 4))
        tk.Label(ctrl, text="Preprocessing:").pack(side=tk.LEFT)

        self.mode_var = tk.StringVar()
        self.mode_combo = ttk.Combobox(
            ctrl, textvariable=self.mode_var, state="readonly", width=26
        )
        self.mode_combo["values"] = PREPROCESSING_OPTIONS
        self.mode_combo.current(0)
        self.mode_combo.pack(side=tk.LEFT, padx=6)
        self.mode_combo.bind("<<ComboboxSelected>>", self._update_result)

        tk.Label(right, text="OCR Result:", anchor="w").pack(fill=tk.X)
        self.result_text = scrolledtext.ScrolledText(
            right, wrap=tk.WORD, font=("Consolas", 10), state=tk.DISABLED
        )
        self.result_text.pack(fill=tk.BOTH, expand=True)
        self._set_result_text("Process an image to see OCR results here.")

        pane.add(right, minsize=320)

        # ---- Status bar ----
        self.status_var = tk.StringVar(value="Ready")
        tk.Label(
            self.root, textvariable=self.status_var, anchor="w",
            relief=tk.SUNKEN, fg="gray", padx=6
        ).pack(fill=tk.X, side=tk.BOTTOM, pady=(0, 4), padx=10)

    # ------------------------------------------------------------------
    def _browse(self):
        path = filedialog.askopenfilename(
            filetypes=[
                ("Image files", "*.jpg *.jpeg *.png *.bmp *.tiff *.tif"),
                ("All files", "*.*"),
            ]
        )
        if not path:
            return
        self.image_path = path
        self.path_var.set(os.path.basename(path))
        self.process_btn.config(state=tk.NORMAL)
        self._show_preview(path)
        self.results = {}
        self._set_result_text("Click 'Process' to run OCR.")
        self.status_var.set("Image loaded.")

    def _show_preview(self, path):
        img = Image.open(path)
        img.thumbnail((400, 500))
        self.photo = ImageTk.PhotoImage(img)
        self.img_label.config(image=self.photo, text="")

    # ------------------------------------------------------------------
    def _process(self):
        if not self.image_path:
            return
        self.process_btn.config(state=tk.DISABLED)
        self.status_var.set("Processing...")
        self._set_result_text("Running OCR, please wait...")
        threading.Thread(target=self._run_ocr, daemon=True).start()

    def _run_ocr(self):
        try:
            image = cv2.imread(self.image_path)
            if image is None:
                self.root.after(
                    0, lambda: messagebox.showerror("Error", "Could not load image.")
                )
                return

            if self.deskew_var.get():
                image = ocr.deskew(image)

            self.results = ocr.run_pipeline(image)
            self.root.after(0, self._on_done)
        except Exception as e:
            self.root.after(0, lambda: messagebox.showerror("Error", str(e)))
        finally:
            self.root.after(0, lambda: self.process_btn.config(state=tk.NORMAL))

    def _on_done(self):
        self.status_var.set("Done.")
        self._update_result()

    # ------------------------------------------------------------------
    def _update_result(self, _event=None):
        if not self.results:
            return
        text = self.results.get(self.mode_var.get(), "(no result)")
        self._set_result_text(text.strip() or "(no text detected)")

    def _set_result_text(self, text):
        self.result_text.config(state=tk.NORMAL)
        self.result_text.delete("1.0", tk.END)
        self.result_text.insert(tk.END, text)
        self.result_text.config(state=tk.DISABLED)


# ------------------------------------------------------------------
if __name__ == "__main__":
    root = tk.Tk()
    app = OCRApp(root)
    root.mainloop()
