import tkinter as tk
from tkinter import ttk, filedialog, messagebox
from PIL import Image, ImageTk
import cv2
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import os
from typing import Optional

# --- Constants ---
BG_COLOR = "#2e2e2e"
FRAME_COLOR = "#3c3c3c"
TEXT_COLOR = "#dcdcdc"
ACCENT_COLOR = "#007acc"
ACCENT_HOVER = "#009cff"
ERROR_COLOR = "#e74c3c"
CANVAS_BG = "#1c1c1c"

HISTOGRAM_PIXEL_LIMIT = 250000
OUTPUT_DIR = "output"

class ImageEnhancerApp:
    def __init__(self, root: tk.Tk):
        self.root = root
        self.root.title("Medical Image Enhancement Studio")
        self.root.geometry("1600x900")
        self.root.configure(bg=BG_COLOR)

        # --- Instance Variables ---
        self.original_image: Optional[np.ndarray] = None
        self.processed_image: Optional[np.ndarray] = None
        self.tk_original_image: Optional[ImageTk.PhotoImage] = None
        self.tk_processed_image: Optional[ImageTk.PhotoImage] = None
        self.original_filename: Optional[str] = None
        
        self.debounce_timer: Optional[str] = None
        self.lut_cache: dict = {}
        self.last_canvas_sizes: dict = {}
        
        self.setup_styles()
        self.setup_gui()
        
        # --- Create output directory on startup ---
        if not os.path.exists(OUTPUT_DIR):
            os.makedirs(OUTPUT_DIR)

    def setup_styles(self):
        style = ttk.Style()
        style.theme_use('clam')
        style.configure('.', background=BG_COLOR, foreground=TEXT_COLOR, fieldbackground=FRAME_COLOR, borderwidth=1)
        style.configure('TFrame', background=BG_COLOR)
        style.configure('TLabel', background=BG_COLOR, foreground=TEXT_COLOR, font=('Segoe UI', 10))
        style.configure('TRadiobutton', background=FRAME_COLOR, foreground=TEXT_COLOR, font=('Segoe UI', 10))
        style.map('TRadiobutton', background=[('active', BG_COLOR)])
        style.configure('TButton', background=ACCENT_COLOR, foreground='white', font=('Segoe UI', 10, 'bold'), borderwidth=0)
        style.map('TButton', background=[('active', ACCENT_HOVER)])
        style.configure('TLabelframe', background=FRAME_COLOR, bordercolor=FRAME_COLOR)
        style.configure('TLabelframe.Label', background=FRAME_COLOR, foreground=TEXT_COLOR, font=('Segoe UI', 11, 'bold'))
        style.configure('Horizontal.TScale', background=FRAME_COLOR, troughcolor=BG_COLOR)

    def setup_gui(self):
        main_pane = ttk.PanedWindow(self.root, orient=tk.HORIZONTAL)
        main_pane.pack(fill=tk.BOTH, expand=True, padx=15, pady=15)
        
        control_frame = ttk.Labelframe(main_pane, text="Controls", style='TLabelframe')
        main_pane.add(control_frame, weight=2)
        
        display_frame = ttk.Frame(main_pane, style='TFrame')
        main_pane.add(display_frame, weight=5)
        
        self.setup_control_widgets(control_frame)
        self.setup_display_widgets(display_frame)

    def setup_control_widgets(self, parent_frame: ttk.Labelframe):
        parent_frame['padding'] = (20, 15)
        
        file_frame = ttk.LabelFrame(parent_frame, text="File Operations", style='TLabelframe', padding=10)
        file_frame.pack(fill=tk.X, pady=(0, 20))
        self.load_button = ttk.Button(file_frame, text="Load Image", command=self.load_image, style='TButton')
        self.load_button.pack(side=tk.LEFT, expand=True, fill=tk.X, padx=5, ipady=5)
        self.save_button = ttk.Button(file_frame, text="Save to 'output' Folder", command=self.save_output, state=tk.DISABLED, style='TButton')
        self.save_button.pack(side=tk.LEFT, expand=True, fill=tk.X, padx=5, ipady=5)
        
        tech_frame = ttk.LabelFrame(parent_frame, text="Enhancement Techniques", style='TLabelframe', padding=15)
        tech_frame.pack(fill=tk.X, pady=15)
        self.technique_var = tk.StringVar(value="None")
        
        techniques = [("None (Show Original)", "None"), ("Histogram Equalization", "hist_eq"), ("Power-Law (Gamma)", "gamma")]
        for text, value in techniques:
            ttk.Radiobutton(tech_frame, text=text, variable=self.technique_var, value=value, command=self.on_technique_change).pack(anchor=tk.W, pady=3)
        
        self.params_frame = ttk.LabelFrame(parent_frame, text="Parameters", style='TLabelframe', padding=15)
        self.params_frame.pack(fill=tk.X, pady=15)
        
        self.gamma_label = ttk.Label(self.params_frame, text="Gamma (γ): 1.00")
        self.gamma_var = tk.DoubleVar(value=1.0)
        self.gamma_slider = ttk.Scale(self.params_frame, from_=0.1, to=5.0, variable=self.gamma_var, orient=tk.HORIZONTAL, command=self.on_slider_change)
        
        self.reset_button = ttk.Button(parent_frame, text="Reset to Original", command=self.reset_image, state=tk.DISABLED, style='TButton')
        self.reset_button.pack(fill=tk.X, side=tk.BOTTOM, pady=10, ipady=5)
        self.on_technique_change()

    def setup_display_widgets(self, parent_frame: ttk.Frame):
        parent_frame.rowconfigure(0, weight=1); parent_frame.rowconfigure(1, weight=1)
        parent_frame.columnconfigure(0, weight=3); parent_frame.columnconfigure(1, weight=2)
        
        original_frame, self.original_canvas, self.original_info_label = self._create_display_canvas(parent_frame, "Original Image")
        original_frame.grid(row=0, column=0, sticky="nsew", padx=(0, 10), pady=(0, 10))
        processed_frame, self.processed_canvas, self.processed_info_label = self._create_display_canvas(parent_frame, "Enhanced Image")
        processed_frame.grid(row=1, column=0, sticky="nsew", padx=(0, 10), pady=(10, 0))
        
        self.hist_original_frame = self._create_histogram_frame(parent_frame, "Original Histogram")
        self.hist_original_frame.grid(row=0, column=1, sticky="nsew", padx=(10, 0), pady=(0, 10))
        self.hist_processed_frame = self._create_histogram_frame(parent_frame, "Enhanced Histogram")
        self.hist_processed_frame.grid(row=1, column=1, sticky="nsew", padx=(10, 0), pady=(10, 0))
        
        self.original_canvas.bind('<Configure>', self.on_canvas_resize)
        self.processed_canvas.bind('<Configure>', self.on_canvas_resize)

    def _create_display_canvas(self, parent, title):
        frame = ttk.Frame(parent, style='TFrame')
        label = ttk.Label(frame, text=title, font=('Segoe UI', 14, 'bold'))
        label.pack(pady=(0, 5))
        canvas = tk.Canvas(frame, bg=CANVAS_BG, relief=tk.FLAT, bd=0, highlightthickness=0)
        canvas.pack(fill=tk.BOTH, expand=True)
        info_label = ttk.Label(frame, text="", font=('Segoe UI', 9))
        info_label.pack(pady=(5, 0))
        return frame, canvas, info_label

    def _create_histogram_frame(self, parent, title):
        return ttk.Labelframe(parent, text=title, style='TLabelframe', padding=10)

    def on_canvas_resize(self, event: tk.Event):
        canvas = event.widget
        canvas_w, canvas_h = canvas.winfo_width(), canvas.winfo_height()
        if self.last_canvas_sizes.get(id(canvas)) == (canvas_w, canvas_h): return
        self.last_canvas_sizes[id(canvas)] = (canvas_w, canvas_h)
        
        if canvas == self.original_canvas and self.original_image is not None:
            self.display_image(self.original_image, self.original_canvas, 'original', self.original_info_label)
        elif canvas == self.processed_canvas and self.processed_image is not None:
            self.display_image(self.processed_image, self.processed_canvas, 'processed', self.processed_info_label)

    def on_slider_change(self, _=None):
        if self.technique_var.get() == "gamma":
            self.gamma_label.config(text=f"Gamma (γ): {self.gamma_var.get():.2f}")
        if self.debounce_timer:
            self.root.after_cancel(self.debounce_timer)
        self.debounce_timer = self.root.after(100, self.apply_enhancement)

    def on_technique_change(self):
        self.gamma_label.pack_forget(); self.gamma_slider.pack_forget()
        if self.technique_var.get() == "gamma":
            self.gamma_label.pack(anchor=tk.W)
            self.gamma_slider.pack(fill=tk.X, pady=(0, 10))
        self.apply_enhancement()

    def apply_enhancement(self):
        if self.original_image is None: return
        technique = self.technique_var.get()
        
        if technique == "hist_eq":
            self.processed_image = cv2.equalizeHist(self.original_image)
        elif technique == "gamma":
            gamma = round(self.gamma_var.get(), 2)
            cache_key = f"gamma_{gamma}"
            if cache_key not in self.lut_cache:
                inv_gamma = 1.0 / gamma
                table = np.array([((i / 255.0) ** inv_gamma) * 255 for i in np.arange(256)]).astype("uint8")
                self.lut_cache[cache_key] = table
            self.processed_image = cv2.LUT(self.original_image, self.lut_cache[cache_key])
        else:
            self.processed_image = self.original_image.copy()
        
        self.display_image(self.processed_image, self.processed_canvas, 'processed', self.processed_info_label)
        self.update_histograms()

    def display_image(self, image_data, canvas, image_type, info_label):
        canvas.delete("all")
        canvas_w, canvas_h = canvas.winfo_width(), canvas.winfo_height()
        if image_data is None or canvas_w < 2 or canvas_h < 2: return

        img_h, img_w = image_data.shape[:2]; aspect = img_w / img_h
        new_w, new_h = (canvas_w, int(canvas_w / aspect)) if (canvas_w / aspect) <= canvas_h else (int(canvas_h * aspect), canvas_h)
        if new_w < 1 or new_h < 1: return
        
        resized_img = cv2.resize(image_data, (int(new_w), int(new_h)), interpolation=cv2.INTER_AREA)
        photo_img = ImageTk.PhotoImage(image=Image.fromarray(resized_img))
        
        if image_type == 'original': self.tk_original_image = photo_img
        else: self.tk_processed_image = photo_img
            
        x, y = (canvas_w - new_w) / 2, (canvas_h - new_h) / 2
        canvas.create_image(x, y, anchor=tk.NW, image=photo_img)
        info_label.config(text=f"Dimensions: {img_w} x {img_h} px")

    def calculate_histogram_fast(self, image):
        if image.size > HISTOGRAM_PIXEL_LIMIT:
            pixels, is_sampled = np.random.choice(image.ravel(), HISTOGRAM_PIXEL_LIMIT, replace=False), True
        else:
            pixels, is_sampled = image.ravel(), False
        counts, bins = np.histogram(pixels, bins=256, range=[0, 256])
        return counts, bins, is_sampled

    def update_histograms(self):
        if self.original_image is not None:
            counts, bins, sampled = self.calculate_histogram_fast(self.original_image)
            self.plot_histogram(self.hist_original_frame, counts, bins, ACCENT_COLOR, "Original Histogram", sampled)
        if self.processed_image is not None:
            counts, bins, sampled = self.calculate_histogram_fast(self.processed_image)
            self.plot_histogram(self.hist_processed_frame, counts, bins, ERROR_COLOR, "Enhanced Histogram", sampled)

    def plot_histogram(self, parent_frame, counts, bins, color, title, sampled):
        for widget in parent_frame.winfo_children(): widget.destroy()
        parent_frame['text'] = title + (" (Sampled)" if sampled else "")
        fig, ax = plt.subplots(facecolor=FRAME_COLOR)
        ax.set_facecolor(BG_COLOR)
        ax.bar(bins[:-1], counts, width=1, color=color)
        ax.set_xlim([0, 255]); ax.tick_params(colors=TEXT_COLOR, which='both')
        ax.set_xlabel("Pixel Intensity", color=TEXT_COLOR, fontsize=8)
        ax.set_ylabel("Frequency", color=TEXT_COLOR, fontsize=8)
        ax.tick_params(axis='both', which='major', labelsize=8)
        ax.grid(True, linestyle='--', alpha=0.2); fig.tight_layout(pad=0.5)
        canvas = FigureCanvasTkAgg(fig, master=parent_frame)
        canvas.draw()
        canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True, padx=5, pady=5)
        plt.close(fig)

    def load_image(self):
        file_path = filedialog.askopenfilename(filetypes=[("Image Files", "*.png;*.jpg;*.jpeg;*.bmp;*.tif;*.dcm")])
        if not file_path: return
        try:
            img = cv2.imread(file_path, cv2.IMREAD_GRAYSCALE)
            if img is None: raise ValueError("File is not a valid image.")
            self.original_image = img
            self.original_filename = os.path.basename(file_path)
            self.lut_cache.clear(); self.last_canvas_sizes = {}
            self.reset_image()
            self.save_button['state'] = tk.NORMAL
            self.reset_button['state'] = tk.NORMAL
        except Exception as e:
            messagebox.showerror("Error", f"Failed to load image: {e}")

    def save_output(self):
        if self.processed_image is None or self.original_filename is None:
            messagebox.showwarning("Warning", "No enhanced image to save.")
            return
        technique = self.technique_var.get()
        if technique == "None":
            messagebox.showinfo("Info", "No enhancement applied. Cannot save.")
            return

        base_name, _ = os.path.splitext(self.original_filename)
        suffix = f"{technique}_gamma{self.gamma_var.get():.2f}" if technique == "gamma" else technique
            
        new_image_filename = f"{base_name}_{suffix}.png"
        new_hist_filename = f"{base_name}_{suffix}_hist.png"
        image_save_path = os.path.join(OUTPUT_DIR, new_image_filename)
        hist_save_path = os.path.join(OUTPUT_DIR, new_hist_filename)

        try:
            cv2.imwrite(image_save_path, self.processed_image)
            self.save_histogram_to_file(self.processed_image, hist_save_path, ERROR_COLOR, "Enhanced Histogram")
            messagebox.showinfo("Success", f"Outputs saved to '{OUTPUT_DIR}' folder:\n- {new_image_filename}\n- {new_hist_filename}")
        except Exception as e:
            messagebox.showerror("Error", f"Failed to save files: {e}")

    def save_histogram_to_file(self, image_data, file_path, color, title):
        counts, bins, sampled = self.calculate_histogram_fast(image_data)
        full_title = title + (" (Sampled)" if sampled else "")

        fig, ax = plt.subplots(facecolor=FRAME_COLOR, figsize=(6, 4))
        fig.suptitle(full_title, color=TEXT_COLOR, fontsize=12)
        ax.set_facecolor(BG_COLOR)
        ax.bar(bins[:-1], counts, width=1.0, color=color)
        ax.set_xlim([0, 255]); ax.tick_params(colors=TEXT_COLOR, which='both')
        ax.set_xlabel("Pixel Intensity", color=TEXT_COLOR)
        ax.set_ylabel("Frequency", color=TEXT_COLOR)
        ax.grid(True, linestyle='--', alpha=0.2)
        fig.tight_layout(rect=[0, 0, 1, 0.95])
        
        fig.savefig(file_path, facecolor=FRAME_COLOR, dpi=150)
        plt.close(fig)

    def reset_image(self):
        if self.original_image is not None:
            self.technique_var.set("None")
            self.last_canvas_sizes = {}
            self.display_image(self.original_image, self.original_canvas, 'original', self.original_info_label)
            self.apply_enhancement()

if __name__ == "__main__":
    root = tk.Tk()
    app = ImageEnhancerApp(root)
    root.mainloop()