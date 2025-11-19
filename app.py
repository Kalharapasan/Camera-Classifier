import tkinter as tk
from tkinter import simpledialog, messagebox, ttk, filedialog
import ttkbootstrap as ttkb
from ttkbootstrap.constants import *
import cv2 as cv
import os
import PIL.Image, PIL.ImageTk
import model
import camera
from datetime import datetime
import csv
from pathlib import Path

class App:

    def __init__(self):
        self.root = ttkb.Window(themename="darkly")
        self.root.title("Camera Classifier v0.3 Pro")
        self.root.geometry("1400x900")
        self.root.minsize(1200, 800)

        self.counters = [1, 1]
        self.model = model.Model('svm')
        self.auto_predict = False
        self.camera = None
        self.photo = None
        self.batch_capture = False
        self.batch_count = 0
        self.current_camera_id = 0
        self.predictions_history = []

        try:
            self.camera = camera.Camera(self.current_camera_id)
        except ValueError as e:
            messagebox.showerror("Camera Error", str(e))
            self.root.destroy()
            return

        self.delay = 15
        self.init_gui()
        self.model.load_model()
        self.update()
        self.root.mainloop()

    def init_gui(self):
        """Initialize comprehensive GUI"""
        
        # Main container
        main_container = ttk.Frame(self.root)
        main_container.pack(fill=BOTH, expand=True, padx=5, pady=5)

        # Top section - Video and Info
        top_section = ttk.Frame(main_container)
        top_section.pack(fill=BOTH, expand=True, side=TOP)

        # Video feed panel
        video_panel = ttk.LabelFrame(top_section, text="Video Feed", bootstyle="info")
        video_panel.pack(side=LEFT, fill=BOTH, expand=True, padx=5, pady=5)

        self.canvas = tk.Canvas(video_panel, bg="black", width=640, height=480)
        self.canvas.pack(padx=5, pady=5)

        # Right panel - Info and Controls
        right_panel = ttk.Frame(top_section)
        right_panel.pack(side=RIGHT, fill=BOTH, padx=5, pady=5)

        # Model Info Panel
        info_frame = ttk.LabelFrame(right_panel, text="Model Information", bootstyle="success")
        info_frame.pack(fill=X, pady=5)

        self.info_label = tk.Label(info_frame, text="", justify=LEFT, font=("Courier", 9))
        self.info_label.pack(padx=5, pady=5)

        # Prediction Display
        pred_frame = ttk.LabelFrame(right_panel, text="Prediction", bootstyle="warning")
        pred_frame.pack(fill=X, pady=5)

        self.class_label = tk.Label(pred_frame, text="Ready", font=("Arial", 20, "bold"), 
                                    fg="white", bg="#FFA500", wraplength=200, pady=10)
        self.class_label.pack(padx=5, pady=5, fill=BOTH)

        self.confidence_label = tk.Label(pred_frame, text="Confidence: N/A", font=("Arial", 11))
        self.confidence_label.pack(padx=5, pady=2)

        # Statistics Panel
        stats_frame = ttk.LabelFrame(right_panel, text="Capture Statistics", bootstyle="info")
        stats_frame.pack(fill=X, pady=5)

        stats_grid = ttk.Frame(stats_frame)
        stats_grid.pack(padx=5, pady=5, fill=X)

        ttk.Label(stats_grid, text="Class 1:").grid(row=0, column=0, sticky=W, padx=5)
        self.label_counter_1 = tk.Label(stats_grid, text="0", font=("Arial", 12, "bold"), fg="#4CAF50")
        self.label_counter_1.grid(row=0, column=1, sticky=W, padx=5)

        ttk.Label(stats_grid, text="Class 2:").grid(row=1, column=0, sticky=W, padx=5)
        self.label_counter_2 = tk.Label(stats_grid, text="0", font=("Arial", 12, "bold"), fg="#2196F3")
        self.label_counter_2.grid(row=1, column=1, sticky=W, padx=5)

        self.fps_label = tk.Label(stats_grid, text="FPS: 0", font=("Arial", 10))
        self.fps_label.grid(row=2, column=0, columnspan=2, sticky=W, padx=5, pady=5)

        # Get class names
        self.classname_one = simpledialog.askstring(
            "Class One Name", 
            "Enter name for Class 1:", 
            parent=self.root
        ) or "Class 1"
        
        self.classname_two = simpledialog.askstring(
            "Class Two Name", 
            "Enter name for Class 2:", 
            parent=self.root
        ) or "Class 2"

        # Bottom section - Controls
        bottom_section = ttk.Frame(main_container)
        bottom_section.pack(fill=BOTH, side=BOTTOM, padx=5, pady=5)

        # Notebook for tabs
        self.notebook = ttk.Notebook(bottom_section)
        self.notebook.pack(fill=BOTH, expand=True)

        self._create_capture_tab()
        self._create_training_tab()
        self._create_settings_tab()
        self._create_advanced_tab()

    def _create_capture_tab(self):
        """Create capture controls tab"""
        frame = ttk.Frame(self.notebook)
        self.notebook.add(frame, text="Capture")

        # Capture buttons
        btn_frame = ttk.Frame(frame)
        btn_frame.pack(fill=X, padx=5, pady=5)

        self.btn_class_one = ttk.Button(
            btn_frame, 
            text=f"📷 Capture {self.classname_one}", 
            command=lambda: self.save_for_class(1),
            bootstyle="success"
        )
        self.btn_class_one.pack(side=LEFT, padx=2, fill=X, expand=True)

        self.btn_class_two = ttk.Button(
            btn_frame, 
            text=f"📷 Capture {self.classname_two}", 
            command=lambda: self.save_for_class(2),
            bootstyle="info"
        )
        self.btn_class_two.pack(side=LEFT, padx=2, fill=X, expand=True)

        # Batch capture
        batch_frame = ttk.LabelFrame(frame, text="Batch Capture Mode", bootstyle="warning")
        batch_frame.pack(fill=X, padx=5, pady=5)

        ttk.Label(batch_frame, text="Capture every (ms):").pack(side=LEFT, padx=5)
        self.batch_interval = ttk.Spinbox(batch_frame, from_=100, to=5000, width=10)
        self.batch_interval.set(500)
        self.batch_interval.pack(side=LEFT, padx=5)

        ttk.Label(batch_frame, text="Total captures:").pack(side=LEFT, padx=5)
        self.batch_total = ttk.Spinbox(batch_frame, from_=1, to=100, width=10)
        self.batch_total.set(10)
        self.batch_total.pack(side=LEFT, padx=5)

        self.btn_batch_one = ttk.Button(
            batch_frame, 
            text=f"🎬 Batch {self.classname_one}", 
            command=lambda: self.start_batch_capture(1),
            bootstyle="warning"
        )
        self.btn_batch_one.pack(side=LEFT, padx=2)

        self.btn_batch_two = ttk.Button(
            batch_frame, 
            text=f"🎬 Batch {self.classname_two}", 
            command=lambda: self.start_batch_capture(2),
            bootstyle="warning"
        )
        self.btn_batch_two.pack(side=LEFT, padx=2)

    def _create_training_tab(self):
        """Create training and prediction tab"""
        frame = ttk.Frame(self.notebook)
        self.notebook.add(frame, text="Training & Prediction")

        train_frame = ttk.Frame(frame)
        train_frame.pack(fill=X, padx=5, pady=5)

        self.btn_train = ttk.Button(
            train_frame, 
            text="🚀 Train Model", 
            command=self.train_model_handler,
            bootstyle="success",
            width=20
        )
        self.btn_train.pack(side=LEFT, padx=2, fill=X, expand=True)

        self.btn_predict = ttk.Button(
            train_frame, 
            text="🎯 Single Prediction", 
            command=self.predict_handler,
            bootstyle="info",
            width=20
        )
        self.btn_predict.pack(side=LEFT, padx=2, fill=X, expand=True)

        self.btn_toggleauto = ttk.Button(
            train_frame, 
            text="▶️ Auto Prediction", 
            command=self.auto_predict_toggle,
            bootstyle="warning",
            width=20
        )
        self.btn_toggleauto.pack(side=LEFT, padx=2, fill=X, expand=True)

        # Algorithm selection
        algo_frame = ttk.LabelFrame(frame, text="Algorithm Selection", bootstyle="info")
        algo_frame.pack(fill=X, padx=5, pady=5)

        ttk.Label(algo_frame, text="Select Algorithm:").pack(side=LEFT, padx=5)
        
        self.algo_var = tk.StringVar(value="svm")
        self.algo_combo = ttk.Combobox(
            algo_frame, 
            textvariable=self.algo_var, 
            values=["svm", "rf"],
            state="readonly",
            width=20
        )
        self.algo_combo.pack(side=LEFT, padx=5)
        self.algo_combo.bind("<<ComboboxSelected>>", self.switch_algorithm)

        ttk.Label(algo_frame, text="(SVM = Support Vector Machine, RF = Random Forest)").pack(side=LEFT, padx=5)

        # Utility buttons
        util_frame = ttk.Frame(frame)
        util_frame.pack(fill=X, padx=5, pady=5)

        ttk.Button(util_frame, text="📊 Export Results", command=self.export_results, 
                  bootstyle="secondary").pack(side=LEFT, padx=2, fill=X, expand=True)
        
        ttk.Button(util_frame, text="🔄 Reset All", command=self.reset, 
                  bootstyle="danger").pack(side=LEFT, padx=2, fill=X, expand=True)

    def _create_settings_tab(self):
        """Create image adjustment settings tab"""
        frame = ttk.Frame(self.notebook)
        self.notebook.add(frame, text="Image Settings")

        settings_frame = ttk.LabelFrame(frame, text="Camera Adjustments", bootstyle="warning")
        settings_frame.pack(fill=BOTH, expand=True, padx=5, pady=5)

        # Brightness
        ttk.Label(settings_frame, text="Brightness:").grid(row=0, column=0, sticky=W, padx=5, pady=5)
        self.brightness_scale = ttk.Scale(settings_frame, from_=-100, to=100, command=self.update_brightness)
        self.brightness_scale.grid(row=0, column=1, sticky=EW, padx=5, pady=5)
        self.brightness_label = ttk.Label(settings_frame, text="0")
        self.brightness_label.grid(row=0, column=2, padx=5)

        # Contrast
        ttk.Label(settings_frame, text="Contrast:").grid(row=1, column=0, sticky=W, padx=5, pady=5)
        self.contrast_scale = ttk.Scale(settings_frame, from_=50, to=300, command=self.update_contrast)
        self.contrast_scale.set(100)
        self.contrast_scale.grid(row=1, column=1, sticky=EW, padx=5, pady=5)
        self.contrast_label = ttk.Label(settings_frame, text="100%")
        self.contrast_label.grid(row=1, column=2, padx=5)

        # Saturation
        ttk.Label(settings_frame, text="Saturation:").grid(row=2, column=0, sticky=W, padx=5, pady=5)
        self.saturation_scale = ttk.Scale(settings_frame, from_=0, to=200, command=self.update_saturation)
        self.saturation_scale.set(100)
        self.saturation_scale.grid(row=2, column=1, sticky=EW, padx=5, pady=5)
        self.saturation_label = ttk.Label(settings_frame, text="100%")
        self.saturation_label.grid(row=2, column=2, padx=5)

        settings_frame.columnconfigure(1, weight=1)

        # Flip options
        flip_frame = ttk.LabelFrame(frame, text="Flip Options", bootstyle="info")
        flip_frame.pack(fill=X, padx=5, pady=5)

        self.flip_h_var = tk.BooleanVar()
        self.flip_v_var = tk.BooleanVar()

        ttk.Checkbutton(flip_frame, text="Flip Horizontal", variable=self.flip_h_var,
                       command=self.update_flip).pack(anchor=W, padx=5, pady=5)
        ttk.Checkbutton(flip_frame, text="Flip Vertical", variable=self.flip_v_var,
                       command=self.update_flip).pack(anchor=W, padx=5, pady=5)

        # Camera selection
        camera_frame = ttk.LabelFrame(frame, text="Camera Selection", bootstyle="secondary")
        camera_frame.pack(fill=X, padx=5, pady=5)

        available_cameras = self.camera.get_available_cameras()
        ttk.Label(camera_frame, text=f"Available cameras: {available_cameras}").pack(padx=5, pady=5)

        if len(available_cameras) > 1:
            ttk.Label(camera_frame, text="Select Camera:").pack(side=LEFT, padx=5)
            self.camera_combo = ttk.Combobox(camera_frame, values=available_cameras, width=10)
            self.camera_combo.set(self.current_camera_id)
            self.camera_combo.pack(side=LEFT, padx=5)
            ttk.Button(camera_frame, text="Switch Camera", 
                      command=self.switch_camera, bootstyle="info").pack(side=LEFT, padx=5)

    def _create_advanced_tab(self):
        """Create advanced features tab"""
        frame = ttk.Frame(self.notebook)
        self.notebook.add(frame, text="Advanced")

        # Statistics
        stats_frame = ttk.LabelFrame(frame, text="Model Statistics", bootstyle="info")
        stats_frame.pack(fill=BOTH, expand=True, padx=5, pady=5)

        self.stats_text = tk.Text(stats_frame, height=15, width=60, state=DISABLED)
        scrollbar = ttk.Scrollbar(stats_frame, orient=VERTICAL, command=self.stats_text.yview)
        self.stats_text.config(yscrollcommand=scrollbar.set)
        self.stats_text.pack(side=LEFT, fill=BOTH, expand=True, padx=5, pady=5)
        scrollbar.pack(side=RIGHT, fill=Y)

        # Utility
        util_frame = ttk.Frame(frame)
        util_frame.pack(fill=X, padx=5, pady=5)

        ttk.Button(util_frame, text="🔄 Refresh Stats", command=self.refresh_stats, 
                  bootstyle="info").pack(side=LEFT, padx=2, fill=X, expand=True)
        ttk.Button(util_frame, text="💾 Save Model", command=self.save_model_manual, 
                  bootstyle="success").pack(side=LEFT, padx=2, fill=X, expand=True)
        ttk.Button(util_frame, text="📂 Open Training Folder", command=self.open_training_folder, 
                  bootstyle="secondary").pack(side=LEFT, padx=2, fill=X, expand=True)

    def update_brightness(self, value):
        self.brightness_label.config(text=f"{int(float(value))}")
        self.camera.set_brightness(int(float(value)))

    def update_contrast(self, value):
        self.contrast_label.config(text=f"{int(float(value))}%")
        self.camera.set_contrast(float(value))

    def update_saturation(self, value):
        self.saturation_label.config(text=f"{int(float(value))}%")
        self.camera.set_saturation(float(value))

    def update_flip(self):
        self.camera.set_flip(self.flip_h_var.get(), self.flip_v_var.get())

    def switch_algorithm(self, event=None):
        if self.model.is_trained:
            messagebox.showwarning("Warning", "Model already trained. Training will be reset.")
        self.model.switch_algorithm(self.algo_var.get())

    def switch_camera(self):
        try:
            new_id = int(self.camera_combo.get())
            self.camera.switch_camera(new_id)
            self.current_camera_id = new_id
            messagebox.showinfo("Success", f"Switched to camera {new_id}")
        except Exception as e:
            messagebox.showerror("Error", str(e))

    def start_batch_capture(self, class_num):
        try:
            interval = int(self.batch_interval.get())
            total = int(self.batch_total.get())
            self.batch_capture = True
            self.batch_count = 0
            self.batch_class = class_num
            self.batch_total_target = total
            self.batch_interval_ms = interval
            messagebox.showinfo("Batch Capture", f"Starting batch capture for {self.classname_one if class_num == 1 else self.classname_two}")
        except Exception as e:
            messagebox.showerror("Error", str(e))

    def save_for_class(self, class_num):
        ret, frame = self.camera.get_frame()
        
        if not ret or frame is None:
            messagebox.showerror("Error", "Failed to capture frame")
            return

        for folder in ['1', '2']:
            if not os.path.exists(folder):
                os.mkdir(folder)

        gray_frame = cv.cvtColor(frame, cv.COLOR_RGB2GRAY)
        img_path = f'{class_num}/frame{self.counters[class_num-1]}.jpg'
        cv.imwrite(img_path, gray_frame)

        img = PIL.Image.open(img_path)
        img.thumbnail((150, 150), PIL.Image.LANCZOS)
        img.save(img_path)

        self.counters[class_num - 1] += 1
        self.update_counters()

    def update_counters(self):
        self.label_counter_1.config(text=str(self.counters[0] - 1))
        self.label_counter_2.config(text=str(self.counters[1] - 1))

    def train_model_handler(self):
        if self.counters[0] < 2 or self.counters[1] < 2:
            messagebox.showwarning("Insufficient Data", 
                                  "Please capture at least 1 image for each class")
            return

        success, accuracy = self.model.train_model(self.counters)
        if success:
            messagebox.showinfo("Success", f"Model trained!\nAccuracy: {accuracy:.2%}")
            self.refresh_stats()
        else:
            messagebox.showerror("Error", "Failed to train model")

    def predict_handler(self):
        if not self.model.is_trained:
            messagebox.showwarning("Model Not Trained", "Please train the model first!")
            return

        frame = self.camera.get_frame()
        prediction, confidence = self.model.predict(frame)

        if prediction is not None:
            class_name = self.classname_one if prediction == 1 else self.classname_two
            self.class_label.config(text=class_name)
            self.confidence_label.config(text=f"Confidence: {confidence:.3f}")

    def auto_predict_toggle(self):
        self.auto_predict = not self.auto_predict
        status = "ON" if self.auto_predict else "OFF"
        self.btn_toggleauto.config(text=f"▶️ Auto Prediction ({status})")

    def refresh_stats(self):
        info = self.model.get_model_info()
        stats_text = f"""
Model Information:
{'='*40}
Algorithm: {info['algorithm'].upper()}
Trained: {info['is_trained']}
Accuracy: {info['accuracy']:.2%}
Training Samples: {info['samples']}
Last Trained: {info['last_trained'] or 'Never'}

Capture Statistics:
{'='*40}
Class 1 ({self.classname_one}): {self.counters[0]-1}
Class 2 ({self.classname_two}): {self.counters[1]-1}
Total Samples: {sum(self.counters)-2}

Predictions: {len(self.predictions_history)}
"""
        self.stats_text.config(state=NORMAL)
        self.stats_text.delete(1.0, tk.END)
        self.stats_text.insert(1.0, stats_text)
        self.stats_text.config(state=DISABLED)

    def export_results(self):
        if not self.predictions_history:
            messagebox.showwarning("No Data", "No predictions to export yet")
            return

        file_path = filedialog.asksaveasfilename(defaultextension=".csv", 
                                                 filetypes=[("CSV", "*.csv")])
        if not file_path:
            return

        try:
            with open(file_path, 'w', newline='') as f:
                writer = csv.writer(f)
                writer.writerow(['Timestamp', 'Prediction', 'Confidence'])
                writer.writerows(self.predictions_history)
            messagebox.showinfo("Success", f"Results exported to {file_path}")
        except Exception as e:
            messagebox.showerror("Error", str(e))

    def save_model_manual(self):
        file_path = filedialog.asksaveasfilename(defaultextension=".pkl",
                                                filetypes=[("Pickle", "*.pkl")])
        if file_path:
            try:
                joblib.dump(self.model.model, file_path)
                messagebox.showinfo("Success", "Model saved successfully")
            except Exception as e:
                messagebox.showerror("Error", str(e))

    def open_training_folder(self):
        try:
            os.startfile('.')
        except:
            try:
                os.system('open .')
            except:
                messagebox.showinfo("Folder", "Training data in: 1/ and 2/ folders")

    def reset(self):
        if messagebox.askyesno("Confirm", "Delete all training data?"):
            for folder in ['1', '2']:
                if os.path.exists(folder):
                    for file in os.listdir(folder):
                        try:
                            os.unlink(os.path.join(folder, file))
                        except:
                            pass
            self.counters = [1, 1]
            self.model = model.Model(self.algo_var.get())
            self.class_label.config(text="Ready")
            self.confidence_label.config(text="Confidence: N/A")
            self.update_counters()
            self.refresh_stats()

    def update(self):
        if self.batch_capture and self.batch_count < self.batch_total_target:
            self.save_for_class(self.batch_class)
            self.batch_count += 1
            if self.batch_count >= self.batch_total_target:
                self.batch_capture = False
                messagebox.showinfo("Batch Complete", f"Captured {self.batch_count} images")
            self.root.after(self.batch_interval_ms, lambda: None)

        if self.auto_predict and self.model.is_trained:
            frame = self.camera.get_frame()
            prediction, confidence = self.model.predict(frame)
            
            if prediction is not None:
                class_name = self.classname_one if prediction == 1 else self.classname_two
                self.class_label.config(text=class_name)
                self.confidence_label.config(text=f"Confidence: {confidence:.3f}")
                self.predictions_history.append((datetime.now().isoformat(), class_name, f"{confidence:.3f}"))

        ret, frame = self.camera.get_frame()
        if ret and frame is not None:
            self.photo = PIL.ImageTk.PhotoImage(image=PIL.Image.fromarray(frame))
            self.canvas.create_image(0, 0, image=self.photo, anchor=tk.NW)

        # Update info
        info = self.model.get_model_info()
        info_text = f"Algorithm: {info['algorithm'].upper()}\nStatus: {'✓ Trained' if info['is_trained'] else '✗ Not Trained'}\nAccuracy: {info['accuracy']:.2%}"
        self.info_label.config(text=info_text)

        self.root.after(self.delay, self.update)