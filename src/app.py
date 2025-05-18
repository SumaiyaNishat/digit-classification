# src/app.py

import os
import re # For parsing filenames
import tkinter as tk
from tkinter import filedialog, ttk, messagebox
from PIL import Image, ImageTk, ImageOps
import cv2
import numpy as np
# No ARFF imports needed: from scipy.io import arff / import arff as liac_arff
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, accuracy_score
from skimage.feature import hog
import joblib
import pandas as pd # Still useful for label distribution, etc.
import threading
import traceback

# --- Configuration ---
APP_NAME = "Digit Classification"
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(SCRIPT_DIR)
IMAGE_BASE_FOLDER = os.path.join(PROJECT_ROOT, 'images') # Folder with all training images
# ARFF_FILE_PATH - Not needed
MODEL_SAVE_DIR = os.path.join(PROJECT_ROOT, 'models')
MODEL_FILENAME = os.path.join(MODEL_SAVE_DIR, 'digit_classifier_from_filenames.joblib')
FEATURE_PARAMS_FILENAME = os.path.join(MODEL_SAVE_DIR, 'feature_params_from_filenames.joblib')
IMAGE_SIZE = (32, 32)
USE_HOG_FEATURES = True

# --- Helper Functions (ensure_dir, preprocess_image_for_model, extract_features) ---
# (These functions remain the same as in the previous "complete app.py" response)
def ensure_dir(directory):
    if not os.path.exists(directory):
        os.makedirs(directory)

def preprocess_image_for_model(image_input, target_size, for_training=False):
    try:
        if isinstance(image_input, str):
            pil_image = Image.open(image_input)
        elif isinstance(image_input, Image.Image):
            pil_image = image_input
        else:
            print(f"Error: Invalid image_input type: {type(image_input)}")
            return None
        pil_image = pil_image.convert('L')
        img_array_for_check = np.array(pil_image)
        if not for_training and np.mean(img_array_for_check) > 128:
             pil_image = ImageOps.invert(pil_image)
        img_np = np.array(pil_image)
        img_resized = cv2.resize(img_np, target_size, interpolation=cv2.INTER_AREA)
        img_normalized = img_resized / 255.0
        return img_normalized
    except Exception as e:
        print(f"ERROR in preprocess_image_for_model for '{image_input if isinstance(image_input, str) else 'PIL object'}': {e}\n{traceback.format_exc()}")
        return None

def extract_features(image_array, use_hog, img_size_for_hog):
    if use_hog:
        ppc_x = max(1, img_size_for_hog[0] // 4)
        ppc_y = max(1, img_size_for_hog[1] // 4)
        cells_per_block = (2, 2)
        min_hog_height = ppc_y * cells_per_block[0]
        min_hog_width = ppc_x * cells_per_block[1]
        if image_array.shape[0] < min_hog_height or image_array.shape[1] < min_hog_width:
            print(f"DEBUG: Warning: Image size {image_array.shape} too small for HOG. Using raw pixels.")
            return image_array.flatten()
        features = hog(image_array, orientations=9, pixels_per_cell=(ppc_x, ppc_y),
                       cells_per_block=cells_per_block, visualize=False, block_norm='L2-Hys')
    else:
        features = image_array.flatten()
    return features

def parse_label_from_filename(filename):
    """
    Extracts a digit label from a filename.
    Looks for the last single digit character (English or Bengali) before the extension.
    Example: "img_B_7.png" -> "7", "pic_BN_৩.jpg" -> "৩"
    Adjust regex if your naming convention is different.
    """
    name_part = os.path.splitext(filename)[0]
    # Regex to find the last occurrence of a single digit (English or Bengali)
    # \d matches English digits. Add Bengali digits explicitly.
    # This regex looks for one of these characters, optionally preceded by _ or - or a letter.
    # A simpler approach might be to find the last of these characters in the string.
    
    # More robust: find the last character that is a digit (English or Bengali)
    # This assumes the digit is indeed the very last relevant character before extension.
    bengali_digits = "০১২৩৪৫৬৭৮৯"
    all_digit_chars = "0123456789" + bengali_digits
    
    found_digit = None
    for char in reversed(name_part): # Iterate from end of filename (without ext)
        if char in all_digit_chars:
            found_digit = char
            break # Found the last digit character
            
    if found_digit:
        return found_digit
    else:
        print(f"Warning: Could not parse digit label from filename: {filename}")
        return None
# --- End Helper Functions ---

class DigitRecognizerApp:
    def __init__(self, root_window):
        self.root = root_window
        self.root.title(APP_NAME)
        ensure_dir(MODEL_SAVE_DIR)
        self.model = None
        self.feature_params = None
        self.all_labels_from_training = []
        self.load_saved_model_and_params()

        image_frame = ttk.LabelFrame(self.root, text="Image Preview & Upload", padding=10)
        image_frame.pack(padx=10, pady=10, fill="x", expand=False)
        self.image_label = ttk.Label(image_frame, text="Upload an image to see preview")
        self.image_label.pack(pady=5)
        self.uploaded_image_pil = None
        self.btn_upload = ttk.Button(image_frame, text="Upload Image", command=self.upload_image_and_predict)
        self.btn_upload.pack(pady=5)

        predict_frame = ttk.LabelFrame(self.root, text="Prediction", padding=10)
        predict_frame.pack(padx=10, pady=5, fill="x", expand=False)
        self.lbl_prediction_text = ttk.Label(predict_frame, text="Predicted Digit:")
        self.lbl_prediction_text.pack(side=tk.LEFT, padx=5)
        self.lbl_prediction_result = ttk.Label(predict_frame, text="N/A", font=("Helvetica", 16, "bold"))
        self.lbl_prediction_result.pack(side=tk.LEFT, padx=5)

        train_frame = ttk.LabelFrame(self.root, text="Model Training (from image filenames)", padding=10)
        train_frame.pack(padx=10, pady=10, fill="x", expand=False)
        self.btn_train = ttk.Button(train_frame, text="Train Model", command=self.start_training_thread)
        self.btn_train.pack(pady=5)
        self.train_status_label = ttk.Label(train_frame, text="Training Status: Idle")
        self.train_status_label.pack(pady=2)
        self.train_progress_bar = ttk.Progressbar(train_frame, orient="horizontal", length=300, mode="determinate")
        self.train_progress_bar.pack(pady=5)

        self.status_bar = ttk.Label(self.root, text="Ready.", relief=tk.SUNKEN, anchor=tk.W)
        self.status_bar.pack(side=tk.BOTTOM, fill=tk.X)
        self.update_status_bar("App started. Model " + ("loaded." if self.model else "not found. Train model."))

    def update_status_bar(self, message):
        self.status_bar.config(text=message)
        self.root.update_idletasks()

    def update_training_status_label(self, message):
        self.train_status_label.config(text=f"Training Status: {message}")
        self.root.update_idletasks()

    def load_saved_model_and_params(self):
        if os.path.exists(MODEL_FILENAME) and os.path.exists(FEATURE_PARAMS_FILENAME):
            try:
                self.model = joblib.load(MODEL_FILENAME)
                self.feature_params = joblib.load(FEATURE_PARAMS_FILENAME)
                self.all_labels_from_training = self.feature_params.get('all_labels', [])
                print("Previously trained model and parameters loaded.")
                return True
            except Exception as e:
                print(f"Error loading saved model/params: {e}\n{traceback.format_exc()}")
                self.model = None; self.feature_params = None
        return False

    def upload_image_and_predict(self):
        filepath = filedialog.askopenfilename(
            title="Select Digit Image",
            filetypes=(("Image files", "*.png *.jpg *.jpeg *.bmp *.gif"), ("All files", "*.*"))
        )
        if not filepath: return
        try:
            self.uploaded_image_pil = Image.open(filepath)
            img_for_display = self.uploaded_image_pil.copy()
            img_for_display.thumbnail((200, 200))
            self.img_tk = ImageTk.PhotoImage(img_for_display)
            self.image_label.config(image=self.img_tk, text="")
            self.update_status_bar(f"Image loaded: {os.path.basename(filepath)}. Predicting...")
            self.predict_current_uploaded_image()
        except Exception as e:
            messagebox.showerror("Image Error", f"Failed to load or display image: {e}")
            self.update_status_bar(f"Error loading image: {e}")
            print(f"Image loading error: {e}\n{traceback.format_exc()}")

    def predict_current_uploaded_image(self):
        if not self.uploaded_image_pil:
            messagebox.showwarning("Prediction Warning", "Please upload an image first.")
            return
        if not self.model or not self.feature_params:
            messagebox.showwarning("Model Warning", "Model not trained or loaded. Please train.")
            self.update_status_bar("Model not ready. Please train.")
            return

        _target_size = tuple(self.feature_params.get('image_size', IMAGE_SIZE))
        _use_hog = self.feature_params.get('use_hog', USE_HOG_FEATURES)
        processed_img_array = preprocess_image_for_model(self.uploaded_image_pil, _target_size, for_training=False)
        if processed_img_array is None:
            self.lbl_prediction_result.config(text="Error"); self.update_status_bar("Preproc. error.")
            return

        features = extract_features(processed_img_array, _use_hog, _target_size)
        features_reshaped = features.reshape(1, -1)
        try:
            prediction_array = self.model.predict(features_reshaped)
            predicted_label = str(prediction_array[0])
            self.lbl_prediction_result.config(text=predicted_label)
            self.update_status_bar(f"Prediction successful: {predicted_label}")
        except Exception as e:
            self.lbl_prediction_result.config(text="Error"); self.update_status_bar(f"Predict err: {e}")
            print(f"Prediction error: {e}\n{traceback.format_exc()}")

    def start_training_thread(self):
        self.btn_train.config(state=tk.DISABLED)
        self.update_training_status_label("Starting...")
        self.train_progress_bar['value'] = 0
        threading.Thread(target=self.train_model_logic, daemon=True).start()

    def train_model_logic(self):
        try:
            self.update_training_status_label("Scanning image folder...")
            self.train_progress_bar['value'] = 5; self.root.update_idletasks()

            if not os.path.isdir(IMAGE_BASE_FOLDER):
                messagebox.showerror("Image Folder Error", f"Training image folder not found: {IMAGE_BASE_FOLDER}")
                self.update_training_status_label(f"Image folder missing."); return
            
            image_files = [f for f in os.listdir(IMAGE_BASE_FOLDER) if f.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.gif'))]

            if not image_files:
                messagebox.showerror("Image Folder Error", f"No image files found in: {IMAGE_BASE_FOLDER}")
                self.update_training_status_label("No images in folder."); return

            self.update_training_status_label(f"Found {len(image_files)} images. Processing...")
            self.train_progress_bar['value'] = 10
            
            features_list, labels_list = [], []
            valid_images_processed_count = 0

            total_images_to_process = len(image_files)
            self.train_progress_bar['maximum'] = total_images_to_process + 30 # For splitting, training, saving

            print(f"DEBUG: Starting image processing loop for {total_images_to_process} files in folder.")

            for i, img_filename in enumerate(image_files):
                label = parse_label_from_filename(img_filename)
                if label is None:
                    print(f"DEBUG: Skipping '{img_filename}' - could not parse label.")
                    continue # Skip if label can't be parsed

                img_path = os.path.join(IMAGE_BASE_FOLDER, img_filename)
                # print(f"DEBUG: Processing image: {img_path}") # Can be verbose
                
                processed_img_array = preprocess_image_for_model(img_path, IMAGE_SIZE, for_training=True)
                
                if processed_img_array is not None:
                    img_features = extract_features(processed_img_array, USE_HOG_FEATURES, IMAGE_SIZE)
                    features_list.append(img_features); labels_list.append(label)
                    valid_images_processed_count += 1
                    # print(f"DEBUG: Successfully processed '{img_filename}'. Label: '{label}'. Total valid: {valid_images_processed_count}")
                else:
                    print(f"DEBUG: Skipping '{img_filename}' - Preprocessing failed.")
                
                self.train_progress_bar['value'] = 10 + i + 1
                if (i + 1) % 20 == 0 or (i + 1) == total_images_to_process:
                    self.update_training_status_label(f"Processed image {i+1}/{total_images_to_process}...")

            print(f"DEBUG: Image processing loop finished.")
            print(f"DEBUG: Total valid images processed: {valid_images_processed_count}")
            print(f"DEBUG: Length of features_list: {len(features_list)}")

            if not features_list or not labels_list:
                msg = "No valid image data could be processed or labeled. Check filenames and image formats."
                messagebox.showerror("Train Data Err", msg)
                self.update_training_status_label("No image data processed."); return

            X, y = np.array(features_list), np.array(labels_list)
            print(f"DEBUG: Shape of X (features array): {X.shape}")
            print(f"DEBUG: Shape of y (labels array): {y.shape}")
            print(f"DEBUG: Unique labels found from filenames: {np.unique(y)}")


            if X.shape[0] < 2 :
                msg = f"Not enough samples to split for training. Found only {X.shape[0]} valid & labeled samples. Need at least 2."
                messagebox.showerror("Insufficient Data", msg)
                self.update_training_status_label(f"Need more data ({X.shape[0]} found)."); print(msg); return

            self.all_labels_from_training = sorted(list(np.unique(y)))
            self.update_training_status_label(f"Data processed. {X.shape[0]} samples. Unique labels: {len(self.all_labels_from_training)}")
            self.train_progress_bar['value'] = 10 + total_images_to_process + 10
            
            self.update_training_status_label("Splitting data & training...")
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42, stratify=y)
            
            print(f"DEBUG: X_train shape: {X_train.shape}, y_train shape: {y_train.shape}")
            print(f"DEBUG: X_test shape: {X_test.shape}, y_test shape: {y_test.shape}")

            model_instance = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1, class_weight='balanced')
            model_instance.fit(X_train, y_train)
            self.train_progress_bar['value'] = 10 + total_images_to_process + 20; self.update_training_status_label("Model fit. Evaluating...")

            y_pred = model_instance.predict(X_test)
            accuracy = accuracy_score(y_test, y_pred)
            print(f"\n--- Model Evaluation (Console) ---"); print(f"Test Accuracy: {accuracy*100:.2f}%")
            print(classification_report(y_test, y_pred, labels=self.all_labels_from_training, zero_division=0))
            
            self.model = model_instance
            self.feature_params = {'image_size': IMAGE_SIZE, 'use_hog': USE_HOG_FEATURES, 'all_labels': self.all_labels_from_training}
            joblib.dump(self.model, MODEL_FILENAME); joblib.dump(self.feature_params, FEATURE_PARAMS_FILENAME)
            self.train_progress_bar['value'] = 10 + total_images_to_process + 30
            self.update_training_status_label(f"Done! Accuracy: {accuracy*100:.2f}%")
            messagebox.showinfo("Training Complete", f"Model training finished.\nTest Accuracy: {accuracy*100:.2f}%\nModel saved.")

        except Exception as e:
            detailed_error_msg = f"Critical error during training: {e}"
            messagebox.showerror("Training Failed", detailed_error_msg)
            self.update_training_status_label(f"Failed! Check console.")
            print(f"Full training error trace:\n{traceback.format_exc()}")
        finally:
            self.btn_train.config(state=tk.NORMAL)
            self.train_progress_bar['value'] = 0

if __name__ == "__main__":
    print("--- Path Configuration (No ARFF Mode) ---")
    print(f"SCRIPT_DIR: {SCRIPT_DIR}"); print(f"PROJECT_ROOT: {PROJECT_ROOT}")
    print(f"IMAGE_BASE_FOLDER: {IMAGE_BASE_FOLDER}") # Training images are read from here
    print(f"MODEL_SAVE_DIR: {MODEL_SAVE_DIR}"); print(f"MODEL_FILENAME: {MODEL_FILENAME}")
    print("--------------------------")
    print(f"Image base folder '{IMAGE_BASE_FOLDER}' exists? {os.path.exists(IMAGE_BASE_FOLDER)}")

    main_window = tk.Tk()
    app_instance = DigitRecognizerApp(main_window)
    main_window.mainloop()