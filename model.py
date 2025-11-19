from sklearn.svm import LinearSVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_score
from sklearn.preprocessing import StandardScaler
import numpy as np
import cv2 as cv
import PIL.Image
import os
import joblib
from datetime import datetime

class Model:
    
    def _create_model(self, algorithm):
        """Create model based on algorithm choice"""
        if algorithm == 'svm':
            return LinearSVC(max_iter=2000, dual=False, random_state=42)
        elif algorithm == 'rf':
            return RandomForestClassifier(n_estimators=100, random_state=42)
        else:
            return LinearSVC(max_iter=2000, dual=False, random_state=42)
    
    def switch_algorithm(self, algorithm):
        """Switch to a different algorithm"""
        self.algorithm = algorithm
        self.model = self._create_model(algorithm)
        self.is_trained = False
        self.accuracy = 0.0
    
    def extract_features(self, img_path):
        """Extract features from image"""
        try:
            img = cv.imread(img_path)
            if img is None:
                return None
            
            # Convert to grayscale
            gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
            
            # Resize to 150x150
            resized = cv.resize(gray, (150, 150))
            
            # Flatten
            features = resized.reshape(-1)
            
            # Optional: Apply histogram equalization for better features
            # equalized = cv.equalizeHist(gray)
            # resized = cv.resize(equalized, (150, 150))
            # features = resized.reshape(-1)
            
            return features
        except Exception as e:
            print(f"Error extracting features: {str(e)}")
            return None
    
    def train_model(self, counters):
        """Train the model with cross-validation"""
        img_list = []
        class_list = []

        try:
            # Load class 1 images
            if counters[0] > 1 and os.path.exists('1'):
                for i in range(1, counters[0]):
                    img_path = f'1/frame{i}.jpg'
                    if os.path.exists(img_path):
                        features = self.extract_features(img_path)
                        if features is not None:
                            img_list.append(features)
                            class_list.append(1)

            # Load class 2 images
            if counters[1] > 1 and os.path.exists('2'):
                for i in range(1, counters[1]):
                    img_path = f'2/frame{i}.jpg'
                    if os.path.exists(img_path):
                        features = self.extract_features(img_path)
                        if features is not None:
                            img_list.append(features)
                            class_list.append(2)

            if len(img_list) < 2:
                raise ValueError("Insufficient training data. Need at least 2 images.")

            # Convert to numpy arrays
            img_array = np.array(img_list)
            class_array = np.array(class_list)
            
            # Scale features
            img_array = self.scaler.fit_transform(img_array)
            
            # Train model
            self.model.fit(img_array, class_array)
            
            # Calculate accuracy with cross-validation
            cv_scores = cross_val_score(self.model, img_array, class_array, cv=min(5, len(img_array)))
            self.accuracy = cv_scores.mean()
            self.training_samples = len(img_array)
            self.last_trained = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            
            self.is_trained = True
            
            # Save model and scaler
            joblib.dump(self.model, 'trained_model.pkl')
            joblib.dump(self.scaler, 'scaler.pkl')
            
            print(f"Model successfully trained!")
            print(f"Accuracy: {self.accuracy:.2%}")
            print(f"Training samples: {self.training_samples}")
            return True, self.accuracy

        except Exception as e:
            print(f"Error during training: {str(e)}")
            return False, 0.0
    
    def load_model(self):
        """Load a previously trained model"""
        try:
            if os.path.exists('trained_model.pkl') and os.path.exists('scaler.pkl'):
                self.model = joblib.load('trained_model.pkl')
                self.scaler = joblib.load('scaler.pkl')
                self.is_trained = True
                print("Model loaded successfully!")
                return True
        except Exception as e:
            print(f"Error loading model: {str(e)}")
        return False