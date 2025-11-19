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
    