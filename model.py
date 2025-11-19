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