import cv2 as cv
import threading
from collections import deque

class Camera:
    def __init__(self, camera_id=0):
        self.camera = cv.VideoCapture(camera_id)
        if not self.camera.isOpened():
            raise ValueError(f"Unable to open camera {camera_id}!")

        self.width = int(self.camera.get(cv.CAP_PROP_FRAME_WIDTH))
        self.height = int(self.camera.get(cv.CAP_PROP_FRAME_HEIGHT))
        
        