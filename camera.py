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
        
    
    def __del__(self):
        if self.camera.isOpened():
            self.camera.release()   
    
    def set_brightness(self, value):
        """Set brightness (-100 to 100)"""
        self.brightness = value
    
    def set_contrast(self, value):
        """Set contrast (0.5 to 3.0)"""
        self.contrast = max(0.5, min(3.0, value / 100.0))

    def set_saturation(self, value):
        """Set saturation (0.0 to 2.0)"""
        self.saturation = max(0.0, min(2.0, value / 100.0))

    def set_flip(self, horizontal=False, vertical=False):
        """Set flip options"""
        self.flip_horizontal = horizontal
        self.flip_vertical = vertical
    
    def adjust_image(self, frame):
        """Apply brightness, contrast, and saturation adjustments"""
        # Apply brightness
        if self.brightness != 0:
            frame = cv.convertScaleAbs(frame, alpha=1, beta=self.brightness)
        
        # Apply contrast
        frame = cv.convertScaleAbs(frame, alpha=self.contrast, beta=0)
        
        # Apply saturation
        if self.saturation != 1.0:
            hsv = cv.cvtColor(frame, cv.COLOR_RGB2HSV).astype('float32')
            hsv[:, :, 1] = hsv[:, :, 1] * self.saturation
            hsv[:, :, 1] = cv.threshold(hsv[:, :, 1], 255, 255, cv.THRESH_TRUNC)[1]
            frame = cv.cvtColor(hsv.astype('uint8'), cv.COLOR_HSV2RGB)
        
        return frame
    
    def get_frame(self):
        """Retrieve and process frame from camera"""
        if self.camera.isOpened():
            ret, frame = self.camera.read()

            if ret:
                # Convert BGR to RGB
                frame = cv.cvtColor(frame, cv.COLOR_BGR2RGB)
                
                # Apply adjustments
                frame = self.adjust_image(frame)
                
                # Apply flips
                if self.flip_horizontal:
                    frame = cv.flip(frame, 1)
                if self.flip_vertical:
                    frame = cv.flip(frame, 0)
                
                self.frame_count += 1
                return (ret, frame)
            else:
                return (ret, None)
        else:
            return (False, None)
    
    
    def get_available_cameras(self):
        """Detect available cameras"""
        available = []
        for i in range(5):
            cap = cv.VideoCapture(i)
            if cap.isOpened():
                available.append(i)
                cap.release()
        return available