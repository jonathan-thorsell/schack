import cv2

last_frame = None

class Camera:
    def __init__(self, index=1):
        self.cap = cv2.VideoCapture(index)
        if not self.cap.isOpened():
            raise ValueError("Cannot open camera")
    
    def get_frame(self):
        ret, frame = self.cap.read()
        if not ret:
            raise ValueError("Can't receive frame (stream end?). Exiting ...")
        return frame
    
    def release(self):
        self.cap.release()