import cv2

last_frame = None

class Camera:
    def __init__(self, index=1):
        print("Waiting for camera...", end="\r")
        self.cap = cv2.VideoCapture(index)
        if not self.cap.isOpened():
            raise ValueError("Cannot open camera")
        print("Camera connected.         ")
    
    def get_frame(self):
        ret, frame = self.cap.read()
        if not ret:
            raise ValueError("Can't receive frame (stream end?). Exiting ...")
        frame = cv2.flip(frame, -1)
        return frame
    
    def get_processed_frame(self, settings):
        frame = self.get_frame()
        cropped_frame = frame[settings['y']:settings['y'] + settings['width'], settings['x']:settings['x'] + settings['width']]
        return cropped_frame
    
    def release(self):
        self.cap.release()