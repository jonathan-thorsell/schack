import cv2

last_frame = None

class Camera:
    def __init__(self, settings, index=1):
        print("Waiting for camera...", end="\r")
        self.cap = cv2.VideoCapture(index)
        self.settings = settings
        if not self.cap.isOpened():
            raise ValueError("Cannot open camera")
        print("Camera connected.         ")
    
    def get_frame(self):
        ret, frame = self.cap.read()
        if not ret:
            raise ValueError("Can't receive frame (stream end?). Exiting ...")
        frame = cv2.flip(frame, -1)
        return frame
    
    def get_processed_frame(self):
        frame = self.get_frame()
        cropped_frame = frame[self.settings['y']:self.settings['y'] + self.settings['width'], self.settings['x']:self.settings['x'] + self.settings['width']]
        square_size = self.settings['width'] // 8
        squares = []
        for row in range(8):
            for col in range(8):
                y_start = row * square_size
                x_start = col * square_size
                square = cropped_frame[y_start:y_start + square_size, x_start:x_start + square_size]
                squares.append(square)
        return squares
    
    def release(self):
        self.cap.release()