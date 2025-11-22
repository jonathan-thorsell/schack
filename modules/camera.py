import cv2
import time

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

    def focus(self, autofocus=True, value=None, wait=0.2):
        """
        Toggle autofocus or set manual focus.
        - autofocus=True: enable autofocus (will try to trigger an autofocus cycle).
        - autofocus=False and value is not None: disable autofocus and set focus to 'value' (camera-dependent range).
        - wait: seconds to wait after toggling autofocus to allow it to settle (optional).
        Returns True if at least one property was successfully set, False otherwise.
        """
        # property ids: try OpenCV constants, fallback to common numeric ids
        af_prop = getattr(cv2, 'CAP_PROP_AUTOFOCUS', 39)
        focus_prop = getattr(cv2, 'CAP_PROP_FOCUS', 28)

        success = False
        try:
            if autofocus:
                # Enable autofocus; some cameras require toggling to trigger a re-focus
                ok = self.cap.set(af_prop, 1)
                success = success or bool(ok)
                # small wait for the camera to adjust
                if wait:
                    time.sleep(wait)
                # sometimes toggling off after on triggers a single autofocus action; don't force here
            else:
                # disable autofocus first if possible
                ok = self.cap.set(af_prop, 0)
                success = success or bool(ok)
                if value is not None:
                    ok2 = self.cap.set(focus_prop, float(value))
                    success = success or bool(ok2)
            return success
        except Exception:
            return False
    
    def release(self):
        self.cap.release()