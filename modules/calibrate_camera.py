from turtle import width
import cv2
import keyboard
from camera import Camera
import json

with open('config.json', 'r') as f:
    config = json.load(f)
    camera_config = config.get('camera', {})

camera_config.setdefault('width', 500)
camera_config.setdefault('x', 10)
camera_config.setdefault('y', 10)
def change_x(val):
    camera_config['x'] += val
def change_y(val):
    camera_config['y'] += val
def change_width(val):
    camera_config['width'] += val
    camera_config['width'] = max(1, camera_config['width'])

keyboard.add_hotkey('a', lambda: change_x(-10))
keyboard.add_hotkey('shift+a', lambda: change_x(-1))
keyboard.add_hotkey('d', lambda: change_x(10))
keyboard.add_hotkey('shift+d', lambda: change_x(1))
keyboard.add_hotkey('w', lambda: change_y(-10))
keyboard.add_hotkey('shift+w', lambda: change_y(-1))
keyboard.add_hotkey('s', lambda: change_y(10))
keyboard.add_hotkey('shift+s', lambda: change_y(1))
keyboard.add_hotkey('left', lambda: change_width(-10))
keyboard.add_hotkey('shift+left', lambda: change_width(-1))
keyboard.add_hotkey('right', lambda: change_width(10))
keyboard.add_hotkey('shift+right', lambda: change_width(1))

print("Camera Calibration".center(22))
print("----------------------")
print("Press 'a'/'d' to move left/right")
print("Press 'w'/'s' to move up/down")
print("Press 'left'/'right' to resize")
print("Use 'shift' for finer adjustments")
print("Press 'q' to quit")

camera = Camera(camera_config,0)

while True:
    frame = camera.get_frame()

    square_frame = frame[camera_config['y']:camera_config['y']+camera_config['width'], camera_config['x']:camera_config['x']+camera_config['width']]
    cv2.rectangle(frame, (camera_config['x'], camera_config['y']), (camera_config['x'] + camera_config['width'], camera_config['y'] + camera_config['width']), (255, 0, 0), 1)

    cv2.imshow('Camera Calibration', frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cv2.destroyAllWindows()

config['camera'] = camera_config
with open('config.json', 'w') as f:
    json.dump(config, f, indent=4)


print("Calibration complete. Settings saved to config.json.")