from turtle import width
import cv2
import keyboard
from camera import Camera
import json

with open('config.json', 'r') as f:
    config = json.load(f).get('camera', {})

config.setdefault('width', 500)
config.setdefault('x', 10)
config.setdefault('y', 10)

def change_x(val):
    config['x'] += val
def change_y(val):
    config['y'] += val
def change_width(val):
    config['width'] += val
    config['width'] = max(1, config['width'])

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

camera = Camera(config,0)

while True:
    frame = camera.get_frame()

    square_frame = frame[config['y']:config['y']+config['width'], config['x']:config['x']+config['width']]
    cv2.rectangle(frame, (config['x'], config['y']), (config['x'] + config['width'], config['y'] + config['width']), (255, 0, 0), 1)

    cv2.imshow('Camera Calibration', frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cv2.destroyAllWindows()

with open('config.json', 'w') as f:
    try:
        with open('config.json', 'r') as f:
            full = json.load(f)
    except (FileNotFoundError, json.JSONDecodeError):
        full = {}

    full['camera'] = config

    with open('config.json', 'w') as f:
        json.dump(full, f, indent=4)


print("Calibration complete. Settings saved to config.json.")