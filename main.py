import cv2
import os
import secrets
from modules.camera import Camera
from modules.drawing import draw_frame
from modules.position import Position
# from tensorflow.keras.models import load_model
import numpy as np
import time
import json  
import keyboard

config_path = "config.json"
if not os.path.exists(config_path):
    print(f"❌ Config file not found: {config_path}")
    exit(1)
with open('config.json', 'r') as f:
    config = json.load(f)

settings = config['camera']


os.system('cls')
print("Performing setup...")
camera = Camera()



print("Loading model...", end="\r")
# model = load_model("model.h5")
print("Model loaded.          ")

print("Setting up board -> ", end="")
board = Position(config['stockfish']['path'])
print("Board initialized.                       ")


print("Setup complete.")

while True:
    frame = camera.get_frame()
    frame = draw_frame(frame, settings, board.get_position())

    #show frame
    cv2.imshow('SCHACK', frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break