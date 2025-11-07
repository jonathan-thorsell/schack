import chess
import chess.engine
import cv2
import numpy as np
from modules.camera import Camera
from tensorflow.keras.models import load_model

labels = ["empty", "black", "white"]

initial_position = [
    2, 2, 2, 2, 2, 2, 2, 2,
    2, 2, 2, 2, 2, 2, 2, 2,
    0, 0, 0, 0, 0, 0, 0, 0,
    0, 0, 0, 0, 0, 0, 0, 0,
    0, 0, 0, 0, 0, 0, 0, 0,
    0, 0, 0, 0, 0, 0, 0, 0,
    1, 1, 1, 1, 1, 1, 1, 1,
    1, 1, 1, 1, 1, 1, 1, 1
]

class Position:
    def __init__(self, path):
        print("Starting engine...", end="\b"*len("Starting engine..."))
        self.engine = chess.engine.SimpleEngine.popen_uci(path)

        print('Loading model...', end="\r")
        self.model = load_model(r"training\models\occupation_classifier.keras")

        self.board = chess.Board()
        self.num_array_position = initial_position.copy()
    
    def get_position(self):
        return str(self.board).replace('\n',' ').split(' ')

    def piece_can_move(self, square_index):
        self.board.turn = chess.WHITE
        return any(m.from_square == square_index for m in self.board.pseudo_legal_moves)


    def process_position(self, camera: Camera):
        print("processing began")
        squares = camera.get_processed_frame()
        # self.board.turn = chess.WHITE # gotta do this to make the legal moves work, weird right?
        indices = []
        for i in range(len(self.num_array_position)):
            if self.num_array_position[i] != 2: continue # not white piece
            if not self.piece_can_move(i): continue # piece couldn't move anyway, no need to check it
            indices.append(i)
        
        processed_square_indices = []
        processed_square_images = []
        for i in indices:
            square = cv2.cvtColor(squares[i], cv2.COLOR_BGR2RGB)
            # Prepare the square image for prediction
            square_resized = cv2.resize(square, (64, 64))
            square_resized = square_resized.astype("float32") / 255.0
            processed_square_indices.append(i)
            processed_square_images.append(square_resized)
        
        if len(processed_square_images) == 0:
            print("no processed images")
            return

        processed_square_images = np.asarray(processed_square_images, dtype=np.float32)

        predictions = self.model.predict(processed_square_images)
        predicted_labels = np.argmax(predictions, axis=1)

        for index, label in enumerate(predicted_labels):
            print(index, label)
            if labels[label] == "empty":
                self.board.remove_piece_at(processed_square_indices[index])
                print(f"removed piece at {processed_square_indices[index]}")
        print("predictions complete.")