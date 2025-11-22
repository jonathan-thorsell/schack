import os
import chess
import chess.engine
import cv2
import numpy as np
import secrets
from modules.camera import Camera
from tensorflow.keras.models import load_model

labels = ["empty", "black", "white"]
ord_labels = {
    "w": "white",
    "b": "black",
    "e": "empty"
}

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

def add_to_training_set(square_image, square_color):
        def save_image(key):
            dir = f"training/data/testing/{ord_labels[key]}"
            file_name = f"{square_color}_{secrets.token_hex(12)}.jpg"
            cv2.imwrite(os.path.join(dir, file_name), square_image)
            print(f"Piece identified as {ord_labels[key]} and saved to {file_name}")
        
        shown_image = cv2.resize(square_image, (500, 500))

        cv2.imshow("W / WHITE | B / BLACK | E / EMPTY", shown_image)
        key = cv2.waitKey(0) & 0xFF
        if key == ord('w'):
            save_image('w')
        elif key == ord('b'):
            save_image('b')
        elif key == ord('e'):
            save_image('e')
        cv2.destroyWindow("W / WHITE | B / BLACK | E / EMPTY")


class Position:
    def __init__(self, path, process_type):
        print("Starting engine...", end="\b"*len("Starting engine..."))
        self.engine = chess.engine.SimpleEngine.popen_uci(path)

        print('Loading model...', end="\r")
        self.occupation_model = load_model(r"training\models\occupation_classifier.keras")
        self.color_model = load_model(r"training\models\color_classifier.keras")

        self.board = chess.Board()
        self.process_type = process_type
        if process_type == 0:
            self.num_array_position = initial_position.copy()
        else:
            self.num_array_position = [0]*64
    
    def get_position(self):
        if self.process_type == 0:
            return str(self.board).replace('\n',' ').split(' ')
        else:
            chars = []
            for val in self.num_array_position:
                match val:
                    case 0:
                        chars.append('.')
                    case 1:
                        chars.append('#')  # black piece
                    case 2:
                        chars.append('%')  # white piece
            return chars

    def piece_can_move(self, square_index):
        self.board.turn = chess.WHITE
        return any(m.from_square == square_index for m in self.board.pseudo_legal_moves)

    def process_position(self, camera: Camera, debug=False):
        if debug:
            print("processing began")
        squares = camera.get_processed_frame()
        # self.board.turn = chess.WHITE # gotta do this to make the legal moves work, weird right?

        indices = []
        match(self.process_type):
            case 0:
                for i in range(len(self.num_array_position)):
                    if self.num_array_position[i] != 2: continue # not white piece
                    if not self.piece_can_move(i): continue # piece couldn't move anyway, no need to check it
                    indices.append(i)
            case 1:
                indices = list(range(64))  # process all squares
        
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

        npsquares = np.asarray(processed_square_images, dtype=np.float32)

        predictions = self.occupation_model.predict(npsquares, verbose = 1 if debug else 0)
        predicted_labels = (predictions > 0.5).astype(int).flatten()

        occupied_squares = [i for i, label in enumerate(predicted_labels) if label == 1]
        if len(occupied_squares) > 0:
            # determine color for occupied squares and convert labels to 0=empty, 1=black, 2=white
            occupied_idxs = [i for i, label in enumerate(predicted_labels) if label == 1]
            if len(occupied_idxs) > 0:
                occupied_images = npsquares[occupied_idxs]
                color_preds = self.color_model.predict(occupied_images, verbose = 1 if debug else 0)

                # Handle both single-output (sigmoid) and two-output (softmax) models
                if color_preds.ndim == 1 or (hasattr(color_preds, "shape") and color_preds.shape[1] == 1):
                    whites = (color_preds.flatten() > 0.5)
                else:
                    whites = np.argmax(color_preds, axis=1) == 1

                # Build final integer labels: 0 empty, 1 black, 2 white
                final_labels = predicted_labels.astype(int).copy()
                for k, occ in enumerate(occupied_idxs):
                    final_labels[occ] = 2 if whites[k] else 1

                predicted_labels = final_labels
                predictions = predicted_labels.copy()

        match(self.process_type):
            case 0:
                empty_count = sum(1 for label in predicted_labels if labels[label] == "empty")
                if empty_count > 0 or empty_count >= 2:
                    input("This prediction had unreliable results. Would you like to add it to the training set? (Press Enter to continue)")
                    for i, square_image in enumerate(squares):
                        rank = 7 - (i // 8)
                        file = i % 8
                        square_color = 255 if (rank + file) % 2 == 1 else 0
                        add_to_training_set(square_image, square_color)
                    return False

                print("Raw predictions:", predictions[:10].flatten())
                print("Predicted labels:", predicted_labels[:10])

                for index, label in enumerate(predicted_labels):
                    if labels[label] == "empty":
                        self.board.remove_piece_at(processed_square_indices[index])
                        print(f"removed piece at {processed_square_indices[index]}")
            case 1:
                #update self.num_array_position
                for index, label in enumerate(predicted_labels):
                    self.num_array_position[processed_square_indices[index]] = label
        if debug:
            print("predictions complete.")