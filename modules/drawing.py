import cv2
import numpy as np
import chess

def draw_frame(frame: np.ndarray, settings: dict, board, engine):
    #draw 8x8 grid
    step = settings['width'] // 8
    for i in range(9):
        cv2.line(frame, (settings['x'] + i * step, settings['y']), (settings['x'] + i * step, settings['y'] + settings['width']), (0, 0, 175), 1)
        cv2.line(frame, (settings['x'], settings['y'] + i * step), (settings['x'] + settings['width'], settings['y'] + i * step), (0, 0, 175), 1)
    
    for i in range(len(board)):
        char = board[i]
        if char == '.':
            continue
        row, col = divmod(i, 8)
        step = settings['width'] // 8
        sx = settings['x'] + col * step
        sy = settings['y'] + row * step
        cx = sx + step // 2
        cy = sy + step // 2
        font = cv2.QT_FONT_NORMAL
        scale = step / 80.0
        thickness = max(1, int(step / 50))
        (tw, th), baseline = cv2.getTextSize(char, font, scale, thickness)
        text_x = int(cx - tw / 2)
        text_y = int(cy + th / 2)
        color = (255,255,255) if (str.isupper(char) or char=='%') else (100,100,100)
        cv2.putText(frame, str.lower(char), (text_x, text_y), font, scale, color, thickness, cv2.LINE_AA)


    return frame