import time
from collections import deque
import cv2
import numpy as np
import chess

def draw_frame(frame: np.ndarray, settings: dict, board, engine):
    camera_settings = settings["camera"]
    # Initialize persistent state (stored on the function object)
    if not hasattr(draw_frame, "_prev_gray"):
        draw_frame._prev_gray = None
        draw_frame._stable_count = 0
        # configurable via settings
        draw_frame._frames_needed = int(settings.get('stable_frames', 20))
        draw_frame._change_threshold = float(settings.get('change_threshold', 0.01))  # fraction of pixels
        draw_frame._min_interval = float(settings.get('min_stable_interval', 5.0))  # seconds
        draw_frame._last_stable_time = 0.0
        # additional guard: require that there were no "stable" frames in the pre-gap before triggering prediction
        draw_frame._pre_stable_gap = int(settings.get('pre_stable_gap', 10))  # number of frames to look back before the stable run
        # history buffer holds the recent per-frame "similar" booleans
        draw_frame._history = deque(maxlen=draw_frame._frames_needed + draw_frame._pre_stable_gap)

    # draw 8x8 grid
    if settings["show_grid_lines"]:
        step = camera_settings['width'] / 8
        for i in range(9):
            w = int(i*step)
            cv2.line(frame, (camera_settings['x'] + w, camera_settings['y']), (camera_settings['x'] + w, camera_settings['y'] + camera_settings['width']), (0, 0, 175), 1)
            cv2.line(frame, (camera_settings['x'], camera_settings['y'] + w), (camera_settings['x'] + camera_settings['width'], camera_settings['y'] + w), (0, 0, 175), 1)
    
    for i in range(len(board)):
        char = board[i]
        if char == '.':
            continue
        row, col = divmod(i, 8)
        step = camera_settings['width'] / 8
        sx = camera_settings['x'] + col * step
        sy = camera_settings['y'] + row * step
        cx = sx + step / 2
        cy = sy + step / 2
        font = cv2.QT_FONT_NORMAL
        scale = step / 80.0
        thickness = max(1, int(step / 50))
        (tw, th), baseline = cv2.getTextSize(char, font, scale, thickness)
        text_x = int(cx - tw / 2)
        text_y = int(cy + th / 2)
        color = (255,255,255) if (str.isupper(char) or char=='%') else (100,100,100)
        cv2.putText(frame, str.lower(char), (text_x, text_y), font, scale, color, thickness, cv2.LINE_AA)

    # --- stability detection ---
    x, y, w = camera_settings['x'], camera_settings['y'], camera_settings['width']
    roi = frame[y:y+w, x:x+w]
    gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
    gray = cv2.GaussianBlur(gray, (5, 5), 0)

    stable = False
    if draw_frame._prev_gray is None:
        draw_frame._prev_gray = gray.copy()
        draw_frame._stable_count = 0
        # also append initial history value (treat first frame as unstable to avoid false positives)
        draw_frame._history.append(False)
    else:
        diff = cv2.absdiff(gray, draw_frame._prev_gray)
        _, diff_thresh = cv2.threshold(diff, 25, 255, cv2.THRESH_BINARY)
        changed = cv2.countNonZero(diff_thresh)
        area = diff_thresh.shape[0] * diff_thresh.shape[1]
        frac_changed = changed / float(area) if area > 0 else 1.0

        frame_similar = (frac_changed <= draw_frame._change_threshold)
        # update history
        draw_frame._history.append(frame_similar)

        if frame_similar:
            draw_frame._stable_count += 1
        else:
            draw_frame._stable_count = 0

        draw_frame._prev_gray = gray.copy()
        if draw_frame._stable_count >= draw_frame._frames_needed:
            # check the pre-stable gap: ensure there were NO stable frames in the pre_stable_gap frames
            need_len = draw_frame._frames_needed + draw_frame._pre_stable_gap
            if len(draw_frame._history) >= need_len:
                # the slice before the stable run:
                # history layout: [..., pre_gap frames, frames_needed frames]
                hist_list = list(draw_frame._history)
                pre_slice = hist_list[-need_len:-draw_frame._frames_needed]  # length == pre_stable_gap
                if any(pre_slice):
                    # there was stability in the pre-gap => do not trigger prediction now
                    stable = False
                else:
                    stable = True
            else:
                # not enough history to check pre-gap; permit triggering (or change as desired)
                stable = True

    # enforce minimum interval between stable events
    if stable:
        now = time.time()
        if now - draw_frame._last_stable_time >= draw_frame._min_interval:
            draw_frame._last_stable_time = now
        else:
            stable = False

    # Return the frame and a boolean indicating if it has been stable for the configured number of frames.
    return frame, stable