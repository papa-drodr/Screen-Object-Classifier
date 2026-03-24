from collections import deque

import cv2
import mss
import numpy as np
import torch
from model import ScreenClassifier

# ------------------------
# Constants
# ------------------------
CAPTURE_W, CAPTURE_H = 1080, 720
MAP_W = 240
SMOOTHING_WINDOW = 10
TOP_K = 3  # the top 3 predicted classes

COLOR_GREEN = (0, 255, 0)
COLOR_ORANGE = (0, 165, 255)
COLOR_GRAY = (50, 50, 50)
COLOR_WHITE = (255, 255, 255)
COLOR_RED = (0, 0, 255)
COLOR_BLACK = (0, 0, 0)


# ------------------------
# Render Helpers
# ------------------------
def draw_outlined_text(img, text, pos, scale, color, thickness=1):
    """draw outline text"""
    cv2.putText(
        img,
        text,
        pos,
        cv2.FONT_HERSHEY_SIMPLEX,
        scale,
        COLOR_BLACK,
        thickness + 2,
        cv2.LINE_AA,
    )
    cv2.putText(
        img, text, pos, cv2.FONT_HERSHEY_SIMPLEX, scale, color, thickness, cv2.LINE_AA
    )


def draw_bar(img, x, y, prob, color, max_width=200):
    """draw prob bar"""
    bar_width = int(max_width * prob)
    cv2.rectangle(img, (x, y), (x + max_width, y + 10), COLOR_GRAY, -1)
    cv2.rectangle(img, (x, y), (x + bar_width, y + 10), color, -1)


def get_bar_color(rank, score):
    """convert color"""
    if rank == 0:
        return COLOR_GREEN if score > 50 else COLOR_ORANGE
    # 2nd, 3rd are orange
    return COLOR_ORANGE


def draw_predictions(img, top_probs, top_catids, class_names, origin_x, origin_y):
    """draw bar and label."""
    for i in range(TOP_K):
        prob = top_probs[i].item()
        score = prob * 100
        label = class_names[top_catids[i]]
        color = get_bar_color(i, score)

        y_offset = origin_y + (i * 40)
        text = f"{i + 1}. {label}: {score:.1f}%"
        draw_outlined_text(img, text, (origin_x, y_offset), scale=0.5, color=color)
        draw_bar(img, origin_x, y_offset + 10, prob, color)


def draw_minimap(
    img,
    minimap_bg,
    map_x,
    map_y,
    map_w,
    map_h,
    monitor_top,
    monitor_left,
    full_w,
    full_h,
):
    """make mini-map and capture area"""
    # mini-map background
    img[map_y : map_y + map_h, map_x : map_x + map_w] = minimap_bg
    cv2.rectangle(img, (map_x, map_y), (map_x + map_w, map_y + map_h), COLOR_WHITE, 1)

    # capture area
    box_x = map_x + int(monitor_left * (map_w / full_w))
    box_y = map_y + int(monitor_top * (map_h / full_h))
    box_w = int(CAPTURE_W * (map_w / full_w))
    box_h = int(CAPTURE_H * (map_h / full_h))
    cv2.rectangle(img, (box_x, box_y), (box_x + box_w, box_y + box_h), COLOR_RED, 2)

    # text position
    monitor_text = f"({monitor_top}, {monitor_left})"
    draw_outlined_text(
        img,
        monitor_text,
        (map_x, map_y + map_h + 20),
        scale=0.5,
        color=COLOR_WHITE,
        thickness=2,
    )


# ------------------------
# Inference
# ------------------------
def predict_smoothed(classifier, rgb_frame, prob_history):
    """Inferring the frame and returning smoothed probabilities"""
    input_tensor = classifier.preprocess(rgb_frame).unsqueeze(0)
    with torch.no_grad():
        output = classifier.model(input_tensor)

    current_prob = torch.nn.functional.softmax(output[0], dim=0)
    prob_history.append(current_prob)

    smoothed_probs = torch.stack(list(prob_history)).mean(dim=0)
    return torch.topk(smoothed_probs, TOP_K)


# ------------------------
# Capture Loop
# ------------------------
def capture(classifier):
    sct = mss.mss()

    primary_monitor = sct.monitors[1]
    full_w, full_h = primary_monitor["width"], primary_monitor["height"]
    monitor_top, monitor_left = 150, 150

    map_h = int(MAP_W * (full_h / full_w))
    map_x = CAPTURE_W - MAP_W - 10
    map_y = 10
    pred_origin_y = map_y + map_h + 40  # Start prediction result y

    full_monitor = {"top": 0, "left": 0, "width": full_w, "height": full_h}
    full_sct_img = sct.grab(full_monitor)
    full_bgr = cv2.cvtColor(np.array(full_sct_img), cv2.COLOR_BGRA2BGR)
    minimap_bg = cv2.resize(full_bgr, (MAP_W, map_h))

    prob_history = deque(maxlen=SMOOTHING_WINDOW)

    print("Start capturing(Termination: 'q' / Mini-map Update: 'n' / Move: WASD)")

    while True:
        monitor = {
            "top": monitor_top,
            "left": monitor_left,
            "width": CAPTURE_W,
            "height": CAPTURE_H,
        }
        sct_img = sct.grab(monitor)
        frame = np.array(sct_img)
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGRA2RGB)
        bgr_frame = cv2.cvtColor(frame, cv2.COLOR_BGRA2BGR)

        # inference
        top_probs, top_catids = predict_smoothed(classifier, rgb_frame, prob_history)

        # rendering
        draw_predictions(
            bgr_frame,
            top_probs,
            top_catids,
            classifier.class_names,
            map_x,
            pred_origin_y,
        )
        draw_minimap(
            bgr_frame,
            minimap_bg,
            map_x,
            map_y,
            MAP_W,
            map_h,
            monitor_top,
            monitor_left,
            full_w,
            full_h,
        )

        cv2.imshow("Screen Object Classifier", bgr_frame)

        # keys
        key = cv2.waitKey(1) & 0xFF
        if key in (ord("q"), 27):
            break
        elif key == ord("w") and monitor_top > 0:
            monitor_top -= 2
        elif key == ord("s") and monitor_top < full_h:
            monitor_top += 2
        elif key == ord("a") and monitor_left > 0:
            monitor_left -= 2
        elif key == ord("d") and monitor_left < full_w:
            monitor_left += 2
        elif key == ord("n"):  # 미니맵 배경 갱신
            full_sct_img = sct.grab(full_monitor)
            full_bgr = cv2.cvtColor(np.array(full_sct_img), cv2.COLOR_BGRA2BGR)
            minimap_bg = cv2.resize(full_bgr, (MAP_W, map_h))

    cv2.destroyAllWindows()


if __name__ == "__main__":
    classifier = ScreenClassifier()
    capture(classifier)
