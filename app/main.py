import base64
import os
import tempfile

import cv2
import mediapipe as mp
import torch
from fastapi import FastAPI
from pydantic import BaseModel

from app.vendor.classification import classify
from app.vendor.detection import detect
from app.vendor.model import UNet
from app.vendor.rectification import warp
from app.vendor.tools import remove_background, resize

RESIZE_VALUE = 256
MODEL_PATH = os.environ.get("MODEL_PATH", "/app/checkpoint/checkpoint_aug_epoch70.pth")

app = FastAPI(title="palm-line-detector")

_net = UNet(n_channels=3, n_classes=1)
_net.load_state_dict(torch.load(MODEL_PATH, map_location=torch.device("cpu")))
_net.eval()


class DetectRequest(BaseModel):
    image: str  # data URL (data:image/jpeg;base64,...) or raw base64 JPEG/PNG


def decode_image_bytes(image_str: str) -> bytes:
    if image_str.startswith("data:"):
        image_str = image_str.split(",", 1)[1]
    return base64.b64decode(image_str)


def length_class(heart_line, head_line, life_line, warped_mini_path):
    """Reproduces the upstream measurement.py threshold logic, long/short only."""
    mp_hands = mp.solutions.hands
    with mp_hands.Hands(static_image_mode=True, max_num_hands=1, min_detection_confidence=0.5) as hands:
        image = cv2.flip(cv2.imread(warped_mini_path), 1)
        image_height, image_width, _ = image.shape
        results = hands.process(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
        if not results.multi_hand_landmarks:
            return None
        hand_landmarks = results.multi_hand_landmarks[0]

        zero = hand_landmarks.landmark[mp_hands.HandLandmark(0).value].y
        one = hand_landmarks.landmark[mp_hands.HandLandmark(1).value].y
        five = hand_landmarks.landmark[mp_hands.HandLandmark(5).value].x
        nine = hand_landmarks.landmark[mp_hands.HandLandmark(9).value].x
        thirteen = hand_landmarks.landmark[mp_hands.HandLandmark(13).value].x

        heart_thres_x = image_width * (1 - (nine + (five - nine) * 2 / 5))
        head_thres_x = image_width * (1 - (thirteen + (nine - thirteen) / 3))
        life_thres_y = image_height * (one + (zero - one) / 3)

    heart_line_points = [tuple(reversed(l[:2])) for l in heart_line]
    head_line_points = [tuple(reversed(l[:2])) for l in head_line]
    life_line_points = [tuple(reversed(l[:2])) for l in life_line]

    heart_tip = heart_line_points[0]
    head_tip = head_line_points[-1]
    life_tip = life_line_points[-1]

    return {
        "heart": "long" if heart_tip[0] < heart_thres_x else "short",
        "head": "long" if head_tip[0] > head_thres_x else "short",
        "life": "long" if life_tip[1] > life_thres_y else "short",
    }


def to_points(line):
    return [{"x": float(p[1]) / RESIZE_VALUE, "y": float(p[0]) / RESIZE_VALUE} for p in line]


@app.get("/health")
def health():
    return {"ok": True}


@app.post("/detect")
def detect_endpoint(req: DetectRequest):
    try:
        raw = decode_image_bytes(req.image)
    except Exception:
        return {"ok": False, "reason": "invalid_image"}

    with tempfile.TemporaryDirectory() as d:
        input_path = os.path.join(d, "input.jpg")
        with open(input_path, "wb") as f:
            f.write(raw)

        if cv2.imread(input_path) is None:
            return {"ok": False, "reason": "invalid_image"}

        warped_path = os.path.join(d, "warped.jpg")
        warp_result = warp(input_path, warped_path)
        if warp_result is None:
            return {"ok": False, "reason": "no_hand_detected"}

        warped_clean_path = os.path.join(d, "warped_clean.jpg")
        remove_background(warped_path, warped_clean_path)

        warped_mini_path = os.path.join(d, "warped_mini.jpg")
        warped_clean_mini_path = os.path.join(d, "warped_clean_mini.jpg")
        resize(warped_path, warped_clean_path, warped_mini_path, warped_clean_mini_path, RESIZE_VALUE)

        lines_path = os.path.join(d, "lines.png")
        detect(_net, warped_clean_path, lines_path, RESIZE_VALUE)

        try:
            lines = classify(lines_path)
        except Exception:
            return {"ok": False, "reason": "classification_failed"}

        if lines is None or len(lines) < 3 or any(l is None for l in lines):
            return {"ok": False, "reason": "lines_not_found"}

        heart_line, head_line, life_line = lines

        classes = length_class(heart_line, head_line, life_line, warped_mini_path)
        if classes is None:
            return {"ok": False, "reason": "no_hand_detected"}

        return {
            "ok": True,
            "lines": {
                "heart": {"points": to_points(heart_line), "lengthClass": classes["heart"]},
                "head": {"points": to_points(head_line), "lengthClass": classes["head"]},
                "life": {"points": to_points(life_line), "lengthClass": classes["life"]},
            },
        }
