# make_known_face.py
import sys
import numpy as np
import cv2
import face_recognition
import os

def make_embedding(image_path, out_path):
    img = cv2.imread(image_path)
    if img is None:
        raise RuntimeError("Could not read image: " + image_path)
    rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    encs = face_recognition.face_encodings(rgb)
    if not encs:
        raise RuntimeError("No face found in the image")
    enc = encs[0]
    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    np.savez(out_path, encoding=enc)
    print("Saved embedding to:", out_path)

if __name__ == "__main__":
    if len(sys.argv) < 3:
        print("Usage: python make_known_face.py input.jpg /full/path/to/uploads/known_face.npz")
        sys.exit(1)
    make_embedding(sys.argv[1], sys.argv[2])
