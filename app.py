import os
import base64
import json
import time
import io
import gc
from flask import Flask, request, render_template, jsonify, redirect, session, send_file, Response
from flask_cors import CORS

# ==========================================
# 1. SETUP
# ==========================================
UPLOAD_FOLDER = "uploads"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(os.path.join(UPLOAD_FOLDER, "panorama"), exist_ok=True)
os.makedirs(os.path.join("static", "mod2", "temps"), exist_ok=True)
os.makedirs(os.path.join("static", "marker"), exist_ok=True)

app = Flask(__name__, static_folder="static", template_folder="templates")
CORS(app)
app.config["UPLOAD_FOLDER"] = UPLOAD_FOLDER
app.config["MAX_CONTENT_LENGTH"] = 200 * 1024 * 1024
app.secret_key = "supersecretkey"


# ==========================================
# 2. HELPERS
# ==========================================
def allowed_image(filename):
    ext = filename.lower().rsplit(".", 1)[-1]
    return ext in ("jpg", "jpeg", "png", "bmp", "tiff")


def read_image_from_bytes(file_storage):
    import numpy as np
    import cv2
    data = file_storage.read()
    arr = np.frombuffer(data, dtype=np.uint8)
    return cv2.imdecode(arr, cv2.IMREAD_COLOR)


# ==========================================
# 3. AUTH & HOME
# ==========================================
@app.route("/")
def index():
    session.clear()
    return render_template("face_unlock.html")


@app.route("/skip_login")
def skip_login():
    session["logged_in"] = True
    return redirect("/home")


@app.route("/auth_face", methods=["POST"])
def auth_face():
    time.sleep(1.0)
    return jsonify({
        "match": False,
        "distance": 0.85,
        "message": "Face not recognized (Cloud Demo Mode)."
    })


@app.route("/home")
def home():
    if not session.get("logged_in"):
        return redirect("/")
    return render_template("home.html")


@app.route("/logout")
def logout():
    session.clear()
    return redirect("/")


# ==========================================
# 4. MODULE 1 – CALIBRATION
# ==========================================
@app.route("/module1")
def module1():
    return render_template("index.html")


@app.route("/calibrate", methods=["POST"])
def calibrate():
    import cv2
    import numpy as np

    files = request.files.getlist("files[]")
    if not files:
        return jsonify({"error": "No files"}), 400

    cols = int(request.form.get("pattern_cols", "9"))
    rows = int(request.form.get("pattern_rows", "6"))
    sq = float(request.form.get("square_size", "1.0"))

    objp = np.zeros((rows * cols, 3), np.float32)
    objp[:, :2] = np.mgrid[0:cols, 0:rows].T.reshape(-1, 2) * sq

    objpoints, imgpoints = [], []
    img_shape = None

    for f in files:
        img = read_image_from_bytes(f)
        if img is None:
            continue
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        if img_shape is None:
            img_shape = gray.shape[::-1]

        found, corners = cv2.findChessboardCorners(gray, (cols, rows))
        if found:
            refined = cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1),
                                        (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.1))
            objpoints.append(objp)
            imgpoints.append(refined)

    ret, K, _, _, _ = cv2.calibrateCamera(objpoints, imgpoints, img_shape, None, None)
    return jsonify({
        "fx": float(K[0, 0]),
        "fy": float(K[1, 1]),
        "cx": float(K[0, 2]),
        "cy": float(K[1, 2])
    })


# ==========================================
# 5. MODULE 2 – TEMPLATE MATCHING
# ==========================================
@app.route("/module2")
def module2():
    return render_template("module2.html")


@app.route("/match", methods=["POST"])
def match():
    import cv2
    from scene import run_multi_template_match

    scene = read_image_from_bytes(request.files["scene"])
    tdir = os.path.join("static", "mod2", "temps")

    templates = []
    for f in os.listdir(tdir):
        if f.endswith((".jpg", ".png")):
            t = cv2.imread(os.path.join(tdir, f))
            if t is not None:
                templates.append(t)

    result, boxes = run_multi_template_match(scene, templates)
    _, buf = cv2.imencode(".png", result)
    return jsonify({
        "image": base64.b64encode(buf).decode("ascii"),
        "boxes": boxes
    })


# ==========================================
# 6. MODULE 3 – IMAGE PROCESSING
# ==========================================
@app.route("/module3")
def module3():
    return render_template("module3.html")


@app.route("/process_gradient_log", methods=["POST"])
def process_gradient_log():
    import cv2
    import numpy as np

    files = request.files.getlist("images[]")
    outputs = []

    for f in files:
        img = read_image_from_bytes(f)
        if img is None:
            continue

        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        gx = cv2.Sobel(gray, cv2.CV_32F, 1, 0)
        gy = cv2.Sobel(gray, cv2.CV_32F, 0, 1)
        mag = cv2.magnitude(gx, gy)
        mag = cv2.normalize(mag, None, 0, 255, cv2.NORM_MINMAX).astype("uint8")

        blur = cv2.GaussianBlur(gray, (5, 5), 1)
        log = cv2.Laplacian(blur, cv2.CV_32F)
        log = cv2.normalize(abs(log), None, 0, 255, cv2.NORM_MINMAX).astype("uint8")

        combined = np.hstack([gray, mag, log])
        outputs.append(combined)

    final = np.vstack(outputs)
    _, buf = cv2.imencode(".png", final)
    return jsonify({"image": base64.b64encode(buf).decode("ascii")})


# ==========================================
# 7. MODULE 4 – PANORAMA + SIFT (UPLOAD)
# ==========================================
@app.route("/module4")
def module4():
    return render_template("module4.html")


@app.route("/process_stitch", methods=["POST"])
def process_stitch():
    import cv2
    images = []

    for f in request.files.getlist("images"):
        img = read_image_from_bytes(f)
        if img is not None:
            images.append(img)

    stitcher = cv2.Stitcher_create()
    status, pano = stitcher.stitch(images)

    if status != 0:
        return jsonify({"error": "Stitch failed"}), 400

    _, buf = cv2.imencode(".jpg", pano)
    return jsonify({"stitched": base64.b64encode(buf).decode("utf-8")})


@app.route("/process_sift", methods=["POST"])
def process_sift():
    import cv2
    import numpy as np

    img1 = read_image_from_bytes(request.files["img1"])
    img2 = read_image_from_bytes(request.files["img2"])

    g1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
    g2 = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)

    sift = cv2.SIFT_create()
    kp1, des1 = sift.detectAndCompute(g1, None)
    kp2, des2 = sift.detectAndCompute(g2, None)

    bf = cv2.BFMatcher()
    matches = bf.knnMatch(des1, des2, k=2)
    good = [m for m, n in matches if m.distance < 0.75 * n.distance]

    res = cv2.drawMatches(g1, kp1, g2, kp2, good, None)
    _, buf = cv2.imencode(".jpg", res)

    return jsonify({"opencv": base64.b64encode(buf).decode("utf-8")})


# ==========================================
# 8. MODULE 6 – MOTION TRACKING
# ==========================================
@app.route("/module6")
def module6():
    return render_template("module6.html")


@app.route("/stream_sam2")
def stream_sam2():
    return jsonify({"error": "SAM2 video not configured"})


# ==========================================
# 9. MODULE 7 – POSE TRACKING
# ==========================================
mp_holistic = None
mp_drawing = None
holistic_net = None
CSV_FILENAME = None


def get_model():
    global mp_holistic, mp_drawing, holistic_net
    if holistic_net is None:
        import mediapipe as mp
        mp_holistic = mp.solutions.holistic
        mp_drawing = mp.solutions.drawing_utils
        holistic_net = mp_holistic.Holistic()
    return holistic_net, mp_drawing, mp_holistic


@app.route("/module7")
def module7():
    return render_template("module7.html")


@app.route("/process_frame", methods=["POST"])
def process_frame():
    import numpy as np
    import cv2
    global CSV_FILENAME

    data = request.json
    image_data = data["image"].split(",")[1]
    binary = base64.b64decode(image_data)
    frame = cv2.imdecode(np.frombuffer(binary, np.uint8), cv2.IMREAD_COLOR)

    model, drawing, mp_ref = get_model()
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = model.process(rgb)

    if results.pose_landmarks:
        drawing.draw_landmarks(frame, results.pose_landmarks, mp_ref.POSE_CONNECTIONS)

    _, buf = cv2.imencode(".jpg", frame)
    return jsonify({"image": base64.b64encode(buf).decode("utf-8")})


# ==========================================
# 10. MAIN
# ==========================================
if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)
