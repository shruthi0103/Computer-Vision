import os
import io
import base64
import json
import cv2
import numpy as np
from PIL import Image
from flask import Flask, request, render_template, jsonify, redirect, session, Response
import cv2.aruco as aruco
import threading
import time
from queue import Queue

from scene import run_multi_template_match, reconstruct_blurred_regions
import matplotlib
matplotlib.use('Agg')
import uuid
import glob
from flask_socketio import SocketIO, emit
import mediapipe as mp
# -------------------------
# Setup
# -------------------------
UPLOAD_FOLDER = "uploads"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

app = Flask(__name__, static_folder="static", template_folder="templates")
app.config["UPLOAD_FOLDER"] = UPLOAD_FOLDER
app.config["MAX_CONTENT_LENGTH"] = 200 * 1024 * 1024
app.secret_key = "supersecretkey"

# -------------------------
# Helpers
# -------------------------
def allowed_image(filename):
    ext = filename.lower().rsplit(".", 1)[-1]
    return ext in ("jpg", "jpeg", "png", "bmp", "tiff")

def read_image_from_bytes(file_storage):
    data = file_storage.read()
    arr = np.frombuffer(data, dtype=np.uint8)
    return cv2.imdecode(arr, cv2.IMREAD_COLOR)

def image_to_dataurl(img_bgr):
    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    pil = Image.fromarray(img_rgb)
    buff = io.BytesIO()
    pil.save(buff, format="PNG")
    b64 = base64.b64encode(buff.getvalue()).decode()
    return f"data:image/png;base64,{b64}"

# -------------------------
# Root → Face Unlock
# -------------------------
@app.route("/")
def face_unlock_page():
    session.clear()
    return render_template("face_unlock.html")

# -------------------------
# Skip Login → Direct Entry
# -------------------------
@app.route("/skip_login")
def skip_login():
    session["logged_in"] = True
    return redirect("/home")

# -------------------------
# Face Recognition Login
# -------------------------
# -------------------------
# Face Recognition Login (MOCKED -> ALWAYS FAILS)
# -------------------------
@app.route("/auth_face", methods=["POST"])
def auth_face():
    # 1. Simulate the time it takes to "think"
    time.sleep(1.0)

    # 2. Return a "No Match" result
    # The frontend will see match=False and ask the user to enter manually.
    return jsonify({
        "match": False, 
        "distance": 0.85,  # High distance = no match
        "message": "Face not recognized. Please use manual entry."
    })
# -------------------------
# Home Page (requires login)
# -------------------------
@app.route("/home")
def home():
    if not session.get("logged_in"):
        return redirect("/")
    return render_template("home.html")

@app.route("/logout")
def logout():
    session.clear()
    return redirect("/")

# -------------------------
# Module 1: Camera Calibration
# -------------------------
# -------------------------
# Module 1: Camera Calibration & Object Measurement
# -------------------------

@app.route("/module1")
def module1():
    if not session.get("logged_in"):
        return redirect("/")
    return render_template("index.html")


@app.route("/calibrate", methods=["POST"])
def calibrate():
    files = request.files.getlist("files[]")
    if not files:
        return jsonify({"error": "No files uploaded"}), 400

    try:
        pattern_cols = int(request.form.get("pattern_cols", "9"))
        pattern_rows = int(request.form.get("pattern_rows", "6"))
        square_size = float(request.form.get("square_size", "1.0"))
    except Exception:
        return jsonify({"error": "Invalid pattern parameters"}), 400

    pattern_size = (pattern_cols, pattern_rows)

    # Prepare object points (0..pattern_cols-1, 0..pattern_rows-1)
    objp = np.zeros((pattern_rows * pattern_cols, 3), np.float32)
    objp[:, :2] = np.mgrid[0:pattern_cols, 0:pattern_rows].T.reshape(-1, 2) * square_size

    objpoints = []   # 3D points in real world space
    imgpoints = []   # 2D points in image plane
    messages = []
    image_shape = None

    for f in files:
        if not allowed_image(f.filename):
            messages.append({"file": f.filename, "status": "skipped"})
            continue

        img = read_image_from_bytes(f)
        if img is None:
            messages.append({"file": f.filename, "status": "read_failed"})
            continue

        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        if image_shape is None:
            image_shape = gray.shape[::-1]

        found, corners = cv2.findChessboardCorners(
            gray, pattern_size,
            cv2.CALIB_CB_ADAPTIVE_THRESH +
            cv2.CALIB_CB_NORMALIZE_IMAGE +
            cv2.CALIB_CB_FAST_CHECK
        )

        if found:
            refined = cv2.cornerSubPix(
                gray,
                corners,
                (11, 11),
                (-1, -1),
                (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 1e-6)
            )
            objpoints.append(objp)
            imgpoints.append(refined)
            messages.append({"file": f.filename, "status": "ok"})
        else:
            messages.append({"file": f.filename, "status": "no_chessboard"})

    if not objpoints:
        return jsonify({"error": "No valid chessboard detections"}), 400

    # Calibrate camera
    ret, K, dist, rvecs, tvecs = cv2.calibrateCamera(
        objpoints, imgpoints, image_shape, None, None
    )

    fx = float(K[0, 0])
    fy = float(K[1, 1])
    cx = float(K[0, 2])
    cy = float(K[1, 2])

    # NO server-side globals – we just return fx, fy, cx, cy
    return jsonify({
        "fx": fx,
        "fy": fy,
        "cx": cx,
        "cy": cy,
        "messages": messages
    })


@app.route("/measure", methods=["POST"])
def measure():
    """
    Uses 4 clicked points + fx, fy, distance_cm (always sent from client)
    to estimate real-world width & height.
    """
    if "image" not in request.files:
        return jsonify({"error": "No image uploaded"}), 400

    img = read_image_from_bytes(request.files["image"])
    if img is None:
        return jsonify({"error": "Could not read image"}), 400

    # Parse points
    try:
        pts = np.array(json.loads(request.form["points"]), dtype=np.float32)
    except Exception:
        return jsonify({"error": "Invalid points"}), 400

    if pts.shape != (4, 2):
        return jsonify({"error": "Exactly 4 points required"}), 400

    # Sort into top/bottom rows by y, then left/right by x (same algorithm as before)
    pts_sorted = pts[np.argsort(pts[:, 1])]
    top_two, bottom_two = pts_sorted[:2], pts_sorted[2:]
    top_left, top_right = sorted(top_two, key=lambda x: x[0])
    bottom_left, bottom_right = sorted(bottom_two, key=lambda x: x[0])

    pixel_width = float(np.linalg.norm(top_right - top_left))
    h_left = np.linalg.norm(bottom_left - top_left)
    h_right = np.linalg.norm(bottom_right - top_right)
    pixel_height = float((h_left + h_right) / 2.0)

    # Read fx, fy ALWAYS from client (option B)
    try:
        fx = float(request.form["fx"])
        fy = float(request.form["fy"])
        distance_cm = float(request.form["distance_cm"])
    except KeyError as e:
        return jsonify({"error": f"Missing field: {str(e)}"}), 400
    except ValueError:
        return jsonify({"error": "fx, fy, distance_cm must be numeric"}), 400

    # Perspective projection size estimate
    estimated_width_cm = (pixel_width * distance_cm) / fx
    estimated_height_cm = (pixel_height * distance_cm) / fy

    return jsonify({
        "pixel_width": pixel_width,
        "pixel_height": pixel_height,
        "estimated_width_cm": estimated_width_cm,
        "estimated_height_cm": estimated_height_cm,
        "fx": fx,
        "fy": fy
    })


# -------------------------
# Module 2 — Multi Template Matching
# -------------------------
# -------------------------
# Module 2: Template Matching + Reconstruction
# -------------------------
# -------------------------
# Module 2: Template Matching + Reconstruction
# -------------------------

@app.route("/module2")
def module2():
    if not session.get("logged_in"):
        return redirect("/")
    return render_template("module2.html")


# ========== LIST TEMPLATES ==========
@app.route("/list_templates_mod2")
def list_templates_mod2():
    template_dir = os.path.join("static", "mod2", "temps")

    if not os.path.exists(template_dir):
        return jsonify({"error": "templates folder missing"}), 400

    files = sorted([
        f for f in os.listdir(template_dir)
        if f.lower().endswith((".jpg", ".jpeg", ".png", ".bmp"))
    ])

    urls = [f"/static/mod2/temps/{f}" for f in files]

    return jsonify({"templates": urls})


# ========== MATCH ROUTE ==========
@app.route("/match", methods=["POST"])
def match():
    # 1) check uploaded scene
    if "scene" not in request.files:
        return jsonify({"error": "Scene image missing"}), 400

    scene_file = request.files["scene"]
    scene = read_image_from_bytes(scene_file)

    if scene is None:
        return jsonify({"error": "Failed to read scene image"}), 400

    # 2) load templates from dataset
    template_dir = os.path.join("static", "mod2", "temps")

    if not os.path.exists(template_dir):
        return jsonify({"error": "templates folder missing"}), 400

    templates = []
    for fname in sorted(os.listdir(template_dir)):
        if fname.lower().endswith((".jpg", ".jpeg", ".png", ".bmp")):
            img = cv2.imread(os.path.join(template_dir, fname))
            if img is not None:
                templates.append(img)

    if not templates:
        return jsonify({"error": "No templates available"}), 400

    # 3) run template matching
    try:
        result, boxes = run_multi_template_match(scene, templates)

        success, buffer = cv2.imencode(".png", result)
        if not success:
            return jsonify({"error": "Encoding failed"}), 500

        img_b64 = base64.b64encode(buffer).decode("ascii")

        return jsonify({
            "image": img_b64,
            "boxes": boxes
        })

    except Exception as e:
        return jsonify({"error": str(e)}), 500


# ========== RECONSTRUCTION ROUTE ==========
@app.route("/reconstruct", methods=["POST"])
def reconstruct_route():
    if "image" not in request.files:
        return jsonify({"error": "No image provided"}), 400

    img_file = request.files["image"]
    img = read_image_from_bytes(img_file)

    if img is None:
        return jsonify({"error": "Invalid image"}), 400

    try:
        boxes = json.loads(request.form.get("boxes", "[]"))
    except:
        return jsonify({"error": "Invalid boxes list"}), 400

    # call original reconstruction logic
    result = reconstruct_blurred_regions(img, boxes)

    ok, buf = cv2.imencode(".png", result)
    if not ok:
        return jsonify({"error": "Encoding failed"}), 500

    b64 = base64.b64encode(buf).decode("ascii")
    return jsonify({"image": b64})
# -------------------------
# Module 3: Image Processing & Segmentation (upload-based)
# -------------------------

@app.route("/module3")
def module3():
    if not session.get("logged_in"):
        return redirect("/")
    return render_template("module3.html")


@app.route("/process_gradient_log", methods=["POST"])
def process_gradient_log():
    """
    Accepts multiple uploaded images (images[]), computes:
    - Original
    - Gradient magnitude
    - Gradient angle
    - LoG edge map
    and returns a single collage image.
    """
    import base64

    files = request.files.getlist("images[]")
    if not files:
        return jsonify({"error": "No images uploaded"}), 400

    target_width = 300  # width of each small image
    processed_images = []

    for f in files:
        if not allowed_image(f.filename):
            continue

        img = read_image_from_bytes(f)
        if img is None:
            continue

        h, w = img.shape[:2]
        scale = target_width / w
        original_resized = cv2.resize(img, (target_width, int(h * scale)))

        # Grayscale for processing
        img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        img_f = img_gray.astype(np.float32) / 255.0

        # ---- Gradient ----
        gx = cv2.Sobel(img_f, cv2.CV_32F, 1, 0, ksize=3)
        gy = cv2.Sobel(img_f, cv2.CV_32F, 0, 1, ksize=3)
        grad_mag = cv2.magnitude(gx, gy)
        grad_angle = cv2.phase(gx, gy, angleInDegrees=True)
        grad_mag_disp = cv2.normalize(grad_mag, None, 0, 1, cv2.NORM_MINMAX)
        grad_angle_disp = grad_angle / 360.0

        # Gradient magnitude → BGR
        grad_mag_uint8 = (grad_mag_disp * 255).astype(np.uint8)
        grad_mag_resized = cv2.resize(grad_mag_uint8, (target_width, int(h * scale)))
        grad_mag_bgr = cv2.cvtColor(grad_mag_resized, cv2.COLOR_GRAY2BGR)

        # Gradient angle → HSV colormap
        grad_angle_uint8 = (grad_angle_disp * 255).astype(np.uint8)
        grad_angle_color = cv2.applyColorMap(grad_angle_uint8, cv2.COLORMAP_HSV)
        grad_angle_resized = cv2.resize(grad_angle_color, (target_width, int(h * scale)))

        # ---- LoG Edge Map ----
        blur = cv2.GaussianBlur(img_f, (5, 5), sigmaX=1)
        log = cv2.Laplacian(blur, cv2.CV_32F, ksize=5)
        log_abs = np.abs(log)
        log_norm = cv2.normalize(log_abs, None, 0, 255, cv2.NORM_MINMAX)
        log_uint8 = log_norm.astype(np.uint8)
        _, log_edges = cv2.threshold(log_uint8, 30, 255, cv2.THRESH_BINARY)
        log_resized = cv2.resize(log_edges, (target_width, int(h * scale)))
        log_bgr = cv2.cvtColor(log_resized, cv2.COLOR_GRAY2BGR)

        # Stack original + outputs horizontally
        combined_row = np.hstack([original_resized, grad_mag_bgr, grad_angle_resized, log_bgr])
        processed_images.append(combined_row)

    if not processed_images:
        return jsonify({"error": "No valid images processed"}), 400

    # ---- Header row ----
    num_cols = 4
    header_height = 40
    header_img = np.full((header_height, target_width * num_cols, 3), 255, dtype=np.uint8)
    labels = ["Original", "Grad Mag", "Grad Angle", "LoG"]
    positions = [
        (target_width // 2 - 50, 28),
        (target_width + target_width // 2 - 50, 28),
        (2 * target_width + target_width // 2 - 50, 28),
        (3 * target_width + target_width // 2 - 50, 28)
    ]
    for pos, text in zip(positions, labels):
        cv2.putText(header_img, text, pos, cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 2, cv2.LINE_AA)

    final_output = np.vstack([header_img] + processed_images)

    ok, buf = cv2.imencode(".png", final_output)
    if not ok:
        return jsonify({"error": "Encoding failed"}), 500

    return jsonify({"image": base64.b64encode(buf).decode("ascii")})


@app.route("/process_edge_corner", methods=["POST"])
def process_edge_corner():
    """
    Accepts multiple uploaded images (images[]),
    returns per-image edge map + corner overlay.
    """
    import base64

    files = request.files.getlist("images[]")
    if not files:
        return jsonify({"error": "No images uploaded"}), 400

    output = []
    target_width = 260  # resize width for display

    for f in files:
        if not allowed_image(f.filename):
            continue

        img = read_image_from_bytes(f)
        if img is None:
            continue

        h, w = img.shape[:2]
        scale = target_width / w

        # ===== Edge Map =====
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY).astype(np.float32)

        Ix = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
        Iy = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
        grad_mag = np.sqrt(Ix**2 + Iy**2)

        edge_thresh = np.percentile(grad_mag, 90)
        edge_map = (grad_mag > edge_thresh).astype(np.uint8) * 255

        edge_map = cv2.morphologyEx(edge_map, cv2.MORPH_OPEN, np.ones((3,3), np.uint8))
        edge_resized = cv2.resize(edge_map, (target_width, int(h * scale)))

        # ===== Corner Map (Harris + NMS) =====
        gray_blur = cv2.GaussianBlur(gray, (7,7), 1)

        Ix = cv2.Sobel(gray_blur, cv2.CV_64F, 1, 0, 3)
        Iy = cv2.Sobel(gray_blur, cv2.CV_64F, 0, 1, 3)

        Ix2 = cv2.GaussianBlur(Ix*Ix, (5,5), 1)
        Iy2 = cv2.GaussianBlur(Iy*Iy, (5,5), 1)
        Ixy = cv2.GaussianBlur(Ix*Iy, (5,5), 1)

        k = 0.04
        R = (Ix2 * Iy2 - Ixy**2) - k * (Ix2 + Iy2)**2

        corner_thresh = np.percentile(R, 99.8)
        raw_corners = (R > corner_thresh)

        dilated = cv2.dilate(R, None)
        nms_mask = (R == dilated) & raw_corners

        corner_img = img.copy()
        ys, xs = np.where(nms_mask)
        for x, y in zip(xs, ys):
            cv2.circle(corner_img, (x, y), 5, (0, 0, 255), -1)

        corner_resized = cv2.resize(corner_img, (target_width, int(h * scale)))

        ok1, buf1 = cv2.imencode(".png", edge_resized)
        ok2, buf2 = cv2.imencode(".png", corner_resized)
        if ok1 and ok2:
            output.append({
                "filename": f.filename,
                "edges": base64.b64encode(buf1).decode("ascii"),
                "corners": base64.b64encode(buf2).decode("ascii")
            })

    if not output:
        return jsonify({"error": "No valid images processed"}), 400

    return jsonify({"images": output})


@app.route("/process_aruco", methods=["POST"])
def process_aruco():
    """
    Accepts multiple uploaded images (images[]),
    detects ArUco markers and draws hull boundaries.
    Returns list of combined before/after images.
    """
    import base64

    files = request.files.getlist("images[]")
    if not files:
        return jsonify({"error": "No images uploaded"}), 400

    aruco_dict = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_50)
    params = cv2.aruco.DetectorParameters()
    detector = cv2.aruco.ArucoDetector(aruco_dict, params)

    output_images = []
    target_width = 260

    for f in files:
        if not allowed_image(f.filename):
            continue

        img = read_image_from_bytes(f)
        if img is None:
            continue

        h, w = img.shape[:2]
        scale = target_width / w
        img_resized = cv2.resize(img, (target_width, int(h * scale)))

        vis = img.copy()
        corners, ids, _ = detector.detectMarkers(img)

        if ids is not None and len(ids) > 0:
            cv2.aruco.drawDetectedMarkers(vis, corners, ids)

            all_pts = []
            for c in corners:
                pts = c[0].astype(int)
                all_pts.extend(pts)
            all_pts = np.array(all_pts)

            hull = cv2.convexHull(all_pts)

            cv2.polylines(vis, [hull], True, (0, 0, 255), 8)
            for (x, y) in hull.reshape(-1, 2):
                cv2.circle(vis, (x, y), 6, (0, 0, 255), -1)

        vis_resized = cv2.resize(vis, (target_width, int(h * scale)))
        combined = np.hstack([img_resized, vis_resized])

        ok, buf = cv2.imencode(".png", combined)
        if ok:
            output_images.append(base64.b64encode(buf).decode("ascii"))

    if not output_images:
        return jsonify({"error": "No valid images processed"}), 400

    return jsonify({"images": output_images})


@app.route("/process_boundary", methods=["POST"])
def process_boundary():
    """
    Accepts multiple uploaded images (images[]),
    applies improved color + edge–based segmentation and draws continuous object boundaries.
    Returns a single mosaic image (2 per row).
    """
    import base64

    files = request.files.getlist("images[]")
    if not files:
        return jsonify({"error": "No images uploaded"}), 400

    results = []

    for f in files:
        if not allowed_image(f.filename):
            continue

        img = read_image_from_bytes(f)
        if img is None:
            continue

        original = img.copy()

        # Convert to HSV for color segmentation
        hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

        # --- TUNED COLOR RANGE ---
        # Wider range catches **all shades** of your teal/pink box edges
        lower = np.array([70, 30, 30])
        upper = np.array([110, 255, 255])
        mask_color = cv2.inRange(hsv, lower, upper)

        # --- EDGE MAP (Canny) ---
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        edges = cv2.Canny(gray, 60, 150)

        # Combine color mask + edges
        combined = cv2.bitwise_or(mask_color, edges)

        # --- Strong morphological cleaning ---
        kernel = np.ones((5, 5), np.uint8)
        combined = cv2.morphologyEx(combined, cv2.MORPH_CLOSE, kernel, iterations=3)
        combined = cv2.morphologyEx(combined, cv2.MORPH_OPEN, kernel, iterations=1)

        # Find contours
        contours, _ = cv2.findContours(combined, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        if contours:
            # Pick LARGEST external contour = the object
            c = max(contours, key=cv2.contourArea)

            # --- Smooth boundary for clarity ---
            epsilon = 0.01 * cv2.arcLength(c, True)
            approx = cv2.approxPolyDP(c, epsilon, True)

            # Draw bold, clear boundary
            cv2.polylines(original, [approx], True, (0, 0, 255), 5, cv2.LINE_AA)

        results.append(original)

    if not results:
        return jsonify({"error": "No valid images processed"}), 400

    # ---- MOSAIC (unchanged) ----
    target_width = 350
    resized = [
        cv2.resize(r, (target_width, int(r.shape[0] * target_width / r.shape[1])))
        for r in results
    ]

    max_h = max(r.shape[0] for r in resized)
    normalized = []
    for r in resized:
        h, w = r.shape[:2]
        if h < max_h:
            pad = np.full((max_h - h, w, 3), 255, dtype=np.uint8)
            r = np.vstack([r, pad])
        normalized.append(r)

    rows = []
    for i in range(0, len(normalized), 2):
        if i + 1 < len(normalized):
            row = np.hstack([normalized[i], normalized[i + 1]])
        else:
            blank = np.full_like(normalized[i], 255)
            row = np.hstack([normalized[i], blank])
        rows.append(row)

    final_output = np.vstack(rows)

    ok, buf = cv2.imencode(".png", final_output)
    if not ok:
        return jsonify({"error": "Encoding failed"}), 500

    return jsonify({"image": base64.b64encode(buf).decode("ascii")})

# ===============================
# Module 3 – SAM2 Offline Results
# ===============================
@app.route("/get_sam2_results", methods=["GET"])
def get_sam2_results():
    folder = os.path.join("static", "marker")

    if not os.path.exists(folder):
        return jsonify({"error": "static/marker folder not found"}), 400

    files = sorted([
        f for f in os.listdir(folder)
        if f.lower().endswith((".jpg", ".jpeg", ".png"))
    ])

    if not files:
        return jsonify({"error": "No SAM2 images found"}), 400

    urls = [f"/static/marker/{f}" for f in files]
    return jsonify({"images": urls})


# ==============================
# IMAGE STITCHING — all images from uploads folder
# ==============================
@app.route("/module4")
def module4():
    if not session.get("logged_in"):
        return redirect("/")
    return render_template("module4.html")


@app.route("/process_stitch")
def process_stitch():
    pano_folder = os.path.join(app.config["UPLOAD_FOLDER"], "panorama")
    print("Looking for images in:", pano_folder)

    # Collect all images
    image_files = sorted([
        os.path.join(pano_folder, f)
        for f in os.listdir(pano_folder)
        if f.lower().endswith((".jpg", ".jpeg", ".png"))
    ])

    print("Found images:", image_files)

    if len(image_files) < 2:
        return jsonify({"error": "Need at least 2 images for stitching"}), 400

    images = []
    for p in image_files:
        img = cv2.imread(p)
        if img is not None:
            images.append(img)

    if len(images) < 2:
        return jsonify({"error": "Failed to load images"}), 400

    # -------------------------------------------------------------------
    # Create stitcher
    # -------------------------------------------------------------------
    try:
        stitcher = cv2.Stitcher_create(cv2.Stitcher_PANORAMA)
    except:
        stitcher = cv2.createStitcher(cv2.Stitcher_PANORAMA)

    stitcher.setPanoConfidenceThresh(0.5)
    stitcher.setWaveCorrection(True)

    status, panorama = stitcher.stitch(images)
    print("STITCH STATUS:", status)

    if status != cv2.Stitcher_OK:
        return jsonify({"error": f"Stitching failed with code {status}"}), 400

    # -------------------------------------------------------------------
    # Auto crop black borders
    # -------------------------------------------------------------------
    gray = cv2.cvtColor(panorama, cv2.COLOR_BGR2GRAY)
    _, thresh = cv2.threshold(gray, 1, 255, cv2.THRESH_BINARY)
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    x, y, w, h = cv2.boundingRect(contours[0])
    cropped = panorama[y:y+h, x:x+w]

    # -------------------------------------------------------------------
    # Encode cropped stitched image
    # -------------------------------------------------------------------
    ok, buf = cv2.imencode(".jpg", cropped)
    if not ok:
        return jsonify({"error": "Failed to encode panorama image"}), 500

    stitched_b64 = base64.b64encode(buf).decode("utf-8")
    print("Encoded panorama size (bytes):", len(buf))

    # -------------------------------------------------------------------
    # Generate small thumbnails for each input image
    # -------------------------------------------------------------------
    thumbs_b64 = []
    for img in images:
        thumb = cv2.resize(img, (200, 120))
        ok2, buf2 = cv2.imencode(".jpg", thumb)
        if ok2:
            thumbs_b64.append(base64.b64encode(buf2).decode("utf-8"))

    # -------------------------------------------------------------------
    # Load phone panorama (optional)
    # -------------------------------------------------------------------
    phone_pano_folder = os.path.join(app.config["UPLOAD_FOLDER"], "phone_pano")
    phone_pano_b64 = None

    if os.path.isdir(phone_pano_folder):
        for f in os.listdir(phone_pano_folder):
            if f.lower().endswith((".jpg", ".jpeg", ".png")):
                ph = cv2.imread(os.path.join(phone_pano_folder, f))
                if ph is not None:
                    ok3, buf3 = cv2.imencode(".jpg", ph)
                    if ok3:
                        phone_pano_b64 = base64.b64encode(buf3).decode("utf-8")
                break

    # -------------------------------------------------------------------
    # Return to frontend
    # -------------------------------------------------------------------
    return jsonify({
        "stitched": stitched_b64,
        "width": cropped.shape[1],
        "height": cropped.shape[0],
        "thumbnails": thumbs_b64,
        "phone_pano": phone_pano_b64
    })



# ---------------------------------------------------
# SIFT + RANSAC COMPARISON
@app.route("/process_sift")
def process_sift():
    folder = os.path.join(app.config['UPLOAD_FOLDER'], "sift_images")
    img1_path = os.path.join(folder, "1.jpeg")
    img2_path = os.path.join(folder, "5.jpeg")

    img1 = cv2.imread(img1_path, cv2.IMREAD_GRAYSCALE)
    img2 = cv2.imread(img2_path, cv2.IMREAD_GRAYSCALE)
    if img1 is None or img2 is None:
        return jsonify({"error": "Images not found in uploads/sift_images"})

    img1 = cv2.resize(img1, (600, 400))
    img2 = cv2.resize(img2, (600, 400))

    # ---- Custom SIFT ----
    def custom_sift_keypoints(image, num_scales=4, sigma=1.6):
        gaussians = [image.astype(np.float32)]
        for i in range(1, num_scales):
            sigma_i = sigma * (2 ** i)
            blur = cv2.GaussianBlur(image, (0, 0), sigmaX=sigma_i)
            gaussians.append(blur.astype(np.float32))
        dogs = [gaussians[i+1] - gaussians[i] for i in range(num_scales - 1)]
        keypoints = []
        for i in range(1, num_scales - 2):
            prev, curr, next_ = dogs[i-1], dogs[i], dogs[i+1]
            for y in range(1, curr.shape[0]-1):
                for x in range(1, curr.shape[1]-1):
                    patch = np.concatenate([
                        prev[y-1:y+2, x-1:x+2].flatten(),
                        curr[y-1:y+2, x-1:x+2].flatten(),
                        next_[y-1:y+2, x-1:x+2].flatten()
                    ])
                    val = curr[y, x]
                    if val == np.max(patch) or val == np.min(patch):
                        keypoints.append(cv2.KeyPoint(x, y, 3))
        return keypoints

    kp1_c = custom_sift_keypoints(img1)
    kp2_c = custom_sift_keypoints(img2)
    orb = cv2.ORB_create()
    kp1_c, des1_c = orb.compute(img1, kp1_c)
    kp2_c, des2_c = orb.compute(img2, kp2_c)

    # ---- RANSAC on custom SIFT ----
    bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
    matches_c = bf.match(des1_c, des2_c)
    matches_c = sorted(matches_c, key=lambda x: x.distance)

    src_pts = np.float32([kp1_c[m.queryIdx].pt for m in matches_c]).reshape(-1,1,2)
    dst_pts = np.float32([kp2_c[m.trainIdx].pt for m in matches_c]).reshape(-1,1,2)
    H, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)
    matches_mask = mask.ravel().tolist() if mask is not None else None

    custom_result = cv2.drawMatches(
        img1, kp1_c, img2, kp2_c,
        [m for i, m in enumerate(matches_c) if matches_mask is None or matches_mask[i]],
        None, matchColor=(0,255,0), flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS
    )

    # ---- OpenCV SIFT ----
    sift = cv2.SIFT_create()
    kp1, des1 = sift.detectAndCompute(img1, None)
    kp2, des2 = sift.detectAndCompute(img2, None)
    bf2 = cv2.BFMatcher()
    matches = bf2.knnMatch(des1, des2, k=2)
    good = [m for m,n in matches if m.distance < 0.75*n.distance]

    src_pts2 = np.float32([kp1[m.queryIdx].pt for m in good]).reshape(-1,1,2)
    dst_pts2 = np.float32([kp2[m.trainIdx].pt for m in good]).reshape(-1,1,2)
    H2, mask2 = cv2.findHomography(src_pts2, dst_pts2, cv2.RANSAC, 5.0)
    matches_mask2 = mask2.ravel().tolist() if mask2 is not None else None

    sift_result = cv2.drawMatches(
        img1, kp1, img2, kp2,
        [m for i,m in enumerate(good) if matches_mask2 is None or matches_mask2[i]],
        None, matchColor=(255,0,0), flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS
    )

    _, buf1 = cv2.imencode('.jpg', custom_result)
    custom_b64 = base64.b64encode(buf1).decode('utf-8')

    _, buf2 = cv2.imencode('.jpg', sift_result)
    opencv_b64 = base64.b64encode(buf2).decode('utf-8')

    return jsonify({
        "custom": custom_b64,
        "opencv": opencv_b64
    })







# ------------------- Helper Functions -------------------

def load_or_generate_initial_mask(frame):
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    lower_range = np.array([15, 80, 80])
    upper_range = np.array([35, 255, 255])
    mask = cv2.inRange(hsv, lower_range, upper_range)
    kernel = np.ones((7, 7), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=2)
    mask = cv2.dilate(mask, kernel, iterations=1)
    return mask

def get_bbox_from_mask(initial_mask):
    contours, _ = cv2.findContours(initial_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return None
    largest_contour = max(contours, key=cv2.contourArea)
    return cv2.boundingRect(largest_contour)

def checkpoint(msg):
    timestamp = time.strftime('%H:%M:%S')
    return f"[{timestamp}] {msg}"
def fix_orientation(frame):
    """Rotate portrait videos to landscape automatically."""
    h, w = frame.shape[:2]
    if h > w:  # iPhone portrait video
        frame = cv2.rotate(frame, cv2.ROTATE_90_CLOCKWISE)
    return frame

# ------------------- Markerless Tracker -------------------

def run_csrt_with_mask_init_stream(video_path):
    SCALE = 0.4  # speed boost

    cap = cv2.VideoCapture(video_path)
    yield checkpoint("Starting Markerless Tracker") + "\n"

    ret, frame = cap.read()
    if not ret:
        yield checkpoint(f"Failed to read video: {video_path}") + "\n"
        return
    frame = cv2.rotate(frame, cv2.ROTATE_90_CLOCKWISE)

    yield checkpoint("First frame read") + "\n"

    # Fast mask (no grabcut)
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    mask = cv2.inRange(hsv, np.array([15,80,80]), np.array([35,255,255]))

    contours,_ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        yield checkpoint("No valid object found") + "\n"
        return

    x,y,w,h = cv2.boundingRect(max(contours, key=cv2.contourArea))
    yield checkpoint(f"Bounding box: {(x,y,w,h)}") + "\n"

    small = cv2.resize(frame, None, fx=SCALE, fy=SCALE)
    small_bbox = (int(x*SCALE), int(y*SCALE), int(w*SCALE), int(h*SCALE))

    tracker = cv2.legacy.TrackerCSRT_create()
    tracker.init(small, small_bbox)

    yield checkpoint("Tracker initialized") + "\n"

    frame_count = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            yield checkpoint("End of video") + "\n"
            break

        frame = cv2.rotate(frame, cv2.ROTATE_90_CLOCKWISE)
        small = cv2.resize(frame, None, fx=SCALE, fy=SCALE)

        success, box = tracker.update(small)
        frame_count += 1

        if frame_count % 30 == 0:
            yield checkpoint(f"Processed {frame_count} frames") + "\n"

        if success:
            sx, sy, sw, sh = [int(v/SCALE) for v in box]
            cv2.rectangle(frame, (sx, sy), (sx+sw, sy+sh), (0,255,0), 2)

        _, buf = cv2.imencode(".jpg", frame)
        yield (b"--frame\r\nContent-Type: image/jpeg\r\n\r\n" + buf.tobytes() + b"\r\n")

    cap.release()
    yield checkpoint("Tracking finished.") + "\n"


# ------------------- ArUco Tracker -------------------
def run_aruco_tracker_stream(video_path):
    import cv2.aruco as aruco
    MARKER_LENGTH = 0.03
    aruco_dict = aruco.getPredefinedDictionary(aruco.DICT_4X4_1000)
    parameters = cv2.aruco.DetectorParameters()
    detector = aruco.ArucoDetector(aruco_dict, parameters)

    cap = cv2.VideoCapture(video_path)
    yield checkpoint("Starting ArUco Tracker") + "\n"

    ret, frame = cap.read()
    if not ret:
        yield checkpoint(f"Failed to read video: {video_path}") + "\n"
        return

    frame = cv2.rotate(frame, cv2.ROTATE_90_CLOCKWISE)
   # 🔥 FIX ORIENTATION

    yield checkpoint("First frame read") + "\n"

    frame_count = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            yield checkpoint("End of video reached") + "\n"
            break

        frame = cv2.rotate(frame, cv2.ROTATE_90_CLOCKWISE)
   # 🔥 ALWAYS FIX

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        corners, ids, _ = detector.detectMarkers(gray)

        if ids is not None and len(ids) > 0:
            corner_points = corners[0][0]
            x_min, y_min = int(np.min(corner_points[:,0])), int(np.min(corner_points[:,1]))
            x_max, y_max = int(np.max(corner_points[:,0])), int(np.max(corner_points[:,1]))

            center_x = (x_min + x_max) // 2
            center_y = (y_min + y_max) // 2

            expansion_factor_w = 3.5
            expansion_factor_h = 3.5
            final_w = (x_max - x_min) * expansion_factor_w
            final_h = (y_max - y_min) * expansion_factor_h
            final_x_min = center_x - int(final_w // 2)
            final_y_min = center_y - int(final_h // 2)

            cv2.rectangle(frame, (int(final_x_min), int(final_y_min)),
                          (int(final_x_min + final_w), int(final_y_min + final_h)),
                          (0, 255, 0), 2)

            cv2.circle(frame, (center_x, center_y), 5, (255, 0, 0), -1)
            cv2.putText(frame, "ArUco Tracker", (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,255,0),2)
        else:
            cv2.putText(frame, "No Marker Detected", (10,30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,0,255),2)

        frame_count += 1
        if frame_count % 30 == 0:
            yield checkpoint(f"Processed {frame_count} frames") + "\n"

        _, buffer = cv2.imencode('.jpg', frame)
        frame_bytes = buffer.tobytes()
        yield (b'--frame\r\nContent-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')

    cap.release()
    yield checkpoint(f"Tracking finished. Total frames: {frame_count}") + "\n"

# ------------------- Flask Routes -------------------

from flask import stream_with_context

@app.route("/video_feed_markerless")
def video_feed_markerless():
    return Response(run_csrt_with_mask_init_stream("static/motion.mp4"),
                    mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route("/checkpoints_markerless")
def checkpoints_markerless():
    def generate_logs():
        for line in run_csrt_with_mask_init_stream("static/motion.mp4"):
            if isinstance(line, str):
                yield f"data: {line}\n\n"
    return Response(generate_logs(), mimetype='text/event-stream')

@app.route("/video_feed_aruco")
def video_feed_aruco():
    return Response(run_aruco_tracker_stream("static/aruco_video.mp4"),
                    mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route("/checkpoints_aruco")
def checkpoints_aruco():
    def generate_logs():
        for line in run_aruco_tracker_stream("static/aruco_video.mp4"):
            if isinstance(line, str):
                yield f"data: {line}\n\n"
    return Response(generate_logs(), mimetype='text/event-stream')

MASK_PATH = "static/masks.npz"
VIDEO_PATH = "static/output_tracking.mp4"
FRAME_ALPHA = 0.5

def load_masks(npz_path):
    data = np.load(npz_path)
    if "masks" not in data:
        raise ValueError("NPZ must contain 'masks' array")
    print(f"Loaded {len(data['masks'])} SAM2 masks")
    return data["masks"]

sam2_masks = load_masks(MASK_PATH)   # load once
def sam2_generator(video_path, masks):
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"Cannot open video: {video_path}")
        return

    frame_idx = 0
    total_masks = len(masks)
    start_time = time.time()

    while True:
        ret, frame = cap.read()
        if not ret:
            print("End of video reached")
            break

        # Select mask for this frame
        mask = masks[frame_idx % total_masks]

        # Convert mask → uint8 → resize → BGR
        mask_uint8 = (mask * 255).astype(np.uint8)
        mask_resized = cv2.resize(mask_uint8, (frame.shape[1], frame.shape[0]))
        colored_mask = cv2.cvtColor(mask_resized, cv2.COLOR_GRAY2BGR)

        # Overlay
        overlay = cv2.addWeighted(frame, 1.0, colored_mask, FRAME_ALPHA, 0)

        # Encode for HTTP streaming
        ret, jpeg = cv2.imencode(".jpg", overlay)
        frame = jpeg.tobytes()

        yield (b"--frame\r\n"
               b"Content-Type: image/jpeg\r\n\r\n" + frame + b"\r\n")

        frame_idx += 1

    cap.release()
@app.route("/stream_sam2")
def stream_sam2():
    def generate():
        cap = cv2.VideoCapture(VIDEO_PATH)

        if not cap.isOpened():
            print("Failed to open SAM2 video")
            return

        frame_idx = 0

        while True:
            ret, frame = cap.read()
            if not ret:
                break  # End of video → stop streaming so browser can restart next time

            mask = sam2_masks[frame_idx % len(sam2_masks)]
            mask_uint8 = (mask * 255).astype(np.uint8)
            mask_resized = cv2.resize(mask_uint8, (frame.shape[1], frame.shape[0]))
            colored_mask = cv2.cvtColor(mask_resized, cv2.COLOR_GRAY2BGR)

            overlay = cv2.addWeighted(frame, 1.0, colored_mask, FRAME_ALPHA, 0)

            ret_jpg, buffer = cv2.imencode('.jpg', overlay)
            frame_bytes = buffer.tobytes()

            yield (b"--frame\r\n"
                   b"Content-Type: image/jpeg\r\n\r\n" + frame_bytes + b"\r\n")

            frame_idx += 1

        cap.release()

    return Response(generate(),
                    mimetype="multipart/x-mixed-replace; boundary=frame")


@app.route("/module6")
def module6():
    return render_template("module6.html")
import cv2
import mediapipe as mp
import csv
import time
from flask import Flask, render_template, Response, send_file

# -------------------------
# Module 7: WEBSOCKET POSE TRACKING (REPLACES OLD STREAM)
# -------------------------
import csv
import time

# --- REPLACEMENT CODE (Lazy Loading) ---

# Global variables (initially None)
mp_holistic = None
mp_drawing = None
holistic_net = None

def get_model():
    """Load the model only when needed to save startup RAM."""
    global mp_holistic, mp_drawing, holistic_net
    
    if holistic_net is None:
        import mediapipe as mp
        mp_holistic = mp.solutions.holistic
        mp_drawing = mp.solutions.drawing_utils
        holistic_net = mp_holistic.Holistic(
            static_image_mode=False,
            model_complexity=0,  # CHANGED to 0 (Lite model) to save RAM!
            smooth_landmarks=True,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5
        )
        print("MediaPipe Model Loaded!")
    
    return holistic_net, mp_drawing, mp_holistic

# Initialize the model globally
holistic_net = mp_holistic.Holistic(
    static_image_mode=False,
    model_complexity=1,
    smooth_landmarks=True,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5
)

# Global CSV filename for the current session
CSV_FILENAME = None
csv_file = None
csv_writer = None

def init_csv():
    global CSV_FILENAME, csv_file, csv_writer
    timestamp = int(time.time())
    CSV_FILENAME = os.path.join(app.config["UPLOAD_FOLDER"], f'pose_hands_{timestamp}.csv')
    
    pose_landmarks_names = [
        "nose","left_eye_inner","left_eye","left_eye_outer","right_eye_inner","right_eye","right_eye_outer",
        "left_ear","right_ear","mouth_left","mouth_right","left_shoulder","right_shoulder","left_elbow","right_elbow",
        "left_wrist","right_wrist","left_pinky","right_pinky","left_index","right_index","left_thumb","right_thumb",
        "left_hip","right_hip","left_knee","right_knee","left_ankle","right_ankle","left_heel","right_heel",
        "left_foot_index","right_foot_index","pelvis"
    ]
    hand_landmark_count = 21
    header = ['timestamp_ms','frame_index','pose_present','left_hand_present','right_hand_present']
    for lm_name in pose_landmarks_names:
        header += [f'{lm_name}_x', f'{lm_name}_y', f'{lm_name}_z', f'{lm_name}_vis']
    for side in ['left','right']:
        for i in range(hand_landmark_count):
            header += [f'{side}_w{i}_x', f'{side}_w{i}_y', f'{side}_w{i}_z']

    csv_file = open(CSV_FILENAME, mode='w', newline='')
    csv_writer = csv.writer(csv_file)
    csv_writer.writerow(header)

@app.route("/module7")
def module7():
    # Initialize a new CSV whenever the page is loaded/refreshed
    init_csv()
    return render_template("module7.html")

@app.route("/download_csv_module7")
def download_csv_module7():
    global CSV_FILENAME, csv_file
    if CSV_FILENAME is None or not os.path.exists(CSV_FILENAME):
        return "CSV not ready or file missing", 404
    
    # Flush file to ensure all data is written before download
    if csv_file:
        csv_file.flush()
        
    return send_file(CSV_FILENAME, as_attachment=True)

# --- SOCKET IO EVENT LISTENER ---
@socketio.on('image')
def handle_image(data_image):
    global csv_writer
    
    # 1. Decode image (Same as before)
    try:
        header, encoded = data_image.split(",", 1)
        data = base64.b64decode(encoded)
        np_arr = np.frombuffer(data, np.uint8)
        frame = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
        if frame is None: return
    except Exception:
        return

    # 2. GET THE MODEL (Lazy Load)
    model, drawing, mp_ref = get_model()

    # 3. Process
    image_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = model.process(image_rgb)
    
    # 4. Draw Landmarks (Use the returned variables)
    if results.pose_landmarks:
        drawing.draw_landmarks(frame, results.pose_landmarks, mp_ref.POSE_CONNECTIONS)
    if results.left_hand_landmarks:
        drawing.draw_landmarks(frame, results.left_hand_landmarks, mp_ref.HAND_CONNECTIONS)
    if results.right_hand_landmarks:
        drawing.draw_landmarks(frame, results.right_hand_landmarks, mp_ref.HAND_CONNECTIONS)

    # ... (Rest of the CSV logging and emit code remains the same) ...
    # 5. Encode and Emit
    _, buffer = cv2.imencode('.jpg', frame)
    frame_encoded = base64.b64encode(buffer).decode('utf-8')
    emit('response_back', f"data:image/jpeg;base64,{frame_encoded}")


@app.route('/module7_calc', methods=['POST'])
def module7_calc():
    # This is the calculation endpoint (Stereo Vision logic)
    # Renamed from /module7 POST to avoid conflict with the GET route name
    import numpy as np
    import cv2, json

    try:
        # --- CAMERA PARAMETERS from form ---
        fx = float(request.form['fx'])
        fy = float(request.form['fy'])
        cx = float(request.form['cx'])
        cy = float(request.form['cy'])
        baseline_cm = float(request.form['baseline'])  # Keep in cm

        # --- Points from frontend ---
        left_pts = json.loads(request.form['left_points'])
        right_pts = json.loads(request.form['right_points'])

        if len(left_pts) != 4 or len(right_pts) != 4:
            return "Error: Need exactly 4 points on each image"

        left = np.array(left_pts, dtype=np.float32)
        right = np.array(right_pts, dtype=np.float32)

        # --- Compute disparities ---
        disparities = left[:, 0] - right[:, 0]
        # Avoid divide-by-zero
        disparities[disparities == 0] = 0.0001

        # --- Compute 3D coordinates in cm ---
        Z_cm = fx * baseline_cm / disparities
        X_cm = (left[:, 0] - cx) * Z_cm / fx
        Y_cm = (left[:, 1] - cy) * Z_cm / fy
        pts3d_cm = np.column_stack([X_cm, Y_cm, Z_cm])

        # --- Fix point order: TL, TR, BR, BL ---
        pts3d_cm = np.array([pts3d_cm[0], pts3d_cm[1], pts3d_cm[3], pts3d_cm[2]])

        TL, TR, BR, BL = pts3d_cm

        # --- Average width and height (like standalone code) ---
        width_cm = (np.linalg.norm(TR - TL) + np.linalg.norm(BR - BL)) / 2
        height_cm = (np.linalg.norm(BL - TL) + np.linalg.norm(BR - TR)) / 2

        # --- Build readable output ---
        out = []
        for i, p in enumerate(pts3d_cm):
            out.append(f"Point {i+1}: X={p[0]:.2f} cm, Y={p[1]:.2f} cm, Z={p[2]:.2f} cm")

        out.append(f"\nWidth:  {width_cm:.2f} cm")
        out.append(f"Height: {height_cm:.2f} cm")
        out.append(f"Depths (Z): {Z_cm.round(2).tolist()}")

        return "\n".join(out)

    except Exception as e:
        return f"Error: {str(e)}"



import os
if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)




