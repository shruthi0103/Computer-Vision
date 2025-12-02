import os
import base64
import json
import time
import io
import gc  # Added explicit import for garbage collection
from flask import Flask, request, render_template, jsonify, redirect, session, send_file, Response
from flask_cors import CORS # Ensure flask-cors is in requirements.txt

# -------------------------
# Setup
# -------------------------
UPLOAD_FOLDER = "uploads"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(os.path.join(UPLOAD_FOLDER, "panorama"), exist_ok=True)
os.makedirs(os.path.join(UPLOAD_FOLDER, "phone_pano"), exist_ok=True)
os.makedirs(os.path.join("static", "mod2", "temps"), exist_ok=True)
os.makedirs(os.path.join("static", "marker"), exist_ok=True)

app = Flask(__name__, static_folder="static", template_folder="templates")
CORS(app) # Enable CORS to allow your frontend to send POST requests freely
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
    # LAZY IMPORT
    import numpy as np
    import cv2
    data = file_storage.read()
    arr = np.frombuffer(data, dtype=np.uint8)
    return cv2.imdecode(arr, cv2.IMREAD_COLOR)

# -------------------------
# Root & Auth (MOCKED FOR STABILITY)
# -------------------------
@app.route("/")
def index():
    # Clear session so they have to "login" (even if mocked)
    session.clear()
    # Renders the landing page with the webcam
    return render_template("face_unlock.html")

@app.route("/skip_login")
def skip_login():
    # Allow manual entry button to work
    session["logged_in"] = True
    return redirect("/home")

@app.route("/auth_face", methods=["POST"])
def auth_face():
    # Mock response to prevent RAM crash
    time.sleep(1.0)
    
    # Return "No Match" so the frontend asks for manual entry
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

# -------------------------
# Module 1: Camera Calibration
# -------------------------
@app.route("/module1")
def module1():
    return render_template("index.html")

@app.route("/calibrate", methods=["POST"])
def calibrate():
    import numpy as np
    import cv2

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
    objp = np.zeros((pattern_rows * pattern_cols, 3), np.float32)
    objp[:, :2] = np.mgrid[0:pattern_cols, 0:pattern_rows].T.reshape(-1, 2) * square_size

    objpoints = []
    imgpoints = []
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
            cv2.CALIB_CB_ADAPTIVE_THRESH + cv2.CALIB_CB_NORMALIZE_IMAGE + cv2.CALIB_CB_FAST_CHECK
        )

        if found:
            refined = cv2.cornerSubPix(
                gray, corners, (11, 11), (-1, -1),
                (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 1e-6)
            )
            objpoints.append(objp)
            imgpoints.append(refined)
            messages.append({"file": f.filename, "status": "ok"})
        else:
            messages.append({"file": f.filename, "status": "no_chessboard"})

    if not objpoints:
        return jsonify({"error": "No valid chessboard detections"}), 400

    ret, K, dist, rvecs, tvecs = cv2.calibrateCamera(
        objpoints, imgpoints, image_shape, None, None
    )

    return jsonify({
        "fx": float(K[0, 0]),
        "fy": float(K[1, 1]),
        "cx": float(K[0, 2]),
        "cy": float(K[1, 2]),
        "messages": messages
    })

@app.route("/measure", methods=["POST"])
def measure():
    import numpy as np
    
    if "image" not in request.files:
        return jsonify({"error": "No image uploaded"}), 400

    img = read_image_from_bytes(request.files["image"])
    if img is None: return jsonify({"error": "Could not read image"}), 400

    try:
        pts = np.array(json.loads(request.form["points"]), dtype=np.float32)
        fx = float(request.form["fx"])
        fy = float(request.form["fy"])
        distance_cm = float(request.form["distance_cm"])
    except Exception as e:
        return jsonify({"error": str(e)}), 400

    if pts.shape != (4, 2):
        return jsonify({"error": "4 points required"}), 400

    pts_sorted = pts[np.argsort(pts[:, 1])]
    top_two, bottom_two = pts_sorted[:2], pts_sorted[2:]
    top_left, top_right = sorted(top_two, key=lambda x: x[0])
    bottom_left, bottom_right = sorted(bottom_two, key=lambda x: x[0])

    pixel_width = float(np.linalg.norm(top_right - top_left))
    h_left = np.linalg.norm(bottom_left - top_left)
    h_right = np.linalg.norm(bottom_right - top_right)
    pixel_height = float((h_left + h_right) / 2.0)

    estimated_width_cm = (pixel_width * distance_cm) / fx
    estimated_height_cm = (pixel_height * distance_cm) / fy

    return jsonify({
        "pixel_width": pixel_width,
        "pixel_height": pixel_height,
        "estimated_width_cm": estimated_width_cm,
        "estimated_height_cm": estimated_height_cm,
        "fx": fx, "fy": fy
    })

# -------------------------
# Module 2: Template Matching
# -------------------------
@app.route("/module2")
def module2(): return render_template("module2.html")

@app.route("/list_templates_mod2")
def list_templates_mod2():
    template_dir = os.path.join("static", "mod2", "temps")
    if not os.path.exists(template_dir): return jsonify({"templates": []})
    files = sorted([f for f in os.listdir(template_dir) if f.lower().endswith((".jpg", ".png", ".jpeg"))])
    return jsonify({"templates": [f"/static/mod2/temps/{f}" for f in files]})

@app.route("/match", methods=["POST"])
def match():
    import cv2
    from scene import run_multi_template_match

    if "scene" not in request.files: return jsonify({"error": "Missing scene"}), 400
    scene = read_image_from_bytes(request.files["scene"])
    
    template_dir = os.path.join("static", "mod2", "temps")
    templates = []
    if os.path.exists(template_dir):
        for fname in sorted(os.listdir(template_dir)):
            if fname.lower().endswith((".jpg", ".png", ".jpeg")):
                t = cv2.imread(os.path.join(template_dir, fname))
                if t is not None: templates.append(t)
    
    if not templates: return jsonify({"error": "No templates found"}), 400

    try:
        result, boxes = run_multi_template_match(scene, templates)
        _, buffer = cv2.imencode(".png", result)
        img_b64 = base64.b64encode(buffer).decode("ascii")
        return jsonify({"image": img_b64, "boxes": boxes})
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route("/reconstruct", methods=["POST"])
def reconstruct_route():
    import cv2
    import json
    from scene import reconstruct_blurred_regions
    
    if "image" not in request.files: return jsonify({"error": "No image"}), 400
    img = read_image_from_bytes(request.files["image"])
    boxes = json.loads(request.form.get("boxes", "[]"))
    
    result = reconstruct_blurred_regions(img, boxes)
    _, buf = cv2.imencode(".png", result)
    return jsonify({"image": base64.b64encode(buf).decode("ascii")})

# -------------------------
# Module 3: Image Processing
# -------------------------
@app.route("/module3")
def module3(): return render_template("module3.html")

@app.route("/process_gradient_log", methods=["POST"])
def process_gradient_log():
    import cv2
    import numpy as np
    
    files = request.files.getlist("images[]")
    processed_images = []
    target_width = 300

    for f in files:
        if not allowed_image(f.filename): continue
        img = read_image_from_bytes(f)
        if img is None: continue

        h, w = img.shape[:2]
        scale = target_width / w
        original_resized = cv2.resize(img, (target_width, int(h * scale)))

        img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        img_f = img_gray.astype(np.float32) / 255.0

        gx = cv2.Sobel(img_f, cv2.CV_32F, 1, 0, ksize=3)
        gy = cv2.Sobel(img_f, cv2.CV_32F, 0, 1, ksize=3)
        grad_mag = cv2.magnitude(gx, gy)
        grad_angle = cv2.phase(gx, gy, angleInDegrees=True)
        
        grad_mag_disp = cv2.normalize(grad_mag, None, 0, 1, cv2.NORM_MINMAX)
        grad_mag_uint8 = (grad_mag_disp * 255).astype(np.uint8)
        grad_mag_bgr = cv2.cvtColor(cv2.resize(grad_mag_uint8, (target_width, int(h*scale))), cv2.COLOR_GRAY2BGR)

        grad_angle_uint8 = (grad_angle / 360.0 * 255).astype(np.uint8)
        grad_angle_color = cv2.applyColorMap(grad_angle_uint8, cv2.COLORMAP_HSV)
        grad_angle_resized = cv2.resize(grad_angle_color, (target_width, int(h*scale)))

        blur = cv2.GaussianBlur(img_f, (5, 5), sigmaX=1)
        log = cv2.Laplacian(blur, cv2.CV_32F, ksize=5)
        log_norm = cv2.normalize(np.abs(log), None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
        _, log_edges = cv2.threshold(log_norm, 30, 255, cv2.THRESH_BINARY)
        log_bgr = cv2.cvtColor(cv2.resize(log_edges, (target_width, int(h*scale))), cv2.COLOR_GRAY2BGR)

        processed_images.append(np.hstack([original_resized, grad_mag_bgr, grad_angle_resized, log_bgr]))

    if not processed_images: return jsonify({"error": "No images"}), 400
    
    # Create Header
    num_cols = 4
    header_height = 40
    header_img = np.full((header_height, target_width * num_cols, 3), 255, dtype=np.uint8)
    labels = ["Original", "Grad Mag", "Grad Angle", "LoG"]
    for i, text in enumerate(labels):
        pos = (i * target_width + target_width // 2 - 50, 28)
        cv2.putText(header_img, text, pos, cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 2, cv2.LINE_AA)
        
    final_output = np.vstack([header_img] + processed_images)
    _, buf = cv2.imencode(".png", final_output)
    return jsonify({"image": base64.b64encode(buf).decode("ascii")})

@app.route("/process_edge_corner", methods=["POST"])
def process_edge_corner():
    import cv2
    import numpy as np
    
    files = request.files.getlist("images[]")
    output = []
    target_width = 260

    for f in files:
        if not allowed_image(f.filename): continue
        img = read_image_from_bytes(f)
        if img is None: continue
        
        h, w = img.shape[:2]
        scale = target_width / w
        
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY).astype(np.float32)
        Ix = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
        Iy = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
        grad_mag = np.sqrt(Ix**2 + Iy**2)
        edge_map = (grad_mag > np.percentile(grad_mag, 90)).astype(np.uint8) * 255
        edge_resized = cv2.resize(edge_map, (target_width, int(h * scale)))

        # Harris Corner
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

        _, buf1 = cv2.imencode(".png", edge_resized)
        _, buf2 = cv2.imencode(".png", corner_resized)
        output.append({
            "filename": f.filename,
            "edges": base64.b64encode(buf1).decode("ascii"),
            "corners": base64.b64encode(buf2).decode("ascii")
        })

    return jsonify({"images": output})

@app.route("/process_aruco", methods=["POST"])
def process_aruco():
    import cv2
    import numpy as np
    
    files = request.files.getlist("images[]")
    output_images = []
    target_width = 260
    
    aruco_dict = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_50)
    params = cv2.aruco.DetectorParameters()
    detector = cv2.aruco.ArucoDetector(aruco_dict, params)

    for f in files:
        if not allowed_image(f.filename): continue
        img = read_image_from_bytes(f)
        if img is None: continue
        
        h, w = img.shape[:2]
        scale = target_width / w
        img_resized = cv2.resize(img, (target_width, int(h * scale)))
        
        vis = img.copy()
        corners, ids, _ = detector.detectMarkers(img)
        if ids is not None and len(ids) > 0:
            cv2.aruco.drawDetectedMarkers(vis, corners, ids)
            all_pts = []
            for c in corners: all_pts.extend(c[0].astype(int))
            hull = cv2.convexHull(np.array(all_pts))
            cv2.polylines(vis, [hull], True, (0,0,255), 8)
            for (x,y) in hull.reshape(-1,2): cv2.circle(vis, (x,y), 6, (0,0,255), -1)
            
        vis_resized = cv2.resize(vis, (target_width, int(h * scale)))
        combined = np.hstack([img_resized, vis_resized])
        _, buf = cv2.imencode(".png", combined)
        output_images.append(base64.b64encode(buf).decode("ascii"))
        
    return jsonify({"images": output_images})

@app.route("/process_boundary", methods=["POST"])
def process_boundary():
    import cv2
    import numpy as np
    
    files = request.files.getlist("images[]")
    results = []
    for f in files:
        if not allowed_image(f.filename): continue
        img = read_image_from_bytes(f)
        if img is None: continue
        
        original = img.copy()
        hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
        mask_color = cv2.inRange(hsv, np.array([70, 30, 30]), np.array([110, 255, 255]))
        edges = cv2.Canny(cv2.cvtColor(img, cv2.COLOR_BGR2GRAY), 60, 150)
        combined = cv2.bitwise_or(mask_color, edges)
        
        kernel = np.ones((5,5), np.uint8)
        combined = cv2.morphologyEx(combined, cv2.MORPH_CLOSE, kernel, iterations=3)
        
        contours, _ = cv2.findContours(combined, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if contours:
            c = max(contours, key=cv2.contourArea)
            approx = cv2.approxPolyDP(c, 0.01 * cv2.arcLength(c, True), True)
            cv2.polylines(original, [approx], True, (0,0,255), 5, cv2.LINE_AA)
        results.append(original)

    if not results: return jsonify({"error": "No results"}), 400
    
    target_width = 350
    resized = [cv2.resize(r, (target_width, int(r.shape[0]*target_width/r.shape[1]))) for r in results]
    final = np.vstack(resized)
    _, buf = cv2.imencode(".png", final)
    return jsonify({"image": base64.b64encode(buf).decode("ascii")})

# -------------------------
# Module 4: Stitching
# -------------------------
@app.route("/module4")
def module4(): return render_template("module4.html")

@app.route("/process_stitch")
def process_stitch():
    import cv2
    
    pano_folder = os.path.join(app.config["UPLOAD_FOLDER"], "panorama")
    if not os.path.exists(pano_folder): 
        os.makedirs(pano_folder, exist_ok=True)
        return jsonify({"error": "No images in uploads/panorama folder"}), 400
    
    image_files = sorted([os.path.join(pano_folder, f) for f in os.listdir(pano_folder) if f.lower().endswith((".jpg", ".png"))])
    if len(image_files) < 2: return jsonify({"error": "Need 2+ images"}), 400

    images = [cv2.imread(p) for p in image_files if cv2.imread(p) is not None]
    
    try:
        stitcher = cv2.Stitcher_create(cv2.Stitcher_PANORAMA)
    except:
        stitcher = cv2.createStitcher(cv2.Stitcher_PANORAMA)
    
    stitcher.setPanoConfidenceThresh(0.5)
    status, panorama = stitcher.stitch(images)
    if status != cv2.Stitcher_OK: return jsonify({"error": f"Stitch failed: {status}"}), 400

    gray = cv2.cvtColor(panorama, cv2.COLOR_BGR2GRAY)
    _, thresh = cv2.threshold(gray, 1, 255, cv2.THRESH_BINARY)
    cnts, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if cnts:
        x,y,w,h = cv2.boundingRect(cnts[0])
        panorama = panorama[y:y+h, x:x+w]

    _, buf = cv2.imencode(".jpg", panorama)
    
    thumbs = []
    for img in images:
        _, tbuf = cv2.imencode(".jpg", cv2.resize(img, (200,120)))
        thumbs.append(base64.b64encode(tbuf).decode("utf-8"))

    return jsonify({
        "stitched": base64.b64encode(buf).decode("utf-8"),
        "width": panorama.shape[1], "height": panorama.shape[0],
        "thumbnails": thumbs
    })

# -------------------------
# Module 6: SIFT/RANSAC
# -------------------------
@app.route("/module6")
def module6(): return render_template("module6.html")

@app.route("/process_sift")
def process_sift():
    import cv2
    import numpy as np
    
    folder = os.path.join(app.config['UPLOAD_FOLDER'], "sift_images")
    if not os.path.exists(folder): return jsonify({"error": "sift_images folder missing"})
    
    p1 = os.path.join(folder, "1.jpeg")
    p2 = os.path.join(folder, "5.jpeg")
    if not os.path.exists(p1) or not os.path.exists(p2):
        return jsonify({"error": "Images 1.jpeg/5.jpeg not found"})

    img1 = cv2.imread(p1, cv2.IMREAD_GRAYSCALE)
    img2 = cv2.imread(p2, cv2.IMREAD_GRAYSCALE)
    
    img1 = cv2.resize(img1, (600, 400))
    img2 = cv2.resize(img2, (600, 400))
    
    sift = cv2.SIFT_create()
    kp1, des1 = sift.detectAndCompute(img1, None)
    kp2, des2 = sift.detectAndCompute(img2, None)
    
    bf = cv2.BFMatcher()
    matches = bf.knnMatch(des1, des2, k=2)
    good = [m for m,n in matches if m.distance < 0.75*n.distance]
    
    src = np.float32([kp1[m.queryIdx].pt for m in good]).reshape(-1,1,2)
    dst = np.float32([kp2[m.trainIdx].pt for m in good]).reshape(-1,1,2)
    
    H, mask = cv2.findHomography(src, dst, cv2.RANSAC, 5.0)
    matches_mask = mask.ravel().tolist() if mask is not None else None
    
    res = cv2.drawMatches(img1, kp1, img2, kp2, good, None, 
                         matchColor=(0,255,0), matchesMask=matches_mask, flags=2)
    
    _, buf = cv2.imencode('.jpg', res)
    return jsonify({"opencv": base64.b64encode(buf).decode("utf-8"), "custom": ""})

# -------------------------
# Module 7: Pose Tracking (HTTP POST)
# -------------------------
mp_holistic = None
mp_drawing = None
holistic_net = None
CSV_FILENAME = None

def get_model():
    """Lazy loader to prevent startup crash"""
    global mp_holistic, mp_drawing, holistic_net
    if holistic_net is None:
        print("Loading MediaPipe (Lazy)...")
        import mediapipe as mp
        mp_holistic = mp.solutions.holistic
        mp_drawing = mp.solutions.drawing_utils
        # Use LITE model to save RAM
        holistic_net = mp_holistic.Holistic(
            static_image_mode=False,
            model_complexity=0, 
            smooth_landmarks=True
        )
    return holistic_net, mp_drawing, mp_holistic

@app.route("/module7")
def module7(): return render_template("module7.html")

@app.route("/download_csv_module7")
def download_csv_module7():
    if CSV_FILENAME and os.path.exists(CSV_FILENAME):
        return send_file(CSV_FILENAME, as_attachment=True)
    return "No CSV generated yet", 404

# --- THE NEW HTTP POST ROUTE ---
@app.route('/process_frame', methods=['POST'])
def process_frame():
    import numpy as np
    import cv2
    import gc
    global CSV_FILENAME

    try:
        data = request.json
        image_data = data.get('image')
        
        # Decode Base64
        header, encoded = image_data.split(",", 1)
        binary = base64.b64decode(encoded)
        np_arr = np.frombuffer(binary, np.uint8)
        frame = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
        
        if frame is None: 
            return jsonify({"error": "Empty frame"}), 400

        # Process
        model, drawing, mp_ref = get_model()
        image_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = model.process(image_rgb)
        
        # Draw
        if results.pose_landmarks:
            drawing.draw_landmarks(frame, results.pose_landmarks, mp_ref.POSE_CONNECTIONS)
        if results.left_hand_landmarks:
            drawing.draw_landmarks(frame, results.left_hand_landmarks, mp_ref.HAND_CONNECTIONS)
        if results.right_hand_landmarks:
            drawing.draw_landmarks(frame, results.right_hand_landmarks, mp_ref.HAND_CONNECTIONS)

        # CSV Logging
        if results.pose_landmarks and CSV_FILENAME is None:
             ts = int(time.time())
             CSV_FILENAME = os.path.join(UPLOAD_FOLDER, f"pose_{ts}.csv")
             with open(CSV_FILENAME, 'w') as f: f.write("timestamp,pose_present\n")
        
        if results.pose_landmarks:
            with open(CSV_FILENAME, 'a') as f: f.write(f"{int(time.time()*1000)},1\n")

        # Encode
        _, buffer = cv2.imencode('.jpg', frame)
        response_b64 = base64.b64encode(buffer).decode('utf-8')
        
        # Explicit cleanup
        del frame, image_rgb, results, binary
        gc.collect()

        return jsonify({"image": f"data:image/jpeg;base64,{response_b64}"})

    except Exception as e:
        print(f"Error: {e}")
        return jsonify({"error": str(e)}), 500

@app.route('/module7_calc', methods=['POST'])
def module7_calc():
    import numpy as np
    import json
    try:
        fx = float(request.form['fx'])
        baseline = float(request.form['baseline'])
        left = np.array(json.loads(request.form['left_points']), dtype=np.float32)
        right = np.array(json.loads(request.form['right_points']), dtype=np.float32)
        
        disparities = left[:, 0] - right[:, 0]
        disparities[disparities == 0] = 0.0001
        Z_cm = fx * baseline / disparities
        
        return f"Calculated Depths: {Z_cm.tolist()}"
    except Exception as e:
        return f"Error: {str(e)}"

@app.route("/get_sam2_results", methods=["GET"])
def get_sam2_results():
    return jsonify({"images": []})

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)