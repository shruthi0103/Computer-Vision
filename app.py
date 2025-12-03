import os
import base64
import json
import time
import io
<<<<<<< Updated upstream
import gc  
from flask import Flask, request, render_template, jsonify, redirect, session, send_file, Response
from flask_cors import CORS 
=======
import gc
from flask import Flask, request, render_template, jsonify, redirect, session, send_file, Response
from flask_cors import CORS
>>>>>>> Stashed changes

# ==========================================
# 1. IMPORTS & SETUP
# ==========================================
UPLOAD_FOLDER = "uploads"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
# Create all subfolders to prevent FileNotFoundError
os.makedirs(os.path.join(UPLOAD_FOLDER, "panorama"), exist_ok=True)
os.makedirs(os.path.join(UPLOAD_FOLDER, "phone_pano"), exist_ok=True)
os.makedirs(os.path.join("static", "mod2", "temps"), exist_ok=True)
os.makedirs(os.path.join("static", "marker"), exist_ok=True)

app = Flask(__name__, static_folder="static", template_folder="templates")
<<<<<<< Updated upstream
CORS(app) 
=======
CORS(app)  # Allow cross-origin requests for the frontend fetch
>>>>>>> Stashed changes
app.config["UPLOAD_FOLDER"] = UPLOAD_FOLDER
app.config["MAX_CONTENT_LENGTH"] = 200 * 1024 * 1024
app.secret_key = "supersecretkey"

# ==========================================
# 2. HELPER FUNCTIONS
# ==========================================
def allowed_image(filename):
    ext = filename.lower().rsplit(".", 1)[-1]
    return ext in ("jpg", "jpeg", "png", "bmp", "tiff")

def read_image_from_bytes(file_storage):
    # LAZY IMPORT: Saves RAM on startup
    import numpy as np
    import cv2
    data = file_storage.read()
    arr = np.frombuffer(data, dtype=np.uint8)
    return cv2.imdecode(arr, cv2.IMREAD_COLOR)

<<<<<<< Updated upstream
# -------------------------
# Root & Auth
# -------------------------
@app.route("/")
def index():

    session.clear()

=======
# ==========================================
# 3. CORE ROUTES (AUTH & HOME)
# ==========================================
@app.route("/")
def index():
    session.clear()
>>>>>>> Stashed changes
    return render_template("face_unlock.html")

@app.route("/skip_login")
def skip_login():
    session["logged_in"] = True
    return redirect("/home")

@app.route("/auth_face", methods=["POST"])
def auth_face():
    # Mocked response to save RAM (Face Rec needs 500MB+)
    time.sleep(1.0)
<<<<<<< Updated upstream
    

=======
>>>>>>> Stashed changes
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
# 4. MODULE 1: CALIBRATION
# ==========================================
@app.route("/module1")
def module1():
    return render_template("index.html")

@app.route("/calibrate", methods=["POST"])
def calibrate():
    import numpy as np
    import cv2

    files = request.files.getlist("files[]")
    if not files: return jsonify({"error": "No files"}), 400

    try:
        cols = int(request.form.get("pattern_cols", "9"))
        rows = int(request.form.get("pattern_rows", "6"))
        sq_size = float(request.form.get("square_size", "1.0"))
    except: return jsonify({"error": "Invalid params"}), 400

    objp = np.zeros((rows * cols, 3), np.float32)
    objp[:, :2] = np.mgrid[0:cols, 0:rows].T.reshape(-1, 2) * sq_size

    objpoints, imgpoints = [], []
    messages = []
    img_shape = None

    for f in files:
        if not allowed_image(f.filename): continue
        img = read_image_from_bytes(f)
        if img is None: continue
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        if img_shape is None: img_shape = gray.shape[::-1]

        found, corners = cv2.findChessboardCorners(gray, (cols, rows), None)
        if found:
            refined = cv2.cornerSubPix(gray, corners, (11,11), (-1,-1), (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 1e-6))
            objpoints.append(objp)
            imgpoints.append(refined)
            messages.append({"file": f.filename, "status": "ok"})
        else:
            messages.append({"file": f.filename, "status": "no_chessboard"})

    if not objpoints: return jsonify({"error": "No valid patterns found"}), 400

    ret, K, dist, rvecs, tvecs = cv2.calibrateCamera(objpoints, imgpoints, img_shape, None, None)
    
    return jsonify({
        "fx": float(K[0,0]), "fy": float(K[1,1]), "cx": float(K[0,2]), "cy": float(K[1,2]),
        "messages": messages
    })

@app.route("/measure", methods=["POST"])
def measure():
    import numpy as np
    try:
        pts = np.array(json.loads(request.form["points"]), dtype=np.float32)
        fx = float(request.form["fx"])
        fy = float(request.form["fy"])
        dist_cm = float(request.form["distance_cm"])
        
        if pts.shape != (4, 2): return jsonify({"error": "4 points needed"}), 400
        
        pts_sorted = pts[np.argsort(pts[:, 1])]
        top, bottom = pts_sorted[:2], pts_sorted[2:]
        top = top[np.argsort(top[:, 0])]
        bottom = bottom[np.argsort(bottom[:, 0])]
        
        tl, tr, bl, br = top[0], top[1], bottom[0], bottom[1]
        
        pix_w = float(np.linalg.norm(tr - tl))
        pix_h = float((np.linalg.norm(bl - tl) + np.linalg.norm(br - tr)) / 2.0)
        
        return jsonify({
            "estimated_width_cm": (pix_w * dist_cm) / fx,
            "estimated_height_cm": (pix_h * dist_cm) / fy
        })
    except Exception as e: return jsonify({"error": str(e)}), 400

# ==========================================
# 5. MODULE 2: TEMPLATE MATCHING
# ==========================================
@app.route("/module2")
def module2(): return render_template("module2.html")

@app.route("/list_templates_mod2")
def list_templates_mod2():
    d = os.path.join("static", "mod2", "temps")
    if not os.path.exists(d): return jsonify({"templates": []})
    return jsonify({"templates": [f"/static/mod2/temps/{f}" for f in sorted(os.listdir(d)) if f.endswith(('.jpg','.png'))]})

@app.route("/match", methods=["POST"])
def match():
    import cv2
    from scene import run_multi_template_match
    
    if "scene" not in request.files: return jsonify({"error": "No scene"}), 400
    scene = read_image_from_bytes(request.files["scene"])
    
    t_dir = os.path.join("static", "mod2", "temps")
    temps = [cv2.imread(os.path.join(t_dir, f)) for f in os.listdir(t_dir) if f.endswith(('.jpg','.png'))]
    if not temps: return jsonify({"error": "No templates"}), 400
    
    try:
        res, boxes = run_multi_template_match(scene, temps)
        _, buf = cv2.imencode(".png", res)
        return jsonify({"image": base64.b64encode(buf).decode("ascii"), "boxes": boxes})
    except Exception as e: return jsonify({"error": str(e)}), 500

@app.route("/reconstruct", methods=["POST"])
def reconstruct_route():
    import cv2
    import json
    from scene import reconstruct_blurred_regions
    
    img = read_image_from_bytes(request.files["image"])
    boxes = json.loads(request.form.get("boxes", "[]"))
    res = reconstruct_blurred_regions(img, boxes)
    _, buf = cv2.imencode(".png", res)
    return jsonify({"image": base64.b64encode(buf).decode("ascii")})

# ==========================================
# 6. MODULE 3: FEATURE ANALYSIS (Grad, Edge, ArUco)
# ==========================================
@app.route("/module3")
def module3(): return render_template("module3.html")

@app.route("/process_gradient_log", methods=["POST"])
def process_gradient_log():
    import cv2
    import numpy as np
    
    files = request.files.getlist("images[]")
    processed = []
    
    for f in files:
        img = read_image_from_bytes(f)
        if img is None: continue
        
        # Resize for RAM
        h, w = img.shape[:2]
        img = cv2.resize(img, (300, int(h * 300/w)))
        
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY).astype(np.float32) / 255.0
        gx = cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3)
        gy = cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=3)
        mag = cv2.magnitude(gx, gy)
        
        mag_norm = cv2.normalize(mag, None, 0, 1, cv2.NORM_MINMAX)
        mag_bgr = cv2.cvtColor((mag_norm*255).astype(np.uint8), cv2.COLOR_GRAY2BGR)
        
        blur = cv2.GaussianBlur(gray, (5,5), 1)
        log = cv2.Laplacian(blur, cv2.CV_32F)
        log_norm = cv2.normalize(np.abs(log), None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
        _, log_bin = cv2.threshold(log_norm, 30, 255, cv2.THRESH_BINARY)
        log_bgr = cv2.cvtColor(log_bin, cv2.COLOR_GRAY2BGR)
        
        processed.append(np.hstack([img, mag_bgr, log_bgr])) # Simplified stack
        
    if not processed: return jsonify({"error": "No images"}), 400
    final = np.vstack(processed)
    _, buf = cv2.imencode(".png", final)
    return jsonify({"image": base64.b64encode(buf).decode("ascii")})

@app.route("/process_edge_corner", methods=["POST"])
def process_edge_corner():
    import cv2
    import numpy as np
    
    files = request.files.getlist("images[]")
    output = []
    
    for f in files:
        img = read_image_from_bytes(f)
        if img is None: continue
        h, w = img.shape[:2]
        img = cv2.resize(img, (260, int(h * 260/w)))
        
<<<<<<< Updated upstream
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY).astype(np.float32)
        Ix = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
        Iy = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
        grad_mag = np.sqrt(Ix**2 + Iy**2)
        edge_map = (grad_mag > np.percentile(grad_mag, 90)).astype(np.uint8) * 255
        edge_resized = cv2.resize(edge_map, (target_width, int(h * scale)))

  
        gray_blur = cv2.GaussianBlur(gray, (7,7), 1)
        Ix = cv2.Sobel(gray_blur, cv2.CV_64F, 1, 0, 3)
        Iy = cv2.Sobel(gray_blur, cv2.CV_64F, 0, 1, 3)
        Ix2 = cv2.GaussianBlur(Ix*Ix, (5,5), 1)
        Iy2 = cv2.GaussianBlur(Iy*Iy, (5,5), 1)
        Ixy = cv2.GaussianBlur(Ix*Iy, (5,5), 1)
        k = 0.04
        R = (Ix2 * Iy2 - Ixy**2) - k * (Ix2 + Iy2)**2
=======
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
>>>>>>> Stashed changes
        
        # Canny
        edges = cv2.Canny(gray, 100, 200)
        
        # Harris
        gray_f = np.float32(gray)
        dst = cv2.cornerHarris(gray_f, 2, 3, 0.04)
        corn = img.copy()
        corn[dst > 0.01*dst.max()] = [0,0,255]
        
        _, b1 = cv2.imencode(".png", edges)
        _, b2 = cv2.imencode(".png", corn)
        output.append({
            "filename": f.filename,
            "edges": base64.b64encode(b1).decode("ascii"),
            "corners": base64.b64encode(b2).decode("ascii")
        })
    return jsonify({"images": output})

@app.route("/process_aruco", methods=["POST"])
def process_aruco():
    import cv2
    import numpy as np
    
    files = request.files.getlist("images[]")
    output = []
    dct = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_50)
    det = cv2.aruco.ArucoDetector(dct, cv2.aruco.DetectorParameters())
    
    for f in files:
        img = read_image_from_bytes(f)
        if img is None: continue
        h, w = img.shape[:2]
        img = cv2.resize(img, (260, int(h * 260/w)))
        
        cnr, ids, _ = det.detectMarkers(img)
        vis = img.copy()
        if ids is not None:
            cv2.aruco.drawDetectedMarkers(vis, cnr, ids)
            pts = []
            for c in cnr: pts.extend(c[0].astype(int))
            if pts:
                hull = cv2.convexHull(np.array(pts))
                cv2.polylines(vis, [hull], True, (0,0,255), 2)
        
        _, b = cv2.imencode(".png", vis)
        output.append(base64.b64encode(b).decode("ascii"))
    return jsonify({"images": output})

@app.route("/process_boundary", methods=["POST"])
def process_boundary():
    import cv2
    import numpy as np
    files = request.files.getlist("images[]")
    res = []
    for f in files:
        img = read_image_from_bytes(f)
        if img is None: continue
        h, w = img.shape[:2]
        img = cv2.resize(img, (300, int(h*300/w)))
        
        hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
        mask = cv2.inRange(hsv, (70,30,30), (110,255,255)) # Default teal/box range
        cnts, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if cnts:
            c = max(cnts, key=cv2.contourArea)
            cv2.drawContours(img, [c], -1, (0,0,255), 2)
        res.append(img)
    
    if not res: return jsonify({"error": "No results"}), 400
    _, b = cv2.imencode(".png", np.vstack(res))
    return jsonify({"image": base64.b64encode(b).decode("ascii")})

# ==========================================
# 7. MODULE 4: STITCHING & SIFT
# ==========================================


@app.route("/module4")
def module4():
    if not session.get("logged_in"):
        return redirect("/")
    return render_template("module4.html")


# -----------------------------------
# ✅ UPLOAD IMAGES FOR STITCHING
# -----------------------------------
@app.route("/upload_panorama", methods=["POST"])
def upload_panorama():
    files = request.files.getlist("images[]")

    pano_folder = os.path.join(app.config["UPLOAD_FOLDER"], "panorama")
    os.makedirs(pano_folder, exist_ok=True)

    # ✅ Clear old images to avoid mixing runs
    for f in os.listdir(pano_folder):
        os.remove(os.path.join(pano_folder, f))

    if not files:
        return jsonify({"error": "No images uploaded"}), 400

    saved_files = []
    for f in files:
        if allowed_image(f.filename):
            filepath = os.path.join(pano_folder, f.filename)
            f.save(filepath)
            saved_files.append(f.filename)

    return jsonify({
        "message": "Images uploaded successfully",
        "files": saved_files
    })


# -----------------------------------
# ✅ STITCH IMAGES AFTER UPLOAD
# -----------------------------------
# --- REPLACE THIS FUNCTION IN APP.PY ---
@app.route("/process_stitch", methods=["POST"])
def process_stitch():
    import cv2
    import numpy as np
    import gc

    # 1. Get uploaded files
    uploaded_files = request.files.getlist("images")
    phone_pano_file = request.files.get("phone_pano")

    if not uploaded_files or len(uploaded_files) < 2:
        return jsonify({"error": "Please upload at least 2 images to stitch."}), 400

    images = []
    # 2. Read stitch images into memory
    for f in uploaded_files:
        img = read_image_from_bytes(f)
        if img is not None:
            # Resize huge images to prevent RAM crash
            if img.shape[1] > 800:
                scale = 800 / img.shape[1]
                img = cv2.resize(img, (0,0), fx=scale, fy=scale)
            images.append(img)

    if len(images) < 2:
        return jsonify({"error": "Could not read images or images corrupted."}), 400

    # 3. Stitch
    try:
        stitcher = cv2.Stitcher_create(cv2.Stitcher_PANORAMA)
    except:
        stitcher = cv2.createStitcher(cv2.Stitcher_PANORAMA)
    
    stitcher.setPanoConfidenceThresh(0.3)
    status, panorama = stitcher.stitch(images)

    if status != cv2.Stitcher_OK:
        return jsonify({"error": f"Stitching failed (Code {status}). Ensure images have overlapping features."}), 400

    # 4. Crop black borders
    gray = cv2.cvtColor(panorama, cv2.COLOR_BGR2GRAY)
    _, thresh = cv2.threshold(gray, 1, 255, cv2.THRESH_BINARY)
    cnts, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if cnts:
        x,y,w,h = cv2.boundingRect(cnts[0])
        panorama = panorama[y:y+h, x:x+w]

    # 5. Encode Result
    _, buf = cv2.imencode(".jpg", panorama)
    stitched_b64 = base64.b64encode(buf).decode("utf-8")

    # 6. Encode Thumbnails
    thumbs = []
    for img in images:
        _, tbuf = cv2.imencode(".jpg", cv2.resize(img, (150, 100)))
        thumbs.append(base64.b64encode(tbuf).decode("utf-8"))

    # 7. Process Phone Panorama (If uploaded)
    phone_pano_b64 = None
    if phone_pano_file:
        phone_img = read_image_from_bytes(phone_pano_file)
        if phone_img is not None:
            # Resize just for display
            if phone_img.shape[1] > 1000:
                s = 1000 / phone_img.shape[1]
                phone_img = cv2.resize(phone_img, (0,0), fx=s, fy=s)
            _, pbuf = cv2.imencode(".jpg", phone_img)
            phone_pano_b64 = base64.b64encode(pbuf).decode("utf-8")

    # Cleanup
    del images, gray, thresh, panorama
    gc.collect()

    return jsonify({
        "stitched": stitched_b64,
        "thumbnails": thumbs,
        "phone_pano": phone_pano_b64
    })
@app.route("/process_sift", methods=["POST"])
def process_sift_upload():
    import cv2
    import numpy as np

    if "img1" not in request.files or "img2" not in request.files:
        return jsonify({"error": "Both images are required"}), 400

    img1 = read_image_from_bytes(request.files["img1"])
    img2 = read_image_from_bytes(request.files["img2"])

    if img1 is None or img2 is None:
        return jsonify({"error": "Failed to read one or both images"}), 400

    # Convert to grayscale
    gray1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
    gray2 = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)

    gray1 = cv2.resize(gray1, (600, 400))
    gray2 = cv2.resize(gray2, (600, 400))

    # ===========================
    # OpenCV SIFT + RANSAC
    # ===========================
    sift = cv2.SIFT_create()
    kp1, des1 = sift.detectAndCompute(gray1, None)
    kp2, des2 = sift.detectAndCompute(gray2, None)

    bf = cv2.BFMatcher()
    matches = bf.knnMatch(des1, des2, k=2)
    good = [m for m, n in matches if m.distance < 0.75 * n.distance]

    if len(good) < 8:
        return jsonify({"error": "Not enough good matches"}), 400

    src = np.float32([kp1[m.queryIdx].pt for m in good]).reshape(-1, 1, 2)
    dst = np.float32([kp2[m.trainIdx].pt for m in good]).reshape(-1, 1, 2)

    H, mask = cv2.findHomography(src, dst, cv2.RANSAC, 5.0)
    matches_mask = mask.ravel().tolist() if mask is not None else None

    opencv_result = cv2.drawMatches(
        gray1, kp1, gray2, kp2,
        [m for i, m in enumerate(good) if matches_mask is None or matches_mask[i]],
        None,
        matchColor=(0, 255, 0),
        flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS
    )

    # ===========================
    # Custom (ORB fallback)
    # ===========================
    orb = cv2.ORB_create(3000)
    kp1c, des1c = orb.detectAndCompute(gray1, None)
    kp2c, des2c = orb.detectAndCompute(gray2, None)

    bf2 = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
    matches_c = bf2.match(des1c, des2c)
    matches_c = sorted(matches_c, key=lambda x: x.distance)[:60]

    custom_result = cv2.drawMatches(
        gray1, kp1c, gray2, kp2c,
        matches_c,
        None,
        matchColor=(255, 0, 0),
        flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS
    )

    # ===========================
    # Encode to Base64
    # ===========================
    _, buf1 = cv2.imencode(".jpg", custom_result)
    _, buf2 = cv2.imencode(".jpg", opencv_result)

    return jsonify({
        "custom": base64.b64encode(buf1).decode("utf-8"),
        "opencv": base64.b64encode(buf2).decode("utf-8")
    })

# ==========================================
# 8. MODULE 6: MOTION TRACKING (Video Streams)
# ==========================================
@app.route("/module6")
def module6(): return render_template("module6.html")

def gen_markerless():
    # Lazy generator for motion tracking
    import cv2
    import numpy as np
    cap = cv2.VideoCapture("static/motion.mp4")
    tracker = cv2.legacy.TrackerCSRT_create()
    init = False
    
    while True:
        ret, frame = cap.read()
        if not ret: break
        
        if not init:
            # Auto-detect object (simple blob) for demo
            hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
            mask = cv2.inRange(hsv, (0, 100, 100), (20, 255, 255)) # Orange-ish
            cnts, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            if cnts:
                x,y,w,h = cv2.boundingRect(max(cnts, key=cv2.contourArea))
                tracker.init(frame, (x,y,w,h))
                init = True
        else:
            success, box = tracker.update(frame)
            if success:
                x,y,w,h = [int(v) for v in box]
                cv2.rectangle(frame, (x,y), (x+w, y+h), (0,255,0), 2)
                
        _, jpeg = cv2.imencode('.jpg', frame)
        yield (b'--frame\r\n' b'Content-Type: image/jpeg\r\n\r\n' + jpeg.tobytes() + b'\r\n')

@app.route("/video_feed_markerless")
def video_feed_markerless():
    return Response(gen_markerless(), mimetype='multipart/x-mixed-replace; boundary=frame')

<<<<<<< Updated upstream
# -------------------------
# Module 7: Pose Tracking 
# -------------------------
=======
def gen_aruco_stream():
    import cv2
    import numpy as np
    cap = cv2.VideoCapture("static/aruco_video.mp4")
    dct = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_50)
    det = cv2.aruco.ArucoDetector(dct, cv2.aruco.DetectorParameters())
    
    while True:
        ret, frame = cap.read()
        if not ret: break
        cnr, ids, _ = det.detectMarkers(frame)
        if ids is not None: cv2.aruco.drawDetectedMarkers(frame, cnr, ids)
        _, jpeg = cv2.imencode('.jpg', frame)
        yield (b'--frame\r\n' b'Content-Type: image/jpeg\r\n\r\n' + jpeg.tobytes() + b'\r\n')

@app.route("/video_feed_aruco")
def video_feed_aruco():
    return Response(gen_aruco_stream(), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route("/stream_sam2")
def stream_sam2():
    # Placeholder for SAM2 stream if file doesn't exist
    return jsonify({"error": "SAM2 video not found"})

# ==========================================
# 9. MODULE 7: POSE TRACKING (HTTP POST)
# ==========================================
>>>>>>> Stashed changes
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
<<<<<<< Updated upstream

=======
>>>>>>> Stashed changes
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

<<<<<<< Updated upstream

=======
>>>>>>> Stashed changes
@app.route('/process_frame', methods=['POST'])
def process_frame():
    import numpy as np
    import cv2
    import gc
    global CSV_FILENAME

    try:
        data = request.json
        image_data = data.get('image')
<<<<<<< Updated upstream
        

=======
>>>>>>> Stashed changes
        header, encoded = image_data.split(",", 1)
        binary = base64.b64decode(encoded)
        np_arr = np.frombuffer(binary, np.uint8)
        frame = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
        
        if frame is None: return jsonify({"error": "Empty"}), 400

<<<<<<< Updated upstream
    
        model, drawing, mp_ref = get_model()
        image_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = model.process(image_rgb)
        
 
        landmark_style = drawing.DrawingSpec(color=(0, 255, 0), thickness=1, circle_radius=1)
        connection_style = drawing.DrawingSpec(color=(255, 255, 255), thickness=1, circle_radius=1)


        if results.pose_landmarks:
            drawing.draw_landmarks(
                frame, results.pose_landmarks, mp_ref.POSE_CONNECTIONS,
                landmark_drawing_spec=landmark_style,
                connection_drawing_spec=connection_style
            )
            

            if CSV_FILENAME is None:
                 ts = int(time.time())
                 CSV_FILENAME = os.path.join(UPLOAD_FOLDER, f"pose_{ts}.csv")
                 with open(CSV_FILENAME, 'w') as f: f.write("timestamp,pose_present\n")
            with open(CSV_FILENAME, 'a') as f: f.write(f"{int(time.time()*1000)},1\n")


        if results.left_hand_landmarks:
            drawing.draw_landmarks(
                frame, results.left_hand_landmarks, mp_ref.HAND_CONNECTIONS,
                landmark_drawing_spec=landmark_style,
                connection_drawing_spec=connection_style
            )


        if results.right_hand_landmarks:
            drawing.draw_landmarks(
                frame, results.right_hand_landmarks, mp_ref.HAND_CONNECTIONS,
                landmark_drawing_spec=landmark_style,
                connection_drawing_spec=connection_style
            )

     
        _, buffer = cv2.imencode('.jpg', frame)
        response_b64 = base64.b64encode(buffer).decode('utf-8')
        

        del frame, image_rgb, results, binary
=======
        # Try-Catch for MediaPipe RAM safety
        try:
            model, drawing, mp_ref = get_model()
            image_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = model.process(image_rgb)
            
            # Styles
            lm_style = drawing.DrawingSpec(color=(0,255,0), thickness=1, circle_radius=1)
            con_style = drawing.DrawingSpec(color=(255,255,255), thickness=1, circle_radius=1)

            if results.pose_landmarks:
                drawing.draw_landmarks(frame, results.pose_landmarks, mp_ref.POSE_CONNECTIONS, lm_style, con_style)
                
                # CSV
                if CSV_FILENAME is None:
                     ts = int(time.time())
                     CSV_FILENAME = os.path.join(UPLOAD_FOLDER, f"pose_{ts}.csv")
                     with open(CSV_FILENAME, 'w') as f: f.write("timestamp,pose_present\n")
                with open(CSV_FILENAME, 'a') as f: f.write(f"{int(time.time()*1000)},1\n")

            if results.left_hand_landmarks:
                drawing.draw_landmarks(frame, results.left_hand_landmarks, mp_ref.HAND_CONNECTIONS, lm_style, con_style)
            if results.right_hand_landmarks:
                drawing.draw_landmarks(frame, results.right_hand_landmarks, mp_ref.HAND_CONNECTIONS, lm_style, con_style)
                
        except Exception as e:
            print(f"AI Skipped (RAM): {e}")

        _, buffer = cv2.imencode('.jpg', frame)
        response_b64 = base64.b64encode(buffer).decode('utf-8')
        
        del frame, binary
>>>>>>> Stashed changes
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
    except Exception as e: return f"Error: {str(e)}"

# ==========================================
# 10. MAIN EXECUTION
# ==========================================
if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)
