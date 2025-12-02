import cv2
import numpy as np
import time
from pathlib import Path

# =========================
# Utility: colors + labeling
# =========================
PALETTE = [
    (0,255,255), (255,0,255), (0,255,0), (255,255,0), (255,0,0),
    (0,128,255), (255,128,0), (0,200,150), (180,0,255), (255,50,150)
]
def pick_color(i): 
    return PALETTE[i % len(PALETTE)]

def nice_label(i, name="TEMPLATE"):
    return f"{name.upper()} {i+1}"

# -------------------------
# Auto-crop (optimized GrabCut)
# -------------------------
def autocrop_template(img_bgr, scale=0.3):
    start = time.time()
    h, w = img_bgr.shape[:2]
    if h < 40 or w < 40:
        return img_bgr

    img_small = cv2.resize(img_bgr, None, fx=scale, fy=scale)
    hs, ws = img_small.shape[:2]
    mask = np.zeros((hs, ws), np.uint8)
    shrink = 0.08
    rect = (int(ws*shrink), int(hs*shrink), int(ws*(1-shrink)-ws*shrink), int(hs*(1-shrink)-hs*shrink))
    bgdModel = np.zeros((1,65), np.float64)
    fgdModel = np.zeros((1,65), np.float64)
    cv2.grabCut(img_small, mask, rect, bgdModel, fgdModel, 1, cv2.GC_INIT_WITH_RECT)

    bin_mask = np.where((mask==cv2.GC_FGD) | (mask==cv2.GC_PR_FGD), 255, 0).astype('uint8')
    cnts, _ = cv2.findContours(bin_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not cnts:
        return img_bgr

    c = max(cnts, key=cv2.contourArea)
    x,y,wc,hc = cv2.boundingRect(c)
    pad = int(max(wc,hc)*0.02)
    x = max(0, x-pad); y = max(0, y-pad)
    x2 = min(ws, x+wc+pad); y2 = min(hs, y+hc+pad)
    cropped_small = img_small[y:y2, x:x2]
    cropped = cv2.resize(cropped_small, None, fx=1/scale, fy=1/scale)
    print(f"[⏱️ GrabCut + crop] {time.time() - start:.3f} s")
    return cropped

# -------------------------
# Feature detection + matching + BLUR
# -------------------------
def detect_in_scene(scene_bgr, templ_bgr, color=(0,255,255)):
    out = scene_bgr.copy()
    sg = cv2.cvtColor(scene_bgr, cv2.COLOR_BGR2GRAY)
    tg = cv2.cvtColor(templ_bgr, cv2.COLOR_BGR2GRAY)

    t0 = time.time()
    akaze = cv2.AKAZE_create()
    kpt, dest = akaze.detectAndCompute(tg, None)
    kps, dess = akaze.detectAndCompute(sg, None)
    print(f"[⏱️ AKAZE detect+compute] {time.time() - t0:.3f} s")

    if dest is None or dess is None or len(kpt) < 4 or len(kps) < 4:
        orb = cv2.ORB_create(nfeatures=6000)
        kpt, dest = orb.detectAndCompute(tg, None)
        kps, dess = orb.detectAndCompute(sg, None)

    if dest is None or dess is None:
        return out, False

    bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=False)
    matches = bf.knnMatch(dest, dess, k=2)
    good = [m for m,n in matches if m.distance < 0.70 * n.distance]
    if len(good) < 10:
        return out, False

    src = np.float32([kpt[m.queryIdx].pt for m in good]).reshape(-1,1,2)
    dst = np.float32([kps[m.trainIdx].pt for m in good]).reshape(-1,1,2)
    H, mask = cv2.findHomography(src, dst, cv2.RANSAC, 3.5)
    if H is None:
        return out, False

    h, w = tg.shape
    corners = np.float32([[0,0],[w,0],[w,h],[0,h]]).reshape(-1,1,2)
    proj = cv2.perspectiveTransform(corners, H)

    # --- Draw polygon ---
    cv2.polylines(out, [np.int32(proj)], True, color, 6, cv2.LINE_AA)

    # --- Create mask for the detected region ---
    region_mask = np.zeros(out.shape[:2], np.uint8)
    cv2.fillPoly(region_mask, [np.int32(proj)], 255)

    # --- Apply Gaussian blur to only detected object area ---
    blurred = cv2.GaussianBlur(out, (41, 41), 10)
    out = np.where(region_mask[..., None] == 255, blurred, out)

    return out, True

# -------------------------
# Multi-template handler
# -------------------------
def run_multi_template_match(scene_bgr, template_images):
    canvas = scene_bgr.copy()
    boxes = []
    for i, temp in enumerate(template_images):
        print(f"\n[🔍 Matching template {i+1}]")
        cropped = autocrop_template(temp)
        color = pick_color(i)
        before = canvas.copy()
        canvas, ok = detect_in_scene(canvas, cropped, color=color)
        if ok:
            diff = cv2.absdiff(before, canvas)
            gray = cv2.cvtColor(diff, cv2.COLOR_BGR2GRAY)
            cnts, _ = cv2.findContours(gray, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            for c in cnts:
                x, y, w, h = cv2.boundingRect(c)
                boxes.append([x, y, x+w, y+h])
        print(f"[{'OK' if ok else 'MISS'}] Template {i+1}")
    return canvas, boxes

# ============================================================
# Reconstruction (Fourier / Wiener deblurring)
# ============================================================
def make_gaussian_psf(ksize=71, sigma=10):
    if ksize % 2 == 0: ksize += 1
    psf_1d = cv2.getGaussianKernel(ksize, sigma)
    psf = psf_1d @ psf_1d.T
    psf /= psf.sum()
    return psf

def wiener_deconv_fft(channel, psf, K=0.02):
    H, W = channel.shape
    kh, kw = psf.shape
    psf_padded = np.zeros((H, W), np.float32)
    psf_padded[:kh, :kw] = psf
    psf_padded = np.roll(psf_padded, -kh//2, axis=0)
    psf_padded = np.roll(psf_padded, -kw//2, axis=1)
    G = np.fft.fft2(channel)
    Hf = np.fft.fft2(psf_padded)
    F_est = (np.conj(Hf)/(Hf*np.conj(Hf)+K))*G
    rec = np.real(np.fft.ifft2(F_est))
    rec = cv2.normalize(rec, None, 0, 255, cv2.NORM_MINMAX)
    return rec.astype(np.uint8)
def reconstruct_blurred_regions(img_bgr, boxes, ksize=71, sigma=10, K=0.02):
    """
    Reconstruct blurred regions using Wiener deconvolution.
    Fully replaces blurred region (NO blending → avoids gray overlay).
    """

    print("\n[🔁 Reconstructing blurred regions...]")
    psf = make_gaussian_psf(ksize, sigma)
    psf_h, psf_w = psf.shape

    out = img_bgr.copy()

    for (x1, y1, x2, y2) in boxes:

        patch = img_bgr[y1:y2, x1:x2]

        if patch.size == 0:
            continue

        ph, pw = patch.shape[:2]

        # Ensure patch ≥ PSF size
        if ph < psf_h or pw < psf_w:
            scale_y = max(1.0, psf_h / ph)
            scale_x = max(1.0, psf_w / pw)
            scale = max(scale_x, scale_y)
            new_w = int(pw * scale) + 1
            new_h = int(ph * scale) + 1
            patch = cv2.resize(patch, (new_w, new_h), interpolation=cv2.INTER_CUBIC)
            print(f"[⚠️ Patch too small → resized to {new_w}×{new_h}]")

        # Apply Wiener deconv per channel
        r, g, b = cv2.split(patch)
        r_rec = wiener_deconv_fft(r, psf, K)
        g_rec = wiener_deconv_fft(g, psf, K)
        b_rec = wiener_deconv_fft(b, psf, K)
        rec = cv2.merge([r_rec, g_rec, b_rec])

        # Crop if patch was resized
        h_rec, w_rec = rec.shape[:2]
        h_use = min(h_rec, y2 - y1)
        w_use = min(w_rec, x2 - x1)

        # ----------------------------
        # FULL OVERWRITE (no blending)
        # ----------------------------
        out[y1:y1+h_use, x1:x1+w_use] = rec[:h_use, :w_use]

    print("[✅ Reconstruction complete]")
    return out
