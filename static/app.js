// --------------------
// Canvas + Globals
// --------------------
let canvas = document.getElementById("imgCanvas");
let ctx = canvas.getContext("2d");
let img = new Image();
let points = [];

const fileInput = document.getElementById("objFile");
const measureBtn = document.getElementById("measureBtn");
const clearBtn = document.getElementById("clearPoints");
const annotatedImg = document.getElementById("annotated");
const messages = document.getElementById("messages");

// --------------------
// Image upload
// --------------------
fileInput?.addEventListener("change", async (e) => {
  const f = e.target.files[0];
  if (!f) return;
  const url = URL.createObjectURL(f);

  img.onload = () => {
    const maxWidth = 900;
    let scale = img.width > maxWidth ? maxWidth / img.width : 1;

    canvas.width = img.width * scale;
    canvas.height = img.height * scale;

    ctx.drawImage(img, 0, 0, canvas.width, canvas.height);
    points = [];
    drawPoints();
    messages.textContent = "";
  };

  img.src = url;
  annotatedImg.src = "";
});

// --------------------
// Canvas Clicks
// --------------------
canvas?.addEventListener("click", (ev) => {
  if (!img.src) {
    alert("Please upload an image first.");
    return;
  }

  const rect = canvas.getBoundingClientRect();
  const x = ev.clientX - rect.left;
  const y = ev.clientY - rect.top;

  const scaleX = img.width / canvas.width;
  const scaleY = img.height / canvas.height;

  const imgX = x * scaleX;
  const imgY = y * scaleY;

  points.push([imgX, imgY]);
  if (points.length > 4) points.shift();
  drawPoints();

  let coordsText = "Clicked points (x, y):\n";
  for (let i = 0; i < points.length; i++) {
    coordsText += `(${points[i][0].toFixed(1)}, ${points[i][1].toFixed(1)})\n`;
  }
  messages.textContent = coordsText;
});

// --------------------
// Draw Clicked Points
// --------------------
function drawPoints() {
  if (!canvas || !ctx) return;
  ctx.clearRect(0, 0, canvas.width, canvas.height);
  if (img.src) {
    ctx.drawImage(img, 0, 0, canvas.width, canvas.height);
  }

  if (!points) return;

  const scaleX = canvas.width / img.width;
  const scaleY = canvas.height / img.height;

  ctx.fillStyle = "red";
  ctx.strokeStyle = "lime";
  ctx.lineWidth = 2;

  for (let i = 0; i < points.length; i++) {
    const [px, py] = points[i];
    const cx = px * scaleX;
    const cy = py * scaleY;
    ctx.beginPath();
    ctx.arc(cx, cy, 6, 0, Math.PI * 2);
    ctx.fill();
  }

  if (points.length === 4) {
    ctx.beginPath();
    ctx.moveTo(points[0][0] * scaleX, points[0][1] * scaleY);
    for (let i = 1; i < 4; i++) {
      ctx.lineTo(points[i][0] * scaleX, points[i][1] * scaleY);
    }
    ctx.closePath();
    ctx.stroke();
  }
}

// --------------------
// Clear Points
// --------------------
clearBtn?.addEventListener("click", () => {
  points = [];
  if (annotatedImg) annotatedImg.src = "";
  drawPoints();
  messages.textContent = "";
});

// --------------------
// Measure  (OPTION B: always send fx, fy)
// --------------------
measureBtn?.addEventListener("click", async () => {
  messages.textContent = "";
  if (!img.src) {
    alert("Upload an image first");
    return;
  }
  if (points.length !== 4) {
    alert("Click exactly 4 corners (any order).");
    return;
  }

  const fd = new FormData();

  if (fileInput && fileInput.files && fileInput.files[0]) {
    fd.append("image", fileInput.files[0], fileInput.files[0].name);
  } else {
    const blob = await new Promise((r) => canvas.toBlob(r, "image/png"));
    fd.append("image", blob, "capture.png");
  }

  fd.append("points", JSON.stringify(points));

  const useCalibCheckbox = document.getElementById("useCalib");
  const useCalib = useCalibCheckbox ? useCalibCheckbox.checked : true;
  fd.append("use_calib", useCalib ? "true" : "false");

  // 🔵 OPTION B: always use fx/fy from inputs (either from calibration or manual)
  const fxInput = document.getElementById("fxInput");
  const fyInput = document.getElementById("fyInput");
  const fx = fxInput ? fxInput.value : "";
  const fy = fyInput ? fyInput.value : "";

  if (!fx || !fy) {
    if (useCalib) {
      alert("Run calibration first so fx and fy are auto-filled.");
    } else {
      alert("Enter fx and fy (or enable calibration).");
    }
    return;
  }

  fd.append("fx", fx);
  fd.append("fy", fy);

  const distanceInput = document.getElementById("distanceInput");
  const distance = distanceInput ? distanceInput.value : "";
  if (!distance) {
    alert("Enter distance (cm)");
    return;
  }
  fd.append("distance_cm", distance);

  messages.textContent = "Measuring...";

  try {
    const resp = await fetch("/measure", { method: "POST", body: fd });
    const j = await resp.json();

    if (!resp.ok) {
      messages.textContent = "Error: " + (j.error || JSON.stringify(j));
      return;
    }

    messages.textContent =
      `Pixel width: ${j.pixel_width.toFixed(1)} px\n` +
      `Pixel height: ${j.pixel_height.toFixed(1)} px\n` +
      `Estimated W: ${j.estimated_width_cm.toFixed(2)} cm\n` +
      `Estimated H: ${j.estimated_height_cm.toFixed(2)} cm\n` +
      `(using fx=${j.fx}, fy=${j.fy})`;

  } catch (err) {
    messages.textContent = "Request failed: " + err;
  }
});

// --------------------
// Calibration
// --------------------
const runCalibBtn = document.getElementById("runCalib");
const calibForm = document.getElementById("calibForm");
const calibFiles = document.getElementById("calibFiles");
const calibResult = document.getElementById("calibResult");

calibForm?.addEventListener("submit", async (ev) => {
  ev.preventDefault();
  const fd = new FormData();

  if (!calibFiles.files.length) {
    alert("Select chessboard images first");
    return;
  }

  for (let i = 0; i < calibFiles.files.length; i++) {
    fd.append("files[]", calibFiles.files[i], calibFiles.files[i].name);
  }

  fd.append("pattern_cols", calibForm.pattern_cols.value);
  fd.append("pattern_rows", calibForm.pattern_rows.value);
  fd.append("square_size", 1.0);

  runCalibBtn.disabled = true;
  calibResult.textContent = "Running calibration...";

  try {
    const res = await fetch("/calibrate", { method: "POST", body: fd });
    const j = await res.json();

    if (!res.ok) {
      calibResult.textContent =
        "Calibration failed: " + (j.error || JSON.stringify(j));
    } else {
      calibResult.innerHTML = `
        <b>Calibration done</b><br>
        fx=${j.fx.toFixed(2)}, fy=${j.fy.toFixed(2)}<br>
        cx=${j.cx.toFixed(2)}, cy=${j.cy.toFixed(2)}<br>
      `;

      // 🔵 Auto-fill fx, fy inputs for measurement step
      const fxInput = document.getElementById("fxInput");
      const fyInput = document.getElementById("fyInput");
      if (fxInput) fxInput.value = j.fx.toFixed(2);
      if (fyInput) fyInput.value = j.fy.toFixed(2);
    }
  } catch (err) {
    calibResult.textContent = "Request failed: " + err;
  } finally {
    runCalibBtn.disabled = false;
  }
});

// ----------------------------
// Panorama Stitching (if present elsewhere)
// ----------------------------
const stitchBtn = document.getElementById("runStitch");
const panoImg = document.getElementById("panoramaResult");

if (stitchBtn && panoImg) {
  stitchBtn.addEventListener("click", async () => {
    try {
      const res = await fetch("/process_stitch");
      const data = await res.json();
      if (data.error) {
        alert("Error: " + data.error);
        return;
      }
      panoImg.src = "data:image/jpeg;base64," + data.stitched;
    } catch (err) {
      alert("Request failed: " + err);
    }
  });
}

// ----------------------------
// Face Unlock (if on current page)
// ----------------------------
const startCamBtn = document.getElementById("startCamBtn");
const captureAuthBtn = document.getElementById("captureAuthBtn");
const video = document.getElementById("video");
const captureCanvas = document.getElementById("captureCanvas");
const faceMsg = document.getElementById("faceMsg");
let stream = null;

startCamBtn?.addEventListener("click", async () => {
  try {
    if (stream) {
      stream.getTracks().forEach((t) => t.stop());
      stream = null;
      video.srcObject = null;
      startCamBtn.textContent = "Start Camera";
      return;
    }
    stream = await navigator.mediaDevices.getUserMedia({ video: true });
    video.srcObject = stream;
    startCamBtn.textContent = "Stop Camera";
  } catch (err) {
    alert("Cannot open camera: " + err);
  }
});

captureAuthBtn?.addEventListener("click", async () => {
  if (!stream) {
    alert("Start camera first");
    return;
  }
  const ctx2 = captureCanvas.getContext("2d");
  captureCanvas.width = video.videoWidth || 360;
  captureCanvas.height = video.videoHeight || 270;
  ctx2.drawImage(video, 0, 0, captureCanvas.width, captureCanvas.height);

  const blob = await new Promise((r) => captureCanvas.toBlob(r, "image/png"));
  const fd = new FormData();
  fd.append("image", blob, "capture.png");

  faceMsg.textContent = "Checking face...";

  try {
    const resp = await fetch("/auth_face", { method: "POST", body: fd });
    const j = await resp.json();

    if (!resp.ok) {
      faceMsg.textContent = "Error: " + (j.error || JSON.stringify(j));
      return;
    }

    faceMsg.textContent = j.match
      ? `✅ Unlocked (distance=${j.distance.toFixed(4)})`
      : `❌ Not recognized (distance=${j.distance.toFixed(4)})`;
  } catch (err) {
    faceMsg.textContent = "Request failed: " + err;
  }
});

// ----------------------------
// Logout
// ----------------------------
const logoutBtn = document.getElementById("logoutBtn");
logoutBtn?.addEventListener("click", () => {
  if (confirm("Log out?")) window.location.href = "/";
});
