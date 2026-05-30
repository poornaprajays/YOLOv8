/**
 * YOLOv8 Explorer v2 — Frontend JavaScript
 * ==========================================
 * Organized into clear sections:
 *   1. Constants & State
 *   2. Sidebar Navigation
 *   3. Confidence Sliders
 *   4. Image Detection (upload, detect, render)
 *   5. Download (image + JSON)
 *   6. Detection Results (list, distribution chart)
 *   7. Model Switcher
 *   8. Webcam (stream, snapshot, gallery)
 *   9. History (list, detail overlay)
 *  10. Learn Cards (accordion)
 *  11. Toast Notifications
 *  12. Init
 */

"use strict";

/* ════════════════════════════════════════════════════════
   1. CONSTANTS & STATE
   ════════════════════════════════════════════════════════ */

// 20 visually distinct colors for COCO classes
const CLASS_COLORS = [
  "#38bdf8","#818cf8","#34d399","#f472b6","#fb923c",
  "#a78bfa","#4ade80","#fbbf24","#60a5fa","#e879f9",
  "#2dd4bf","#f87171","#a3e635","#fb7185","#c084fc",
  "#22d3ee","#86efac","#fde68a","#93c5fd","#f9a8d4",
];

function classColor(id) {
  return CLASS_COLORS[id % CLASS_COLORS.length];
}

// Mutable state
const State = {
  selectedFile:     null,
  lastResultData:   null,   // Full detection response for download
  currentModel:     "yolov8n.pt",
  webcamActive:     false,
  historyCount:     0,
};

/* ════════════════════════════════════════════════════════
   2. SIDEBAR NAVIGATION
   ════════════════════════════════════════════════════════ */

const PANEL_TITLES = {
  detect:  "Image Detection",
  webcam:  "Live Webcam",
  history: "Detection History",
  learn:   "Learn",
  models:  "Model Comparison",
};

document.querySelectorAll(".nav-item").forEach(btn => {
  btn.addEventListener("click", () => {
    const target = btn.dataset.panel;

    // Update nav buttons
    document.querySelectorAll(".nav-item").forEach(b => {
      b.classList.toggle("active", b === btn);
    });

    // Update panels
    document.querySelectorAll(".panel").forEach(p => {
      p.classList.toggle("active", p.id === `panel-${target}`);
    });

    // Update topbar title
    document.getElementById("topbar-title").textContent = PANEL_TITLES[target] || "";

    // Lazy-load history when switching to it
    if (target === "history") refreshHistory();
  });
});

// Sidebar toggle (collapse/expand)
const sidebar    = document.getElementById("sidebar");
const sidebarBtn = document.getElementById("sidebar-toggle");

sidebarBtn.addEventListener("click", () => {
  // Mobile: toggle class; Desktop: toggle width via class
  if (window.innerWidth <= 640) {
    sidebar.classList.toggle("mobile-open");
  } else {
    sidebar.classList.toggle("collapsed");
  }
});

/* ════════════════════════════════════════════════════════
   3. CONFIDENCE SLIDERS
   ════════════════════════════════════════════════════════ */

function initSlider(sliderId, displayId) {
  const slider  = document.getElementById(sliderId);
  const display = document.getElementById(displayId);
  if (!slider || !display) return;

  function update() {
    const val = parseFloat(slider.value) / 100;
    display.textContent = val.toFixed(2);
    // Drive the CSS gradient via custom property
    const pct = slider.value + "%";
    slider.style.background =
      `linear-gradient(90deg, var(--accent) ${pct}, var(--border-h) ${pct})`;
  }

  slider.addEventListener("input", update);
  update();
}

initSlider("conf-slider",        "conf-value");
initSlider("webcam-conf-slider", "webcam-conf-value");

/* ════════════════════════════════════════════════════════
   4. IMAGE DETECTION
   ════════════════════════════════════════════════════════ */

const dropZone    = document.getElementById("drop-zone");
const fileInput   = document.getElementById("file-input");
const previewWrap = document.getElementById("preview-wrap");
const previewImg  = document.getElementById("preview-img");
const btnDetect   = document.getElementById("btn-detect");
const btnClear    = document.getElementById("btn-clear");

// Click to open file browser
dropZone.addEventListener("click", () => fileInput.click());
dropZone.addEventListener("keydown", e => { if (e.key === "Enter") fileInput.click(); });

// File selected
fileInput.addEventListener("change", e => {
  if (e.target.files[0]) loadPreview(e.target.files[0]);
});

// Drag & drop
dropZone.addEventListener("dragover",  e => { e.preventDefault(); dropZone.classList.add("over"); });
dropZone.addEventListener("dragleave", ()  => dropZone.classList.remove("over"));
dropZone.addEventListener("drop", e => {
  e.preventDefault();
  dropZone.classList.remove("over");
  const f = e.dataTransfer.files[0];
  if (f && f.type.startsWith("image/")) loadPreview(f);
  else toast("Please drop an image file (JPG, PNG, BMP, WebP)", "warn");
});

function loadPreview(file) {
  State.selectedFile = file;
  const reader = new FileReader();
  reader.onload = e => {
    previewImg.src = e.target.result;
    dropZone.style.display   = "none";
    previewWrap.style.display = "block";
    btnDetect.disabled = false;
    resetResults();
  };
  reader.readAsDataURL(file);
}

btnClear.addEventListener("click", () => {
  State.selectedFile   = null;
  State.lastResultData = null;
  fileInput.value      = "";
  previewImg.src       = "";
  previewWrap.style.display = "none";
  dropZone.style.display    = "block";
  btnDetect.disabled = true;
  resetResults();
});

btnDetect.addEventListener("click", runDetection);

async function runDetection() {
  if (!State.selectedFile) return;

  const conf = parseFloat(document.getElementById("conf-slider").value) / 100;

  setStatus("busy", "Detecting…");
  showLoading(true);

  const form = new FormData();
  form.append("image", State.selectedFile);
  form.append("confidence", conf);

  try {
    const res  = await fetch("/detect/image", { method: "POST", body: form });
    const data = await res.json();

    if (!res.ok || data.error) {
      toast(`Detection failed: ${data.error}`, "danger");
      return;
    }

    State.lastResultData = data;
    renderResults(data);
    updateHistoryBadge(data.history_count);

  } catch (err) {
    toast(`Network error: ${err.message}`, "danger");
  } finally {
    showLoading(false);
    setStatus("ready", "Ready");
  }
}

/* ════════════════════════════════════════════════════════
   5. DOWNLOADS
   ════════════════════════════════════════════════════════ */

document.getElementById("btn-dl-img").addEventListener("click", downloadImage);
document.getElementById("btn-dl-json").addEventListener("click", downloadJSON);

function downloadImage() {
  if (!State.lastResultData?.annotated_image) return;
  const a = document.createElement("a");
  a.href     = State.lastResultData.annotated_image;
  a.download = `yolov8_detection_${Date.now()}.jpg`;
  a.click();
}

function downloadJSON() {
  if (!State.lastResultData) return;
  const payload = {
    model:       State.lastResultData.model,
    timestamp:   new Date().toISOString(),
    count:       State.lastResultData.count,
    inference_ms:State.lastResultData.inference_ms,
    class_summary: State.lastResultData.class_summary,
    detections:  State.lastResultData.detections,
  };
  const blob = new Blob([JSON.stringify(payload, null, 2)], { type: "application/json" });
  const a    = document.createElement("a");
  a.href     = URL.createObjectURL(blob);
  a.download = `yolov8_results_${Date.now()}.json`;
  a.click();
  URL.revokeObjectURL(a.href);
}

/* ════════════════════════════════════════════════════════
   6. DETECTION RESULTS RENDERING
   ════════════════════════════════════════════════════════ */

function renderResults(data) {
  // Show result image
  document.getElementById("result-img").src = data.annotated_image;

  // Metrics
  document.getElementById("stat-count").textContent       = data.count;
  document.getElementById("stat-ms").textContent          = data.inference_ms + "ms";
  document.getElementById("stat-classes").textContent     = Object.keys(data.class_summary || {}).length;
  document.getElementById("stat-model-badge").textContent = (data.model || "n").replace("yolov8","").replace(".pt","");

  // Detection list
  renderDetectionList(data.detections);

  // Distribution chart
  renderDistChart(data.class_summary);

  // Show results section
  document.getElementById("results-empty").style.display = "none";
  document.getElementById("results-fill").style.display  = "block";
}

function renderDetectionList(detections) {
  const list = document.getElementById("detection-list");
  list.innerHTML = "";

  if (!detections || detections.length === 0) {
    list.innerHTML = `<p style="color:var(--txt-3);font-size:.8rem;padding:12px 0;">No objects detected above threshold.</p>`;
    return;
  }

  detections.forEach((det, i) => {
    const color = classColor(det.class_id);
    const pct   = Math.round(det.confidence * 100);
    const item  = document.createElement("div");
    item.className = "det-item";
    item.style.setProperty("--c", color);
    item.style.animationDelay = `${i * 30}ms`;
    item.innerHTML = `
      <span class="det-rank">${i + 1}</span>
      <span class="det-name">${det.class_name}</span>
      <div class="det-bar"><div class="det-fill" style="width:${pct}%"></div></div>
      <span class="det-conf">${det.confidence_pct}</span>
    `;
    list.appendChild(item);
  });
}

function renderDistChart(classSummary) {
  const chart = document.getElementById("dist-chart");
  chart.innerHTML = "";

  if (!classSummary || Object.keys(classSummary).length === 0) {
    chart.innerHTML = `<p style="color:var(--txt-3);font-size:.8rem;">—</p>`;
    return;
  }

  const entries = Object.entries(classSummary).sort((a, b) => b[1] - a[1]);
  const maxVal  = entries[0][1];

  entries.forEach(([cls, cnt], i) => {
    const pct  = Math.round((cnt / maxVal) * 100);
    const row  = document.createElement("div");
    row.className = "dc-row";
    row.style.animationDelay = `${i * 40}ms`;
    row.innerHTML = `
      <span class="dc-name" title="${cls}">${cls}</span>
      <div class="dc-bar-wrap">
        <div class="dc-bar" style="width:${pct}%"><span>${cnt}</span></div>
      </div>
    `;
    chart.appendChild(row);
  });
}

function resetResults() {
  document.getElementById("results-empty").style.display = "block";
  document.getElementById("results-fill").style.display  = "none";
  document.getElementById("detection-list").innerHTML    = "";
  document.getElementById("dist-chart").innerHTML        = "";
}

/* ════════════════════════════════════════════════════════
   7. MODEL SWITCHER
   ════════════════════════════════════════════════════════ */

document.getElementById("model-switcher").addEventListener("change", async function () {
  const model = this.value;
  if (model === State.currentModel) return;

  const loadingWrap = document.getElementById("model-loading-wrap");
  loadingWrap.style.display = "flex";
  this.disabled = true;
  setStatus("busy", "Loading model…");

  try {
    const res  = await fetch("/api/set-model", {
      method:  "POST",
      headers: { "Content-Type": "application/json" },
      body:    JSON.stringify({ model }),
    });
    const data = await res.json();

    if (data.success) {
      State.currentModel = model;
      const shortName = model.replace("yolov8", "").replace(".pt", "");

      // Update sidebar model display
      document.getElementById("sf-model-name").textContent = model.replace(".pt", "");

      // Update "active" tags in the models table
      document.querySelectorAll(".active-tag").forEach(el => el.style.display = "none");
      const activeTag = document.getElementById(`active-tag-${shortName}`);
      if (activeTag) activeTag.style.display = "";

      // Highlight current row in table
      document.querySelectorAll(".spec-table tr[data-model]").forEach(row => {
        row.classList.toggle("current-model-row", row.dataset.model === shortName);
      });

      toast(`Switched to ${model.replace(".pt", "")} (loaded in ${data.load_ms}ms)`, "success");
    } else {
      toast(`Failed to load model: ${data.error}`, "danger");
      this.value = State.currentModel; // Revert
    }
  } catch (err) {
    toast(`Network error: ${err.message}`, "danger");
    this.value = State.currentModel;
  } finally {
    loadingWrap.style.display = "none";
    this.disabled = false;
    setStatus("ready", "Ready");
  }
});

/* ════════════════════════════════════════════════════════
   8. WEBCAM
   ════════════════════════════════════════════════════════ */

const webcamImg   = document.getElementById("webcam-img");
const webcamIdle  = document.getElementById("webcam-idle");
const btnStart    = document.getElementById("btn-webcam-start");
const btnStop     = document.getElementById("btn-webcam-stop");
const btnSnap     = document.getElementById("btn-snapshot");

btnStart.addEventListener("click", startWebcam);
btnStop.addEventListener("click",  stopWebcam);
btnSnap.addEventListener("click",  takeSnapshot);

function startWebcam() {
  const conf = parseFloat(document.getElementById("webcam-conf-slider").value) / 100;
  const url  = `/webcam/stream?confidence=${conf}&t=${Date.now()}`;

  webcamImg.src          = url;
  webcamImg.style.display = "block";
  webcamIdle.style.display = "none";

  State.webcamActive    = true;
  btnStart.disabled     = true;
  btnStop.disabled      = false;
  btnSnap.disabled      = false;

  setStatus("streaming", "Streaming");
}

async function stopWebcam() {
  State.webcamActive    = false;
  webcamImg.src          = "";
  webcamImg.style.display = "none";
  webcamIdle.style.display = "block";

  btnStart.disabled = false;
  btnStop.disabled  = true;
  btnSnap.disabled  = true;

  setStatus("ready", "Ready");

  try { await fetch("/webcam/stop", { method: "POST" }); } catch (_) {}
}

async function takeSnapshot() {
  if (!State.webcamActive) return;

  const conf = parseFloat(document.getElementById("webcam-conf-slider").value) / 100;
  btnSnap.disabled = true;

  try {
    const res  = await fetch("/webcam/snapshot", {
      method:  "POST",
      headers: { "Content-Type": "application/json" },
      body:    JSON.stringify({ confidence: conf }),
    });
    const data = await res.json();

    if (data.success) {
      addSnapshotToGallery(data);
      toast(`📸 Snapshot: ${data.count} object${data.count !== 1 ? "s" : ""} detected`, "success");
    } else {
      toast(data.error || "Snapshot failed", "warn");
    }
  } catch (err) {
    toast(`Snapshot error: ${err.message}`, "danger");
  } finally {
    btnSnap.disabled = false;
  }
}

function addSnapshotToGallery(data) {
  const list    = document.getElementById("snapshot-list");
  const empty   = document.getElementById("snap-empty");
  if (empty) empty.style.display = "none";

  const topClass = data.class_summary ? Object.keys(data.class_summary)[0] || "—" : "—";
  const item     = document.createElement("div");
  item.className = "snap-item";
  item.innerHTML = `
    <img src="${data.thumbnail}" class="snap-thumb" alt="snapshot" />
    <div class="snap-info">
      <div class="snap-count">${data.count} object${data.count !== 1 ? "s" : ""}</div>
      <div class="snap-time">${data.timestamp} · ${topClass}</div>
    </div>
  `;

  // Click to expand full image
  item.addEventListener("click", () => showSnapshotFull(data));
  list.insertBefore(item, list.firstChild);
}

function showSnapshotFull(data) {
  // Reuse history detail overlay
  document.getElementById("hd-img").src = data.annotated_image;
  document.getElementById("hd-title").textContent = "Webcam Snapshot";
  document.getElementById("hd-meta").innerHTML = `
    <span>Time: ${data.timestamp}</span>
    <span>Objects: ${data.count}</span>
    <span>Inference: ${data.inference_ms}ms</span>
  `;
  document.getElementById("history-detail").style.display = "flex";
}

/* ════════════════════════════════════════════════════════
   9. HISTORY
   ════════════════════════════════════════════════════════ */

async function refreshHistory() {
  try {
    const res  = await fetch("/api/history");
    const data = await res.json();
    renderHistory(data.history);
    updateHistoryBadge(data.count);
  } catch (err) {
    console.error("History fetch failed:", err);
  }
}

function renderHistory(history) {
  const grid  = document.getElementById("history-grid");
  const empty = document.getElementById("history-empty");

  grid.innerHTML = "";

  if (!history || history.length === 0) {
    grid.appendChild(empty);
    empty.style.display = "block";
    return;
  }

  empty.style.display = "none";
  history.forEach((item, i) => {
    const card = document.createElement("div");
    card.className = "history-item";
    card.style.animationDelay = `${i * 40}ms`;
    card.innerHTML = `
      <img src="${item.thumbnail}" class="hi-thumb" alt="Detection thumbnail" />
      <div class="hi-body">
        <div class="hi-count">${item.count}<span>objects</span></div>
        <div class="hi-top-class">${item.top_class}</div>
        <div class="hi-meta">
          <span>${item.date} ${item.timestamp}</span>
          <span>${item.inference_ms}ms</span>
        </div>
      </div>
    `;
    card.addEventListener("click", () => showHistoryDetail(item));
    grid.appendChild(card);
  });
}

function showHistoryDetail(item) {
  document.getElementById("hd-img").src             = item.full_image;
  document.getElementById("hd-title").textContent   = `Detection — ${item.date} ${item.timestamp}`;
  document.getElementById("hd-meta").innerHTML = `
    <span>Model: ${item.model}</span>
    <span>Objects: ${item.count}</span>
    <span>Inference: ${item.inference_ms}ms</span>
    <span>Top: ${item.top_class}</span>
  `;
  document.getElementById("history-detail").style.display = "flex";
}

// Close overlay
document.getElementById("hd-close").addEventListener("click", () => {
  document.getElementById("history-detail").style.display = "none";
});
document.getElementById("history-detail").addEventListener("click", function (e) {
  if (e.target === this) this.style.display = "none";
});

// Clear history
document.getElementById("btn-clear-history").addEventListener("click", async () => {
  await fetch("/api/history", { method: "DELETE" });
  renderHistory([]);
  updateHistoryBadge(0);
  toast("History cleared", "success");
});

function updateHistoryBadge(count) {
  State.historyCount = count;
  const badge = document.getElementById("history-badge");
  badge.textContent    = count;
  badge.style.display  = count > 0 ? "inline-flex" : "none";
}

/* ════════════════════════════════════════════════════════
   10. LEARN CARDS (ACCORDION)
   ════════════════════════════════════════════════════════ */

function toggleCard(id) {
  const card   = document.getElementById(id);
  const body   = card.querySelector(".lc-body");
  const header = card.querySelector(".lc-header");
  const isOpen = body.classList.contains("open");

  body.classList.toggle("open", !isOpen);
  header.setAttribute("aria-expanded", !isOpen);
}

// Open first card by default (already open via HTML class)
window.toggleCard = toggleCard; // Expose to onclick

/* ════════════════════════════════════════════════════════
   11. STATUS HELPERS
   ════════════════════════════════════════════════════════ */

function setStatus(state, label) {
  const dot  = document.getElementById("status-dot");
  const text = document.getElementById("status-text");
  dot.className  = "status-dot " + (state !== "ready" ? state : "");
  text.textContent = label;
}

function showLoading(show) {
  document.getElementById("loading-overlay").style.display = show ? "flex" : "none";
}

/* ════════════════════════════════════════════════════════
   12. TOAST NOTIFICATIONS
   ════════════════════════════════════════════════════════ */

function toast(message, type = "info") {
  document.querySelectorAll(".x-toast").forEach(t => t.remove());

  const colors = {
    info:    { bg: "rgba(108,99,255,.92)", icon: "ℹ️" },
    success: { bg: "rgba(0,229,160,.88)",  icon: "✅" },
    warn:    { bg: "rgba(255,179,71,.92)", icon: "⚠️" },
    danger:  { bg: "rgba(255,77,106,.92)", icon: "❌" },
  };
  const { bg, icon } = colors[type] || colors.info;

  const el = document.createElement("div");
  el.className = "x-toast";
  Object.assign(el.style, {
    position:     "fixed",
    bottom:       "24px",
    right:        "24px",
    background:   bg,
    backdropFilter: "blur(12px)",
    color:        "#fff",
    padding:      "11px 18px",
    borderRadius: "12px",
    fontFamily:   "var(--font, sans-serif)",
    fontWeight:   "600",
    fontSize:     ".85rem",
    zIndex:       "9999",
    display:      "flex",
    alignItems:   "center",
    gap:          "10px",
    boxShadow:    "0 8px 28px rgba(0,0,0,.35)",
    maxWidth:     "340px",
    animation:    "fade-up .25s ease",
  });
  el.innerHTML = `<span>${icon}</span><span>${message}</span>`;
  document.body.appendChild(el);

  setTimeout(() => {
    el.style.opacity    = "0";
    el.style.transition = "opacity .3s";
    setTimeout(() => el.remove(), 300);
  }, 3500);
}

/* ════════════════════════════════════════════════════════
   12. INIT
   ════════════════════════════════════════════════════════ */

(async function init() {
  // Fetch model info to display current state
  try {
    const res  = await fetch("/api/model-info");
    const data = await res.json();
    const name = (data.model || "yolov8n.pt").replace(".pt", "");
    document.getElementById("sf-model-name").textContent = name;
    const switcher = document.getElementById("model-switcher");
    if (switcher) switcher.value = data.model || "yolov8n.pt";
    State.currentModel = data.model || "yolov8n.pt";
  } catch (_) {}

  // Refresh history count
  try {
    const res  = await fetch("/api/history");
    const data = await res.json();
    updateHistoryBadge(data.count);
  } catch (_) {}

  console.log(
    "%c🎯 YOLOv8 Explorer v2%c loaded · Ultralytics + Flask + Vanilla JS",
    "color:#6c63ff;font-size:14px;font-weight:800",
    "color:#7b85ab;font-size:12px"
  );
})();
