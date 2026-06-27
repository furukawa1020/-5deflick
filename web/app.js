const state = {
  settings: {},
  calibrated: false,
  recalibrating: false,
  markerSampling: false,
  lastEventAt: 0,
  initializedEvents: false,
};

const $ = (id) => document.getElementById(id);

let audioContext = null;

function unlockAudio() {
  if (audioContext) return;
  const AudioContextClass = window.AudioContext || window.webkitAudioContext;
  if (!AudioContextClass) return;
  audioContext = new AudioContextClass();
}

function playTone(kind = "text") {
  if (!$("sound-enabled").checked) return;
  unlockAudio();
  if (!audioContext) return;
  if (audioContext.state === "suspended") audioContext.resume();

  const now = audioContext.currentTime;
  const oscillator = audioContext.createOscillator();
  const gain = audioContext.createGain();
  const frequency = kind === "text" ? 880 : kind === "key" ? 520 : 660;
  oscillator.type = "sine";
  oscillator.frequency.setValueAtTime(frequency, now);
  gain.gain.setValueAtTime(0.0001, now);
  gain.gain.exponentialRampToValueAtTime(0.12, now + 0.01);
  gain.gain.exponentialRampToValueAtTime(0.0001, now + 0.09);
  oscillator.connect(gain);
  gain.connect(audioContext.destination);
  oscillator.start(now);
  oscillator.stop(now + 0.1);
}

async function postJSON(path, payload = {}) {
  const response = await fetch(path, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(payload),
  });
  return response.json();
}

function setControlValues(settings, sendUnicode) {
  const pairs = [
    ["input-mode", settings.input_mode],
    ["arm-mode", settings.arm_mode],
    ["direction-mode", settings.direction_mode],
    ["motion-threshold", settings.motion_threshold],
    ["min-motion-area", settings.min_motion_area],
    ["deadzone", settings.deadzone],
    ["marker-min-area", settings.marker_min_area],
  ];
  for (const [id, value] of pairs) {
    const node = $(id);
    if (node && document.activeElement !== node) node.value = value;
  }
  $("send-unicode").checked = sendUnicode;
  $("motion-threshold-value").textContent = settings.motion_threshold;
  $("min-motion-area-value").textContent = settings.min_motion_area;
  $("deadzone-value").textContent = Number(settings.deadzone).toFixed(2);
  $("marker-min-area-value").textContent = settings.marker_min_area;
  $("mask-mode").textContent = settings.input_mode === "mediapipe" ? "landmarks" : (settings.input_mode === "marker" ? "marker" : settings.arm_mode);
  $("sample-marker").classList.toggle("active", state.markerSampling);
  $("sample-marker").textContent = state.markerSampling ? "クリック待ち" : "マーカー";
}

function renderEvents(events) {
  $("events").replaceChildren(...events.map((event) => {
    const item = document.createElement("div");
    item.className = "event";
    const value = event.kind === "key" ? `<${event.value}>` : event.value;
    item.textContent = value;
    return item;
  }));
}

function playNewEventSounds(events) {
  const ordered = [...events].sort((a, b) => (a.at || 0) - (b.at || 0));
  if (!state.initializedEvents) {
    state.lastEventAt = ordered.length ? ordered[ordered.length - 1].at || 0 : 0;
    state.initializedEvents = true;
    return;
  }
  for (const event of ordered) {
    if ((event.at || 0) <= state.lastEventAt) continue;
    if (event.kind === "text") playTone("text");
    else if (event.kind === "key") playTone("key");
    state.lastEventAt = Math.max(state.lastEventAt, event.at || 0);
  }
}

async function refreshState() {
  const response = await fetch("/api/state", { cache: "no-store" });
  const data = await response.json();
  state.settings = data.settings;
  state.calibrated = data.calibrated;
  state.recalibrating = data.recalibrating;
  $("status").textContent = data.status;
  $("corner-state").textContent = data.recalibrating ? `${data.calibrationPoints}/4` : "ready";
  $("last-label").textContent = data.lastLabel || "";
  $("output").value = data.outputText || "";
  setControlValues(data.settings, data.sendUnicode);
  renderEvents(data.events || []);
  playNewEventSounds(data.events || []);
}

function scaledClick(event) {
  const image = $("camera");
  const rect = image.getBoundingClientRect();
  const naturalWidth = image.naturalWidth || rect.width;
  const naturalHeight = image.naturalHeight || rect.height;
  return {
    x: (event.clientX - rect.left) * naturalWidth / rect.width,
    y: (event.clientY - rect.top) * naturalHeight / rect.height,
  };
}

function settingPayload() {
  return {
    arm_mode: $("arm-mode").value,
    input_mode: $("input-mode").value,
    direction_mode: $("direction-mode").value,
    motion_threshold: Number($("motion-threshold").value),
    min_motion_area: Number($("min-motion-area").value),
    deadzone: Number($("deadzone").value),
    marker_min_area: Number($("marker-min-area").value),
    send_unicode: $("send-unicode").checked,
  };
}

function bind() {
  document.addEventListener("pointerdown", unlockAudio, { once: true });
  $("camera").addEventListener("click", async (event) => {
    if (state.markerSampling) {
      await postJSON("/api/marker/sample", scaledClick(event));
      state.markerSampling = false;
      await refreshState();
      return;
    }
    if (!state.recalibrating) return;
    await postJSON("/api/calibration/click", scaledClick(event));
    await refreshState();
  });
  $("recalibrate").addEventListener("click", async () => {
    state.markerSampling = false;
    await postJSON("/api/calibration/reset");
    await refreshState();
  });
  $("sample-marker").addEventListener("click", async () => {
    state.markerSampling = true;
    await refreshState();
  });
  $("reset-bg").addEventListener("click", async () => {
    await postJSON("/api/background/reset");
    await refreshState();
  });
  $("clear-output").addEventListener("click", async () => {
    await postJSON("/api/output/clear");
    await refreshState();
  });
  $("test-output").addEventListener("click", async () => {
    await postJSON("/api/output/test");
    await refreshState();
  });
  for (const id of ["input-mode", "arm-mode", "direction-mode", "motion-threshold", "min-motion-area", "deadzone", "marker-min-area", "send-unicode"]) {
    $(id).addEventListener("input", async () => {
      await postJSON("/api/settings", settingPayload());
      await refreshState();
    });
  }
}

bind();
refreshState();
setInterval(refreshState, 250);
