// Configuration
const API = "http://localhost:8080";
const POLL_INTERVAL = 1000;   // ms
const MAX_POINTS = 20;        // data points per line
window.threshold = 25;   // initial Mbps - make it global for topology.js
let paused = false;

// A fixed, distinguishable palette
const PALETTE = [
  "#1f78b4", "#33a02c", "#e31a1c", "#ff7f00", "#6a3d9a",
  "#b15928", "#a6cee3", "#b2df8a", "#fb9a99", "#fdbf6f"
];
let paletteIndex = 0;
function nextColor() {
  return PALETTE[paletteIndex++ % PALETTE.length];
}

// Chart.js instance and dataset map
let chart;
const datasets = {};  // linkKey -> dataset object

// Initialize the Chart
function initChart() {
  const ctx = document.getElementById("loadChart").getContext("2d");
  chart = new Chart(ctx, {
    type: "line",
    data: { datasets: [] },
    options: {
      responsive: true,
      animation: false,
      scales: {
        x: {
          type: "time",
          time: { unit: "second", tooltipFormat: "HH:mm:ss" },
          title: { display: true, text: "Time" }
        },
        y: {
          beginAtZero: true,
          title: { display: true, text: "Bytes/sec" }
        }
      },
      plugins: {
        legend: { position: "top", align: "end" },
        title: { display: true, text: "Live Link Load (Moving Avg)" }
      }
    }
  });
}

// Fetch and update the chart data
async function updateChart() {
  if (paused) return;

  const now = new Date();
  const loads = await fetchJSON("/load/links");
  const hot = [];

  Object.entries(loads).forEach(([link, bps]) => {
    if (!datasets[link]) {
      const color = nextColor();
      const ds = {
        label: link,
        data: [],
        borderColor: color,
        backgroundColor: color,
        fill: false,
        tension: 0
      };
      chart.data.datasets.push(ds);
      datasets[link] = ds;
    }
    const ds = datasets[link];
    ds.data.push({ x: now, y: bps });
    if (ds.data.length > MAX_POINTS) ds.data.shift();
    if (bps > (window.threshold * 1000000)) hot.push({ link, bps });
  });

  chart.update();
  updateHotList(hot);
  updatePathsDisplay();
}

// Update the “hot links” list
function updateHotList(hot) {
  const ul = document.getElementById("hotLinks");
  ul.innerHTML = "";
  if (!hot.length) {
    ul.innerHTML = "<li class='ok'>None</li>";
    return;
  }
  hot.forEach(({ link, bps }) => {
    const li = document.createElement("li");
    li.className = "hot";
    li.textContent = `${link} : ${(bps / 1000000).toFixed(1)} Mbps`;
    ul.appendChild(li);
  });
}

// Fetch and display all active flow paths
async function updatePathsDisplay() {
  try {
    const pathsObj = await fetchJSON("/load/path");
    const pathContainer = document.getElementById("path");
    
    if (!pathsObj || Object.keys(pathsObj).length === 0) {
      pathContainer.innerHTML = "<li class='ok'>No active flows</li>";
      return;
    }
    
    // pathsObj is now an object: { "h1→h2": [1,2,3], ... }
    const lines = Object.entries(pathsObj).map(([flow, path]) => {
      if (!path || !Array.isArray(path) || path.length === 0) {
        return `${flow} : No path found`;
      }
      
      // Format path with switch names
      const formattedPath = path.map(dpid => `s${dpid}`).join(" → ");
      return `${flow} : ${formattedPath}`;
    });
    
    pathContainer.innerHTML = lines.map(line => `<li>${line}</li>`).join("");
  } catch (error) {
    console.error("Error updating paths display:", error);
    document.getElementById("path").innerHTML = "<li class='error'>Error loading paths</li>";
  }
}

// Simple fetch wrapper
async function fetchJSON(endpoint, opts) {
  const res = await fetch(API + endpoint, opts);
  return res.json();
}

// Wire up UI controls
function attachUI() {
  // Pause/Resume button
  const btn = document.getElementById("toggle");
  btn.onclick = () => {
    paused = !paused;
    btn.textContent = paused ? "Resume" : "Pause";
    btn.classList.toggle("resume");
  };

  // Threshold slider
  const slider = document.getElementById("thSlider");
  const valSpan = document.getElementById("thVal");
  
  // Mode selector
  const modeSelect = document.getElementById("modeSelect");

  fetchJSON("/config/threshold")
    .then(obj => {
      window.threshold = Math.round(obj.threshold / 1000000); // Convert B/s to Mbps
      slider.value = window.threshold;
      valSpan.textContent = window.threshold;
    });

  slider.oninput = () => { valSpan.textContent = slider.value; };
  slider.onchange = () => {
    const thresholdMbps = parseInt(slider.value, 10);
    window.threshold = thresholdMbps;
    const thresholdBytes = thresholdMbps * 1000000; // Convert Mbps to B/s
    fetch(API + "/config/threshold", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ threshold: thresholdBytes })
    });
  };
  
  // Load current mode
  fetchJSON("/config/mode")
    .then(obj => {
      if (obj.mode) {
        modeSelect.value = obj.mode;
      }
    })
    .catch(() => {}); // Ignore if endpoint doesn't exist yet
  
  // Mode change handler
  modeSelect.onchange = () => {
    fetch(API + "/config/mode", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ mode: modeSelect.value })
    });
  };
}

// Bootstrap everything
window.onload = () => {
  attachUI();
  initChart();
  updateChart();
  setInterval(updateChart, POLL_INTERVAL);
};
