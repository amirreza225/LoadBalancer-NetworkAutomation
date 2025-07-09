// Configuration
const API = "http://localhost:8080";
window.API = API;  // Make API available globally
const POLL_INTERVAL = 500;    // ms (reduced for D-ITG real-time detection)
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
      maintainAspectRatio: false,
      animation: false,
      scales: {
        x: {
          type: "time",
          time: { unit: "second", tooltipFormat: "HH:mm:ss" },
          title: { 
            display: true, 
            text: "Time",
            font: {
              size: 14,
              weight: '500'
            },
            color: '#374151'
          },
          grid: {
            color: '#f3f4f6',
            borderColor: '#e5e7eb'
          },
          ticks: {
            color: '#6b7280'
          }
        },
        y: {
          beginAtZero: true,
          title: { 
            display: true, 
            text: "Bandwidth (Bytes/sec)",
            font: {
              size: 14,
              weight: '500'
            },
            color: '#374151'
          },
          grid: {
            color: '#f3f4f6',
            borderColor: '#e5e7eb'
          },
          ticks: {
            color: '#6b7280'
          }
        }
      },
      plugins: {
        legend: { 
          position: "top", 
          align: "end",
          labels: {
            font: {
              size: 12,
              weight: '500'
            },
            color: '#374151',
            padding: 20
          }
        },
        title: { 
          display: false
        },
        tooltip: {
          backgroundColor: '#1f2937',
          titleColor: '#f9fafb',
          bodyColor: '#f9fafb',
          borderColor: '#374151',
          borderWidth: 1,
          cornerRadius: 8
        }
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
    const uniquePaths = new Set();
    const lines = [];
    
    Object.entries(pathsObj).forEach(([flow, path]) => {
      if (!path || !Array.isArray(path) || path.length === 0) {
        lines.push(`${flow} : No path found`);
        return;
      }
      
      // Create a normalized flow key to avoid duplicates (h1→h2 and h2→h1)
      const flowParts = flow.split("→");
      if (flowParts.length === 2) {
        const [src, dst] = flowParts;
        const normalizedFlow = src.localeCompare(dst) <= 0 ? `${src}→${dst}` : `${dst}→${src}`;
        
        // Check if we've already seen this flow (in either direction)
        if (!uniquePaths.has(normalizedFlow)) {
          uniquePaths.add(normalizedFlow);
          
          // Format path with switch names
          const formattedPath = path.map(dpid => `s${dpid}`).join(" → ");
          lines.push(`${flow} : ${formattedPath}`);
        }
      } else {
        // Fallback for unexpected format
        const formattedPath = path.map(dpid => `s${dpid}`).join(" → ");
        lines.push(`${flow} : ${formattedPath}`);
      }
    });
    
    // Sort lines by host numbers (extract numbers from host names)
    lines.sort((a, b) => {
      const extractHostNumbers = (line) => {
        const match = line.match(/h(\d+)→h(\d+)/);
        if (match) {
          return [parseInt(match[1]), parseInt(match[2])];
        }
        return [0, 0];
      };
      
      const [srcA, dstA] = extractHostNumbers(a);
      const [srcB, dstB] = extractHostNumbers(b);
      
      // Sort by source host first, then by destination host
      if (srcA !== srcB) {
        return srcA - srcB;
      }
      return dstA - dstB;
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

// Show notification to user with professional styling
function showNotification(message, type = "info") {
  // Create notification element
  const notification = document.createElement("div");
  notification.className = `notification ${type}`;
  notification.textContent = message;
  
  // Add to page
  document.body.appendChild(notification);
  
  // Trigger show animation
  setTimeout(() => {
    notification.classList.add("show");
  }, 10);
  
  // Remove after 4 seconds with fade out
  setTimeout(() => {
    notification.classList.remove("show");
    setTimeout(() => {
      if (notification.parentNode) {
        notification.parentNode.removeChild(notification);
      }
    }, 300);
  }, 4000);
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
    // Show notification about mode change and metrics reset
    showNotification("Changing mode - efficiency metrics will be reset", "info");
    
    fetch(API + "/config/mode", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ mode: modeSelect.value })
    })
    .then(response => response.json())
    .then(data => {
      if (data.error) {
        showNotification("Error changing mode: " + data.error, "error");
      } else {
        showNotification("Mode changed successfully - metrics reset", "success");
      }
    })
    .catch(error => {
      showNotification("Error changing mode: " + error.message, "error");
    });
  };
}

// Apply professional animations to dashboard sections
function initProfessionalAnimations() {
  // Add fade-in animations to dashboard sections
  const sections = document.querySelectorAll('.dashboard-section');
  sections.forEach((section, index) => {
    section.style.opacity = '0';
    section.style.transform = 'translateY(20px)';
    
    setTimeout(() => {
      section.style.transition = 'all 0.6s ease-out';
      section.style.opacity = '1';
      section.style.transform = 'translateY(0)';
    }, index * 200 + 300); // Stagger the animations
  });
  
  // Add slide-in animation to controls panel
  const controlsPanel = document.querySelector('.controls-panel');
  if (controlsPanel) {
    controlsPanel.classList.add('slide-in-right');
  }
}

// Bootstrap everything
window.onload = () => {
  attachUI();
  initChart();
  updateChart();
  setInterval(updateChart, POLL_INTERVAL);
  
  // Add professional animations
  setTimeout(initProfessionalAnimations, 100);
};
