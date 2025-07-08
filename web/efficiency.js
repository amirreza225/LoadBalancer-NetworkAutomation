// Efficiency metrics visualization
let efficiencyChart;
let congestionTrendsChart;

// Data storage for congestion avoidance trends
let congestionTrendsData = {
  timestamps: [],
  pathBased: []
};

let currentLoadBalancingMode = "unknown";
const MAX_DATA_POINTS = 60; // Keep last 60 data points (5 minutes at 5-second intervals)

// Initialize efficiency metrics chart
function initEfficiencyChart() {
  const ctx = document.getElementById("efficiencyChart").getContext("2d");
  efficiencyChart = new Chart(ctx, {
    type: "doughnut",
    data: {
      labels: ["Load Balanced", "Shortest Path"],
      datasets: [{
        data: [0, 100],
        backgroundColor: ["#28a745", "#dc3545"],
        borderWidth: 2
      }]
    },
    options: {
      responsive: true,
      plugins: {
        title: {
          display: true,
          text: "Load Balancing vs Shortest Path Routing"
        },
        legend: {
          position: "bottom"
        }
      }
    }
  });
}

// Initialize congestion trends chart with Chart.js
function initCongestionTrendsChart() {
  const container = document.getElementById("congestionTrendsContainer");
  if (!container) {
    console.log("congestionTrendsContainer not found, skipping initialization");
    return;
  }
  
  // Create chart container
  container.innerHTML = `
    <div class="congestion-trends-display">
      <div class="trend-header">
        <h4>Congestion Avoidance Trends</h4>
        <div class="current-value" id="currentCongestionValue">0%</div>
      </div>
      <canvas id="congestionTrendsChart" width="400" height="200"></canvas>
    </div>
    <style>
      .congestion-trends-display {
        background: white;
        border-radius: 8px;
        padding: 20px;
        margin: 10px 0;
      }
      .trend-header {
        display: flex;
        justify-content: space-between;
        align-items: center;
        margin-bottom: 15px;
      }
      .current-value {
        font-size: 1.5em;
        font-weight: bold;
        color: #007bff;
      }
      #congestionTrendsChart {
        max-height: 200px;
      }
    </style>
  `;
  
  // Initialize Chart.js line chart
  const ctx = document.getElementById("congestionTrendsChart").getContext("2d");
  congestionTrendsChart = new Chart(ctx, {
    type: "line",
    data: {
      labels: [],
      datasets: [{
        label: "Congestion Avoidance %",
        data: [],
        borderColor: "#28a745",
        backgroundColor: "rgba(40, 167, 69, 0.15)",
        fill: true,
        tension: 0.4,
        pointRadius: 4,
        pointHoverRadius: 8,
        borderWidth: 3
      }]
    },
    options: {
      responsive: true,
      maintainAspectRatio: false,
      scales: {
        x: {
          type: "time",
          time: {
            unit: "minute",
            tooltipFormat: "HH:mm:ss",
            displayFormats: {
              minute: "HH:mm"
            }
          },
          title: {
            display: true,
            text: "Time"
          }
        },
        y: {
          beginAtZero: true,
          max: 100,
          title: {
            display: true,
            text: "Avoidance Rate (%)"
          },
          ticks: {
            callback: function(value) {
              return value + "%";
            }
          }
        }
      },
      plugins: {
        title: {
          display: true,
          text: "Real-time Congestion Avoidance Trends"
        },
        legend: {
          display: false
        },
        tooltip: {
          mode: "index",
          intersect: false,
          callbacks: {
            label: function(context) {
              return context.dataset.label + ": " + context.parsed.y.toFixed(1) + "%";
            }
          }
        }
      },
      interaction: {
        mode: "nearest",
        intersect: false
      }
    }
  });
}

// Update efficiency metrics display
async function updateEfficiencyMetrics() {
  try {
    const metrics = await fetchJSON("/stats/efficiency");
    
    console.log("Efficiency metrics received:", metrics); // Debug log
    
    // Update efficiency chart
    if (efficiencyChart) {
      if (metrics.total_flows > 0) {
        const loadBalancedPercent = metrics.load_balancing_rate || 0;
        const shortestPathPercent = 100 - loadBalancedPercent;
        
        efficiencyChart.data.datasets[0].data = [loadBalancedPercent, shortestPathPercent];
        efficiencyChart.data.labels = [
          `Load Balanced (${loadBalancedPercent.toFixed(1)}%)`,
          `Shortest Path (${shortestPathPercent.toFixed(1)}%)`
        ];
      } else {
        // No flows yet
        efficiencyChart.data.datasets[0].data = [0, 100];
        efficiencyChart.data.labels = ["Load Balanced (0%)", "Shortest Path (100%)"];
      }
      efficiencyChart.update();
    }
    
    // Update metrics display
    updateMetricsDisplay(metrics);
    
    // Update congestion trends chart with enhanced metrics
    updateCongestionTrends(metrics);
    
  } catch (error) {
    console.error("Error updating efficiency metrics:", error);
    // Show error in UI
    document.getElementById("totalFlows").textContent = "Error";
  }
}

// Update the metrics display in the dashboard
function updateMetricsDisplay(metrics) {
  console.log("Updating metrics display with:", {
    total_flows: metrics.total_flows,
    load_balanced_flows: metrics.load_balanced_flows,
    congestion_avoided: metrics.congestion_avoided,
    load_balancing_rate: metrics.load_balancing_rate,
    congestion_avoidance_rate: metrics.congestion_avoidance_rate
  }); // Debug log
  
  // Total flows
  updateMetric("totalFlows", metrics.total_flows || 0);
  
  // Total flows processed (displayed as "Total Flows Processed")
  updateMetric("loadBalancedFlows", metrics.total_flows || 0);
  
  // Load balancing rate
  updateMetric("loadBalancingRate", 
    `${(metrics.load_balancing_rate || 0).toFixed(1)}%`);
  
  // Congestion avoidance percentage (enhanced algorithm)
  const congestionPercentage = metrics.enhanced_path_congestion_avoidance || metrics.congestion_avoidance_percentage || 0;
  updateMetric("congestionAvoidancePercentage", 
    `${congestionPercentage.toFixed(1)}%`,
    congestionPercentage > 70 ? "excellent" : 
    congestionPercentage > 40 ? "good" : 
    congestionPercentage > 15 ? "fair" : "poor");
  
  // Total reroutes
  updateMetric("totalReroutes", metrics.total_reroutes || 0);
  
  // Variance improvement
  updateMetric("varianceImprovement", 
    `${(metrics.variance_improvement_percent || 0).toFixed(1)}%`);
  
  // Path overhead
  const pathOverhead = metrics.path_overhead_percent || 0;
  updateMetric("pathOverhead", 
    `${pathOverhead.toFixed(1)}%`, 
    pathOverhead > 20 ? "warning" : "good");
  
  // Average path lengths
  updateMetric("avgPathLengthLB", 
    (metrics.avg_path_length_lb || 0).toFixed(1));
  updateMetric("avgPathLengthSP", 
    (metrics.avg_path_length_sp || 0).toFixed(1));
  
  // Runtime
  updateMetric("runtime", 
    `${(metrics.runtime_minutes || 0).toFixed(1)} min`);
  
  // Efficiency score (composite metric)
  const efficiencyScore = calculateEfficiencyScore(metrics);
  updateMetric("efficiencyScore", 
    `${efficiencyScore.toFixed(1)}%`,
    efficiencyScore > 75 ? "excellent" : 
    efficiencyScore > 50 ? "good" : 
    efficiencyScore > 25 ? "fair" : "poor");
}

// Update algorithm information
async function updateAlgorithmInfo() {
  try {
    const algorithmInfo = await fetchJSON("/stats/algorithm");
    
    // Update algorithm mode display
    updateMetric("algorithmMode", algorithmInfo.current_mode || "Unknown");
    updateMetric("alternativePaths", algorithmInfo.algorithm_stats?.alternative_paths_stored || 0);
    updateMetric("congestionTrends", algorithmInfo.algorithm_stats?.congestion_trends_tracked || 0);
    
  } catch (error) {
    console.error("Error updating algorithm info:", error);
  }
}

// Helper function to update individual metrics
function updateMetric(elementId, value, status = null) {
  const element = document.getElementById(elementId);
  if (element) {
    element.textContent = value;
    
    // Remove existing status classes
    element.classList.remove("excellent", "good", "fair", "poor", "warning");
    
    // Add status class if provided
    if (status) {
      element.classList.add(status);
    }
  }
}

// Calculate composite efficiency score based on network engineering principles
function calculateEfficiencyScore(metrics) {
  // Cap variance improvement to realistic values to prevent inflated scores
  const cappedVarianceImprovement = Math.min(75, metrics.variance_improvement_percent || 0);
  
  // Debug logging to investigate high scores
  console.log("Efficiency calculation input:", {
    congestion_avoidance_rate: metrics.congestion_avoidance_rate,
    flows_with_congested_baseline: metrics.flows_with_congested_baseline,
    unique_flows_avoided: metrics.unique_flows_with_congestion_avoidance,
    variance_improvement_percent: metrics.variance_improvement_percent,
    capped_variance: cappedVarianceImprovement,
    load_balancing_rate: metrics.load_balancing_rate,
    path_overhead_percent: metrics.path_overhead_percent,
    total_flows: metrics.total_flows
  });
  
  let weightedScore = 0;
  
  // Congestion avoidance (35% of total) - Most critical for network performance
  if (metrics.total_flows > 0) {
    const congestionContribution = (metrics.congestion_avoidance_rate || 0) * 0.35;
    weightedScore += congestionContribution;
    console.log("Congestion component:", metrics.congestion_avoidance_rate, "% * 35% =", congestionContribution);
  }
  
  // Variance improvement (25% of total) - Load distribution quality (capped at 75%)
  const varianceContribution = cappedVarianceImprovement * 0.25;
  weightedScore += varianceContribution;
  console.log("Variance component:", cappedVarianceImprovement, "% * 25% =", varianceContribution);
  
  // Load balancing utilization (25% of total) - Alternative path usage
  if (metrics.total_flows > 0) {
    const lbContribution = (metrics.load_balancing_rate || 0) * 0.25;
    weightedScore += lbContribution;
    console.log("Load balancing component:", metrics.load_balancing_rate, "% * 25% =", lbContribution);
  }
  
  // Path efficiency (15% of total) - Minimize path overhead
  const pathOverhead = metrics.path_overhead_percent || 0;
  // Penalty for excessive path overhead: 40% overhead = 60% efficiency
  const pathEfficiency = Math.max(0, 100 - pathOverhead);
  const pathContribution = pathEfficiency * 0.15;
  weightedScore += pathContribution;
  console.log("Path efficiency:", pathEfficiency, "% * 15% =", pathContribution);
  
  console.log("Total weighted score:", weightedScore);
  
  // No additional normalization needed - score is already percentage-based
  const finalScore = Math.max(0, Math.min(100, weightedScore));
  
  console.log("Final efficiency score:", finalScore);
  
  return finalScore;
}

// Update congestion trends chart with new data
async function updateCongestionTrends(metrics) {
  if (!congestionTrendsChart) return;
  
  const now = new Date();
  
  // Extract main congestion avoidance metric
  const pathBasedValue = metrics.enhanced_path_congestion_avoidance || 0;
  
  // Add new data point
  congestionTrendsChart.data.labels.push(now);
  congestionTrendsChart.data.datasets[0].data.push(pathBasedValue);
  
  // Keep only last MAX_DATA_POINTS
  if (congestionTrendsChart.data.labels.length > MAX_DATA_POINTS) {
    congestionTrendsChart.data.labels.shift();
    congestionTrendsChart.data.datasets[0].data.shift();
  }
  
  // Update the chart
  congestionTrendsChart.update('none'); // 'none' for better performance
  
  // Update current value display
  const currentValueElement = document.getElementById("currentCongestionValue");
  if (currentValueElement) {
    currentValueElement.textContent = `${pathBasedValue.toFixed(1)}%`;
    // Change color based on value
    if (pathBasedValue > 80) {
      currentValueElement.style.color = "#28a745"; // Green for high
    } else if (pathBasedValue > 50) {
      currentValueElement.style.color = "#007bff"; // Blue for medium
    } else {
      currentValueElement.style.color = "#dc3545"; // Red for low
    }
  }
}


// Reset congestion trends data
function resetCongestionTrends() {
  // Reset chart data
  if (congestionTrendsChart) {
    congestionTrendsChart.data.labels = [];
    congestionTrendsChart.data.datasets[0].data = [];
    congestionTrendsChart.update();
  }
  
  // Reset display elements
  const currentValueElement = document.getElementById("currentCongestionValue");
  if (currentValueElement) {
    currentValueElement.textContent = "0%";
    currentValueElement.style.color = "#007bff";
  }
}

// Reset efficiency display to zero values
function resetEfficiencyDisplay() {
  // Reset chart
  if (efficiencyChart) {
    efficiencyChart.data.datasets[0].data = [0, 100];
    efficiencyChart.data.labels = ["Load Balanced (0%)", "Shortest Path (100%)"];
    efficiencyChart.update();
  }
  
  // Reset all metric displays
  updateMetric("totalFlows", 0);
  updateMetric("loadBalancedFlows", 0);
  updateMetric("loadBalancingRate", "0%");
  updateMetric("congestionAvoidancePercentage", "0%");
  updateMetric("totalReroutes", 0);
  updateMetric("varianceImprovement", "0%");
  updateMetric("pathOverhead", "0%");
  updateMetric("avgPathLengthLB", "0");
  updateMetric("avgPathLengthSP", "0");
  updateMetric("runtime", "0 min");
  updateMetric("efficiencyScore", "0%");
  
  // Reset congestion trends
  resetCongestionTrends();
  
  console.log("Efficiency display reset to zero values");
}

// Listen for mode changes to reset display
function listenForModeChanges() {
  const modeSelect = document.getElementById("modeSelect");
  if (modeSelect) {
    modeSelect.addEventListener("change", function() {
      console.log("Mode changed, resetting efficiency display");
      resetEfficiencyDisplay();
    });
  }
}

// Initialize efficiency tracking when page loads
document.addEventListener('DOMContentLoaded', function() {
  if (document.getElementById("efficiencyChart")) {
    initEfficiencyChart();
    initCongestionTrendsChart(); // This will check if the canvas exists internally
    updateEfficiencyMetrics();
    updateAlgorithmInfo();
    listenForModeChanges();
    setInterval(updateEfficiencyMetrics, 5000); // Update every 5 seconds
    setInterval(updateAlgorithmInfo, 10000); // Update algorithm info every 10 seconds
  }
});