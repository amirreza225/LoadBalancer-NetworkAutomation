// Efficiency metrics visualization
let efficiencyChart;
let congestionTrendsChart;

// Data storage for congestion avoidance trends
let congestionTrendsData = {
  timestamps: [],
  pathBased: [],
};

let currentLoadBalancingMode = "unknown";
const MAX_DATA_POINTS = 300; // Keep last 300 data points (5 minutes at 1-second intervals)

// 30-second rolling average tracking
let avg30sMetrics = {
  history: [], // Array of {timestamp, efficiency, loadBalancing, congestionAvoidance, varianceImprovement, reroutesPerSec}
  lastTotalReroutes: 0,
};

// Initialize efficiency metrics chart
function initEfficiencyChart() {
  const ctx = document.getElementById("efficiencyChart").getContext("2d");
  efficiencyChart = new Chart(ctx, {
    type: "doughnut",
    data: {
      labels: ["Load Balanced", "Shortest Path"],
      datasets: [
        {
          data: [0, 100],
          backgroundColor: ["#3b82f6", "#e5e7eb"],
          borderColor: ["#2563eb", "#d1d5db"],
          borderWidth: 3,
          hoverBackgroundColor: ["#2563eb", "#d1d5db"],
          hoverBorderColor: ["#1d4ed8", "#9ca3af"],
          hoverBorderWidth: 4,
        },
      ],
    },
    options: {
      responsive: true,
      maintainAspectRatio: false,
      plugins: {
        title: {
          display: false,
        },
        legend: {
          position: "bottom",
          labels: {
            padding: 20,
            font: {
              size: 14,
              weight: "500",
            },
            color: "#374151",
          },
        },
      },
      elements: {
        arc: {
          borderJoinStyle: "round",
        },
      },
    },
  });
}

// Initialize congestion trends chart with Chart.js
function initCongestionTrendsChart() {
  const container = document.getElementById("congestionTrendsContainer");
  if (!container) {
    console.log("congestionTrendsContainer not found, skipping initialization");
    return;
  }

  // Create chart container with new professional styling
  container.innerHTML = `
    <div class="chart-container">
      <div class="chart-header">
        <h4 class="chart-title">Real-time Congestion Avoidance</h4>
        <div class="metric-value" id="currentCongestionValue">0%</div>
      </div>
      <div style="height: 200px; position: relative;">
        <canvas id="congestionTrendsChart" class="chart-canvas"></canvas>
      </div>
    </div>
  `;

  // Initialize Chart.js line chart
  const ctx = document.getElementById("congestionTrendsChart").getContext("2d");
  congestionTrendsChart = new Chart(ctx, {
    type: "line",
    data: {
      labels: [],
      datasets: [
        {
          label: "Congestion Avoidance %",
          data: [],
          borderColor: "#3b82f6",
          backgroundColor: "rgba(59, 130, 246, 0.1)",
          fill: true,
          tension: 0.4,
          pointRadius: 5,
          pointHoverRadius: 8,
          borderWidth: 3,
          pointBackgroundColor: "#3b82f6",
          pointBorderColor: "#ffffff",
          pointBorderWidth: 2,
          pointHoverBackgroundColor: "#2563eb",
          pointHoverBorderColor: "#ffffff",
          pointHoverBorderWidth: 3,
        },
      ],
    },
    options: {
      responsive: true,
      maintainAspectRatio: false,
      scales: {
        x: {
          type: "time",
          time: {
            unit: "second",
            tooltipFormat: "HH:mm:ss",
            displayFormats: {
              second: "mm:ss",
            },
          },
          title: {
            display: true,
            text: "Time",
            font: {
              size: 14,
              weight: "500",
            },
            color: "#374151",
          },
          grid: {
            color: "#f3f4f6",
            borderColor: "#e5e7eb",
          },
          ticks: {
            color: "#6b7280",
          },
        },
        y: {
          beginAtZero: true,
          max: 100,
          title: {
            display: true,
            text: "Avoidance Rate (%)",
            font: {
              size: 14,
              weight: "500",
            },
            color: "#374151",
          },
          grid: {
            color: "#f3f4f6",
            borderColor: "#e5e7eb",
          },
          ticks: {
            color: "#6b7280",
            callback: function (value) {
              return value + "%";
            },
          },
        },
      },
      plugins: {
        title: {
          display: false,
        },
        legend: {
          display: false,
        },
        tooltip: {
          mode: "index",
          intersect: false,
          backgroundColor: "#1f2937",
          titleColor: "#f9fafb",
          bodyColor: "#f9fafb",
          borderColor: "#374151",
          borderWidth: 1,
          cornerRadius: 8,
          callbacks: {
            label: function (context) {
              return "Avoidance Rate: " + context.parsed.y.toFixed(1) + "%";
            },
          },
        },
      },
      interaction: {
        mode: "nearest",
        intersect: false,
      },
    },
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

        efficiencyChart.data.datasets[0].data = [
          loadBalancedPercent,
          shortestPathPercent,
        ];
        efficiencyChart.data.labels = [
          `Load Balanced (${loadBalancedPercent.toFixed(1)}%)`,
          `Shortest Path (${shortestPathPercent.toFixed(1)}%)`,
        ];
      } else {
        // No flows yet
        efficiencyChart.data.datasets[0].data = [0, 100];
        efficiencyChart.data.labels = [
          "Load Balanced (0%)",
          "Shortest Path (100%)",
        ];
      }
      efficiencyChart.update();
    }

    // Update metrics display
    updateMetricsDisplay(metrics);

    // Update congestion trends chart with enhanced metrics
    updateCongestionTrends(metrics);

    // Update 30s rolling averages
    update30sAverages(metrics);
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
    congestion_avoidance_rate: metrics.congestion_avoidance_rate,
  }); // Debug log

  // Total flows
  updateMetric("totalFlows", metrics.total_flows || 0);

  // Total flows processed (displayed as "Total Flows Processed")
  updateMetric("loadBalancedFlows", metrics.total_flows || 0);

  // Load balancing rate - use traffic-based calculation if available
  const loadBalancingRate =
    metrics.load_distribution?.load_balancing_effectiveness ||
    metrics.load_balancing_rate ||
    0;
  const legacyRate = metrics.legacy_load_balancing_rate;

  updateMetric(
    "loadBalancingRate",
    `${loadBalancingRate.toFixed(1)}%`,
    loadBalancingRate > 80
      ? "excellent"
      : loadBalancingRate > 60
      ? "good"
      : loadBalancingRate > 30
      ? "fair"
      : "poor"
  );

  // Log comparison if both metrics are available
  if (
    legacyRate !== undefined &&
    Math.abs(loadBalancingRate - legacyRate) > 5
  ) {
    console.log(
      `Load balancing: Traffic-based: ${loadBalancingRate.toFixed(
        1
      )}%, Flow-based: ${legacyRate.toFixed(1)}%`
    );
  }

  // Congestion avoidance percentage (enhanced algorithm)
  const congestionPercentage =
    metrics.enhanced_path_congestion_avoidance ||
    metrics.congestion_avoidance_percentage ||
    0;
  updateMetric(
    "congestionAvoidancePercentage",
    `${congestionPercentage.toFixed(1)}%`,
    congestionPercentage > 70
      ? "excellent"
      : congestionPercentage > 40
      ? "good"
      : congestionPercentage > 15
      ? "fair"
      : "poor"
  );

  // Total reroutes
  updateMetric("totalReroutes", metrics.total_reroutes || 0);

  // Variance improvement
  updateMetric(
    "varianceImprovement",
    `${(metrics.variance_improvement_percent || 0).toFixed(1)}%`
  );

  // Path overhead
  const pathOverhead = metrics.path_overhead_percent || 0;
  updateMetric(
    "pathOverhead",
    `${pathOverhead.toFixed(1)}%`,
    pathOverhead > 20 ? "warning" : "good"
  );

  // Average path lengths
  updateMetric("avgPathLengthLB", (metrics.avg_path_length_lb || 0).toFixed(1));
  updateMetric("avgPathLengthSP", (metrics.avg_path_length_sp || 0).toFixed(1));

  // Runtime
  updateMetric("runtime", `${(metrics.runtime_minutes || 0).toFixed(1)} min`);

  // Efficiency score (composite metric)
  const efficiencyScore = calculateEfficiencyScore(metrics);
  updateMetric(
    "efficiencyScore",
    `${efficiencyScore.toFixed(1)}%`,
    efficiencyScore > 75
      ? "excellent"
      : efficiencyScore > 50
      ? "good"
      : efficiencyScore > 25
      ? "fair"
      : "poor"
  );

  // Update additional load distribution metrics
  updateLoadDistributionMetrics(metrics);
}

// Update algorithm information
async function updateAlgorithmInfo() {
  try {
    const algorithmInfo = await fetchJSON("/stats/algorithm");

    // Update algorithm mode display
    updateMetric("algorithmMode", algorithmInfo.current_mode || "Unknown");
    updateMetric(
      "alternativePaths",
      algorithmInfo.algorithm_stats?.alternative_paths_stored || 0
    );
    updateMetric(
      "congestionTrends",
      algorithmInfo.algorithm_stats?.congestion_trends_tracked || 0
    );
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
  const cappedVarianceImprovement = Math.min(
    75,
    metrics.variance_improvement_percent || 0
  );

  // Debug logging to investigate high scores
  console.log("Efficiency calculation input:", {
    congestion_avoidance_rate: metrics.congestion_avoidance_rate,
    flows_with_congested_baseline: metrics.flows_with_congested_baseline,
    unique_flows_avoided: metrics.unique_flows_with_congestion_avoidance,
    variance_improvement_percent: metrics.variance_improvement_percent,
    capped_variance: cappedVarianceImprovement,
    load_balancing_rate: metrics.load_balancing_rate,
    path_overhead_percent: metrics.path_overhead_percent,
    total_flows: metrics.total_flows,
  });

  let weightedScore = 0;

  // Congestion avoidance (35% of total) - Most critical for network performance
  if (metrics.total_flows > 0) {
    const congestionContribution =
      (metrics.congestion_avoidance_rate || 0) * 0.35;
    weightedScore += congestionContribution;
    console.log(
      "Congestion component:",
      metrics.congestion_avoidance_rate,
      "% * 35% =",
      congestionContribution
    );
  }

  // Variance improvement (25% of total) - Load distribution quality (capped at 75%)
  const varianceContribution = cappedVarianceImprovement * 0.25;
  weightedScore += varianceContribution;
  console.log(
    "Variance component:",
    cappedVarianceImprovement,
    "% * 25% =",
    varianceContribution
  );

  // Load balancing utilization (25% of total) - Alternative path usage
  if (metrics.total_flows > 0) {
    const lbContribution = (metrics.load_balancing_rate || 0) * 0.25;
    weightedScore += lbContribution;
    console.log(
      "Load balancing component:",
      metrics.load_balancing_rate,
      "% * 25% =",
      lbContribution
    );
  }

  // Path efficiency (15% of total) - Minimize path overhead
  const pathOverhead = metrics.path_overhead_percent || 0;
  // Penalty for excessive path overhead: 40% overhead = 60% efficiency
  const pathEfficiency = Math.max(0, 100 - pathOverhead);
  const pathContribution = pathEfficiency * 0.15;
  weightedScore += pathContribution;
  console.log(
    "Path efficiency:",
    pathEfficiency,
    "% * 15% =",
    pathContribution
  );

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

  // Keep only last 30 data points (30 seconds at 1-second intervals)
  const MAX_CONGESTION_POINTS = 30;
  if (congestionTrendsChart.data.labels.length > MAX_CONGESTION_POINTS) {
    congestionTrendsChart.data.labels.shift();
    congestionTrendsChart.data.datasets[0].data.shift();
  }

  // Update the chart
  congestionTrendsChart.update("none"); // 'none' for better performance

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

// Update additional load distribution metrics display
function updateLoadDistributionMetrics(metrics) {
  if (!metrics.load_distribution) {
    return; // No load distribution data available
  }

  const loadDist = metrics.load_distribution;

  // Log detailed load distribution metrics for debugging
  console.log("Load Distribution Metrics:", {
    coefficient_of_variation: loadDist.coefficient_of_variation,
    distribution_entropy: loadDist.distribution_entropy,
    utilization_balance_score: loadDist.utilization_balance_score,
    variance_reduction: loadDist.variance_reduction,
    avg_utilization: loadDist.avg_utilization,
    utilization_range: loadDist.utilization_range,
  });

  // Add visual indicators for load distribution quality
  const loadBalancingElement = document.getElementById("loadBalancingRate");
  if (loadBalancingElement && loadBalancingElement.parentElement) {
    const parentCard = loadBalancingElement.parentElement;

    // Add a title attribute with detailed explanation
    const cv = loadDist.coefficient_of_variation || 0;
    const entropy = loadDist.distribution_entropy || 0;

    parentCard.title =
      `Traffic-based Load Balancing Effectiveness\n` +
      `• Coefficient of Variation: ${cv.toFixed(3)} (lower = better)\n` +
      `• Distribution Entropy: ${entropy.toFixed(3)} (higher = better)\n` +
      `• Balance Score: ${(loadDist.utilization_balance_score || 0).toFixed(
        1
      )}%\n` +
      `• Variance Reduction: ${(loadDist.variance_reduction || 0).toFixed(
        1
      )}%\n` +
      `• Avg Utilization: ${(loadDist.avg_utilization / 1000000 || 0).toFixed(
        1
      )} Mbps`;

    // Add visual indicator for calculation method
    if (!parentCard.querySelector(".calc-method-indicator")) {
      const indicator = document.createElement("div");
      indicator.className = "calc-method-indicator";
      indicator.style.cssText = `
        position: absolute;
        top: 4px;
        right: 4px;
        width: 8px;
        height: 8px;
        background: #3b82f6;
        border-radius: 50%;
        title: Traffic-based calculation
      `;
      indicator.title = "Traffic-based calculation";
      parentCard.style.position = "relative";
      parentCard.appendChild(indicator);
    }
  }

  // Update variance improvement with new calculation if available
  if (loadDist.variance_reduction !== undefined) {
    updateMetric(
      "varianceImprovement",
      `${loadDist.variance_reduction.toFixed(1)}%`,
      loadDist.variance_reduction > 50
        ? "excellent"
        : loadDist.variance_reduction > 25
        ? "good"
        : loadDist.variance_reduction > 10
        ? "fair"
        : "poor"
    );
  }
}

// Reset efficiency display to zero values
function resetEfficiencyDisplay() {
  // Reset chart
  if (efficiencyChart) {
    efficiencyChart.data.datasets[0].data = [0, 100];
    efficiencyChart.data.labels = [
      "Load Balanced (0%)",
      "Shortest Path (100%)",
    ];
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

// Update 30-second rolling averages
function update30sAverages(metrics) {
  const now = Date.now();

  // Calculate reroutes per second
  const currentReroutesPerSec = Math.max(
    0,
    metrics.total_reroutes - avg30sMetrics.lastTotalReroutes
  );

  // Add current metrics to history
  avg30sMetrics.history.push({
    timestamp: now,
    efficiency: metrics.efficiency_score || 0,
    loadBalancing: metrics.load_balancing_rate || 0,
    congestionAvoidance: metrics.congestion_avoidance_rate || 0,
    varianceImprovement: metrics.variance_improvement_percent || 0,
    reroutesPerSec: currentReroutesPerSec,
  });

  // Keep only last 30 seconds of data
  avg30sMetrics.history = avg30sMetrics.history.filter(
    (item) => now - item.timestamp <= 30000
  );

  // Calculate and display averages
  if (avg30sMetrics.history.length > 0) {
    const avgEfficiency =
      avg30sMetrics.history.reduce((sum, item) => sum + item.efficiency, 0) /
      avg30sMetrics.history.length;
    const avgLoadBalancing =
      avg30sMetrics.history.reduce((sum, item) => sum + item.loadBalancing, 0) /
      avg30sMetrics.history.length;
    const avgCongestionAvoidance =
      avg30sMetrics.history.reduce(
        (sum, item) => sum + item.congestionAvoidance,
        0
      ) / avg30sMetrics.history.length;
    const avgVarianceImprovement =
      avg30sMetrics.history.reduce(
        (sum, item) => sum + item.varianceImprovement,
        0
      ) / avg30sMetrics.history.length;
    const avgReroutesPerSec =
      avg30sMetrics.history.reduce(
        (sum, item) => sum + item.reroutesPerSec,
        0
      ) / avg30sMetrics.history.length;

    // Update display
    document.getElementById("avg30sEfficiency").textContent =
      avgEfficiency.toFixed(1) + "%";
    document.getElementById("avg30sLoadBalancing").textContent =
      avgLoadBalancing.toFixed(1) + "%";
    document.getElementById("avg30sCongestionAvoidance").textContent =
      avgCongestionAvoidance.toFixed(1) + "%";
    document.getElementById("avg30sVarianceImprovement").textContent =
      avgVarianceImprovement.toFixed(1) + "%";
    document.getElementById("avg30sReroutesPerSecond").textContent =
      avgReroutesPerSec.toFixed(2);
  } else {
    // No data yet
    document.getElementById("avg30sEfficiency").textContent = "-";
    document.getElementById("avg30sLoadBalancing").textContent = "-";
    document.getElementById("avg30sCongestionAvoidance").textContent = "-";
    document.getElementById("avg30sVarianceImprovement").textContent = "-";
    document.getElementById("avg30sReroutesPerSecond").textContent = "-";
  }

  // Update tracking values
  avg30sMetrics.lastTotalReroutes = metrics.total_reroutes;
}

// Reset 30s averages (called on mode change)
function reset30sAverages() {
  avg30sMetrics.history = [];
  avg30sMetrics.lastTotalReroutes = 0;

  // Reset display
  document.getElementById("avg30sEfficiency").textContent = "-";
  document.getElementById("avg30sLoadBalancing").textContent = "-";
  document.getElementById("avg30sCongestionAvoidance").textContent = "-";
  document.getElementById("avg30sVarianceImprovement").textContent = "-";
  document.getElementById("avg30sReroutesPerSecond").textContent = "-";

  console.log("30s averages reset");
}

// Listen for mode changes to reset display
function listenForModeChanges() {
  const modeSelect = document.getElementById("modeSelect");
  if (modeSelect) {
    modeSelect.addEventListener("change", function () {
      console.log(
        "Mode changed, resetting efficiency display and 30s averages"
      );
      resetEfficiencyDisplay();
      reset30sAverages();
    });
  }
}

// Initialize efficiency tracking when page loads
document.addEventListener("DOMContentLoaded", function () {
  if (document.getElementById("efficiencyChart")) {
    initEfficiencyChart();
    initCongestionTrendsChart(); // This will check if the canvas exists internally
    updateEfficiencyMetrics();
    updateAlgorithmInfo();
    listenForModeChanges();
    setInterval(updateEfficiencyMetrics, 1000); // Update every 1 second for D-ITG detection
    setInterval(updateAlgorithmInfo, 5000); // Update algorithm info every 5 seconds
  }
});
