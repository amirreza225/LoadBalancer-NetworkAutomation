// Efficiency metrics visualization
let efficiencyChart;

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
  
  // Load balanced flows
  updateMetric("loadBalancedFlows", metrics.load_balanced_flows || 0);
  
  // Load balancing rate
  updateMetric("loadBalancingRate", 
    `${(metrics.load_balancing_rate || 0).toFixed(1)}%`);
  
  // Congestion avoidance rate
  updateMetric("congestionAvoidanceRate", 
    `${(metrics.congestion_avoidance_rate || 0).toFixed(1)}%`);
  
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
  let score = 0;
  let totalWeight = 0;
  
  // Congestion avoidance (35% weight) - Most critical for network performance
  if (metrics.total_flows > 0) {
    const congestionScore = Math.min(100, metrics.congestion_avoidance_rate || 0);
    score += congestionScore * 0.35;
    totalWeight += 0.35;
  }
  
  // Variance improvement (25% weight) - Load distribution quality
  const varianceScore = Math.min(100, metrics.variance_improvement_percent || 0);
  score += varianceScore * 0.25;
  totalWeight += 0.25;
  
  // Load balancing utilization (25% weight) - Alternative path usage
  if (metrics.total_flows > 0) {
    const lbScore = Math.min(100, metrics.load_balancing_rate || 0);
    score += lbScore * 0.25;
    totalWeight += 0.25;
  }
  
  // Path efficiency (15% weight) - Minimize path overhead
  const pathOverhead = metrics.path_overhead_percent || 0;
  // Convert overhead to efficiency: 0% overhead = 100% efficiency
  const pathEfficiency = Math.max(0, 100 - Math.min(100, pathOverhead * 2));
  score += pathEfficiency * 0.15;
  totalWeight += 0.15;
  
  // Normalize score based on available components
  const normalizedScore = totalWeight > 0 ? score / totalWeight * 100 : 0;
  
  return Math.max(0, Math.min(100, normalizedScore));
}

// Initialize efficiency tracking when page loads
document.addEventListener('DOMContentLoaded', function() {
  if (document.getElementById("efficiencyChart")) {
    initEfficiencyChart();
    updateEfficiencyMetrics();
    updateAlgorithmInfo();
    setInterval(updateEfficiencyMetrics, 5000); // Update every 5 seconds
    setInterval(updateAlgorithmInfo, 10000); // Update algorithm info every 10 seconds
  }
});