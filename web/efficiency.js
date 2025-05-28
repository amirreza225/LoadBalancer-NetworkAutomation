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
    
    // Update efficiency chart
    if (efficiencyChart && metrics.total_flows > 0) {
      const loadBalancedPercent = metrics.load_balancing_rate || 0;
      const shortestPathPercent = 100 - loadBalancedPercent;
      
      efficiencyChart.data.datasets[0].data = [loadBalancedPercent, shortestPathPercent];
      efficiencyChart.update();
    }
    
    // Update metrics display
    updateMetricsDisplay(metrics);
    
  } catch (error) {
    console.error("Error updating efficiency metrics:", error);
  }
}

// Update the metrics display in the dashboard
function updateMetricsDisplay(metrics) {
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

// Calculate composite efficiency score
function calculateEfficiencyScore(metrics) {
  let score = 0;
  let factors = 0;
  
  // Load balancing rate (0-40 points)
  if (metrics.total_flows > 0) {
    score += Math.min(40, (metrics.load_balancing_rate || 0) * 0.4);
    factors++;
  }
  
  // Congestion avoidance (0-30 points)
  if (metrics.total_flows > 0) {
    score += Math.min(30, (metrics.congestion_avoidance_rate || 0) * 0.3);
    factors++;
  }
  
  // Variance improvement (0-20 points)
  score += Math.min(20, (metrics.variance_improvement_percent || 0) * 0.2);
  factors++;
  
  // Path overhead penalty (0 to -10 points)
  const pathOverhead = metrics.path_overhead_percent || 0;
  if (pathOverhead > 0) {
    score -= Math.min(10, pathOverhead * 0.5);
  }
  
  // Normalize to percentage
  return Math.max(0, Math.min(100, score));
}

// Initialize efficiency tracking when page loads
document.addEventListener('DOMContentLoaded', function() {
  if (document.getElementById("efficiencyChart")) {
    initEfficiencyChart();
    updateEfficiencyMetrics();
    setInterval(updateEfficiencyMetrics, 5000); // Update every 5 seconds
  }
});