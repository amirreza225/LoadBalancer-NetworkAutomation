const linkThreshold = { low: 0.3, high: 1.0 };
const THRESH_DEFAULT = 100000;
// Use API from app.js

// Error tracking and retry logic
let topologyErrorCount = 0;
let lastTopologyError = null;
const MAX_TOPOLOGY_ERRORS = 3;
const TOPOLOGY_RETRY_DELAY = 5000; // 5 seconds

// Dynamic topology data - will be populated from API
let nodes = [];
let links = [];
let simulation;
let gLink, gNode;

const svg = d3.select("svg"),
  width = +svg.attr("width"),
  height = +svg.attr("height");

const linkColor = (bps) => {
  // Use the global threshold from app.js (convert Mbps to bytes/s)
  const currentThreshold = (window.threshold || 100) * 1000000;
  if (bps > currentThreshold * linkThreshold.high) return "red";
  if (bps > currentThreshold * linkThreshold.low) return "orange";
  return "green";
};

// Initialize the visualization
function initTopology() {
  // Create groups for links and nodes
  gLink = svg
    .append("g")
    .attr("stroke", "#999")
    .attr("stroke-opacity", 0.6)
    .attr("class", "links");

  gNode = svg
    .append("g")
    .attr("stroke", "#fff")
    .attr("stroke-width", 1.5)
    .attr("class", "nodes");

  // Start fetching topology data
  updateTopology();
  setInterval(checkTopologyChanges, 5000); // Check for topology changes every 5 seconds
  setInterval(updateLinkColors, 1000); // Update link colors every second
}

// Check for topology changes and only update if needed
async function checkTopologyChanges() {
  try {
    const response = await fetch(`${API}/topology`);

    if (!response.ok) {
      throw new Error(`HTTP ${response.status}: ${response.statusText}`);
    }

    const newTopologyData = await response.json();

    if (newTopologyData.nodes && newTopologyData.links) {
      // Check if topology has actually changed
      if (hasTopologyChanged(newTopologyData)) {
        console.log("Topology changed, updating visualization...");
        updateTopology(newTopologyData);
      }

      // Reset error count on successful fetch
      topologyErrorCount = 0;
      lastTopologyError = null;
      updateTopologyStatus("ok");
    } else {
      console.warn("Topology API returned invalid data:", newTopologyData);
    }
  } catch (error) {
    topologyErrorCount++;
    lastTopologyError = error;

    console.error(
      `Error checking topology changes (${topologyErrorCount}/${MAX_TOPOLOGY_ERRORS}):`,
      error
    );

    // If we have persistent errors, try to recover
    if (topologyErrorCount >= MAX_TOPOLOGY_ERRORS) {
      console.warn("Too many topology errors, attempting recovery...");
      updateTopologyStatus("error", "Connection Failed");
      setTimeout(attemptTopologyRecovery, TOPOLOGY_RETRY_DELAY);
    } else {
      updateTopologyStatus(
        "warning",
        `Error ${topologyErrorCount}/${MAX_TOPOLOGY_ERRORS}`
      );
    }
  }
}

// Check if topology has changed compared to current state
function hasTopologyChanged(newTopology) {
  // Compare number of nodes and links
  if (
    nodes.length !== newTopology.nodes.length ||
    links.length !== newTopology.links.length
  ) {
    console.log(
      `Topology size changed: nodes ${nodes.length}->${newTopology.nodes.length}, links ${links.length}->${newTopology.links.length}`
    );
    return true;
  }

  // Compare node IDs with more detailed logging
  const currentNodeIds = new Set(nodes.map((n) => n.id));
  const newNodeIds = new Set(newTopology.nodes.map((n) => n.id));
  if (
    currentNodeIds.size !== newNodeIds.size ||
    [...currentNodeIds].some((id) => !newNodeIds.has(id))
  ) {
    console.log(`Node changes detected:`, {
      current: [...currentNodeIds].sort(),
      new: [...newNodeIds].sort(),
    });
    return true;
  }

  // Compare link connections
  const currentLinkIds = new Set(
    links.map((l) => `${l.source.id || l.source}-${l.target.id || l.target}`)
  );
  const newLinkIds = new Set(
    newTopology.links.map((l) => `${l.source}-${l.target}`)
  );
  if (
    currentLinkIds.size !== newLinkIds.size ||
    [...currentLinkIds].some((id) => !newLinkIds.has(id))
  ) {
    console.log(`Link changes detected:`, {
      current: [...currentLinkIds].sort(),
      new: [...newLinkIds].sort(),
    });
    return true;
  }

  return false;
}

// Fetch and update topology from API (only called when topology changes)
async function updateTopology(topologyData = null) {
  try {
    if (!topologyData) {
      const response = await fetch(`${API}/topology`);

      if (!response.ok) {
        throw new Error(`HTTP ${response.status}: ${response.statusText}`);
      }

      topologyData = await response.json();
    }

    if (topologyData.nodes && topologyData.links) {
      // Validate topology data structure
      const validatedData = validateTopologyData(topologyData);
      if (!validatedData.valid) {
        console.warn("Topology data validation failed:", validatedData.errors);
        return;
      }

      // Update nodes and links
      nodes = topologyData.nodes;
      links = topologyData.links;

      // Restart simulation with new data
      if (simulation) {
        simulation.stop();
      }

      simulation = d3
        .forceSimulation(nodes)
        .force(
          "link",
          d3
            .forceLink(links)
            .id((d) => d.id)
            .distance(80)
        )
        .force("charge", d3.forceManyBody().strength(-400))
        .force("center", d3.forceCenter(width / 2, height / 2));

      // Update visualization
      updateVisualization();

      // Log topology summary for debugging
      const hostCount = nodes.filter((n) => n.type === "host").length;
      const switchCount = nodes.filter((n) => n.type === "switch").length;
      console.log(
        `Topology updated: ${nodes.length} nodes (${switchCount} switches, ${hostCount} hosts), ${links.length} links`
      );

      // Log host details for debugging
      const hostNodes = nodes.filter((n) => n.type === "host");
      if (hostNodes.length > 0) {
        console.log(
          "Host nodes:",
          hostNodes.map((h) => h.id)
        );
      }
    } else {
      console.warn("Invalid topology data received:", topologyData);
    }
  } catch (error) {
    console.error("Error fetching topology:", error);

    // Don't retry immediately on fetch errors - let the regular interval handle it
    if (topologyErrorCount < MAX_TOPOLOGY_ERRORS) {
      console.log("Will retry topology fetch on next interval");
    }
  }
}

// Update the D3 visualization with current nodes and links
function updateVisualization() {
  // Update links
  gLink
    .selectAll("line")
    .data(links, (d) => `${d.source.id || d.source}-${d.target.id || d.target}`)
    .join(
      (enter) =>
        enter
          .append("line")
          .attr("class", "link")
          .attr("stroke-width", (d) => (d.type === "host-switch" ? 2 : 3))
          .attr("stroke-dasharray", (d) =>
            d.type === "host-switch" ? "5,5" : null
          ),
      (update) => update,
      (exit) => exit.remove()
    );

  // Update nodes
  const nodeGroups = gNode
    .selectAll("g")
    .data(nodes, (d) => d.id)
    .join(
      (enter) => {
        const nodeGroup = enter.append("g").call(drag(simulation));

        nodeGroup
          .append("circle")
          .attr("r", (d) => (d.type === "host" ? 8 : 12))
          .attr("fill", (d) => (d.type === "host" ? "#bbb" : "#1f77b4"));

        nodeGroup
          .append("text")
          .text((d) => d.id)
          .attr("x", 15)
          .attr("y", 4)
          .attr("stroke", "black")
          .attr("font-size", "12px");

        return nodeGroup;
      },
      (update) => update,
      (exit) => exit.remove()
    );

  // Update simulation
  simulation.nodes(nodes);
  simulation.force("link").links(links);
  simulation.alpha(0.3).restart();

  // Update simulation tick handler
  simulation.on("tick", () => {
    gLink
      .selectAll("line")
      .attr("x1", (d) => (d.source.x || d.source.x === 0 ? d.source.x : 0))
      .attr("y1", (d) => (d.source.y || d.source.y === 0 ? d.source.y : 0))
      .attr("x2", (d) => (d.target.x || d.target.x === 0 ? d.target.x : 0))
      .attr("y2", (d) => (d.target.y || d.target.y === 0 ? d.target.y : 0));

    gNode
      .selectAll("g")
      .attr("transform", (d) => `translate(${d.x || 0},${d.y || 0})`);
  });
}

// Update link colors based on current traffic
async function updateLinkColors() {
  try {
    const response = await fetch(`${API}/load/links`);

    if (!response.ok) {
      throw new Error(`HTTP ${response.status}: ${response.statusText}`);
    }

    const stats = await response.json();

    if (gLink && gLink.selectAll) {
      gLink.selectAll("line").attr("stroke", (d) => {
        // Only color switch-to-switch links based on traffic
        if (d.type !== "switch-switch") {
          return d.type === "host-switch" ? "#666" : "#999";
        }

        // Extract switch IDs from node IDs (e.g., "s1" -> "1")
        const sourceId = (d.source.id || d.source).replace("s", "");
        const targetId = (d.target.id || d.target).replace("s", "");

        // Try both directions for the link key
        const key1 = `${sourceId}-${targetId}`;
        const key2 = `${targetId}-${sourceId}`;
        const bps = stats[key1] || stats[key2] || 0;

        return linkColor(bps);
      });
    }
  } catch (error) {
    console.error("Error updating link colors:", error);
    // Don't stop the visualization if link colors fail
  }
}

// Drag behavior
function drag(simulation) {
  function dragstarted(event) {
    if (!event.active) simulation.alphaTarget(0.3).restart();
    event.subject.fx = event.subject.x;
    event.subject.fy = event.subject.y;
  }

  function dragged(event) {
    event.subject.fx = event.x;
    event.subject.fy = event.y;
  }

  function dragended(event) {
    if (!event.active) simulation.alphaTarget(0);
    event.subject.fx = null;
    event.subject.fy = null;
  }

  return d3
    .drag()
    .on("start", dragstarted)
    .on("drag", dragged)
    .on("end", dragended);
}

// Validate topology data structure
function validateTopologyData(data) {
  const errors = [];

  if (!data.nodes || !Array.isArray(data.nodes)) {
    errors.push("Missing or invalid nodes array");
  }

  if (!data.links || !Array.isArray(data.links)) {
    errors.push("Missing or invalid links array");
  }

  // Check for required node properties
  if (data.nodes) {
    data.nodes.forEach((node, index) => {
      if (!node.id) {
        errors.push(`Node ${index} missing id`);
      }
      if (!node.type) {
        errors.push(`Node ${index} missing type`);
      }
    });
  }

  // Check for required link properties
  if (data.links) {
    data.links.forEach((link, index) => {
      if (!link.source) {
        errors.push(`Link ${index} missing source`);
      }
      if (!link.target) {
        errors.push(`Link ${index} missing target`);
      }
    });
  }

  return {
    valid: errors.length === 0,
    errors: errors,
  };
}

// Attempt topology recovery after persistent errors
function attemptTopologyRecovery() {
  console.log("Attempting topology recovery...");

  // Reset error count
  topologyErrorCount = 0;
  lastTopologyError = null;

  // Clear existing visualization
  if (gLink) {
    gLink.selectAll("*").remove();
  }
  if (gNode) {
    gNode.selectAll("*").remove();
  }

  // Reinitialize topology
  setTimeout(() => {
    console.log("Reinitializing topology visualization...");
    initTopology();
  }, 1000);
}

// Add topology status indicator
function updateTopologyStatus(status, message = "") {
  // Create status indicator if it doesn't exist
  let statusIndicator = document.getElementById("topology-status");
  if (!statusIndicator) {
    statusIndicator = document.createElement("div");
    statusIndicator.id = "topology-status";
    statusIndicator.style.cssText = `
      position: absolute;
      top: 10px;
      right: 10px;
      padding: 5px 10px;
      border-radius: 4px;
      font-size: 12px;
      font-weight: bold;
      z-index: 1000;
    `;
    document.querySelector("svg").parentNode.appendChild(statusIndicator);
  }

  // Update status
  switch (status) {
    case "ok":
      statusIndicator.textContent = "Topology: OK";
      statusIndicator.style.backgroundColor = "#28a745";
      statusIndicator.style.color = "white";
      break;
    case "warning":
      statusIndicator.textContent = `Topology: ${message}`;
      statusIndicator.style.backgroundColor = "#ffc107";
      statusIndicator.style.color = "black";
      break;
    case "error":
      statusIndicator.textContent = `Topology: ${message}`;
      statusIndicator.style.backgroundColor = "#dc3545";
      statusIndicator.style.color = "white";
      break;
  }
}

// Initialize when DOM is loaded
document.addEventListener("DOMContentLoaded", function () {
  // Initialize topology visualization
  initTopology();

  // Set up error monitoring
  window.addEventListener("error", (event) => {
    if (event.filename && event.filename.includes("topology.js")) {
      console.error("Topology visualization error:", event.error);
      updateTopologyStatus("error", "Visualization Error");
    }
  });
});
