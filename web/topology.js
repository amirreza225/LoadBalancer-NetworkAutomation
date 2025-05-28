const linkThreshold = { low: 0.3, high: 1.0 };
const THRESH_DEFAULT = 1000000;

// Dynamic topology data - will be populated from API
let nodes = [];
let links = [];
let simulation;
let gLink, gNode;

const svg = d3.select("svg"), width = +svg.attr("width"), height = +svg.attr("height");

const linkColor = bps => {
  // Use the global threshold from app.js
  const currentThreshold = window.threshold || THRESH_DEFAULT;
  if (bps > currentThreshold * linkThreshold.high) return "red";
  if (bps > currentThreshold * linkThreshold.low) return "orange";
  return "green";
};

// Initialize the visualization
function initTopology() {
  // Create groups for links and nodes
  gLink = svg.append("g")
    .attr("stroke", "#999")
    .attr("stroke-opacity", 0.6)
    .attr("class", "links");

  gNode = svg.append("g")
    .attr("stroke", "#fff")
    .attr("stroke-width", 1.5)
    .attr("class", "nodes");

  // Start fetching topology data
  updateTopology();
  setInterval(updateTopology, 5000); // Update topology every 5 seconds
  setInterval(updateLinkColors, 1000); // Update link colors every second
}

// Fetch and update topology from API
async function updateTopology() {
  try {
    const response = await fetch(`${API}/topology`);
    const topologyData = await response.json();
    
    if (topologyData.nodes && topologyData.links) {
      // Update nodes and links
      nodes = topologyData.nodes;
      links = topologyData.links;
      
      // Restart simulation with new data
      if (simulation) {
        simulation.stop();
      }
      
      simulation = d3.forceSimulation(nodes)
        .force("link", d3.forceLink(links).id(d => d.id).distance(80))
        .force("charge", d3.forceManyBody().strength(-400))
        .force("center", d3.forceCenter(width / 2, height / 2));

      // Update visualization
      updateVisualization();
      
      console.log(`Topology updated: ${nodes.length} nodes, ${links.length} links`);
    }
  } catch (error) {
    console.error("Error fetching topology:", error);
  }
}

// Update the D3 visualization with current nodes and links
function updateVisualization() {
  // Update links
  gLink.selectAll("line")
    .data(links, d => `${d.source.id || d.source}-${d.target.id || d.target}`)
    .join(
      enter => enter.append("line")
        .attr("class", "link")
        .attr("stroke-width", d => d.type === "host-switch" ? 2 : 3)
        .attr("stroke-dasharray", d => d.type === "host-switch" ? "5,5" : null),
      update => update,
      exit => exit.remove()
    );

  // Update nodes
  const nodeGroups = gNode.selectAll("g")
    .data(nodes, d => d.id)
    .join(
      enter => {
        const nodeGroup = enter.append("g")
          .call(drag(simulation));
        
        nodeGroup.append("circle")
          .attr("r", d => d.type === "host" ? 8 : 12)
          .attr("fill", d => d.type === "host" ? "#bbb" : "#1f77b4");
        
        nodeGroup.append("text")
          .text(d => d.id)
          .attr("x", 15)
          .attr("y", 4)
          .attr("stroke", "black")
          .attr("font-size", "12px");
        
        return nodeGroup;
      },
      update => update,
      exit => exit.remove()
    );

  // Update simulation
  simulation.nodes(nodes);
  simulation.force("link").links(links);
  simulation.alpha(0.3).restart();

  // Update simulation tick handler
  simulation.on("tick", () => {
    gLink.selectAll("line")
      .attr("x1", d => (d.source.x || d.source.x === 0) ? d.source.x : 0)
      .attr("y1", d => (d.source.y || d.source.y === 0) ? d.source.y : 0)
      .attr("x2", d => (d.target.x || d.target.x === 0) ? d.target.x : 0)
      .attr("y2", d => (d.target.y || d.target.y === 0) ? d.target.y : 0);
    
    gNode.selectAll("g")
      .attr("transform", d => `translate(${d.x || 0},${d.y || 0})`);
  });
}

// Update link colors based on current traffic
async function updateLinkColors() {
  try {
    const response = await fetch(`${API}/load/links`);
    const stats = await response.json();

    gLink.selectAll("line").attr("stroke", d => {
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
  } catch (error) {
    console.error("Error updating link colors:", error);
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
  
  return d3.drag()
    .on("start", dragstarted)
    .on("drag", dragged)
    .on("end", dragended);
}

// Initialize when DOM is loaded
document.addEventListener('DOMContentLoaded', function() {
  // Initialize topology visualization
  initTopology();
});