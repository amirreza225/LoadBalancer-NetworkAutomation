const linkThreshold = { low: 0.3, high: 1.0 };
const THRESH_DEFAULT = 100000;

// Error tracking and retry logic
let topologyErrorCount = 0;
let lastTopologyError = null;
let lastCongestedUpdate = 0;
const MAX_TOPOLOGY_ERRORS = 3;
const TOPOLOGY_RETRY_DELAY = 5000;

// D3.js variables
let svg, simulation;
let nodes = [],
  links = [];
let topologyData = { nodes: [], links: [] };
let currentLinkLoads = {};
let nodeGroup, linkGroup;

// Professional styling constants for tidy appearance
const STYLES = {
  dimensions: {
    width: 850,
    height: 500,
    margin: { top: 40, right: 20, bottom: 40, left: 20 },
  },
  nodes: {
    switch: {
      radius: 22,
      fill: "#1e40af",
      stroke: "#1e3a8a",
      strokeWidth: 2.5,
    },
    host: {
      radius: 16,
      fill: "#db2777",
      stroke: "#be185d",
      strokeWidth: 2,
    },
  },
  links: {
    normal: { stroke: "#059669", strokeWidth: 3, opacity: 0.8 },
    warning: { stroke: "#f59e0b", strokeWidth: 4, opacity: 0.9 },
    critical: { stroke: "#dc2626", strokeWidth: 5, opacity: 1.0 },
  },
  labels: {
    fontSize: "11px",
    fontWeight: "bold",
    fill: "#ffffff",
    textAnchor: "middle",
    dy: "0.35em",
  },
  linkLabels: {
    fontSize: "9px",
    fill: "#374151",
    textAnchor: "middle",
    dy: "-0.7em",
    fontWeight: "500",
  },
};

// Initialize professional D3.js topology
function initTopology() {
  const container = d3.select("#topologyNetwork");
  if (container.empty()) {
    console.log("topologyNetwork SVG not found");
    return;
  }

  svg = container;

  // Clear any existing content
  svg.selectAll("*").remove();

  // Create professional gradient definitions
  const defs = svg.append("defs");

  // Switch gradient
  const switchGradient = defs
    .append("radialGradient")
    .attr("id", "switchGradient")
    .attr("cx", "30%")
    .attr("cy", "30%");

  switchGradient
    .append("stop")
    .attr("offset", "0%")
    .attr("stop-color", "#3b82f6");

  switchGradient
    .append("stop")
    .attr("offset", "100%")
    .attr("stop-color", "#1e40af");

  // Host gradient
  const hostGradient = defs
    .append("radialGradient")
    .attr("id", "hostGradient")
    .attr("cx", "30%")
    .attr("cy", "30%");

  hostGradient
    .append("stop")
    .attr("offset", "0%")
    .attr("stop-color", "#ec4899");

  hostGradient
    .append("stop")
    .attr("offset", "100%")
    .attr("stop-color", "#db2777");

  // Drop shadow filter
  const filter = defs
    .append("filter")
    .attr("id", "dropshadow")
    .attr("x", "-50%")
    .attr("y", "-50%")
    .attr("width", "200%")
    .attr("height", "200%");

  filter
    .append("feDropShadow")
    .attr("dx", 2)
    .attr("dy", 2)
    .attr("stdDeviation", 3)
    .attr("flood-color", "rgba(0,0,0,0.3)");

  // Create groups for links and nodes
  linkGroup = svg.append("g").attr("class", "links");
  nodeGroup = svg.append("g").attr("class", "nodes");

  // Initialize force simulation with tidy, organized parameters
  simulation = d3
    .forceSimulation()
    .force(
      "link",
      d3
        .forceLink()
        .id((d) => d.id)
        .distance(120)
        .strength(0.6)
    )
    .force("charge", d3.forceManyBody().strength(-1200))
    .force(
      "center",
      d3.forceCenter(STYLES.dimensions.width / 2, STYLES.dimensions.height / 2)
    )
    .force("collision", d3.forceCollide().radius(45))
    .force("x", d3.forceX(STYLES.dimensions.width / 2).strength(0.05))
    .force("y", d3.forceY(STYLES.dimensions.height / 2).strength(0.05))
    .alphaDecay(0.01)
    .velocityDecay(0.9);

  // Setup zoom and pan behavior
  const zoom = d3
    .zoom()
    .scaleExtent([0.5, 3])
    .on("zoom", (event) => {
      nodeGroup.attr("transform", event.transform);
      linkGroup.attr("transform", event.transform);
    });

  svg.call(zoom);

  // Start data updates
  updateTopology();
  setInterval(checkTopologyChanges, 10000); // Check structure changes less frequently
  setInterval(updateLinkColors, 500); // Update colors more frequently
  setInterval(updateCongestedLinksList, 2000);

  // Gentle layout optimization for tidiness
  setInterval(() => {
    if (simulation && nodes.length > 0) {
      // Apply gentle stabilization
      optimizeLayout();
      simulation.alpha(0.1).restart();
    }
  }, 15000);
}

// Check for topology changes - only rebuild if actually changed
async function checkTopologyChanges() {
  try {
    const response = await fetch(`${window.API}/topology`);
    if (!response.ok) {
      throw new Error(`HTTP ${response.status}: ${response.statusText}`);
    }

    const newTopology = await response.json();

    // Only rebuild if there's an actual structural change
    const hasChanged =
      !topologyData ||
      newTopology.nodes.length !== topologyData.nodes.length ||
      newTopology.links.length !== topologyData.links.length ||
      JSON.stringify(newTopology.nodes.map((n) => n.id).sort()) !==
        JSON.stringify(topologyData.nodes.map((n) => n.id).sort()) ||
      JSON.stringify(
        newTopology.links.map((l) => `${l.source}-${l.target}`).sort()
      ) !==
        JSON.stringify(
          topologyData.links.map((l) => `${l.source}-${l.target}`).sort()
        );

    if (hasChanged) {
      console.log("Topology structure changed, rebuilding...");
      topologyData = newTopology;
      buildNetworkFromTopology();
    }

    topologyErrorCount = 0;
  } catch (error) {
    console.error("Error checking topology changes:", error);
    handleTopologyError(error);
  }
}

// Update topology from API
async function updateTopology() {
  try {
    const response = await fetch(`${window.API}/topology`);
    if (!response.ok) {
      throw new Error(`HTTP ${response.status}: ${response.statusText}`);
    }

    topologyData = await response.json();
    buildNetworkFromTopology();
    topologyErrorCount = 0;
  } catch (error) {
    console.error("Error updating topology:", error);
    handleTopologyError(error);
  }
}

// Build beautiful D3.js network from topology data
function buildNetworkFromTopology() {
  if (!svg || !topologyData || !topologyData.nodes || !topologyData.links) {
    console.log("Invalid topology data or SVG not initialized");
    return;
  }

  // Prepare data with smart initial positioning
  nodes = topologyData.nodes.map((d) => ({
    ...d,
    ...getSmartInitialPosition(d, topologyData),
  }));

  links = topologyData.links.map((d) => ({
    ...d,
    load: 0,
    congestionLevel: "normal",
  }));

  // Create links with professional styling
  const link = linkGroup
    .selectAll("line")
    .data(links, (d) => `${d.source}-${d.target}`);

  link.exit().remove();

  const linkEnter = link
    .enter()
    .append("line")
    .attr("class", "link")
    .style("stroke", STYLES.links.normal.stroke)
    .style("stroke-width", STYLES.links.normal.strokeWidth)
    .style("stroke-opacity", STYLES.links.normal.opacity)
    .style("cursor", "pointer");

  // Link hover effects
  linkEnter
    .on("mouseover", function (event, d) {
      d3.select(this)
        .style("stroke", "#fbbf24")
        .style(
          "stroke-width",
          parseInt(d3.select(this).style("stroke-width")) + 2
        );
    })
    .on("mouseout", function (event, d) {
      const style = getLinkStyle(d.congestionLevel);
      d3.select(this)
        .style("stroke", style.stroke)
        .style("stroke-width", style.strokeWidth);
    })
    .on("click", function (event, d) {
      console.log("Clicked link:", d, "Load:", d.load || "0 Mbps");
    });

  const linkUpdate = linkEnter.merge(link);

  // Skip link labels - we don't want speed text on links

  // Create nodes with professional styling
  const node = nodeGroup.selectAll(".node").data(nodes, (d) => d.id);

  node.exit().remove();

  const nodeEnter = node
    .enter()
    .append("g")
    .attr("class", "node")
    .style("cursor", "pointer")
    .call(
      d3
        .drag()
        .on("start", dragstarted)
        .on("drag", dragged)
        .on("end", dragended)
    );

  // Add professional node circles
  nodeEnter
    .append("circle")
    .attr("r", (d) =>
      d.type === "switch"
        ? STYLES.nodes.switch.radius
        : STYLES.nodes.host.radius
    )
    .style("fill", (d) =>
      d.type === "switch" ? "url(#switchGradient)" : "url(#hostGradient)"
    )
    .style("stroke", (d) =>
      d.type === "switch"
        ? STYLES.nodes.switch.stroke
        : STYLES.nodes.host.stroke
    )
    .style("stroke-width", (d) =>
      d.type === "switch"
        ? STYLES.nodes.switch.strokeWidth
        : STYLES.nodes.host.strokeWidth
    )
    .style("filter", "url(#dropshadow)");

  // Add node labels
  nodeEnter
    .append("text")
    .style("font-size", STYLES.labels.fontSize)
    .style("font-weight", STYLES.labels.fontWeight)
    .style("fill", STYLES.labels.fill)
    .style("text-anchor", STYLES.labels.textAnchor)
    .style("dy", STYLES.labels.dy)
    .style("pointer-events", "none")
    .text((d) => d.id);

  const nodeUpdate = nodeEnter.merge(node);

  // Professional node interactions
  nodeUpdate
    .on("mouseover", function (event, d) {
      // Highlight node
      d3.select(this)
        .select("circle")
        .style("stroke", "#fbbf24")
        .style("stroke-width", 4);

      // Highlight connected links
      linkUpdate
        .style("stroke", (l) =>
          l.source.id === d.id || l.target.id === d.id ? "#fbbf24" : null
        )
        .style("stroke-width", (l) =>
          l.source.id === d.id || l.target.id === d.id
            ? parseInt(d3.select(this).style("stroke-width")) + 2
            : null
        );
    })
    .on("mouseout", function (event, d) {
      // Reset node
      d3.select(this)
        .select("circle")
        .style(
          "stroke",
          d.type === "switch"
            ? STYLES.nodes.switch.stroke
            : STYLES.nodes.host.stroke
        )
        .style(
          "stroke-width",
          d.type === "switch"
            ? STYLES.nodes.switch.strokeWidth
            : STYLES.nodes.host.strokeWidth
        );

      // Reset links
      linkUpdate.each(function (l) {
        const style = getLinkStyle(l.congestionLevel);
        d3.select(this)
          .style("stroke", style.stroke)
          .style("stroke-width", style.strokeWidth);
      });
    })
    .on("click", function (event, d) {
      console.log("Clicked node:", d.id, "Type:", d.type);
    });

  // Update simulation
  simulation.nodes(nodes);
  simulation.force("link").links(links);

  // Tick function for smooth animations
  simulation.on("tick", () => {
    linkUpdate
      .attr("x1", (d) => d.source.x)
      .attr("y1", (d) => d.source.y)
      .attr("x2", (d) => d.target.x)
      .attr("y2", (d) => d.target.y);

    nodeUpdate.attr("transform", (d) => `translate(${d.x},${d.y})`);
  });

  simulation.alpha(1).restart();

  console.log(`Updated topology: ${nodes.length} nodes, ${links.length} links`);
}

// Optimize layout for better tidiness
function optimizeLayout() {
  if (!nodes || nodes.length === 0) return;

  const { width, height } = STYLES.dimensions;
  const center = { x: width / 2, y: height / 2 };

  // Apply gentle corrections for better organization
  nodes.forEach((node) => {
    // Prevent nodes from getting too close to edges
    const margin = 60;
    if (node.x < margin) node.x = margin;
    if (node.x > width - margin) node.x = width - margin;
    if (node.y < margin) node.y = margin;
    if (node.y > height - margin) node.y = height - margin;

    // Gentle pull toward center for switches
    if (node.type === "switch") {
      const dx = center.x - node.x;
      const dy = center.y - node.y;
      const distance = Math.sqrt(dx * dx + dy * dy);

      if (distance > 0) {
        node.x += dx * 0.01; // Gentle correction
        node.y += dy * 0.01;
      }
    }
  });
}

// Smart initial positioning for tidy layouts
function getSmartInitialPosition(node, topologyData) {
  const { width, height } = STYLES.dimensions;
  const center = { x: width / 2, y: height / 2 };
  const margin = 80;

  // Separate switches and hosts
  const switches = topologyData.nodes.filter((n) => n.type === "switch");
  const hosts = topologyData.nodes.filter((n) => n.type === "host");

  if (node.type === "switch") {
    const switchIndex = switches.findIndex((n) => n.id === node.id);
    const totalSwitches = switches.length;

    // Arrange switches in a tidy circle or grid
    if (totalSwitches <= 6) {
      // Circular arrangement for small topologies
      const angle = (switchIndex * 2 * Math.PI) / totalSwitches;
      const radius = Math.min(width, height) * 0.25;
      return {
        x: center.x + radius * Math.cos(angle),
        y: center.y + radius * Math.sin(angle),
      };
    } else {
      // Grid arrangement for larger topologies
      const cols = Math.ceil(Math.sqrt(totalSwitches));
      const row = Math.floor(switchIndex / cols);
      const col = switchIndex % cols;
      const spacing = Math.min(
        (width - 2 * margin) / cols,
        (height - 2 * margin) / Math.ceil(totalSwitches / cols)
      );

      return {
        x: margin + col * spacing + spacing / 2,
        y: margin + row * spacing + spacing / 2,
      };
    }
  } else {
    // Position hosts around their connected switches
    const hostIndex = hosts.findIndex((n) => n.id === node.id);
    const totalHosts = hosts.length;

    // Find connected switch
    const connectedLink = topologyData.links.find(
      (l) => l.source === node.id || l.target === node.id
    );

    if (connectedLink) {
      const switchId =
        connectedLink.source === node.id
          ? connectedLink.target
          : connectedLink.source;
      const switchNode = switches.find((s) => s.id === switchId);

      if (switchNode) {
        // Position host near its switch in a tidy manner
        const angle = (hostIndex * 2 * Math.PI) / totalHosts;
        const radius = 48; // Distance from switch (20% closer than 60)
        const switchPos = getSmartInitialPosition(switchNode, topologyData);

        return {
          x: switchPos.x + radius * Math.cos(angle),
          y: switchPos.y + radius * Math.sin(angle),
        };
      }
    }

    // Fallback: outer ring positioning
    const angle = (hostIndex * 2 * Math.PI) / totalHosts;
    const radius = Math.min(width, height) * 0.32; // 20% closer than 0.4
    return {
      x: center.x + radius * Math.cos(angle),
      y: center.y + radius * Math.sin(angle),
    };
  }
}

// Get link style based on congestion level
function getLinkStyle(congestionLevel) {
  switch (congestionLevel) {
    case "critical":
      return STYLES.links.critical;
    case "warning":
      return STYLES.links.warning;
    default:
      return STYLES.links.normal;
  }
}

// Update link colors and congestion indicators
async function updateLinkColors() {
  if (!svg) return;

  try {
    const response = await fetch(`${window.API}/load/links`);
    if (!response.ok) return;

    const linkLoads = await response.json();
    currentLinkLoads = linkLoads;

    // Update each link with current load
    linkGroup.selectAll("line").each(function (d) {
      // Get source and target IDs (handle both object and primitive cases)
      const sourceId = typeof d.source === "object" ? d.source.id : d.source;
      const targetId = typeof d.target === "object" ? d.target.id : d.target;

      // Find matching link load data with better matching logic
      let load = 0;
      let matchedKey = null;

      for (const [linkKey, loadData] of Object.entries(linkLoads)) {
        // Try exact matches first
        if (
          linkKey === `${sourceId}-${targetId}` ||
          linkKey === `${targetId}-${sourceId}`
        ) {
          load =
            typeof loadData === "object" ? loadData.current || 0 : loadData;
          matchedKey = linkKey;
          break;
        }

        // Try with 's' prefix if not found
        if (
          linkKey === `s${sourceId}-s${targetId}` ||
          linkKey === `s${targetId}-s${sourceId}`
        ) {
          load =
            typeof loadData === "object" ? loadData.current || 0 : loadData;
          matchedKey = linkKey;
          break;
        }

        // Try removing 's' prefix if present
        const cleanSource = sourceId.toString().replace("s", "");
        const cleanTarget = targetId.toString().replace("s", "");
        if (
          linkKey === `${cleanSource}-${cleanTarget}` ||
          linkKey === `${cleanTarget}-${cleanSource}`
        ) {
          load =
            typeof loadData === "object" ? loadData.current || 0 : loadData;
          matchedKey = linkKey;
          break;
        }
      }

      const loadMbps = (load / 1000000).toFixed(1);
      const currentThreshold = (window.threshold || 25) * 1000000;

      // Determine congestion level - very sensitive to show orange warning
      let congestionLevel = "normal";
      if (load > currentThreshold * 1.2) {
        congestionLevel = "critical"; // Red at 120% of threshold (over limit)
      } else if (load > currentThreshold * 0.3) {
        congestionLevel = "warning"; // Orange at 30% of threshold (much more sensitive)
      }

      d.load = `${loadMbps} Mbps`;
      d.congestionLevel = congestionLevel;

      // Update link appearance immediately
      const style = getLinkStyle(congestionLevel);
      d3.select(this)
        .style("stroke", style.stroke)
        .style("stroke-width", style.strokeWidth)
        .style("stroke-opacity", style.opacity);
    });

    // Link labels removed - no speed text needed
  } catch (error) {
    console.log("Error updating link colors:", error);
  }
}

// Simple congested links display - just show basic info
async function updateCongestedLinksList() {
  const hotLinksElement = document.getElementById("hotLinks");
  if (!hotLinksElement || !currentLinkLoads) return;

  const currentThreshold = (window.threshold || 25) * 1000000;
  const congestedLinks = [];

  for (const [linkKey, loadData] of Object.entries(currentLinkLoads)) {
    const load =
      typeof loadData === "object" ? loadData.current || 0 : loadData;
    if (load > currentThreshold * 0.7) {
      const loadMbps = (load / 1000000).toFixed(1);
      congestedLinks.push(`${linkKey}: ${loadMbps} Mbps`);
    }
  }

  // Simple display
  if (congestedLinks.length === 0) {
    hotLinksElement.innerHTML = '<li class="ok">None</li>';
  } else {
    hotLinksElement.innerHTML = congestedLinks
      .map((link) => `<li class="hot">${link}</li>`)
      .join("");
  }

  lastCongestedUpdate = Date.now();
}

// Professional drag behavior
function dragstarted(event, d) {
  if (!event.active) simulation.alphaTarget(0.3).restart();
  d.fx = d.x;
  d.fy = d.y;
}

function dragged(event, d) {
  d.fx = event.x;
  d.fy = event.y;
}

function dragended(event, d) {
  if (!event.active) simulation.alphaTarget(0);
  d.fx = null;
  d.fy = null;
}

// Handle topology errors
function handleTopologyError(error) {
  topologyErrorCount++;
  lastTopologyError = error;

  if (topologyErrorCount >= MAX_TOPOLOGY_ERRORS) {
    console.error(
      `Topology update failed ${MAX_TOPOLOGY_ERRORS} times, stopping automatic updates`
    );
    return;
  }

  console.log(
    `Topology error ${topologyErrorCount}/${MAX_TOPOLOGY_ERRORS}, retrying in ${TOPOLOGY_RETRY_DELAY}ms...`
  );
  setTimeout(() => {
    updateTopology();
  }, TOPOLOGY_RETRY_DELAY);
}

// Initialize when DOM is ready
document.addEventListener("DOMContentLoaded", function () {
  if (document.getElementById("topologyNetwork")) {
    initTopology();
  }
});
