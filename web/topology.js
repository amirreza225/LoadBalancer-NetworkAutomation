const linkThreshold = { low: 0.3, high: 1.0 };
// const API = "http://localhost:8080";
const THRESH_DEFAULT = 1000000;
// let threshold = THRESH_DEFAULT;

const nodes = [
  // Switches
  { id: "s1" }, { id: "s2" }, { id: "s3" },
  { id: "s4" }, { id: "s5" }, { id: "s6" },
  // Hosts
  { id: "h1" }, { id: "h2" }, { id: "h3" },
  { id: "h4" }, { id: "h5" }, { id: "h6" },
];

const links = [
  // Switch-Switch
  { source: "s1", target: "s2" }, { source: "s2", target: "s3" },
  { source: "s3", target: "s4" }, { source: "s4", target: "s5" },
  { source: "s5", target: "s6" }, { source: "s6", target: "s1" },
  { source: "s6", target: "s3" }, { source: "s2", target: "s5" },
  { source: "s1", target: "s4" },
  // Hosts
  { source: "h1", target: "s1" },
  { source: "h2", target: "s2" },
  { source: "h3", target: "s3" },
  { source: "h4", target: "s4" },
  { source: "h5", target: "s5" },
  { source: "h6", target: "s6" }
];

const svg = d3.select("svg"), width = +svg.attr("width"), height = +svg.attr("height");
const linkColor = bps => {
  if (bps > threshold * linkThreshold.high) return "red";
  if (bps > threshold * linkThreshold.low) return "orange";
  return "green";
};

const simulation = d3.forceSimulation(nodes)
  .force("link", d3.forceLink(links).id(d => d.id).distance(80))
  .force("charge", d3.forceManyBody().strength(-400))
  .force("center", d3.forceCenter(width / 2, height / 2));

const gLink = svg.append("g")
  .attr("stroke", "#999").attr("stroke-opacity", 0.6)
  .selectAll("line")
  .data(links)
  .join("line")
  .attr("class", "link")
  .attr("stroke-width", 3);

const gNode = svg.append("g")
  .attr("stroke", "#fff").attr("stroke-width", 1.5)
  .selectAll("g")
  .data(nodes)
  .join("g")
  .call(drag(simulation));

gNode.append("circle")
  .attr("r", 10)
  .attr("fill", d => d.id.startsWith("h") ? "#bbb" : "#1f77b4");

gNode.append("text")
  .text(d => d.id)
  .attr("x", 12).attr("y", 4)
  .attr("stroke", "black");

simulation.on("tick", () => {
  gLink
    .attr("x1", d => d.source.x)
    .attr("y1", d => d.source.y)
    .attr("x2", d => d.target.x)
    .attr("y2", d => d.target.y);
  gNode
    .attr("transform", d => `translate(${d.x},${d.y})`);
});

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

async function updateLinkColors() {
  try {
    const res = await fetch(`${API}/load/links`);
    const stats = await res.json();

    gLink.attr("stroke", d => {
      const u = d.source.id.replace("s", ""), v = d.target.id.replace("s", "");
      const key1 = `${u}-${v}`;
      const key2 = `${v}-${u}`;
      const bps = stats[key1] || stats[key2] || 0;
      return linkColor(bps);
    });
  } catch (e) {
    console.error("Error updating link colors:", e);
  }
}

setInterval(updateLinkColors, 1000);

fetch(`${API}/config/threshold`).then(res => res.json()).then(cfg => {
  if (cfg.threshold) threshold = cfg.threshold;
});
