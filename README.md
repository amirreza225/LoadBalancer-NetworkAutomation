# SDN Load Balancer using Ryu and Mininet

This project implements a dynamic load balancer in a Software Defined Network (SDN) environment using the Ryu controller and Mininet. It features real-time traffic monitoring, adaptive path computation, and visualization through a RESTful API and a browser-based UI.

---

## 📚 Overview

- **Controller**: [Ryu SDN Framework](https://osrg.github.io/ryu/)
- **Emulator**: [Mininet](http://mininet.org/)
- **Protocol**: OpenFlow 1.3
- **Topology**: 6-switch hexagonal ring with redundant chordal links
- **Goal**: Minimize congestion and optimize flow paths based on traffic statistics

---

## 🔧 Architecture

- **Mininet Topology**: Custom-built with 6 switches (DPIDs 1–6), each connected to one host (h1–h6), with both ring and chordal links to offer multiple routing paths.
- **Ryu Controller**: Implements monitoring, Dijkstra’s algorithm with moving-average link costs, and proactive flow installation.
- **Web Interface**: Visualizes the network topology with D3.js and displays bandwidth statistics with Chart.js.
- **REST API**: Exposes live switch and flow metrics.

---

## 🗂️ Project Structure

```
loadbalancer/
├── controller/
│   └── lb_stp_ma_rest.py        # Main Ryu app with STP, MA, REST logic
├── topology/
│   └── hexring_topo.py          # Mininet topology (6-switch hex ring)
├── utils/
│   └── portmap.py               # Static port adjacency map
├── web/
│   ├── index.html               # D3.js topology + Chart.js frontend
│   ├── topology.js              # D3-based graph visualizer
│   └── app.js                   # Bandwidth charting with Chart.js
├── LICENSE                       # MIT License
└── README.md                     # Project overview
```

---

## ✅ Features

- 📡 **Topology Discovery** via static mapping
- 📊 **Traffic Monitoring** via port stats polling
- 🔀 **Dynamic Path Selection** with Dijkstra’s algorithm
- 🔁 **Flow Installation** proactive and reactive
- 🌍 **Web Visualization** using D3.js and Chart.js
- 🔌 **REST API Access** for external tools or UI

---

## ⚙️ Requirements

- Python 3.x
- Mininet
- Ryu controller (OpenFlow 1.3 support)
- Web browser (for frontend)
- `iperf` (optional for traffic testing)

Preferably use mininet virtual machine for a more stable environment.

Alternatively:
Install Ryu and Mininet:

```bash
sudo apt install mininet
pip install ryu
```

---

## 🚀 Running the Project

1. **Start Mininet Topology**

```bash
sudo python3 topology/hexring_topo.py
```

2. **Run the Ryu Controller**

```bash
ryu-manager controller/lb_stp_ma_rest.py
```

3. **Launch Web UI**

Simply open `web/index.html` in your browser (Chrome/Firefox).

4. **Generate Traffic (Optional)**

```bash
mininet> h1 iperf -s &
mininet> h6 iperf -c h1
```

---

## 🌐 REST API Endpoints

| Endpoint              | Description                          |
|-----------------------|--------------------------------------|
| `/stats/ports`        | Get per-port byte statistics         |
| `/stats/flows`        | List of active flow entries          |
| `/stats/paths`        | Current least-cost paths             |

Access via:

```bash
curl http://localhost:8080/stats/ports
```

---

## 📈 Visualization

- **D3.js**: Shows live topology with switch-host mapping.
- **Chart.js**: Line charts display real-time bandwidth usage per port.

Both are included via CDN in `web/index.html`.

---

## 🧩 Future Enhancements

- Multi-controller support
- Machine learning-based path prediction
- Integration with external SDN orchestrators
- REST API authentication & rate-limiting

---

## 📄 License

This project is licensed under the MIT License.

---

## 👨‍💻 Authors

Developed by **Amirreza Alibeigi and Reza Ghadiri Abkenari**  
Politecnico di Milano – SDN Network Automation Project  
GitHub: [@amirreza225](https://github.com/amirreza225)
GitHub: [@rghaf](https://github.com/rghaf)
