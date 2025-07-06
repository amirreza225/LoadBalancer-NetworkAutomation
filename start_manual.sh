#!/bin/bash
# Manual startup script for SDN Load Balancer
# This script only starts the controller and web dashboard
# Topology must be started manually

set -e

echo "Starting SDN Load Balancer (Manual Mode)..."

# Fix Python path for Mininet
export PYTHONPATH="/usr/local/lib/python3.8/site-packages:/usr/lib/python3/dist-packages:/usr/local/lib/python3.8/dist-packages:$PYTHONPATH"

# Start OpenVSwitch
service openvswitch-switch start

# Wait for OVS to be ready
sleep 2

# Function to cleanup on exit
cleanup() {
    echo "Cleaning up..."
    pkill -f ryu-manager 2>/dev/null || true
    pkill -f "python3 -m http.server" 2>/dev/null || true
    mn -c 2>/dev/null || true
    exit 0
}

# Set trap for cleanup
trap cleanup SIGINT SIGTERM

# Start the load balancer in background
echo "Starting Ryu controller..."
ryu-manager --observe-links lb_stp_ma_rest.py &
RYU_PID=$!

# Give controller time to start
sleep 5

# Start web dashboard in background
echo "Starting web dashboard..."
cd web && python3 -m http.server 8000 &
WEB_PID=$!
cd /app

echo "=============================================="
echo "SDN Load Balancer Started (Manual Mode)!"
echo "=============================================="
echo "Web Dashboard: http://localhost:8000"
echo "REST API: http://localhost:8080"
echo "Controller: ryu-manager (PID: $RYU_PID)"
echo "=============================================="
echo "Available REST API endpoints:"
echo "  GET  /topology            - Network topology"
echo "  GET  /load/links          - Link utilization"
echo "  GET  /stats/efficiency    - Efficiency metrics"
echo "  GET  /stats/algorithm     - Algorithm info"
echo "  POST /config/mode         - Change load balancing mode"
echo "  POST /config/threshold    - Set congestion threshold"
echo "=============================================="
echo ""
echo "🔧 MANUAL TOPOLOGY STARTUP OPTIONS:"
echo ""
echo "1. Exact Hexring Topology (6 switches, 6 hosts):"
echo "   python3 hexring_topo_mn.py"
echo ""
echo "2. Alternative approaches:"
echo "   python3 start_hexring.py"
echo "   python3 generic_topo.py --topology linear --switches 4"
echo "   mn --topo tree,depth=2,fanout=3 --controller remote,ip=127.0.0.1,port=6653 --switch ovsk --mac"
echo ""
echo "3. Quick test topology:"
echo "   mn --topo linear,4 --controller remote,ip=127.0.0.1,port=6653 --switch ovsk --mac"
echo ""
echo "=============================================="
echo "Ready for manual topology startup!"
echo "Press Ctrl+C to stop the load balancer."

# Wait for controller and web server (topology runs manually)
wait $RYU_PID $WEB_PID