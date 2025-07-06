#!/bin/bash
# Minimal startup script - only sets up environment
# No automatic service startup

set -e

echo "Initializing SDN Load Balancer Environment..."

# Fix Python path for Mininet
export PYTHONPATH="/usr/local/lib/python3.8/site-packages:/usr/lib/python3/dist-packages:/usr/local/lib/python3.8/dist-packages:$PYTHONPATH"

# Start OpenVSwitch (required for any network operations)
service openvswitch-switch start

# Wait for OVS to be ready
sleep 2

echo "=============================================="
echo "SDN Load Balancer Environment Ready!"
echo "=============================================="
echo "Environment setup complete. Ready for manual operations."
echo ""
echo "🔧 MANUAL STARTUP COMMANDS:"
echo ""
echo "1. Start Ryu Controller:"
echo "   ryu-manager --observe-links lb_stp_ma_rest.py"
echo ""
echo "2. Start Web Dashboard:"
echo "   cd web && python3 -m http.server 8000"
echo ""
echo "3. Start Hexring Topology:"
echo "   sudo python3 hexring_topo.py"
echo ""
echo "4. Alternative topologies:"
echo "   sudo python3 generic_topo.py --topology linear --switches 4"
echo "   sudo mn --topo linear,4 --controller remote,ip=127.0.0.1,port=6653 --switch ovsk --mac"
echo ""
echo "5. Test Mininet Python 3:"
echo "   python3 test_mininet_python3.py"
echo ""
echo "=============================================="
echo "All services require MANUAL startup!"

# Keep container running for exec commands
echo "Container ready for docker exec commands..."
# Keep container alive
tail -f /dev/null