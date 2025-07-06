#!/bin/bash
# Minimal startup - only essential setup

set -e

# Fix Python path for Mininet
export PYTHONPATH="/usr/local/lib/python3.8/site-packages:/usr/lib/python3/dist-packages:/usr/local/lib/python3.8/dist-packages:$PYTHONPATH"

# Start OpenVSwitch (required for networking)
service openvswitch-switch start >/dev/null 2>&1

# Wait for OVS to be ready
sleep 2

echo "SDN Load Balancer Environment Ready!"
echo ""
echo "Manual startup commands:"
echo "  Use: ./docker-run.sh shell"
echo "  Then: ryu-manager --observe-links --ofp-tcp-listen-port 6653 --wsapi-port 8080 --wsapi-host 0.0.0.0 lb_stp_ma_rest.py"
echo "  Then: cd web && python3 -m http.server 8000"
echo "  Then: sudo python3 hexring_topo.py"
echo ""
echo "Container ready."

# Keep container running
tail -f /dev/null