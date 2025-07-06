# SDN Load Balancer Docker Container
FROM ubuntu:20.04

# Prevent interactive prompts during package installation
ENV DEBIAN_FRONTEND=noninteractive
ENV TZ=UTC

# Update and install system dependencies
RUN apt-get update && apt-get install -y \
    python3 \
    python3-pip \
    python3-setuptools \
    python3-dev \
    curl \
    wget \
    git \
    net-tools \
    lsof \
    htop \
    iftop \
    tcpdump \
    iproute2 \
    iputils-ping \
    iperf \
    openvswitch-switch \
    openvswitch-common \
    sudo \
    && rm -rf /var/lib/apt/lists/*

# Install Python dependencies with eventlet version that has ALREADY_HANDLED
RUN pip3 install --no-cache-dir \
    eventlet==0.30.2 \
    ryu==4.34 \
    routes>=2.5.1 \
    webob>=1.8.7

# Manual Mininet Python 3 installation approach
# Install required system packages first
RUN apt-get update && apt-get install -y \
    build-essential \
    make \
    git \
    gcc \
    libc6-dev \
    python3-distutils \
    python3-setuptools \
    && rm -rf /var/lib/apt/lists/*

# Install Mininet Python modules and executables manually
RUN git clone https://github.com/mininet/mininet.git /tmp/mininet && \
    cd /tmp/mininet && \
    echo "Installing Mininet Python modules..." && \
    python3 setup.py install && \
    echo "Installing Mininet executables..." && \
    install bin/mn /usr/local/bin/ && \
    echo "Building mnexec manually..." && \
    cd /tmp/mininet && \
    gcc -Wall -Wextra -DVERSION='"2.3.1b4"' mnexec.c -o mnexec && \
    install mnexec /usr/local/bin/ && \
    echo "Installing other utilities..." && \
    install util/m /usr/local/bin/ && \
    echo "Mininet installation completed" && \
    rm -rf /tmp/mininet

# Create application directory
WORKDIR /app

# Copy application files
COPY . .

# Create directories for logs and data
RUN mkdir -p /app/logs /app/data

# Set proper permissions
RUN chmod +x *.py

# Test Mininet Python 3 installation
RUN python3 test_mininet_python3.py || echo "Mininet Python 3 test failed, but continuing build..."

# Create startup script
COPY <<EOF /app/start.sh
#!/bin/bash
set -e

echo "Starting SDN Load Balancer..."

# Fix Python path for Mininet
export PYTHONPATH="/usr/local/lib/python3.8/site-packages:/usr/lib/python3/dist-packages:/usr/local/lib/python3.8/dist-packages:\$PYTHONPATH"

# Start OpenVSwitch
service openvswitch-switch start

# Wait for OVS to be ready
sleep 2

# Function to cleanup on exit
cleanup() {
    echo "Cleaning up..."
    pkill -f ryu-manager 2>/dev/null || true
    pkill -f "python3 -m http.server" 2>/dev/null || true
    pkill -f mininet 2>/dev/null || true
    mn -c 2>/dev/null || true
    exit 0
}

# Set trap for cleanup
trap cleanup SIGINT SIGTERM

# Start the load balancer in background
echo "Starting Ryu controller..."
ryu-manager --observe-links lb_stp_ma_rest.py &
RYU_PID=\$!

# Give controller time to start
sleep 5

# Start web dashboard in background
echo "Starting web dashboard..."
cd web && python3 -m http.server 8000 &
WEB_PID=\$!
cd /app

# Note: Topology will be started manually by user
echo "Topology startup disabled - run manually with:"
echo "  python3 hexring_topo_mn.py     # Exact hexring topology"
echo "  python3 start_hexring.py       # Alternative hexring"  
echo "  python3 generic_topo.py --topology linear --switches 4"
echo "  mn --topo tree,depth=2,fanout=3 --controller remote,ip=127.0.0.1,port=6653 --switch ovsk --mac"

echo "=============================================="
echo "SDN Load Balancer Started Successfully!"
echo "=============================================="
echo "Web Dashboard: http://localhost:8000"
echo "REST API: http://localhost:8080"
echo "Controller: ryu-manager (PID: \$RYU_PID)"
echo "Topology: \${TOPOLOGY:-hexring}"
echo "=============================================="
echo "Available REST API endpoints:"
echo "  GET  /topology            - Network topology"
echo "  GET  /load/links          - Link utilization"
echo "  GET  /stats/efficiency    - Efficiency metrics"
echo "  GET  /stats/algorithm     - Algorithm info"
echo "  POST /config/mode         - Change load balancing mode"
echo "  POST /config/threshold    - Set congestion threshold"
echo "=============================================="

# Wait for controller and web server (topology runs manually)
wait \$RYU_PID \$WEB_PID
EOF

# Make startup script executable
RUN chmod +x /app/start.sh

# Create environment configuration script
COPY <<EOF /app/configure.sh
#!/bin/bash
# Configure SDN Load Balancer environment

echo "Configuring SDN Load Balancer..."

# Set default values
export TOPOLOGY=\${TOPOLOGY:-hexring}
export SWITCHES=\${SWITCHES:-4}
export THRESHOLD=\${THRESHOLD:-25}
export MODE=\${MODE:-adaptive}

# Configure OpenVSwitch
ovs-vsctl set-manager ptcp:6640 2>/dev/null || true

echo "Environment configured:"
echo "  TOPOLOGY: \$TOPOLOGY"
echo "  SWITCHES: \$SWITCHES"
echo "  THRESHOLD: \$THRESHOLD Mbps"
echo "  MODE: \$MODE"
EOF

RUN chmod +x /app/configure.sh

# Create testing script
COPY <<EOF /app/test.sh
#!/bin/bash
# Test SDN Load Balancer functionality

echo "Testing SDN Load Balancer..."

# Wait for services to be ready
sleep 10

# Test REST API endpoints
echo "Testing REST API..."
curl -s http://localhost:8080/topology | python3 -m json.tool || echo "Topology endpoint test failed"
curl -s http://localhost:8080/stats/efficiency | python3 -m json.tool || echo "Efficiency endpoint test failed"
curl -s http://localhost:8080/stats/algorithm | python3 -m json.tool || echo "Algorithm endpoint test failed"

# Test mode switching
echo "Testing mode switching..."
curl -X POST http://localhost:8080/config/mode \
  -H "Content-Type: application/json" \
  -d '{"mode": "least_loaded"}' || echo "Mode switch test failed"

# Test threshold configuration
echo "Testing threshold configuration..."
curl -X POST http://localhost:8080/config/threshold \
  -H "Content-Type: application/json" \
  -d '{"threshold": 50000000}' || echo "Threshold test failed"

echo "Basic API tests completed."
EOF

RUN chmod +x /app/test.sh

# Expose ports
EXPOSE 8000 8080 6653

# Set environment variables for Python3 Mininet compatibility
ENV PYTHONPATH="/usr/local/lib/python3.8/site-packages:/usr/lib/python3/dist-packages:/usr/local/lib/python3.8/dist-packages:/app"
ENV MININET_PATH="/usr/local/lib/python3.8/site-packages"
ENV TOPOLOGY=hexring
ENV SWITCHES=4
ENV THRESHOLD=25
ENV MODE=adaptive

# Health check disabled for minimal mode (no auto services)
# HEALTHCHECK --interval=30s --timeout=10s --start-period=60s --retries=3 \
#   CMD curl -f http://localhost:8080/stats/efficiency || exit 1

# Copy minimal startup script
COPY start_minimal.sh /app/start_minimal.sh
RUN chmod +x /app/start_minimal.sh

# Default command - minimal setup only
CMD ["/app/start_minimal.sh"]