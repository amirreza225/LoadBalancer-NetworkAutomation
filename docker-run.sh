#!/bin/bash
# Docker run script for SDN Load Balancer

set -e

echo "SDN Load Balancer Docker Management Script"
echo "=========================================="

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Function to print colored output
print_info() {
    echo -e "${BLUE}[INFO]${NC} $1"
}

print_success() {
    echo -e "${GREEN}[SUCCESS]${NC} $1"
}

print_warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
}

print_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

# Function to check if Docker is running
check_docker() {
    if ! docker info >/dev/null 2>&1; then
        print_error "Docker is not running. Please start Docker first."
        exit 1
    fi
}

# Function to build the Docker image
build_image() {
    print_info "Building SDN Load Balancer Docker image..."
    docker build -t sdn-load-balancer:latest .
    print_success "Docker image built successfully!"
}

# Function to run the container
run_container() {
    local topology=${1:-hexring}
    local switches=${2:-4}
    local threshold=${3:-25}
    local mode=${4:-adaptive}
    
    print_info "Starting SDN Load Balancer container..."
    print_info "Configuration:"
    print_info "  Topology: $topology"
    print_info "  Switches: $switches"
    print_info "  Threshold: $threshold Mbps"
    print_info "  Mode: $mode"
    
    # Stop existing container if running
    docker stop sdn-lb 2>/dev/null || true
    docker rm sdn-lb 2>/dev/null || true
    
    # Create logs and data directories
    mkdir -p logs data
    
    # Run the container
    docker run -d \
        --name sdn-lb \
        --privileged \
        -v "$(pwd)/logs:/app/logs" \
        -v "$(pwd)/data:/app/data" \
        -v "/lib/modules:/lib/modules:ro" \
        -e TOPOLOGY="$topology" \
        -e SWITCHES="$switches" \
        -e THRESHOLD="$threshold" \
        -e MODE="$mode" \
        -p 8000:8000 \
        -p 8080:8080 \
        -p 6653:6653 \
        sdn-load-balancer:latest
    
    # Wait for container to start
    sleep 5
    
    if docker ps | grep -q sdn-lb; then
        print_success "SDN Load Balancer started successfully!"
        print_info "Access points:"
        print_info "  Web Dashboard: http://localhost:8000"
        print_info "  REST API: http://localhost:8080"
        print_info "  OpenFlow Controller: localhost:6653"
    else
        print_error "Failed to start SDN Load Balancer container"
        print_info "Checking logs..."
        docker logs sdn-lb
        exit 1
    fi
}

# Function to run container in manual mode
run_manual_container() {
    print_info "Starting SDN Load Balancer container in MANUAL mode..."
    print_info "Only controller and web dashboard will start automatically."
    print_info "You will start topologies manually."
    
    # Stop existing container if running
    docker stop sdn-lb 2>/dev/null || true
    docker rm sdn-lb 2>/dev/null || true
    
    # Create logs and data directories
    mkdir -p logs data
    
    # Run the container with manual startup script
    docker run -d \
        --name sdn-lb \
        --privileged \
        -v "$(pwd)/logs:/app/logs" \
        -v "$(pwd)/data:/app/data" \
        -v "/lib/modules:/lib/modules:ro" \
        -p 8000:8000 \
        -p 8080:8080 \
        -p 6653:6653 \
        sdn-load-balancer:latest /app/start_manual.sh
    
    # Wait for container to start
    sleep 5
    
    if docker ps | grep -q sdn-lb; then
        print_success "SDN Load Balancer started in MANUAL mode!"
        print_info "Access points:"
        print_info "  Web Dashboard: http://localhost:8000"
        print_info "  REST API: http://localhost:8080"
        print_info "  OpenFlow Controller: localhost:6653"
        print_info ""
        print_info "To start topologies manually:"
        print_info "  docker exec -it sdn-lb python3 hexring_topo_mn.py    # Exact hexring"
        print_info "  docker exec -it sdn-lb mn --topo linear,4 --controller remote,ip=127.0.0.1,port=6653 --switch ovsk --mac"
        print_info "  docker exec -it sdn-lb /bin/bash    # Enter container shell"
    else
        print_error "Failed to start SDN Load Balancer container"
        print_info "Checking logs..."
        docker logs sdn-lb
        exit 1
    fi
}

# Function to run container in minimal mode (no auto services)
run_minimal_container() {
    print_info "Starting SDN Load Balancer container in MINIMAL mode..."
    print_info "NO services will start automatically."
    print_info "Complete manual control over all components."
    
    # Stop existing container if running
    docker stop sdn-lb 2>/dev/null || true
    docker rm sdn-lb 2>/dev/null || true
    
    # Create logs and data directories
    mkdir -p logs data
    
    # Run the container with minimal startup script
    docker run -d \
        --name sdn-lb \
        --privileged \
        -v "$(pwd)/logs:/app/logs" \
        -v "$(pwd)/data:/app/data" \
        -v "/lib/modules:/lib/modules:ro" \
        -p 8000:8000 \
        -p 8080:8080 \
        -p 6653:6653 \
        sdn-load-balancer:latest /app/start_minimal.sh
    
    # Wait for container to start
    sleep 3
    
    # Check if container exists (it may not show in ps due to interactive mode)
    if docker container inspect sdn-lb >/dev/null 2>&1; then
        print_success "SDN Load Balancer started in MINIMAL mode!"
        print_info "Container is ready with NO auto-started services."
        print_info ""
        print_info "Manual startup commands:"
        print_info "  docker exec -it sdn-lb ryu-manager --observe-links lb_stp_ma_rest.py"
        print_info "  docker exec -it sdn-lb bash -c 'cd web && python3 -m http.server 8000'"
        print_info "  docker exec -it sdn-lb sudo python3 hexring_topo.py"
        print_info "  docker exec -it sdn-lb /bin/bash    # Enter container shell"
        print_info ""
        print_info "Container is running in interactive mode. Use 'docker exec' commands above."
    else
        print_error "Failed to start SDN Load Balancer container"
        print_info "Checking logs..."
        docker logs sdn-lb
        exit 1
    fi
}

# Function to stop the container
stop_container() {
    print_info "Stopping SDN Load Balancer container..."
    docker stop sdn-lb 2>/dev/null || true
    docker rm sdn-lb 2>/dev/null || true
    print_success "Container stopped and removed"
}

# Function to show container logs
show_logs() {
    print_info "Showing SDN Load Balancer logs..."
    docker logs -f sdn-lb
}

# Function to enter container shell
enter_container() {
    print_info "Entering SDN Load Balancer container..."
    docker exec -it sdn-lb /bin/bash
}

# Function to run tests
run_tests() {
    print_info "Running SDN Load Balancer tests..."
    docker exec sdn-lb /app/test.sh
}

# Function to show status
show_status() {
    print_info "SDN Load Balancer Status:"
    
    if docker ps | grep -q sdn-lb; then
        print_success "Container is running"
        
        # Check if services are responding
        if curl -s http://localhost:8000 >/dev/null 2>&1; then
            print_success "Web Dashboard: http://localhost:8000 (✓ Accessible)"
        else
            print_warning "Web Dashboard: http://localhost:8000 (✗ Not accessible)"
        fi
        
        if curl -s http://localhost:8080/stats/efficiency >/dev/null 2>&1; then
            print_success "REST API: http://localhost:8080 (✓ Accessible)"
        else
            print_warning "REST API: http://localhost:8080 (✗ Not accessible)"
        fi
        
        # Show container stats
        docker stats sdn-lb --no-stream
    else
        print_error "Container is not running"
    fi
}

# Function to show help
show_help() {
    echo "Usage: $0 [COMMAND] [OPTIONS]"
    echo ""
    echo "Commands:"
    echo "  build                          Build Docker image"
    echo "  run [topology] [switches] [threshold] [mode]"
    echo "                                 Run container with configuration"
    echo "  run-manual                     Run container with manual topology control"
    echo "  run-minimal                    Run container with NO auto-started services"
    echo "  stop                           Stop and remove container"
    echo "  logs                           Show container logs"
    echo "  shell                          Enter container shell"
    echo "  test                           Run tests"
    echo "  status                         Show container status"
    echo "  help                           Show this help message"
    echo ""
    echo "Default configuration:"
    echo "  topology: hexring"
    echo "  switches: 4"
    echo "  threshold: 25 (Mbps)"
    echo "  mode: adaptive"
    echo ""
    echo "Available topologies: hexring, linear, ring, tree, mesh"
    echo "Available modes: adaptive, least_loaded, weighted_ecmp, round_robin,"
    echo "                 latency_aware, qos_aware, flow_aware"
    echo ""
    echo "Examples:"
    echo "  $0 build"
    echo "  $0 run"
    echo "  $0 run linear 5 50 least_loaded"
    echo "  $0 run-manual                    # Controller + web dashboard auto-start"
    echo "  $0 run-minimal                   # Nothing auto-starts"
    echo "  $0 status"
    echo "  $0 logs"
}

# Main script logic
check_docker

case "${1:-help}" in
    build)
        build_image
        ;;
    run)
        build_image
        run_container "$2" "$3" "$4" "$5"
        ;;
    run-manual)
        build_image
        run_manual_container
        ;;
    run-minimal)
        build_image
        run_minimal_container
        ;;
    stop)
        stop_container
        ;;
    logs)
        show_logs
        ;;
    shell)
        enter_container
        ;;
    test)
        run_tests
        ;;
    status)
        show_status
        ;;
    help|*)
        show_help
        ;;
esac