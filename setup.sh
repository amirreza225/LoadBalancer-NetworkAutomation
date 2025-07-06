#!/bin/bash
# SDN Load Balancer Setup Script
# Automated setup for Docker-based deployment

set -e

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
BOLD='\033[1m'
NC='\033[0m' # No Color

# Function to print colored output
print_header() {
    echo -e "${BOLD}${BLUE}$1${NC}"
    echo -e "${BLUE}$(printf '=%.0s' $(seq 1 ${#1}))${NC}"
}

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

print_step() {
    echo -e "${BOLD}${GREEN}[STEP]${NC} $1"
}

# Function to check if command exists
command_exists() {
    command -v "$1" >/dev/null 2>&1
}

# Function to check Docker
check_docker() {
    print_step "Checking Docker installation..."
    
    if ! command_exists docker; then
        print_error "Docker is not installed. Please install Docker first."
        echo "Visit: https://docs.docker.com/get-docker/"
        exit 1
    fi
    
    if ! docker info >/dev/null 2>&1; then
        print_error "Docker daemon is not running. Please start Docker."
        exit 1
    fi
    
    # Check Docker version
    DOCKER_VERSION=$(docker --version | cut -d' ' -f3 | cut -d',' -f1)
    print_success "Docker $DOCKER_VERSION is installed and running"
    
    # Check Docker Compose
    if command_exists docker-compose; then
        COMPOSE_VERSION=$(docker-compose --version | cut -d' ' -f3 | cut -d',' -f1)
        print_success "Docker Compose $COMPOSE_VERSION is available"
    elif docker compose version >/dev/null 2>&1; then
        COMPOSE_VERSION=$(docker compose version --short)
        print_success "Docker Compose $COMPOSE_VERSION (plugin) is available"
    else
        print_warning "Docker Compose not found. Some features may not be available."
    fi
}

# Function to check system requirements
check_system() {
    print_step "Checking system requirements..."
    
    # Check OS
    if [[ "$OSTYPE" == "linux-gnu"* ]]; then
        print_success "Linux OS detected"
    elif [[ "$OSTYPE" == "darwin"* ]]; then
        print_success "macOS detected"
    elif [[ "$OSTYPE" == "msys" || "$OSTYPE" == "cygwin" ]]; then
        print_warning "Windows detected. WSL2 recommended for best performance."
    else
        print_warning "Unknown OS: $OSTYPE. Compatibility not guaranteed."
    fi
    
    # Check memory
    if command_exists free; then
        MEMORY_GB=$(free -g | awk '/^Mem:/{print $2}')
        if [ "$MEMORY_GB" -ge 8 ]; then
            print_success "Sufficient memory: ${MEMORY_GB}GB"
        else
            print_warning "Low memory: ${MEMORY_GB}GB. 8GB+ recommended."
        fi
    elif command_exists sysctl; then
        # macOS
        MEMORY_BYTES=$(sysctl -n hw.memsize)
        MEMORY_GB=$((MEMORY_BYTES / 1024 / 1024 / 1024))
        if [ "$MEMORY_GB" -ge 8 ]; then
            print_success "Sufficient memory: ${MEMORY_GB}GB"
        else
            print_warning "Low memory: ${MEMORY_GB}GB. 8GB+ recommended."
        fi
    fi
    
    # Check for curl
    if command_exists curl; then
        print_success "curl is available"
    else
        print_warning "curl not found. Some tests may not work."
    fi
}

# Function to prepare environment
prepare_environment() {
    print_step "Preparing environment..."
    
    # Create necessary directories
    mkdir -p logs data
    print_info "Created logs/ and data/ directories"
    
    # Make scripts executable
    chmod +x docker-run.sh 2>/dev/null || true
    chmod +x setup.sh 2>/dev/null || true
    print_info "Made scripts executable"
    
    # Check for Python (for JSON formatting in tests)
    if command_exists python3; then
        print_success "Python 3 is available for testing"
    elif command_exists python; then
        print_success "Python is available for testing"
    else
        print_warning "Python not found. JSON formatting in tests may not work."
    fi
}

# Function to build Docker image
build_image() {
    print_step "Building Docker image..."
    
    if docker build -t sdn-load-balancer:latest .; then
        print_success "Docker image built successfully"
    else
        print_error "Failed to build Docker image"
        exit 1
    fi
}

# Function to test deployment
test_deployment() {
    print_step "Testing deployment..."
    
    # Stop any existing container
    docker stop sdn-lb 2>/dev/null || true
    docker rm sdn-lb 2>/dev/null || true
    
    # Start container
    print_info "Starting test container..."
    docker run -d \
        --name sdn-lb-test \
        --privileged \
        --network host \
        -v "$(pwd)/logs:/app/logs" \
        -v "$(pwd)/data:/app/data" \
        -v "/lib/modules:/lib/modules:ro" \
        -e TOPOLOGY=hexring \
        -e SWITCHES=4 \
        -e THRESHOLD=25 \
        -e MODE=adaptive \
        -p 8000:8000 \
        -p 8080:8080 \
        -p 6653:6653 \
        sdn-load-balancer:latest >/dev/null
    
    # Wait for services to start
    print_info "Waiting for services to start..."
    sleep 15
    
    # Test if container is running
    if docker ps | grep -q sdn-lb-test; then
        print_success "Container is running"
    else
        print_error "Container failed to start"
        docker logs sdn-lb-test
        docker rm sdn-lb-test 2>/dev/null || true
        exit 1
    fi
    
    # Test web dashboard
    if curl -s http://localhost:8000 >/dev/null 2>&1; then
        print_success "Web dashboard is accessible"
    else
        print_warning "Web dashboard is not accessible (may still be starting)"
    fi
    
    # Test REST API
    if curl -s http://localhost:8080/stats/efficiency >/dev/null 2>&1; then
        print_success "REST API is responding"
    else
        print_warning "REST API is not responding (may still be starting)"
    fi
    
    # Cleanup test container
    docker stop sdn-lb-test >/dev/null 2>&1 || true
    docker rm sdn-lb-test >/dev/null 2>&1 || true
    print_info "Test container cleaned up"
}

# Function to show usage information
show_usage() {
    print_header "SDN Load Balancer - Usage Information"
    echo ""
    echo "The setup is complete! Here's how to use the SDN Load Balancer:"
    echo ""
    echo -e "${BOLD}Quick Start:${NC}"
    echo "  ./docker-run.sh run                    # Build and run with default settings"
    echo "  make run                               # Alternative using Makefile"
    echo ""
    echo -e "${BOLD}Configuration Options:${NC}"
    echo "  ./docker-run.sh run linear 5 50 least_loaded   # Custom topology, switches, threshold, mode"
    echo "  make run-custom TOPOLOGY=ring SWITCHES=6       # Using Makefile with custom settings"
    echo ""
    echo -e "${BOLD}Docker Compose:${NC}"
    echo "  docker-compose up -d                   # Start with default configuration"
    echo "  docker-compose --profile linear up -d  # Start with linear topology"
    echo ""
    echo -e "${BOLD}Access Points:${NC}"
    echo "  Web Dashboard: http://localhost:8000"
    echo "  REST API:      http://localhost:8080"
    echo "  OpenFlow:      localhost:6653"
    echo ""
    echo -e "${BOLD}Management Commands:${NC}"
    echo "  ./docker-run.sh status                 # Check status"
    echo "  ./docker-run.sh logs                   # View logs"
    echo "  ./docker-run.sh test                   # Run tests"
    echo "  ./docker-run.sh shell                  # Enter container"
    echo "  ./docker-run.sh stop                   # Stop container"
    echo ""
    echo -e "${BOLD}Available Topologies:${NC}"
    echo "  hexring, linear, ring, tree, mesh"
    echo ""
    echo -e "${BOLD}Available Load Balancing Modes:${NC}"
    echo "  adaptive, least_loaded, weighted_ecmp, round_robin,"
    echo "  latency_aware, qos_aware, flow_aware"
    echo ""
    echo -e "${BOLD}Testing Traffic:${NC}"
    echo "  1. ./docker-run.sh shell"
    echo "  2. In Mininet CLI: pingall"
    echo "  3. Generate traffic: h2 iperf -s & h1 iperf -c 192.168.1.12 -u -b 200M -t 60"
    echo ""
    echo -e "${BOLD}Monitoring:${NC}"
    echo "  Dashboard: Real-time efficiency metrics and topology visualization"
    echo "  API: curl http://localhost:8080/stats/efficiency | python3 -m json.tool"
    echo "  Logs: ./docker-run.sh logs or check logs/ directory"
    echo ""
    echo -e "${BOLD}For more information:${NC}"
    echo "  Read DOCKER.md for comprehensive documentation"
    echo "  Check README.md for project details"
    echo "  Use 'make help' for Makefile options"
}

# Main setup function
main() {
    print_header "SDN Load Balancer Docker Setup"
    echo ""
    print_info "This script will set up the SDN Load Balancer for Docker deployment"
    echo ""
    
    # Check if running in interactive mode
    if [ -t 0 ]; then
        echo -e "${YELLOW}Do you want to proceed with the setup? [Y/n]${NC}"
        read -r response
        case "$response" in
            [nN][oO]|[nN])
                print_info "Setup cancelled by user"
                exit 0
                ;;
        esac
    fi
    
    # Run setup steps
    check_docker
    check_system
    prepare_environment
    build_image
    
    # Ask about testing
    if [ -t 0 ]; then
        echo ""
        echo -e "${YELLOW}Do you want to run a deployment test? [Y/n]${NC}"
        read -r test_response
        case "$test_response" in
            [nN][oO]|[nN])
                print_info "Skipping deployment test"
                ;;
            *)
                test_deployment
                ;;
        esac
    else
        print_info "Non-interactive mode: skipping deployment test"
    fi
    
    echo ""
    print_success "Setup completed successfully!"
    echo ""
    show_usage
}

# Handle command line arguments
case "${1:-setup}" in
    setup|"")
        main
        ;;
    build)
        print_header "Building Docker Image"
        check_docker
        build_image
        print_success "Build completed!"
        ;;
    test)
        print_header "Testing Deployment"
        check_docker
        test_deployment
        print_success "Test completed!"
        ;;
    check)
        print_header "Checking Requirements"
        check_docker
        check_system
        print_success "Check completed!"
        ;;
    usage|help)
        show_usage
        ;;
    *)
        echo "Usage: $0 [setup|build|test|check|usage|help]"
        echo ""
        echo "Commands:"
        echo "  setup (default) - Full setup process"
        echo "  build          - Build Docker image only"
        echo "  test           - Test deployment only"
        echo "  check          - Check requirements only"
        echo "  usage|help     - Show usage information"
        exit 1
        ;;
esac