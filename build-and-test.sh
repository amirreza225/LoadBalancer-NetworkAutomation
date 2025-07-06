#!/bin/bash
# Comprehensive build and test script for SDN Load Balancer with Mininet Python 3

set -e

echo "=========================================="
echo "Building SDN Load Balancer with Python 3 Mininet"
echo "=========================================="

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

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

# Check if Docker is running
check_docker() {
    if ! docker info >/dev/null 2>&1; then
        print_error "Docker is not running. Please start Docker first."
        exit 1
    fi
}

# Build the Docker image
build_image() {
    print_info "Building Docker image with Python 3 Mininet..."
    
    # Clean up existing containers and images
    print_info "Cleaning up existing containers..."
    docker stop sdn-lb 2>/dev/null || true
    docker rm sdn-lb 2>/dev/null || true
    
    # Build the image
    print_info "Building new image..."
    docker build -t sdn-load-balancer:python3 . || {
        print_error "Docker build failed!"
        exit 1
    }
    
    print_success "Docker image built successfully!"
}

# Test the container
test_container() {
    print_info "Testing container with Python 3 Mininet..."
    
    # Start container in test mode
    docker run --rm --privileged \
        -v "$(pwd)/logs:/app/logs" \
        -v "$(pwd)/data:/app/data" \
        -v "/lib/modules:/lib/modules:ro" \
        sdn-load-balancer:python3 \
        python3 test_mininet_python3.py || {
        print_error "Mininet Python 3 test failed!"
        return 1
    }
    
    print_success "Mininet Python 3 test passed!"
}

# Run the container in manual mode
run_container() {
    print_info "Starting container in manual mode..."
    
    # Create logs and data directories
    mkdir -p logs data
    
    # Run the container
    docker run -d \
        --name sdn-lb \
        --privileged \
        -v "$(pwd)/logs:/app/logs" \
        -v "$(pwd)/data:/app/data" \
        -v "/lib/modules:/lib/modules:ro" \
        -p 8000:8000 \
        -p 8080:8080 \
        -p 6653:6653 \
        sdn-load-balancer:python3 /app/start_manual.sh || {
        print_error "Failed to start container!"
        exit 1
    }
    
    # Wait for container to start
    sleep 5
    
    if docker ps | grep -q sdn-lb; then
        print_success "Container started successfully!"
        print_info "Access points:"
        print_info "  Web Dashboard: http://localhost:8000"
        print_info "  REST API: http://localhost:8080"
        print_info "  OpenFlow Controller: localhost:6653"
        
        print_info ""
        print_info "Test Python 3 Mininet in container:"
        print_info "  docker exec -it sdn-lb python3 test_mininet_python3.py"
        
        print_info ""
        print_info "Run hexring topology:"
        print_info "  docker exec -it sdn-lb python3 hexring_topo.py"
        
        print_info ""
        print_info "Enter container shell:"
        print_info "  docker exec -it sdn-lb /bin/bash"
        
    else
        print_error "Container failed to start!"
        print_info "Checking logs..."
        docker logs sdn-lb
        exit 1
    fi
}

# Main execution
main() {
    case "${1:-build-and-run}" in
        build)
            check_docker
            build_image
            ;;
        test)
            check_docker
            test_container
            ;;
        run)
            check_docker
            run_container
            ;;
        build-and-test)
            check_docker
            build_image
            test_container
            ;;
        build-and-run)
            check_docker
            build_image
            test_container
            run_container
            ;;
        *)
            echo "Usage: $0 [build|test|run|build-and-test|build-and-run]"
            echo ""
            echo "Commands:"
            echo "  build           Build Docker image only"
            echo "  test            Test Mininet Python 3 in container"
            echo "  run             Run container in manual mode"
            echo "  build-and-test  Build and test (default)"
            echo "  build-and-run   Build, test, and run"
            ;;
    esac
}

main "$@"