# SDN Load Balancer Makefile

# Variables
DOCKER_IMAGE = sdn-load-balancer:latest
CONTAINER_NAME = sdn-lb
TOPOLOGY ?= hexring
SWITCHES ?= 4
THRESHOLD ?= 25
MODE ?= adaptive

# Colors for output
BLUE = \033[0;34m
GREEN = \033[0;32m
YELLOW = \033[1;33m
RED = \033[0;31m
NC = \033[0m

# Default target
.PHONY: help
help:
	@echo "$(BLUE)SDN Load Balancer Management$(NC)"
	@echo "==============================="
	@echo ""
	@echo "$(GREEN)Available targets:$(NC)"
	@echo "  $(YELLOW)build$(NC)        - Build Docker image"
	@echo "  $(YELLOW)run$(NC)          - Build and run container with default config"
	@echo "  $(YELLOW)run-custom$(NC)   - Run with custom configuration (use TOPOLOGY=, SWITCHES=, etc.)"
	@echo "  $(YELLOW)stop$(NC)         - Stop and remove container"
	@echo "  $(YELLOW)restart$(NC)      - Restart container"
	@echo "  $(YELLOW)logs$(NC)         - Show container logs"
	@echo "  $(YELLOW)shell$(NC)        - Enter container shell"
	@echo "  $(YELLOW)test$(NC)         - Run functionality tests"
	@echo "  $(YELLOW)status$(NC)       - Show container and service status"
	@echo "  $(YELLOW)clean$(NC)        - Clean up containers, images, and volumes"
	@echo "  $(YELLOW)clean-all$(NC)    - Deep clean including system docker cache"
	@echo ""
	@echo "$(GREEN)Docker Compose targets:$(NC)"
	@echo "  $(YELLOW)compose-up$(NC)   - Start with docker-compose"
	@echo "  $(YELLOW)compose-down$(NC) - Stop docker-compose services"
	@echo "  $(YELLOW)compose-logs$(NC) - Show docker-compose logs"
	@echo ""
	@echo "$(GREEN)Testing targets:$(NC)"
	@echo "  $(YELLOW)test-api$(NC)     - Test REST API endpoints"
	@echo "  $(YELLOW)test-traffic$(NC) - Generate test traffic"
	@echo ""
	@echo "$(GREEN)Development targets:$(NC)"
	@echo "  $(YELLOW)dev$(NC)          - Run in development mode (mounted source)"
	@echo "  $(YELLOW)lint$(NC)         - Run code linting (if tools available)"
	@echo ""
	@echo "$(GREEN)Examples:$(NC)"
	@echo "  make run"
	@echo "  make run-custom TOPOLOGY=linear SWITCHES=5 MODE=least_loaded"
	@echo "  make compose-up"

# Build Docker image
.PHONY: build
build:
	@echo "$(BLUE)[INFO]$(NC) Building Docker image..."
	docker build -t $(DOCKER_IMAGE) .
	@echo "$(GREEN)[SUCCESS]$(NC) Docker image built successfully!"

# Run container with default configuration
.PHONY: run
run: build
	@echo "$(BLUE)[INFO]$(NC) Starting SDN Load Balancer..."
	@$(MAKE) _run TOPOLOGY=$(TOPOLOGY) SWITCHES=$(SWITCHES) THRESHOLD=$(THRESHOLD) MODE=$(MODE)

# Run container with custom configuration
.PHONY: run-custom
run-custom: build
	@echo "$(BLUE)[INFO]$(NC) Starting SDN Load Balancer with custom configuration..."
	@echo "$(BLUE)[INFO]$(NC) Topology: $(TOPOLOGY), Switches: $(SWITCHES), Threshold: $(THRESHOLD), Mode: $(MODE)"
	@$(MAKE) _run TOPOLOGY=$(TOPOLOGY) SWITCHES=$(SWITCHES) THRESHOLD=$(THRESHOLD) MODE=$(MODE)

# Internal run target
.PHONY: _run
_run:
	@# Stop existing container if running
	-docker stop $(CONTAINER_NAME) 2>/dev/null
	-docker rm $(CONTAINER_NAME) 2>/dev/null
	@# Create directories
	mkdir -p logs data
	@# Run container
	docker run -d \
		--name $(CONTAINER_NAME) \
		--privileged \
		--network host \
		-v "$$(pwd)/logs:/app/logs" \
		-v "$$(pwd)/data:/app/data" \
		-v "/lib/modules:/lib/modules:ro" \
		-e TOPOLOGY=$(TOPOLOGY) \
		-e SWITCHES=$(SWITCHES) \
		-e THRESHOLD=$(THRESHOLD) \
		-e MODE=$(MODE) \
		-p 8000:8000 \
		-p 8080:8080 \
		-p 6653:6653 \
		$(DOCKER_IMAGE)
	@# Wait and check status
	@sleep 5
	@if docker ps | grep -q $(CONTAINER_NAME); then \
		echo "$(GREEN)[SUCCESS]$(NC) SDN Load Balancer started successfully!"; \
		echo "$(BLUE)[INFO]$(NC) Web Dashboard: http://localhost:8000"; \
		echo "$(BLUE)[INFO]$(NC) REST API: http://localhost:8080"; \
		echo "$(BLUE)[INFO]$(NC) OpenFlow Controller: localhost:6653"; \
	else \
		echo "$(RED)[ERROR]$(NC) Failed to start container"; \
		docker logs $(CONTAINER_NAME); \
		exit 1; \
	fi

# Stop container
.PHONY: stop
stop:
	@echo "$(BLUE)[INFO]$(NC) Stopping SDN Load Balancer..."
	-docker stop $(CONTAINER_NAME) 2>/dev/null
	-docker rm $(CONTAINER_NAME) 2>/dev/null
	@echo "$(GREEN)[SUCCESS]$(NC) Container stopped and removed"

# Restart container
.PHONY: restart
restart: stop run

# Show logs
.PHONY: logs
logs:
	@echo "$(BLUE)[INFO]$(NC) Showing SDN Load Balancer logs..."
	docker logs -f $(CONTAINER_NAME)

# Enter container shell
.PHONY: shell
shell:
	@echo "$(BLUE)[INFO]$(NC) Entering SDN Load Balancer container..."
	docker exec -it $(CONTAINER_NAME) /bin/bash

# Run tests
.PHONY: test
test:
	@echo "$(BLUE)[INFO]$(NC) Running SDN Load Balancer tests..."
	docker exec $(CONTAINER_NAME) /app/test.sh

# Show status
.PHONY: status
status:
	@echo "$(BLUE)[INFO]$(NC) SDN Load Balancer Status:"
	@if docker ps | grep -q $(CONTAINER_NAME); then \
		echo "$(GREEN)[SUCCESS]$(NC) Container is running"; \
		if curl -s http://localhost:8000 >/dev/null 2>&1; then \
			echo "$(GREEN)[SUCCESS]$(NC) Web Dashboard: http://localhost:8000 (✓ Accessible)"; \
		else \
			echo "$(YELLOW)[WARNING]$(NC) Web Dashboard: http://localhost:8000 (✗ Not accessible)"; \
		fi; \
		if curl -s http://localhost:8080/stats/efficiency >/dev/null 2>&1; then \
			echo "$(GREEN)[SUCCESS]$(NC) REST API: http://localhost:8080 (✓ Accessible)"; \
		else \
			echo "$(YELLOW)[WARNING]$(NC) REST API: http://localhost:8080 (✗ Not accessible)"; \
		fi; \
		echo "$(BLUE)[INFO]$(NC) Container stats:"; \
		docker stats $(CONTAINER_NAME) --no-stream; \
	else \
		echo "$(RED)[ERROR]$(NC) Container is not running"; \
	fi

# Clean up
.PHONY: clean
clean:
	@echo "$(BLUE)[INFO]$(NC) Cleaning up..."
	-docker stop $(CONTAINER_NAME) 2>/dev/null
	-docker rm $(CONTAINER_NAME) 2>/dev/null
	-docker rmi $(DOCKER_IMAGE) 2>/dev/null
	@echo "$(GREEN)[SUCCESS]$(NC) Cleanup completed"

# Deep clean
.PHONY: clean-all
clean-all: clean
	@echo "$(BLUE)[INFO]$(NC) Deep cleaning Docker system..."
	docker system prune -f
	docker volume prune -f
	@echo "$(GREEN)[SUCCESS]$(NC) Deep cleanup completed"

# Docker Compose targets
.PHONY: compose-up
compose-up:
	@echo "$(BLUE)[INFO]$(NC) Starting with docker-compose..."
	docker-compose up -d
	@echo "$(GREEN)[SUCCESS]$(NC) Services started with docker-compose"

.PHONY: compose-down
compose-down:
	@echo "$(BLUE)[INFO]$(NC) Stopping docker-compose services..."
	docker-compose down
	@echo "$(GREEN)[SUCCESS]$(NC) Services stopped"

.PHONY: compose-logs
compose-logs:
	@echo "$(BLUE)[INFO]$(NC) Showing docker-compose logs..."
	docker-compose logs -f

# Test API endpoints
.PHONY: test-api
test-api:
	@echo "$(BLUE)[INFO]$(NC) Testing REST API endpoints..."
	@echo "Testing topology endpoint..."
	curl -s http://localhost:8080/topology | python3 -m json.tool || echo "$(RED)Topology test failed$(NC)"
	@echo "Testing efficiency endpoint..."
	curl -s http://localhost:8080/stats/efficiency | python3 -m json.tool || echo "$(RED)Efficiency test failed$(NC)"
	@echo "Testing algorithm endpoint..."
	curl -s http://localhost:8080/stats/algorithm | python3 -m json.tool || echo "$(RED)Algorithm test failed$(NC)"
	@echo "$(GREEN)[SUCCESS]$(NC) API tests completed"

# Generate test traffic
.PHONY: test-traffic
test-traffic:
	@echo "$(BLUE)[INFO]$(NC) Generating test traffic..."
	@echo "$(YELLOW)[WARNING]$(NC) This requires manual interaction with Mininet CLI"
	@echo "$(BLUE)[INFO]$(NC) Run 'make shell' and then execute traffic commands in Mininet"

# Development mode with mounted source
.PHONY: dev
dev: build
	@echo "$(BLUE)[INFO]$(NC) Starting in development mode (source mounted)..."
	@# Stop existing container
	-docker stop $(CONTAINER_NAME)-dev 2>/dev/null
	-docker rm $(CONTAINER_NAME)-dev 2>/dev/null
	@# Create directories
	mkdir -p logs data
	@# Run with mounted source
	docker run -it --rm \
		--name $(CONTAINER_NAME)-dev \
		--privileged \
		--network host \
		-v "$$(pwd):/app:Z" \
		-v "$$(pwd)/logs:/app/logs" \
		-v "$$(pwd)/data:/app/data" \
		-v "/lib/modules:/lib/modules:ro" \
		-e TOPOLOGY=$(TOPOLOGY) \
		-e SWITCHES=$(SWITCHES) \
		-e THRESHOLD=$(THRESHOLD) \
		-e MODE=$(MODE) \
		-p 8000:8000 \
		-p 8080:8080 \
		-p 6653:6653 \
		$(DOCKER_IMAGE) /bin/bash

# Lint code (if tools available)
.PHONY: lint
lint:
	@echo "$(BLUE)[INFO]$(NC) Running code linting..."
	@if command -v flake8 >/dev/null 2>&1; then \
		flake8 *.py; \
	else \
		echo "$(YELLOW)[WARNING]$(NC) flake8 not available"; \
	fi
	@if command -v black >/dev/null 2>&1; then \
		black --check *.py; \
	else \
		echo "$(YELLOW)[WARNING]$(NC) black not available"; \
	fi

# Install Python dependencies locally (for development)
.PHONY: install-deps
install-deps:
	@echo "$(BLUE)[INFO]$(NC) Installing Python dependencies..."
	pip3 install -r requirements.txt
	@echo "$(GREEN)[SUCCESS]$(NC) Dependencies installed"

# Show Docker image info
.PHONY: image-info
image-info:
	@echo "$(BLUE)[INFO]$(NC) Docker image information:"
	docker images | grep sdn-load-balancer || echo "$(YELLOW)[WARNING]$(NC) Image not found"
	@echo ""
	@echo "$(BLUE)[INFO]$(NC) Image layers:"
	docker history $(DOCKER_IMAGE) 2>/dev/null || echo "$(YELLOW)[WARNING]$(NC) Image not built yet"

# Health check
.PHONY: health
health:
	@echo "$(BLUE)[INFO]$(NC) Running health check..."
	@if curl -f http://localhost:8080/stats/efficiency >/dev/null 2>&1; then \
		echo "$(GREEN)[SUCCESS]$(NC) Health check passed"; \
	else \
		echo "$(RED)[ERROR]$(NC) Health check failed"; \
		exit 1; \
	fi