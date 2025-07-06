# SDN Load Balancer Docker Setup

This guide provides complete instructions for running the SDN Load Balancer in Docker containers.

## 🐳 Quick Start

### 1. Build and Run (Simple)
```bash
# Make the script executable (if not already)
chmod +x docker-run.sh

# Build and run with default configuration (hexring topology)
./docker-run.sh run

# Access the dashboard
open http://localhost:8000
```

### 2. Using Docker Compose
```bash
# Start with default configuration
docker-compose up -d

# Start with specific topology profile
docker-compose --profile linear up -d
docker-compose --profile ring up -d
docker-compose --profile mesh up -d
```

## 📋 Prerequisites

- **Docker** (version 20.10+)
- **Docker Compose** (version 2.0+)
- **Linux/macOS** (Windows with WSL2)
- **8GB RAM minimum** (for network emulation)
- **Privileged container access** (required for Mininet)

## 🛠️ Installation Options

### Option 1: Quick Setup Script

The `docker-run.sh` script provides the easiest way to manage the container:

```bash
# Show all available commands
./docker-run.sh help

# Build the Docker image
./docker-run.sh build

# Run with custom configuration
./docker-run.sh run linear 5 50 least_loaded

# Check status
./docker-run.sh status

# View logs
./docker-run.sh logs

# Run tests
./docker-run.sh test

# Stop container
./docker-run.sh stop
```

### Option 2: Docker Compose

```bash
# Default hexring topology
docker-compose up -d

# Linear topology (5 switches)
docker-compose --profile linear up -d

# Ring topology (6 switches)  
docker-compose --profile ring up -d

# Mesh topology (4 switches)
docker-compose --profile mesh up -d

# Stop all containers
docker-compose down
```

### Option 3: Manual Docker Commands

```bash
# Build the image
docker build -t sdn-load-balancer:latest .

# Run with custom configuration
docker run -d \
  --name sdn-lb \
  --privileged \
  --network host \
  -v "$(pwd)/logs:/app/logs" \
  -v "$(pwd)/data:/app/data" \
  -e TOPOLOGY=hexring \
  -e SWITCHES=4 \
  -e THRESHOLD=25 \
  -e MODE=adaptive \
  -p 8000:8000 \
  -p 8080:8080 \
  -p 6653:6653 \
  sdn-load-balancer:latest
```

## 🔧 Configuration Options

### Environment Variables

| Variable | Default | Description | Valid Values |
|----------|---------|-------------|--------------|
| `TOPOLOGY` | `hexring` | Network topology type | `hexring`, `linear`, `ring`, `tree`, `mesh` |
| `SWITCHES` | `4` | Number of switches | `3-15` |
| `THRESHOLD` | `25` | Congestion threshold (Mbps) | `5-1000` |
| `MODE` | `adaptive` | Load balancing algorithm | `adaptive`, `least_loaded`, `weighted_ecmp`, `round_robin`, `latency_aware`, `qos_aware`, `flow_aware` |

### Port Mappings

| Container Port | Host Port | Service |
|----------------|-----------|---------|
| 8000 | 8000 | Web Dashboard |
| 8080 | 8080 | REST API |
| 6653 | 6653 | OpenFlow Controller |

### Volume Mounts

| Host Path | Container Path | Purpose |
|-----------|----------------|---------|
| `./logs` | `/app/logs` | Application logs |
| `./data` | `/app/data` | Persistent data |
| `/lib/modules` | `/lib/modules:ro` | Kernel modules (read-only) |

## 🌐 Access Points

Once the container is running:

- **Web Dashboard**: [http://localhost:8000](http://localhost:8000)
- **REST API**: [http://localhost:8080](http://localhost:8080)
- **API Documentation**: Available in browser at dashboard

### Key REST API Endpoints

```bash
# Get network topology
curl http://localhost:8080/topology

# Get efficiency metrics
curl http://localhost:8080/stats/efficiency

# Get algorithm information
curl http://localhost:8080/stats/algorithm

# Change load balancing mode
curl -X POST http://localhost:8080/config/mode \
  -H "Content-Type: application/json" \
  -d '{"mode": "least_loaded"}'

# Set congestion threshold (in bytes/sec)
curl -X POST http://localhost:8080/config/threshold \
  -H "Content-Type: application/json" \
  -d '{"threshold": 50000000}'
```

## 🧪 Testing Traffic

### Basic Connectivity Test
```bash
# Enter container shell
docker exec -it sdn-lb /bin/bash

# Inside container - test connectivity
mininet> pingall
```

### Generate Load Balancing Traffic
```bash
# Inside Mininet CLI
mininet> h2 iperf -s &
mininet> h1 iperf -c 192.168.1.12 -u -b 200M -t 60

# Multi-flow concurrent test
mininet> h2 iperf -s -p 5001 &
mininet> h3 iperf -s -p 5002 &
mininet> h1 iperf -c 192.168.1.12 -u -p 5001 -b 200M -t 120 &
mininet> h4 iperf -c 192.168.1.13 -u -p 5002 -b 200M -t 120
```

## 📊 Monitoring

### Container Status
```bash
# Check if container is running
docker ps | grep sdn-lb

# View container logs
docker logs -f sdn-lb

# Monitor container resources
docker stats sdn-lb
```

### Application Logs
```bash
# Application logs are mounted to host
tail -f logs/controller.log
tail -f logs/web.log
tail -f logs/mininet.log
```

### Health Check
```bash
# Automatic health check endpoint
curl -f http://localhost:8080/stats/efficiency
```

## 🐛 Troubleshooting

### Common Issues

#### Container Won't Start
```bash
# Check Docker daemon
sudo systemctl status docker

# Check container logs
docker logs sdn-lb

# Ensure privileged mode
docker run --privileged ...
```

#### Web Dashboard Not Accessible
```bash
# Check if port is available
lsof -i :8000

# Verify container networking
docker exec sdn-lb netstat -tulpn | grep 8000
```

#### REST API Not Responding
```bash
# Check Ryu controller
docker exec sdn-lb ps aux | grep ryu-manager

# Check API port
docker exec sdn-lb netstat -tulpn | grep 8080
```

#### OpenVSwitch Issues
```bash
# Check OVS status inside container
docker exec sdn-lb service openvswitch-switch status

# Restart OVS
docker exec sdn-lb service openvswitch-switch restart
```

### Debug Mode

Run with debug output:
```bash
# Enable debug logging
docker run -e DEBUG=1 --name sdn-lb-debug ...

# Or modify existing container
docker exec -it sdn-lb /bin/bash
export DEBUG=1
```

### Clean Reset

```bash
# Stop and remove everything
docker-compose down -v
docker system prune -f

# Remove logs and data
rm -rf logs/ data/

# Rebuild and restart
./docker-run.sh run
```

## 🔧 Advanced Configuration

### Custom Dockerfile

If you need to modify the container:

1. Edit `Dockerfile`
2. Rebuild: `docker build -t sdn-load-balancer:custom .`
3. Run with custom tag: `docker run sdn-load-balancer:custom`

### Network Bridge Mode

For non-host networking:

```bash
docker run -d \
  --name sdn-lb \
  --privileged \
  -p 8000:8000 \
  -p 8080:8080 \
  -p 6653:6653 \
  sdn-load-balancer:latest
```

### Persistent Data

```bash
# Create named volumes
docker volume create sdn-lb-logs
docker volume create sdn-lb-data

# Use named volumes
docker run -v sdn-lb-logs:/app/logs -v sdn-lb-data:/app/data ...
```

## 📈 Performance Optimization

### Resource Limits

```bash
# Limit CPU and memory
docker run --cpus="2.0" --memory="4g" ...
```

### Docker Compose with Limits

```yaml
services:
  sdn-load-balancer:
    # ... other config
    deploy:
      resources:
        limits:
          cpus: '2.0'
          memory: 4G
        reservations:
          cpus: '1.0'
          memory: 2G
```

## 🚀 Production Deployment

### Docker Swarm

```bash
# Initialize swarm
docker swarm init

# Deploy stack
docker stack deploy -c docker-compose.yml sdn-lb-stack
```

### Kubernetes

See `k8s/` directory for Kubernetes manifests (if available).

## 📝 Development

### Local Development

```bash
# Mount source code for development
docker run -v "$(pwd):/app" ...

# Or use docker-compose override
echo "
version: '3.8'
services:
  sdn-load-balancer:
    volumes:
      - ./:/app:Z
" > docker-compose.override.yml
```

### Building Custom Images

```bash
# Build with specific Python version
docker build --build-arg PYTHON_VERSION=3.9 .

# Build for different architecture
docker buildx build --platform linux/amd64,linux/arm64 .
```

---

## 📋 Quick Reference

### Essential Commands

```bash
# Build and run
./docker-run.sh run

# Check status  
./docker-run.sh status

# View logs
./docker-run.sh logs

# Run tests
./docker-run.sh test

# Stop
./docker-run.sh stop

# Enter shell
./docker-run.sh shell
```

### Default Access URLs

- **Dashboard**: http://localhost:8000
- **API**: http://localhost:8080
- **Metrics**: http://localhost:8080/stats/efficiency
- **Topology**: http://localhost:8080/topology

This Docker setup provides a complete, production-ready environment for the SDN Load Balancer with comprehensive monitoring, testing, and configuration options.