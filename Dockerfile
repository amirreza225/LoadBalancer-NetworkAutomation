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

# Create python symlink for compatibility with mn command
RUN ln -s /usr/bin/python3 /usr/bin/python

# Expose ports
EXPOSE 8000 8080 6653

# Set environment variables for Python3 Mininet compatibility
ENV PYTHONPATH="/usr/local/lib/python3.8/site-packages:/usr/lib/python3/dist-packages:/usr/local/lib/python3.8/dist-packages:/app"
ENV MININET_PATH="/usr/local/lib/python3.8/site-packages"

# Copy minimal startup script
COPY start_minimal.sh /app/start_minimal.sh
RUN chmod +x /app/start_minimal.sh

# Default command - minimal setup only
CMD ["/app/start_minimal.sh"]