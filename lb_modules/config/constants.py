"""
Configuration constants for the SDN Load Balancer
"""

# Core timing constants
POLL_PERIOD = 1                    # seconds
MA_WINDOW_SEC = 5                  # seconds
DEFAULT_THRESH = 25_000_000        # bytes/sec
CONGESTION_PREDICTION_WINDOW = 10  # seconds for trend analysis

# Enhanced load balancing constants
ELEPHANT_FLOW_THRESHOLD = 10_000_000  # 10 Mbps threshold for elephant flows
MICE_FLOW_THRESHOLD = 1_000_000       # 1 Mbps threshold for mice flows
EWMA_ALPHA = 0.3                      # Exponential weighted moving average factor
LATENCY_WEIGHT = 0.2                  # Weight for latency in path selection
QOS_WEIGHT = 0.25                     # Weight for QoS in path selection
FLOW_TIMEOUT_SEC = 300                # Flow tracking timeout (5 minutes)

# Load balancing modes
LOAD_BALANCING_MODES = {
    'round_robin': 0,
    'least_loaded': 1,
    'weighted_ecmp': 2,
    'adaptive': 3,
    'latency_aware': 4,
    'qos_aware': 5,
    'flow_aware': 6
}

# QoS Classes
QOS_CLASSES = {
    'CRITICAL': {'priority': 3, 'min_bw': 10_000_000, 'max_latency': 10},     # VoIP, real-time
    'HIGH': {'priority': 2, 'min_bw': 5_000_000, 'max_latency': 50},         # Video streaming
    'NORMAL': {'priority': 1, 'min_bw': 1_000_000, 'max_latency': 200},      # Web browsing
    'BEST_EFFORT': {'priority': 0, 'min_bw': 0, 'max_latency': 1000}         # Background traffic
}

# Flow classification thresholds
FLOW_CLASSIFICATION = {
    'elephant_threshold': ELEPHANT_FLOW_THRESHOLD,
    'mice_threshold': MICE_FLOW_THRESHOLD,
    'burst_threshold': 50_000_000,  # 50 Mbps
    'sustained_duration': 10        # seconds
}

# Congestion prediction parameters
CONGESTION_PARAMS = {
    'prediction_window': CONGESTION_PREDICTION_WINDOW,
    'trend_min_points': 3,
    'ewma_alpha': EWMA_ALPHA,
    'congestion_threshold': 0.8,    # 80% utilization
    'prediction_threshold': 0.7     # 70% utilization for prediction
}

# Path selection weights
PATH_WEIGHTS = {
    'latency': LATENCY_WEIGHT,
    'qos': QOS_WEIGHT,
    'utilization': 0.4,
    'reliability': 0.15
}