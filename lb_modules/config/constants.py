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
    'prediction_threshold': 0.7,    # 70% utilization for prediction
    'approaching_congestion_threshold': 0.7,  # 70% threshold for approaching congestion (was 30%)
    'burst_detection_threshold': 0.5,  # 50% threshold for burst detection
    'adaptive_prediction_weight': 0.6,  # 60% weight for prediction (was 30%)
    'congestion_avoidance_bonus': 0.5,  # 50% bonus for avoiding congestion (was 30%)
    'safety_margin_factor': 1.3,  # 30% safety margin for critical links
    'gradient_weight': 0.2  # 20% weight for congestion gradient analysis
}

# Path selection weights
PATH_WEIGHTS = {
    'latency': LATENCY_WEIGHT,
    'qos': QOS_WEIGHT,
    'utilization': 0.4,
    'reliability': 0.15
}

# Adaptive mode specific parameters
ADAPTIVE_MODE_PARAMS = {
    'congestion_prediction_weight': 0.6,  # Increased from 0.3 to 0.6
    'congestion_avoidance_bonus': 0.5,    # Increased from 0.3 to 0.5
    'approaching_congestion_factor': 0.7,  # Increased from 0.3 to 0.7
    'congestion_penalty_multiplier': 0.8,  # Heavy penalty for congested links
    'burst_detection_sensitivity': 0.3,    # Sensitivity for burst detection
    'gradient_analysis_weight': 0.15,      # Weight for congestion gradient
    'flow_aware_integration': True,        # Enable flow-aware characteristics
    'adaptive_threshold_adjustment': True,  # Enable dynamic threshold adjustment
    'exponential_penalty_factor': 2.0,     # Exponential penalty for heavy congestion
    'multi_level_scoring': True            # Enable multi-level congestion scoring
}

# Load distribution calculation constants
LOAD_DISTRIBUTION_WINDOW_SEC = 60  # Time window for load balancing effectiveness calculation
LOAD_DISTRIBUTION_MIN_SAMPLES = 5  # Minimum samples required for reliable calculation
LOAD_DISTRIBUTION_UPDATE_INTERVAL = 2  # Update interval in seconds

# Load balancing effectiveness thresholds
EXCELLENT_LOAD_BALANCING_THRESHOLD = 80  # Above this % is considered excellent
GOOD_LOAD_BALANCING_THRESHOLD = 60      # Above this % is considered good  
FAIR_LOAD_BALANCING_THRESHOLD = 30      # Above this % is considered fair
# Below fair threshold is considered poor

# Coefficient of variation thresholds (lower is better)
EXCELLENT_CV_THRESHOLD = 0.2  # CV below this is excellent balance
GOOD_CV_THRESHOLD = 0.4       # CV below this is good balance
FAIR_CV_THRESHOLD = 0.8       # CV below this is fair balance
# CV above fair threshold indicates poor balance

# Distribution entropy thresholds (higher is better, normalized 0-1)
EXCELLENT_ENTROPY_THRESHOLD = 0.8  # Entropy above this is excellent
GOOD_ENTROPY_THRESHOLD = 0.6       # Entropy above this is good
FAIR_ENTROPY_THRESHOLD = 0.4       # Entropy above this is fair
# Entropy below fair threshold indicates poor distribution