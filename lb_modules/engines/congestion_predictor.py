"""
Congestion Predictor
====================

Advanced congestion prediction algorithms using multiple techniques:
- Linear regression
- Exponential weighted moving average (EWMA)
- Rate of change analysis
- Burst detection
- Adaptive weight adjustment
- Congestion gradient analysis
"""

import collections
import time
import math
from ..config.constants import (
    CONGESTION_PREDICTION_WINDOW, EWMA_ALPHA, CONGESTION_PARAMS, ADAPTIVE_MODE_PARAMS
)


class CongestionPredictor:
    """
    Predicts network congestion using multiple algorithms
    """
    
    def __init__(self, parent_app):
        self.parent_app = parent_app
        self.logger = parent_app.logger
        
        # EWMA state
        self.congestion_ewma = collections.defaultdict(float)  # (dpid, port) -> EWMA value
        
        # Enhanced prediction state
        self.burst_detection_state = collections.defaultdict(dict)  # (dpid, port) -> burst info
        self.adaptive_weights = collections.defaultdict(dict)  # (dpid, port) -> prediction weights
        self.congestion_gradients = collections.defaultdict(list)  # (dpid, port) -> gradient history
        self.network_congestion_level = 0.0  # Global network congestion level
        self.adaptive_thresholds = collections.defaultdict(float)  # (dpid, port) -> adaptive threshold
        
        # Update parent app references
        self.parent_app.congestion_ewma = self.congestion_ewma
        self.parent_app.burst_detection_state = self.burst_detection_state
        self.parent_app.adaptive_weights = self.adaptive_weights
        self.parent_app.congestion_gradients = self.congestion_gradients
        self.parent_app.network_congestion_level = self.network_congestion_level
        self.parent_app.adaptive_thresholds = self.adaptive_thresholds
    
    def predict_congestion(self, dpid, port, now):
        """
        Enhanced congestion prediction using EWMA and multiple algorithms.
        Combines linear regression, EWMA, rate of change, burst detection, and adaptive weighting.
        """
        trends = self.parent_app.congestion_trends.get((dpid, port), [])
        link_key = (dpid, port)
        
        # Keep only recent trends
        recent_trends = [(t, util) for t, util in trends if now - t <= CONGESTION_PREDICTION_WINDOW]
        
        if len(recent_trends) < 2:
            return 0  # Not enough data for prediction
        
        # Get current utilization
        current_util = recent_trends[-1][1] if recent_trends else 0
        
        # Update EWMA for this link
        if link_key in self.congestion_ewma:
            # Update existing EWMA
            self.congestion_ewma[link_key] = (EWMA_ALPHA * current_util + 
                                            (1 - EWMA_ALPHA) * self.congestion_ewma[link_key])
        else:
            # Initialize EWMA
            self.congestion_ewma[link_key] = current_util
        
        ewma_value = self.congestion_ewma[link_key]
        
        # Enhanced prediction methods
        linear_prediction = self._linear_regression_prediction(recent_trends)
        ewma_prediction = self._ewma_prediction(current_util, ewma_value)
        rate_prediction = self._rate_of_change_prediction(recent_trends)
        
        # New enhancement: Burst detection
        burst_factor = self._detect_burst_pattern(link_key, recent_trends, now)
        
        # New enhancement: Congestion gradient analysis
        gradient_prediction = self._congestion_gradient_prediction(link_key, recent_trends)
        
        # Adaptive weight adjustment based on network conditions
        adaptive_weights = self._calculate_adaptive_weights(link_key, recent_trends)
        
        # Enhanced prediction combination
        combined_prediction = self._combine_enhanced_predictions(
            linear_prediction, ewma_prediction, rate_prediction, 
            gradient_prediction, burst_factor, adaptive_weights
        )
        
        # Dynamic safety margin based on network congestion level
        safety_factor = self._calculate_safety_factor(current_util, link_key)
        combined_prediction *= safety_factor
        
        # Update adaptive threshold for this link
        self._update_adaptive_threshold(link_key, current_util, combined_prediction)
        
        return max(0, combined_prediction)
    
    def _linear_regression_prediction(self, recent_trends):
        """Linear regression prediction method"""
        if len(recent_trends) < 3:
            return 0
        
        times = [t for t, _ in recent_trends]
        utils = [util for _, util in recent_trends]
        
        n = len(recent_trends)
        sum_t = sum(times)
        sum_u = sum(utils)
        sum_tu = sum(t * u for t, u in recent_trends)
        sum_t2 = sum(t * t for t in times)
        
        # Calculate slope
        denominator = n * sum_t2 - sum_t * sum_t
        if denominator == 0:
            return 0
        
        slope = (n * sum_tu - sum_t * sum_u) / denominator
        # Predict utilization in 5 seconds
        return max(0, utils[-1] + slope * 5)
    
    def _ewma_prediction(self, current_util, ewma_value):
        """EWMA-based prediction method"""
        # If current utilization is increasing above EWMA, predict higher congestion
        if current_util > ewma_value * 1.1:  # 10% above EWMA
            growth_factor = current_util / (ewma_value + 1)
            return current_util * min(growth_factor, 2.0)  # Cap at 2x growth
        
        return ewma_value
    
    def _rate_of_change_prediction(self, recent_trends):
        """Rate of change prediction method"""
        if len(recent_trends) < 2:
            return 0
        
        recent_util = recent_trends[-1][1]
        prev_util = recent_trends[-2][1]
        time_diff = recent_trends[-1][0] - recent_trends[-2][0]
        
        if time_diff <= 0:
            return 0
        
        rate_of_change = (recent_util - prev_util) / time_diff
        # Project 5 seconds ahead
        return max(0, recent_util + rate_of_change * 5)
    
    def _combine_predictions(self, linear_prediction, ewma_prediction, rate_prediction):
        """Combine multiple predictions with weights"""
        # Linear regression: 40%, EWMA: 35%, Rate of change: 25%
        if linear_prediction > 0 and rate_prediction > 0:
            return (0.4 * linear_prediction + 
                   0.35 * ewma_prediction + 
                   0.25 * rate_prediction)
        elif linear_prediction > 0:
            return (0.6 * linear_prediction + 0.4 * ewma_prediction)
        else:
            return ewma_prediction
    
    def _detect_burst_pattern(self, link_key, recent_trends, now):
        """Detect burst patterns in traffic for rapid congestion onset"""
        if len(recent_trends) < 3:
            return 1.0  # No burst detected
        
        # Initialize burst state if not exists
        if link_key not in self.burst_detection_state:
            self.burst_detection_state[link_key] = {
                'last_burst_time': 0,
                'burst_intensity': 0,
                'baseline_util': 0
            }
        
        burst_state = self.burst_detection_state[link_key]
        current_util = recent_trends[-1][1]
        
        # Update baseline utilization (slower-changing average)
        if burst_state['baseline_util'] == 0:
            burst_state['baseline_util'] = current_util
        else:
            # Very slow update for baseline
            burst_state['baseline_util'] = (0.95 * burst_state['baseline_util'] + 
                                          0.05 * current_util)
        
        # Detect burst: current utilization significantly above baseline
        burst_threshold = burst_state['baseline_util'] * (1 + ADAPTIVE_MODE_PARAMS['burst_detection_sensitivity'])
        
        if current_util > burst_threshold:
            # Burst detected
            burst_intensity = (current_util - burst_state['baseline_util']) / burst_state['baseline_util']
            burst_state['burst_intensity'] = min(burst_intensity, 3.0)  # Cap at 3x
            burst_state['last_burst_time'] = now
            
            # Return burst factor (higher factor = more aggressive prediction)
            return 1.0 + burst_state['burst_intensity']
        else:
            # Decay burst intensity
            time_since_burst = now - burst_state['last_burst_time']
            if time_since_burst > 5:  # 5 seconds decay
                burst_state['burst_intensity'] *= 0.9
            
            return 1.0 + max(0, burst_state['burst_intensity'])
    
    def _congestion_gradient_prediction(self, link_key, recent_trends):
        """Analyze congestion gradient for trend prediction"""
        if len(recent_trends) < 4:
            return 0
        
        # Calculate gradient over the last few samples
        gradients = []
        for i in range(len(recent_trends) - 3, len(recent_trends)):
            if i > 0:
                dt = recent_trends[i][0] - recent_trends[i-1][0]
                du = recent_trends[i][1] - recent_trends[i-1][1]
                if dt > 0:
                    gradients.append(du / dt)
        
        if not gradients:
            return 0
        
        # Store gradient history
        if link_key not in self.congestion_gradients:
            self.congestion_gradients[link_key] = []
        
        self.congestion_gradients[link_key].extend(gradients)
        
        # Keep only recent gradients
        max_gradient_history = 10
        if len(self.congestion_gradients[link_key]) > max_gradient_history:
            self.congestion_gradients[link_key] = self.congestion_gradients[link_key][-max_gradient_history:]
        
        # Calculate average gradient
        avg_gradient = sum(self.congestion_gradients[link_key]) / len(self.congestion_gradients[link_key])
        
        # Predict utilization based on gradient
        prediction_horizon = 5  # 5 seconds ahead
        current_util = recent_trends[-1][1]
        
        return max(0, current_util + avg_gradient * prediction_horizon)
    
    def _calculate_adaptive_weights(self, link_key, recent_trends):
        """Calculate adaptive weights based on prediction accuracy and network conditions"""
        if link_key not in self.adaptive_weights:
            # Initialize with default weights
            self.adaptive_weights[link_key] = {
                'linear': 0.4,
                'ewma': 0.35,
                'rate': 0.25,
                'gradient': ADAPTIVE_MODE_PARAMS['gradient_analysis_weight']
            }
        
        weights = self.adaptive_weights[link_key]
        
        # Adapt weights based on recent prediction accuracy
        # This is a simplified adaptation - in practice, you'd track prediction errors
        
        # If we have enough data, adjust weights based on volatility
        if len(recent_trends) >= 5:
            # Calculate volatility (standard deviation of recent changes)
            recent_utils = [util for _, util in recent_trends[-5:]]
            if len(recent_utils) > 1:
                mean_util = sum(recent_utils) / len(recent_utils)
                variance = sum((u - mean_util) ** 2 for u in recent_utils) / len(recent_utils)
                volatility = math.sqrt(variance)
                
                # High volatility: increase rate-of-change weight
                if volatility > self.parent_app.THRESHOLD_BPS * 0.1:
                    weights['rate'] = min(0.4, weights['rate'] + 0.05)
                    weights['linear'] = max(0.2, weights['linear'] - 0.025)
                    weights['ewma'] = max(0.2, weights['ewma'] - 0.025)
                
                # Low volatility: increase EWMA weight
                elif volatility < self.parent_app.THRESHOLD_BPS * 0.05:
                    weights['ewma'] = min(0.5, weights['ewma'] + 0.05)
                    weights['rate'] = max(0.15, weights['rate'] - 0.025)
                    weights['linear'] = max(0.25, weights['linear'] - 0.025)
        
        return weights
    
    def _combine_enhanced_predictions(self, linear_prediction, ewma_prediction, rate_prediction, 
                                    gradient_prediction, burst_factor, adaptive_weights):
        """Enhanced prediction combination with adaptive weights and burst detection"""
        
        # Base prediction using adaptive weights
        base_prediction = (
            adaptive_weights['linear'] * linear_prediction +
            adaptive_weights['ewma'] * ewma_prediction +
            adaptive_weights['rate'] * rate_prediction +
            adaptive_weights['gradient'] * gradient_prediction
        )
        
        # Apply burst factor
        enhanced_prediction = base_prediction * burst_factor
        
        # If any prediction method shows high congestion, be more aggressive
        max_prediction = max(linear_prediction, ewma_prediction, rate_prediction, gradient_prediction)
        if max_prediction > self.parent_app.THRESHOLD_BPS * 0.8:
            # Aggressive prediction when any method shows high congestion
            enhanced_prediction = max(enhanced_prediction, max_prediction * 1.2)
        
        return enhanced_prediction
    
    def _calculate_safety_factor(self, current_util, link_key):
        """Calculate dynamic safety factor based on current utilization and network state"""
        base_safety = CONGESTION_PARAMS['safety_margin_factor']
        
        # Increase safety factor for links approaching congestion
        if current_util > self.parent_app.THRESHOLD_BPS * CONGESTION_PARAMS['prediction_threshold']:
            congestion_ratio = current_util / self.parent_app.THRESHOLD_BPS
            additional_safety = 0.3 * congestion_ratio  # Up to 30% additional safety
            return base_safety + additional_safety
        
        return base_safety
    
    def _update_adaptive_threshold(self, link_key, current_util, prediction):
        """Update adaptive threshold for this link based on historical patterns"""
        if not ADAPTIVE_MODE_PARAMS['adaptive_threshold_adjustment']:
            return
        
        # Initialize adaptive threshold if not exists
        if link_key not in self.adaptive_thresholds:
            self.adaptive_thresholds[link_key] = self.parent_app.THRESHOLD_BPS
        
        # Gradually adjust threshold based on utilization patterns
        target_threshold = self.parent_app.THRESHOLD_BPS
        
        # If link consistently operates at high utilization, increase threshold slightly
        if current_util > self.parent_app.THRESHOLD_BPS * 0.8:
            target_threshold = self.parent_app.THRESHOLD_BPS * 1.1
        
        # If link has low utilization, decrease threshold for more sensitive detection
        elif current_util < self.parent_app.THRESHOLD_BPS * 0.3:
            target_threshold = self.parent_app.THRESHOLD_BPS * 0.9
        
        # Slowly adjust toward target (prevents oscillation)
        self.adaptive_thresholds[link_key] = (
            0.95 * self.adaptive_thresholds[link_key] + 
            0.05 * target_threshold
        )
    
    def get_congestion_trend(self, dpid, port):
        """Get the congestion trend for a specific link"""
        return self.parent_app.congestion_trends.get((dpid, port), [])
    
    def is_congestion_predicted(self, dpid, port, threshold_factor=0.7):
        """Check if congestion is predicted for a link"""
        prediction = self.predict_congestion(dpid, port, time.time())
        return prediction > self.parent_app.THRESHOLD_BPS * threshold_factor
    
    def get_congestion_score(self, dpid, port, now):
        """Get a normalized congestion score (0-1)"""
        prediction = self.predict_congestion(dpid, port, now)
        return min(1.0, prediction / self.parent_app.THRESHOLD_BPS)
    
    def get_link_congestion_scores(self, now):
        """Get congestion scores for all links"""
        scores = {}
        
        for (dpid1, dpid2), (port1, port2) in self.parent_app.links.items():
            if dpid1 < dpid2:  # Avoid duplicates
                score1 = self.get_congestion_score(dpid1, port1, now)
                score2 = self.get_congestion_score(dpid2, port2, now)
                
                link_key = f"{dpid1}→{dpid2}"
                scores[link_key] = max(score1, score2)
        
        return scores
    
    def predict_path_congestion(self, path, now):
        """Predict congestion for an entire path"""
        if len(path) < 2:
            return 0
        
        max_congestion = 0
        
        for i in range(len(path) - 1):
            u, v = path[i], path[i + 1]
            
            # Get port numbers for this link
            if (u, v) in self.parent_app.links:
                port_u, port_v = self.parent_app.links[(u, v)]
                
                # Get congestion predictions for both directions
                congestion_u = self.predict_congestion(u, port_u, now)
                congestion_v = self.predict_congestion(v, port_v, now)
                
                # Take the maximum congestion for this link
                link_congestion = max(congestion_u, congestion_v)
                max_congestion = max(max_congestion, link_congestion)
        
        return max_congestion
    
    def get_least_congested_path(self, paths, now):
        """Select the path with least predicted congestion"""
        if not paths:
            return None
        
        if len(paths) == 1:
            return paths[0]
        
        best_path = None
        least_congestion = float('inf')
        
        for path in paths:
            congestion = self.predict_path_congestion(path, now)
            if congestion < least_congestion:
                least_congestion = congestion
                best_path = path
        
        return best_path
    
    def get_prediction_accuracy(self, dpid, port, window_seconds=60):
        """Calculate prediction accuracy for a link"""
        now = time.time()
        trends = self.get_congestion_trend(dpid, port)
        
        # Get trends within the window
        recent_trends = [(t, util) for t, util in trends if now - t <= window_seconds]
        
        if len(recent_trends) < 6:  # Need at least 6 data points
            return 0.0
        
        # Calculate accuracy by comparing predictions with actual values
        accurate_predictions = 0
        total_predictions = 0
        
        for i in range(3, len(recent_trends) - 3):  # Leave room for prediction window
            # Use data up to point i for prediction
            prediction_data = recent_trends[:i]
            actual_time = recent_trends[i + 3][0]  # 3 samples ahead (≈5 seconds)
            actual_util = recent_trends[i + 3][1]
            
            # Make prediction
            predicted_util = self._linear_regression_prediction(prediction_data)
            
            # Check if prediction was accurate (within 20% margin)
            if predicted_util > 0:
                error_margin = abs(predicted_util - actual_util) / predicted_util
                if error_margin <= 0.2:  # Within 20% margin
                    accurate_predictions += 1
            
            total_predictions += 1
        
        if total_predictions == 0:
            return 0.0
        
        return accurate_predictions / total_predictions
    
    def get_prediction_statistics(self):
        """Get statistics about prediction performance"""
        now = time.time()
        stats = {
            'total_links_tracked': len(self.congestion_ewma),
            'links_with_trends': len(self.parent_app.congestion_trends),
            'prediction_accuracies': {},
            'current_predictions': {}
        }
        
        for (dpid, port) in self.congestion_ewma:
            link_key = f"{dpid}:{port}"
            stats['prediction_accuracies'][link_key] = self.get_prediction_accuracy(dpid, port)
            stats['current_predictions'][link_key] = self.predict_congestion(dpid, port, now)
        
        return stats
    
    def reset_prediction_state(self):
        """Reset all prediction state"""
        self.congestion_ewma.clear()
        self.logger.info("Congestion predictor state reset")