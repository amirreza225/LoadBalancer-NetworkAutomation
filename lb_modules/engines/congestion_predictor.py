"""
Congestion Predictor
====================

Advanced congestion prediction algorithms using multiple techniques:
- Linear regression
- Exponential weighted moving average (EWMA)
- Rate of change analysis
"""

import collections
import time
from ..config.constants import CONGESTION_PREDICTION_WINDOW, EWMA_ALPHA


class CongestionPredictor:
    """
    Predicts network congestion using multiple algorithms
    """
    
    def __init__(self, parent_app):
        self.parent_app = parent_app
        self.logger = parent_app.logger
        
        # EWMA state
        self.congestion_ewma = collections.defaultdict(float)  # (dpid, port) -> EWMA value
        
        # Update parent app reference
        self.parent_app.congestion_ewma = self.congestion_ewma
    
    def predict_congestion(self, dpid, port, now):
        """
        Enhanced congestion prediction using EWMA and multiple algorithms.
        Combines linear regression with exponential weighted moving average for better accuracy.
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
        
        # Method 1: Linear regression prediction
        linear_prediction = self._linear_regression_prediction(recent_trends)
        
        # Method 2: EWMA-based prediction
        ewma_prediction = self._ewma_prediction(current_util, ewma_value)
        
        # Method 3: Rate of change prediction
        rate_prediction = self._rate_of_change_prediction(recent_trends)
        
        # Combine predictions with weights
        combined_prediction = self._combine_predictions(
            linear_prediction, ewma_prediction, rate_prediction
        )
        
        # Add safety margin for critical links
        if current_util > self.parent_app.THRESHOLD_BPS * 0.7:
            combined_prediction *= 1.2  # 20% safety margin
        
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