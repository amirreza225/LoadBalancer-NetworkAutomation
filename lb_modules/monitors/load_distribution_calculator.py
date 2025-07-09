#!/usr/bin/env python3
"""
Load Distribution Calculator for SDN Load Balancer
Calculates proper load balancing metrics based on actual traffic distribution
rather than simple flow counts.
"""

import time
import math
from collections import defaultdict, deque
from typing import Dict, List, Tuple, Optional
import logging

class LoadDistributionCalculator:
    """
    Calculates load balancing effectiveness based on actual traffic distribution
    across network links and paths over time windows.
    """
    
    def __init__(self, window_size_sec: int = 60, min_samples: int = 5):
        """
        Initialize the load distribution calculator.
        
        Args:
            window_size_sec: Time window for calculations in seconds
            min_samples: Minimum samples needed for reliable calculations
        """
        self.window_size_sec = window_size_sec
        self.min_samples = min_samples
        
        # Time-series data for traffic distribution analysis
        self.utilization_history = defaultdict(lambda: deque(maxlen=window_size_sec))
        self.timestamp_history = deque(maxlen=window_size_sec)
        
        # Baseline shortest-path distribution for comparison
        self.baseline_distribution = {}
        
        self.logger = logging.getLogger(self.__class__.__name__)
        
    def update_link_utilization(self, link_loads: Dict[Tuple[int, int], float], 
                               timestamp: Optional[float] = None):
        """
        Update link utilization data for distribution calculations.
        
        Args:
            link_loads: Dictionary of (dpid1, dpid2) -> bytes_per_sec
            timestamp: Optional timestamp, uses current time if None
        """
        if timestamp is None:
            timestamp = time.time()
            
        self.timestamp_history.append(timestamp)
        
        # Store utilization for each link
        for link, utilization in link_loads.items():
            self.utilization_history[link].append(utilization)
            
        # Clean old data outside time window
        self._cleanup_old_data(timestamp)
        
    def _cleanup_old_data(self, current_time: float):
        """Remove data outside the time window."""
        cutoff_time = current_time - self.window_size_sec
        
        # Remove old timestamps
        while (self.timestamp_history and 
               self.timestamp_history[0] < cutoff_time):
            self.timestamp_history.popleft()
            
    def calculate_load_balancing_effectiveness(self, 
                                             parallel_paths: Dict[str, List[List[int]]] = None) -> Dict[str, float]:
        """
        Calculate comprehensive load balancing effectiveness metrics.
        
        Args:
            parallel_paths: Dictionary of flow_key -> list of available paths
            
        Returns:
            Dictionary containing various load balancing metrics
        """
        if len(self.utilization_history) < 2:
            return self._get_default_metrics()
            
        # Get current utilization distribution
        current_utilizations = self._get_current_utilizations()
        if not current_utilizations:
            return self._get_default_metrics()
            
        utilization_values = list(current_utilizations.values())
        
        # Calculate multiple load balancing metrics
        metrics = {}
        
        # 1. Coefficient of Variation (CV) - Lower is better balanced
        metrics['coefficient_of_variation'] = self._calculate_coefficient_of_variation(utilization_values)
        
        # 2. Entropy-based distribution measure - Higher is better balanced
        metrics['distribution_entropy'] = self._calculate_distribution_entropy(utilization_values)
        
        # 3. Standard deviation of utilizations - Lower is better balanced
        metrics['utilization_std_dev'] = self._calculate_std_dev(utilization_values)
        
        # 4. Load balancing effectiveness percentage (0-100%)
        metrics['load_balancing_effectiveness'] = self._calculate_effectiveness_percentage(
            metrics['coefficient_of_variation']
        )
        
        # 5. Traffic variance reduction compared to shortest path
        metrics['variance_reduction'] = self._calculate_variance_reduction()
        
        # 6. Utilization balance score (0-100%)
        metrics['utilization_balance_score'] = self._calculate_balance_score(utilization_values)
        
        # 7. Time-weighted load balancing rate
        metrics['time_weighted_lb_rate'] = self._calculate_time_weighted_rate()
        
        # 8. Link utilization statistics
        if utilization_values:
            metrics['avg_utilization'] = sum(utilization_values) / len(utilization_values)
            metrics['max_utilization'] = max(utilization_values)
            metrics['min_utilization'] = min(utilization_values)
            metrics['utilization_range'] = metrics['max_utilization'] - metrics['min_utilization']
        else:
            metrics.update({
                'avg_utilization': 0,
                'max_utilization': 0, 
                'min_utilization': 0,
                'utilization_range': 0
            })
            
        return metrics
        
    def _get_current_utilizations(self) -> Dict[Tuple[int, int], float]:
        """Get the most recent utilization values for all links."""
        current_utils = {}
        
        for link, history in self.utilization_history.items():
            if history:
                current_utils[link] = history[-1]  # Most recent value
                
        return current_utils
        
    def _calculate_coefficient_of_variation(self, utilizations: List[float]) -> float:
        """
        Calculate coefficient of variation (CV = std_dev / mean).
        Lower CV indicates better load balancing.
        """
        if not utilizations or len(utilizations) < 2:
            return 0.0
            
        mean_util = sum(utilizations) / len(utilizations)
        if mean_util == 0:
            return 0.0
            
        std_util = self._calculate_std_dev(utilizations)
        return std_util / mean_util
        
    def _calculate_distribution_entropy(self, utilizations: List[float]) -> float:
        """
        Calculate Shannon entropy of traffic distribution.
        Higher entropy indicates better load balancing.
        """
        if not utilizations:
            return 0.0
            
        # Normalize utilizations to get probabilities
        total_util = sum(utilizations)
        if total_util == 0:
            return 0.0
            
        probabilities = [u / total_util for u in utilizations if u > 0]
        
        if len(probabilities) <= 1:
            return 0.0
            
        # Calculate Shannon entropy
        entropy = -sum(p * math.log2(p) for p in probabilities if p > 0)
        
        # Normalize by maximum possible entropy (log2(n))
        max_entropy = math.log2(len(probabilities))
        
        return entropy / max_entropy if max_entropy > 0 else 0.0
        
    def _calculate_effectiveness_percentage(self, cv: float) -> float:
        """
        Convert coefficient of variation to load balancing effectiveness percentage.
        
        CV of 0 = 100% effectiveness (perfect balance)
        CV of 1+ = 0% effectiveness (very unbalanced)
        """
        if cv <= 0:
            return 100.0
            
        # Use exponential decay: effectiveness = 100 * e^(-cv)
        effectiveness = 100.0 * math.exp(-cv)
        return min(100.0, max(0.0, effectiveness))
        
    def _calculate_variance_reduction(self) -> float:
        """
        Calculate variance reduction compared to baseline shortest-path routing.
        """
        if not self.baseline_distribution:
            return 0.0
            
        current_utils = self._get_current_utilizations()
        if not current_utils:
            return 0.0
            
        current_variance = self._calculate_variance(list(current_utils.values()))
        baseline_variance = self._calculate_variance(list(self.baseline_distribution.values()))
        
        if baseline_variance == 0:
            return 0.0
            
        reduction = (baseline_variance - current_variance) / baseline_variance * 100
        return max(0.0, min(100.0, reduction))
        
    def _calculate_balance_score(self, utilizations: List[float]) -> float:
        """
        Calculate utilization balance score (0-100%).
        Based on how close all utilizations are to the mean.
        """
        if not utilizations or len(utilizations) < 2:
            return 100.0
            
        mean_util = sum(utilizations) / len(utilizations)
        if mean_util == 0:
            return 100.0
            
        # Calculate relative deviations from mean
        relative_deviations = [abs(u - mean_util) / mean_util for u in utilizations]
        avg_deviation = sum(relative_deviations) / len(relative_deviations)
        
        # Convert to score (lower deviation = higher score)
        balance_score = 100.0 * math.exp(-2 * avg_deviation)
        return min(100.0, max(0.0, balance_score))
        
    def _calculate_time_weighted_rate(self) -> float:
        """
        Calculate time-weighted load balancing rate based on traffic volumes.
        Recent traffic has higher weight.
        """
        if len(self.timestamp_history) < self.min_samples:
            return 0.0
            
        # Simple time-weighted average for now
        # Could be enhanced with actual flow-path correlation
        current_utils = self._get_current_utilizations()
        if not current_utils:
            return 0.0
            
        total_traffic = sum(current_utils.values())
        if total_traffic == 0:
            return 0.0
            
        # Placeholder calculation - would need flow-path mapping for accuracy
        # This gives a rough estimate based on traffic distribution
        cv = self._calculate_coefficient_of_variation(list(current_utils.values()))
        return self._calculate_effectiveness_percentage(cv)
        
    def _get_default_metrics(self) -> Dict[str, float]:
        """Return default metrics when insufficient data."""
        return {
            'coefficient_of_variation': 0.0,
            'distribution_entropy': 0.0,
            'utilization_std_dev': 0.0,
            'load_balancing_effectiveness': 0.0,
            'variance_reduction': 0.0,
            'utilization_balance_score': 0.0,
            'time_weighted_lb_rate': 0.0,
            'avg_utilization': 0.0,
            'max_utilization': 0.0,
            'min_utilization': 0.0,
            'utilization_range': 0.0
        }
        
    def set_baseline_distribution(self, baseline_loads: Dict[Tuple[int, int], float]):
        """
        Set baseline traffic distribution for comparison.
        
        Args:
            baseline_loads: Expected traffic distribution with shortest-path routing
        """
        self.baseline_distribution = baseline_loads.copy()
        
    def get_utilization_trends(self, link: Tuple[int, int], 
                              samples: int = 10) -> List[Tuple[float, float]]:
        """
        Get utilization trends for a specific link.
        
        Args:
            link: Link identifier (dpid1, dpid2)
            samples: Number of recent samples to return
            
        Returns:
            List of (timestamp, utilization) tuples
        """
        if link not in self.utilization_history:
            return []
            
        history = self.utilization_history[link]
        timestamps = list(self.timestamp_history)
        
        if len(history) != len(timestamps):
            return []
            
        # Return recent samples
        recent_data = list(zip(timestamps, history))[-samples:]
        return recent_data
        
    def get_network_load_summary(self) -> Dict[str, any]:
        """
        Get comprehensive network load summary.
        
        Returns:
            Dictionary with network-wide load balancing analysis
        """
        metrics = self.calculate_load_balancing_effectiveness()
        current_utils = self._get_current_utilizations()
        
        summary = {
            'timestamp': time.time(),
            'active_links': len(current_utils),
            'total_traffic': sum(current_utils.values()) if current_utils else 0,
            'load_balancing_metrics': metrics,
            'link_utilizations': dict(current_utils),
            'data_quality': {
                'samples_available': len(self.timestamp_history),
                'time_window_sec': self.window_size_sec,
                'sufficient_data': len(self.timestamp_history) >= self.min_samples
            }
        }
        
        return summary
        
    def _calculate_std_dev(self, values: List[float]) -> float:
        """Calculate standard deviation of a list of values."""
        if not values or len(values) < 2:
            return 0.0
            
        mean = sum(values) / len(values)
        variance = sum((x - mean) ** 2 for x in values) / len(values)
        return math.sqrt(variance)
        
    def _calculate_variance(self, values: List[float]) -> float:
        """Calculate variance of a list of values."""
        if not values:
            return 0.0
            
        mean = sum(values) / len(values)
        variance = sum((x - mean) ** 2 for x in values) / len(values)
        return variance