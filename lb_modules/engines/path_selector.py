"""
Path Selection Engine
====================

Advanced path selection algorithms for SDN load balancing including:
- Yen's K-shortest paths algorithm
- Multiple load balancing strategies
- Adaptive congestion-aware selection
"""

import time
import hashlib
from abc import ABC, abstractmethod

from ..config.constants import (
    LOAD_BALANCING_MODES, EWMA_ALPHA, CONGESTION_PREDICTION_WINDOW,
    ADAPTIVE_MODE_PARAMS, CONGESTION_PARAMS
)


class PathSelectionStrategy(ABC):
    """Abstract base class for path selection strategies"""
    
    @abstractmethod
    def select_path(self, paths, cost, flow_key=None, **kwargs):
        """Select best path from available paths"""
        pass


class AdaptivePathSelector(PathSelectionStrategy):
    """Adaptive path selection based on current load and congestion prediction"""
    
    def __init__(self, parent_app):
        self.parent_app = parent_app
        self.logger = parent_app.logger
    
    def select_path(self, paths, cost, flow_key=None, **kwargs):
        if not paths:
            return None
        
        if len(paths) == 1:
            return paths[0]
        
        best_path = None
        best_score = float('inf')
        baseline_path = paths[0]  # First path is hop-count baseline
        
        # Check if baseline path is congested
        baseline_congested = self._is_path_congested(baseline_path, cost)
        
        # Flow-aware classification if available
        flow_type = None
        if flow_key and hasattr(self.parent_app, 'flow_classifier') and ADAPTIVE_MODE_PARAMS['flow_aware_integration']:
            flow_type = self.parent_app.flow_classifier.classify_flow(flow_key)
            self.logger.debug("Flow %s classified as %s", flow_key, flow_type)
        
        for i, path in enumerate(paths):
            score = self._calculate_adaptive_score(path, cost, flow_key, flow_type)
            
            # AGGRESSIVE congestion avoidance bonus (increased from 50% to 70%)
            if baseline_congested and path != baseline_path:
                avoidance_bonus = 1.0 - ADAPTIVE_MODE_PARAMS['congestion_avoidance_bonus']
                # Make adaptive mode more aggressive by increasing the bonus
                avoidance_bonus *= 0.6  # 40% more aggressive than configured value
                
                # Flow-aware adjustment to congestion avoidance bonus
                if flow_type:
                    avoidance_bonus = self._adjust_avoidance_bonus_for_flow(avoidance_bonus, flow_type)
                
                score *= avoidance_bonus
                self.logger.debug("Applying AGGRESSIVE congestion avoidance bonus to path %s (score reduced by %.0f%%) for flow type %s", 
                                path, (1.0 - avoidance_bonus) * 100, flow_type or 'unknown')
            
            # Additional penalty for paths that share links with congested baseline
            if baseline_congested and path != baseline_path:
                shared_congestion_penalty = self._calculate_shared_congestion_penalty(path, baseline_path, cost)
                score += shared_congestion_penalty
            
            if score < best_score:
                best_score = score
                best_path = path
        
        # Log decision for debugging
        if baseline_congested and best_path != baseline_path:
            self.logger.info("Adaptive path selection: avoided congested baseline %s, selected %s for flow type %s", 
                           baseline_path, best_path, flow_type or 'unknown')
        elif baseline_congested and best_path == baseline_path:
            self.logger.warning("Adaptive path selection: baseline %s is congested but no better alternative found for flow type %s", 
                              baseline_path, flow_type or 'unknown')
        
        return best_path
    
    def _is_path_congested(self, path, cost):
        """Enhanced path congestion detection with dynamic thresholds and prediction."""
        if len(path) < 2:
            return False
        
        now = time.time()
        
        for i in range(len(path) - 1):
            u, v = path[i], path[i + 1]
            link_cost = cost.get((u, v), 0)
            
            # Use enhanced congestion detection threshold
            congestion_threshold = self.parent_app.THRESHOLD_BPS * CONGESTION_PARAMS['approaching_congestion_threshold']
            
            # Check current utilization
            if link_cost > congestion_threshold:
                return True
            
            # Check predicted congestion
            if (u, v) in self.parent_app.links:
                port_u, port_v = self.parent_app.links[(u, v)]
                prediction_u = self.parent_app._predict_congestion(u, port_u, now)
                prediction_v = self.parent_app._predict_congestion(v, port_v, now)
                
                # Consider path congested if prediction shows high congestion
                if max(prediction_u, prediction_v) > congestion_threshold:
                    return True
        
        return False
    
    def _calculate_adaptive_score(self, path, cost, flow_key=None, flow_type=None):
        """Enhanced adaptive score with multi-level congestion analysis and flow-aware weighting."""
        if len(path) < 2:
            return 0
        
        score = 0
        now = time.time()
        
        # Multi-level congestion scoring
        congestion_levels = self._analyze_path_congestion_levels(path, cost, now)
        
        for i in range(len(path) - 1):
            u, v = path[i], path[i + 1]
            
            # Current utilization cost
            current_cost = cost.get((u, v), 0)
            score += current_cost
            
            # Enhanced congestion prediction penalty (increased from 30% to 60%)
            if (u, v) in self.parent_app.links:
                pu, pv = self.parent_app.links[(u, v)]
                prediction_u = self.parent_app._predict_congestion(u, pu, now)
                prediction_v = self.parent_app._predict_congestion(v, pv, now)
                prediction_penalty = max(prediction_u, prediction_v) * ADAPTIVE_MODE_PARAMS['congestion_prediction_weight']
                score += prediction_penalty
            
            # Multi-level congestion penalties
            link_congestion_level = congestion_levels.get((u, v), 0)
            if link_congestion_level > 0:
                # Exponential penalty based on congestion level
                congestion_penalty = (
                    self.parent_app.THRESHOLD_BPS * 
                    ADAPTIVE_MODE_PARAMS['congestion_penalty_multiplier'] * 
                    (ADAPTIVE_MODE_PARAMS['exponential_penalty_factor'] ** link_congestion_level)
                )
                score += congestion_penalty
            
            # Burst detection penalty
            if hasattr(self.parent_app, 'burst_detection_state'):
                link_key = (u, pu) if (u, v) in self.parent_app.links else None
                if link_key and link_key in self.parent_app.burst_detection_state:
                    burst_intensity = self.parent_app.burst_detection_state[link_key].get('burst_intensity', 0)
                    if burst_intensity > 0:
                        burst_penalty = current_cost * burst_intensity * 0.5
                        score += burst_penalty
        
        # Path length penalty (prefer shorter paths when congestion is similar)
        path_length_penalty = len(path) * 100  # Small penalty for each hop
        score += path_length_penalty
        
        # Flow-aware adjustment if enabled
        if ADAPTIVE_MODE_PARAMS['flow_aware_integration']:
            score = self._apply_flow_aware_adjustment(score, path, cost, flow_type)
        
        return score
    
    def _analyze_path_congestion_levels(self, path, cost, now):
        """Analyze congestion levels for each link in the path."""
        congestion_levels = {}
        
        if len(path) < 2:
            return congestion_levels
        
        for i in range(len(path) - 1):
            u, v = path[i], path[i + 1]
            current_cost = cost.get((u, v), 0)
            
            # Calculate congestion level (0=green, 1=yellow, 2=red)
            threshold = self.parent_app.THRESHOLD_BPS
            
            if current_cost < threshold * 0.5:
                level = 0  # Green - low congestion
            elif current_cost < threshold * 0.8:
                level = 1  # Yellow - moderate congestion  
            else:
                level = 2  # Red - high congestion
            
            # Check prediction to upgrade level
            if (u, v) in self.parent_app.links:
                pu, pv = self.parent_app.links[(u, v)]
                prediction_u = self.parent_app._predict_congestion(u, pu, now)
                prediction_v = self.parent_app._predict_congestion(v, pv, now)
                predicted_util = max(prediction_u, prediction_v)
                
                if predicted_util > threshold * 0.8:
                    level = max(level, 2)  # Upgrade to red if prediction shows high congestion
                elif predicted_util > threshold * 0.5:
                    level = max(level, 1)  # Upgrade to yellow if prediction shows moderate congestion
            
            congestion_levels[(u, v)] = level
        
        return congestion_levels
    
    def _apply_flow_aware_adjustment(self, score, path, cost, flow_type=None):
        """Apply flow-aware adjustments to the path score based on flow characteristics."""
        # Calculate path capacity and utilization ratio
        total_capacity = len(path) * 1_000_000_000  # Assume 1 Gbps per hop
        path_utilization = sum(cost.get((path[i], path[i+1]), 0) for i in range(len(path)-1))
        
        if path_utilization > 0:
            utilization_ratio = path_utilization / total_capacity
            
            # Flow-type specific adjustments
            if flow_type == 'elephant':
                # Elephant flows: heavily penalize high utilization paths, prefer dedicated capacity
                if utilization_ratio > 0.6:
                    score *= 1.5  # 50% penalty for elephant flows on busy paths
                elif utilization_ratio < 0.2:
                    score *= 0.8  # 20% bonus for elephant flows on low utilization paths
                
                # Elephant flows prefer shorter paths with high capacity
                score *= (1.0 + 0.1 * (len(path) - 2))  # 10% penalty per extra hop
                
            elif flow_type == 'mice':
                # Mice flows: prioritize low latency (shorter paths), less sensitive to utilization
                if utilization_ratio > 0.8:
                    score *= 1.2  # 20% penalty only for very high utilization
                
                # Mice flows strongly prefer shorter paths
                score *= (1.0 + 0.2 * (len(path) - 2))  # 20% penalty per extra hop
                
            elif flow_type == 'normal':
                # Normal flows: balanced approach
                if utilization_ratio > 0.7:
                    score *= 1.25  # 25% penalty for high utilization paths
                elif utilization_ratio < 0.3:
                    score *= 0.9  # 10% bonus for low utilization paths
                
                # Moderate hop count preference
                score *= (1.0 + 0.05 * (len(path) - 2))  # 5% penalty per extra hop
                
            else:
                # Default behavior for unknown flow types
                if utilization_ratio > 0.7:
                    score *= 1.3  # 30% penalty for high utilization paths
                elif utilization_ratio < 0.3:
                    score *= 0.9  # 10% bonus for low utilization paths
        
        return score
    
    def _adjust_avoidance_bonus_for_flow(self, base_avoidance_bonus, flow_type):
        """Adjust congestion avoidance bonus based on flow type."""
        if flow_type == 'elephant':
            # Elephant flows get maximum congestion avoidance bonus
            return base_avoidance_bonus * 0.7  # More aggressive avoidance (30% more bonus)
        elif flow_type == 'mice':
            # Mice flows get moderate congestion avoidance bonus (prefer low latency)
            return base_avoidance_bonus * 0.85  # Moderate avoidance (15% more bonus)
        elif flow_type == 'normal':
            # Normal flows get standard congestion avoidance bonus
            return base_avoidance_bonus
        else:
            # Unknown flow types get standard bonus
            return base_avoidance_bonus
    
    def _calculate_shared_congestion_penalty(self, path, baseline_path, cost):
        """Calculate penalty for paths that share links with congested baseline."""
        if len(path) < 2 or len(baseline_path) < 2:
            return 0
        
        # Create sets of links for each path
        path_links = set((path[i], path[i+1]) for i in range(len(path)-1))
        baseline_links = set((baseline_path[i], baseline_path[i+1]) for i in range(len(baseline_path)-1))
        
        # Find shared links
        shared_links = path_links.intersection(baseline_links)
        
        penalty = 0
        for u, v in shared_links:
            link_cost = cost.get((u, v), 0)
            # Penalty for sharing congested links with baseline
            if link_cost > self.parent_app.THRESHOLD_BPS * 0.5:
                penalty += link_cost * 0.3  # 30% penalty for shared congested links
        
        return penalty


class LeastLoadedPathSelector(PathSelectionStrategy):
    """Enhanced least loaded path selector with aggressive load balancing"""
    
    def __init__(self, parent_app):
        self.parent_app = parent_app
        self.path_performance_history = {}
        self.logger = parent_app.logger
    
    def select_path(self, paths, cost, flow_key=None, **kwargs):
        if not paths:
            return None
        
        if len(paths) == 1:
            return paths[0]
        
        best_path = None
        best_score = float('inf')
        path_analysis = []
        
        for i, path in enumerate(paths):
            # Calculate current path cost
            current_cost = self._calculate_path_cost(path, cost)
            
            # Calculate advanced score considering multiple factors
            score = current_cost
            
            # Factor 1: Penalize paths with bottleneck links heavily
            max_link_util = 0
            bottleneck_penalty = 0
            
            for j in range(len(path) - 1):
                u, v = path[j], path[j + 1]
                link_util = cost.get((u, v), 0)
                max_link_util = max(max_link_util, link_util)
                
                # Heavy penalty for links approaching threshold
                if link_util > self.parent_app.THRESHOLD_BPS * 0.6:  # 60% threshold
                    bottleneck_penalty += link_util * 2  # Double penalty
                elif link_util > self.parent_app.THRESHOLD_BPS * 0.3:  # 30% threshold
                    bottleneck_penalty += link_util * 0.5  # Moderate penalty
            
            score += bottleneck_penalty
            
            # Factor 2: Prefer paths with consistent low utilization (no spikes)
            if len(path) > 1:
                link_utils = [cost.get((path[k], path[k+1]), 0) for k in range(len(path)-1)]
                util_variance = self._calculate_variance(link_utils)
                score += util_variance * 0.1  # Small penalty for high variance
            
            # Factor 3: Historical performance bonus for consistently good paths
            path_str = "->".join(map(str, path))
            if path_str in self.path_performance_history:
                avg_historical_cost = self.path_performance_history[path_str]['avg_cost']
                selection_count = self.path_performance_history[path_str]['count']
                
                # Bonus for paths that have historically performed well
                if avg_historical_cost < current_cost * 1.2 and selection_count > 3:
                    score *= 0.9  # 10% bonus
            
            path_analysis.append({
                'path': path,
                'current_cost': current_cost,
                'bottleneck_penalty': bottleneck_penalty,
                'max_link_util': max_link_util,
                'final_score': score
            })
            
            if score < best_score:
                best_score = score
                best_path = path
        
        # Update performance history
        if best_path:
            path_str = "->".join(map(str, best_path))
            if path_str not in self.path_performance_history:
                self.path_performance_history[path_str] = {'total_cost': 0, 'count': 0}
            
            history = self.path_performance_history[path_str]
            history['total_cost'] += self._calculate_path_cost(best_path, cost)
            history['count'] += 1
            history['avg_cost'] = history['total_cost'] / history['count']
        
        # Log least loaded decision
        sorted_analysis = sorted(path_analysis, key=lambda x: x['final_score'])
        self.logger.debug("Least loaded analysis: best=%s (score=%.1f, cost=%.1fM, max_util=%.1fM), alternatives: %s", 
                         best_path, best_score, 
                         sorted_analysis[0]['current_cost']/1_000_000,
                         sorted_analysis[0]['max_link_util']/1_000_000,
                         [(a['path'], f"{a['final_score']:.1f}") for a in sorted_analysis[1:3]])
        
        return best_path
    
    def _calculate_path_cost(self, path, cost):
        """Calculate total cost of a path."""
        if len(path) < 2:
            return 0
        return sum(cost.get((path[i], path[i+1]), 0) for i in range(len(path)-1))
    
    def _calculate_variance(self, values):
        """Calculate variance of a list of values."""
        if not values or len(values) < 2:
            return 0
        
        mean = sum(values) / len(values)
        variance = sum((x - mean) ** 2 for x in values) / len(values)
        return variance


class WeightedECMPSelector(PathSelectionStrategy):
    """Enhanced weighted ECMP with aggressive load distribution"""
    
    def __init__(self, parent_app):
        self.parent_app = parent_app
        self.path_selection_history = {}
        self.logger = parent_app.logger
    
    def select_path(self, paths, cost, flow_key=None, **kwargs):
        if not paths:
            return None
        
        if len(paths) == 1:
            return paths[0]
        
        # Calculate enhanced weights based on multiple factors
        weights = []
        path_costs = []
        
        for i, path in enumerate(paths):
            path_cost = self._calculate_path_cost(path, cost)
            path_costs.append(path_cost)
            
            # Enhanced weight calculation with exponential preference for low-cost paths
            if path_cost == 0:
                weight = 100.0  # Very high weight for unused paths
            else:
                # Exponential inverse relationship: lower cost = exponentially higher weight
                weight = 1000.0 / (path_cost ** 0.5 + 1)
            
            # Boost weight for underutilized paths in selection history
            path_str = "->".join(map(str, path))
            historical_usage = self.path_selection_history.get(path_str, 0)
            avg_usage = sum(self.path_selection_history.values()) / max(len(self.path_selection_history), 1)
            
            if historical_usage < avg_usage * 0.8:  # If path is underutilized
                weight *= 1.5  # Boost its weight
                
            weights.append(weight)
        
        # Use consistent hashing for flow persistence but with enhanced distribution
        flow_hash = self._hash_flow(flow_key) if flow_key else 0
        
        # Select path based on weighted random selection with bias toward load balancing
        total_weight = sum(weights)
        normalized_hash = (flow_hash % 10000) / 10000.0  # Higher resolution
        
        cumulative_weight = 0
        selected_path = None
        selected_index = 0
        
        for i, weight in enumerate(weights):
            cumulative_weight += weight / total_weight
            if normalized_hash <= cumulative_weight:
                selected_path = paths[i]
                selected_index = i
                break
        
        if selected_path is None:
            selected_path = paths[0]  # Fallback
            selected_index = 0
        
        # Update selection history
        path_str = "->".join(map(str, selected_path))
        self.path_selection_history[path_str] = self.path_selection_history.get(path_str, 0) + 1
        
        # Log weighted ECMP decision
        weight_ratios = [w/total_weight for w in weights]
        self.logger.debug("Weighted ECMP selected path %d (weight %.3f): %s, costs: %s, weights: %s", 
                         selected_index, weight_ratios[selected_index], selected_path, 
                         [f"{c/1_000_000:.1f}M" for c in path_costs], 
                         [f"{w:.1f}" for w in weights])
        
        return selected_path
    
    def _calculate_path_cost(self, path, cost):
        """Calculate total cost of a path."""
        if len(path) < 2:
            return 0
        return sum(cost.get((path[i], path[i+1]), 0) for i in range(len(path)-1))
    
    def _hash_flow(self, flow_key):
        """Generate consistent hash for flow key"""
        if not flow_key:
            return 0
        flow_str = f"{flow_key[0]}-{flow_key[1]}"
        return int(hashlib.md5(flow_str.encode()).hexdigest(), 16)


class RoundRobinSelector(PathSelectionStrategy):
    """Enhanced round-robin path selection with aggressive cycling"""
    
    def __init__(self, parent_app):
        self.parent_app = parent_app
        self.global_counter = 0
        self.path_usage_stats = {}
        self.logger = parent_app.logger
    
    def select_path(self, paths, cost, flow_key=None, **kwargs):
        if not paths:
            return None
        
        if len(paths) == 1:
            return paths[0]
        
        # Aggressive round-robin: always cycle through paths
        # Don't use flow-specific counters to ensure true round-robin behavior
        selected_index = self.global_counter % len(paths)
        selected_path = paths[selected_index]
        
        # Update global counter
        self.global_counter += 1
        
        # Track path usage for analytics
        path_str = "->".join(map(str, selected_path))
        self.path_usage_stats[path_str] = self.path_usage_stats.get(path_str, 0) + 1
        
        # Log round-robin decision every 10 selections for debugging
        if self.global_counter % 10 == 0:
            usage_summary = sorted(self.path_usage_stats.items(), key=lambda x: x[1], reverse=True)
            self.logger.debug("Round-robin stats after %d selections: %s", 
                            self.global_counter, usage_summary[:3])
        
        self.logger.debug("Round-robin selected path %d/%d: %s (global counter: %d)", 
                         selected_index + 1, len(paths), selected_path, self.global_counter)
        
        return selected_path


class LatencyAwarePathSelector(PathSelectionStrategy):
    """Latency-aware path selection for real-time applications"""
    
    def __init__(self, parent_app):
        self.parent_app = parent_app
        self.logger = parent_app.logger
    
    def select_path(self, paths, cost, flow_key=None, **kwargs):
        """Select path with minimum latency"""
        if not paths:
            return None
        
        if len(paths) == 1:
            return paths[0]
        
        # Use flow classifier for latency-aware selection
        if hasattr(self.parent_app, 'flow_classifier'):
            return self.parent_app.flow_classifier.select_latency_aware_path(paths, cost)
        
        # Fallback: select path with minimum hop count and utilization
        best_path = None
        best_score = float('inf')
        
        for path in paths:
            # Estimate latency based on hop count and current utilization
            hop_count = len(path) - 1
            path_cost = self._calculate_path_cost(path, cost)
            
            # Combine hop count (latency proxy) with utilization
            latency_score = hop_count * 1.0  # Each hop adds 1ms base latency
            utilization_score = (path_cost / 1_000_000) * 0.2  # 20% weight for utilization
            
            total_score = latency_score + utilization_score
            
            if total_score < best_score:
                best_score = total_score
                best_path = path
        
        return best_path or paths[0]
    
    def _calculate_path_cost(self, path, cost):
        """Calculate total cost of a path."""
        if len(path) < 2:
            return 0
        return sum(cost.get((path[i], path[i+1]), 0) for i in range(len(path)-1))


class QoSAwarePathSelector(PathSelectionStrategy):
    """QoS-aware path selection based on service requirements"""
    
    def __init__(self, parent_app):
        self.parent_app = parent_app
        self.logger = parent_app.logger
    
    def select_path(self, paths, cost, flow_key=None, **kwargs):
        """Select path that meets QoS requirements"""
        if not paths:
            return None
        
        if len(paths) == 1:
            return paths[0]
        
        # Use flow classifier for QoS-aware selection
        if hasattr(self.parent_app, 'flow_classifier') and flow_key:
            return self.parent_app.flow_classifier.select_qos_aware_path(paths, cost, flow_key)
        
        # Fallback: select path with best balance of latency and capacity
        best_path = None
        best_score = float('inf')
        
        for path in paths:
            path_cost = self._calculate_path_cost(path, cost)
            hop_count = len(path) - 1
            
            # Score based on utilization and hop count
            utilization_score = path_cost / 1_000_000  # Normalize to Mbps
            latency_score = hop_count * 10  # Each hop penalty
            
            total_score = utilization_score + latency_score
            
            if total_score < best_score:
                best_score = total_score
                best_path = path
        
        return best_path or paths[0]
    
    def _calculate_path_cost(self, path, cost):
        """Calculate total cost of a path."""
        if len(path) < 2:
            return 0
        return sum(cost.get((path[i], path[i+1]), 0) for i in range(len(path)-1))


class FlowAwarePathSelector(PathSelectionStrategy):
    """Flow-aware path selection with elephant/mice flow differentiation"""
    
    def __init__(self, parent_app):
        self.parent_app = parent_app
        self.logger = parent_app.logger
    
    def select_path(self, paths, cost, flow_key=None, **kwargs):
        """Select path based on flow characteristics"""
        if not paths:
            return None
        
        if len(paths) == 1:
            return paths[0]
        
        # Use flow classifier for flow-aware selection
        if hasattr(self.parent_app, 'flow_classifier') and flow_key:
            return self.parent_app.flow_classifier.select_flow_aware_path(paths, cost, flow_key)
        
        # Fallback: adaptive selection based on utilization
        best_path = None
        best_score = float('inf')
        
        for path in paths:
            path_cost = self._calculate_path_cost(path, cost)
            
            # Simple score based on utilization
            score = path_cost
            
            if score < best_score:
                best_score = score
                best_path = path
        
        return best_path or paths[0]
    
    def _calculate_path_cost(self, path, cost):
        """Calculate total cost of a path."""
        if len(path) < 2:
            return 0
        return sum(cost.get((path[i], path[i+1]), 0) for i in range(len(path)-1))


class PathSelectionEngine:
    """Main path selection engine with multiple strategies"""
    
    def __init__(self, parent_app):
        self.parent_app = parent_app
        self.logger = parent_app.logger
        
        # Initialize selectors for all 7 load balancing modes
        self.selectors = {
            LOAD_BALANCING_MODES['adaptive']: AdaptivePathSelector(parent_app),
            LOAD_BALANCING_MODES['least_loaded']: LeastLoadedPathSelector(parent_app),
            LOAD_BALANCING_MODES['weighted_ecmp']: WeightedECMPSelector(parent_app),
            LOAD_BALANCING_MODES['round_robin']: RoundRobinSelector(parent_app),
            LOAD_BALANCING_MODES['latency_aware']: LatencyAwarePathSelector(parent_app),
            LOAD_BALANCING_MODES['qos_aware']: QoSAwarePathSelector(parent_app),
            LOAD_BALANCING_MODES['flow_aware']: FlowAwarePathSelector(parent_app),
        }
        
        self.alternative_paths = {}  # (src, dst) -> [path1, path2, ...]
    
    def find_path(self, src, dst, cost):
        """Enhanced path finding with algorithm-specific behavior"""
        flow_key = (src, dst)
        
        # Get current algorithm for specialized behavior
        current_algorithm = self._get_current_algorithm_name()
        
        # Algorithm-specific path discovery
        if current_algorithm == 'round_robin':
            # Round Robin: Always use all available paths for true rotation
            all_paths = self._find_k_shortest_paths(src, dst, cost, k=5)
        elif current_algorithm == 'adaptive':
            # Adaptive: Aggressive path discovery with congestion-aware alternatives
            all_paths = self._find_k_shortest_paths(src, dst, cost, k=5)
            all_paths = self._enhance_paths_for_adaptive(all_paths, cost, src, dst)
        elif current_algorithm == 'least_loaded':
            # Least Loaded: Focus on utilization-diverse paths
            all_paths = self._find_k_shortest_paths(src, dst, cost, k=4)
            all_paths = self._filter_paths_by_utilization_diversity(all_paths, cost)
        elif current_algorithm == 'weighted_ecmp':
            # Weighted ECMP: Ensure path cost variance for meaningful weighting
            all_paths = self._find_k_shortest_paths(src, dst, cost, k=5)
            all_paths = self._ensure_path_cost_variance(all_paths, cost)
        elif current_algorithm == 'latency_aware':
            # Latency Aware: Prioritize path length diversity
            all_paths = self._find_k_shortest_paths(src, dst, cost, k=4)
            all_paths = self._prioritize_by_latency(all_paths, cost)
        else:
            # Default behavior for other algorithms
            all_paths = self._find_k_shortest_paths(src, dst, cost, k=4)
        
        if not all_paths:
            return None
        
        # Store alternative paths for ECMP
        self.alternative_paths[flow_key] = all_paths
        
        # Select best path based on current mode
        mode = self.parent_app.load_balancing_mode
        selector = self.selectors.get(mode)
        
        if selector:
            selected_path = selector.select_path(all_paths, cost, flow_key)
            
            # Log algorithm-specific decision for debugging
            if len(all_paths) > 1:
                path_costs = [self._calculate_path_cost(p, cost) for p in all_paths]
                selected_cost = self._calculate_path_cost(selected_path, cost) if selected_path else 0
                self.logger.debug("%s algorithm selected path %s (cost=%.1fM) from %d alternatives (costs: %s)", 
                                current_algorithm.upper(), selected_path, selected_cost/1_000_000,
                                len(all_paths), [f"{c/1_000_000:.1f}M" for c in path_costs])
            
            return selected_path
        else:
            # Fallback to first path
            return all_paths[0]
    
    def _find_k_shortest_paths(self, src, dst, cost, k=5):
        """Find k diverse shortest paths using enhanced Yen's algorithm"""
        paths = []
        
        # Find first shortest path using hop-count (true baseline for congestion avoidance comparison)
        uniform_cost = {}
        for (u, v) in self.parent_app.links.keys():
            if u < v:  # Avoid duplicates - only add each link once
                uniform_cost[(u, v)] = 1
                uniform_cost[(v, u)] = 1
        
        first_path = self._dijkstra(src, dst, uniform_cost, avoid_congested=False)
        if not first_path:
            return paths
        
        paths.append(first_path)
        candidates = []
        
        # Enhanced path diversity tracking
        path_diversity_threshold = 0.3  # Paths must differ by at least 30% of links
        
        for i in range(k - 1):
            if not paths:
                break
                
            # For each node in the previous shortest path
            for j in range(len(paths[-1]) - 1):
                spur_node = paths[-1][j]
                root_path = paths[-1][:j+1]
                
                # Remove edges that would create duplicate paths
                removed_edges = set()
                for path in paths:
                    if len(path) > j and path[:j+1] == root_path:
                        if j+1 < len(path):
                            edge = (path[j], path[j+1])
                            removed_edges.add(edge)
                
                # Create modified cost without removed edges (use current utilization cost for alternatives)
                modified_cost = dict(cost)
                for edge in removed_edges:
                    if edge in modified_cost:
                        modified_cost[edge] = float('inf')
                
                # Find spur path using current utilization cost for alternative paths
                spur_path = self._dijkstra(spur_node, dst, modified_cost, avoid_congested=False)
                if spur_path:
                    total_path = root_path[:-1] + spur_path
                    if total_path not in paths and total_path not in candidates:
                        # Check path diversity before adding
                        if self._is_path_diverse_enough(total_path, paths, path_diversity_threshold):
                            candidates.append(total_path)
            
            if candidates:
                # Sort candidates by utilization-based cost and diversity
                candidates.sort(key=lambda p: (
                    self._calculate_path_cost(p, cost),
                    -self._calculate_path_diversity_score(p, paths)  # Higher diversity is better
                ))
                
                # Select the best diverse candidate
                best_candidate = candidates[0]
                paths.append(best_candidate)
                candidates.remove(best_candidate)
                
                # Log path diversity for debugging
                diversity = self._calculate_path_diversity_score(best_candidate, paths[:-1])
                self.logger.debug("Added diverse path %s with diversity score %.2f", 
                                best_candidate, diversity)
        
        # If we have fewer than k paths, try different spur nodes to increase diversity
        if len(paths) < k:
            self._generate_additional_diverse_paths(src, dst, cost, paths, k - len(paths))
        
        self.logger.info("Generated %d diverse paths for %s->%s (requested %d)", 
                        len(paths), src, dst, k)
        return paths
    
    def _calculate_path_cost(self, path, cost):
        """Calculate total cost of a path."""
        if len(path) < 2:
            return 0
        return sum(cost.get((path[i], path[i+1]), 0) for i in range(len(path)-1))
    
    def _dijkstra(self, src, dst, cost, avoid_congested=True):
        """Dijkstra's shortest path algorithm"""
        return self.parent_app._dijkstra(src, dst, cost, avoid_congested)
    
    def _is_path_diverse_enough(self, new_path, existing_paths, threshold=0.3):
        """Check if a new path is diverse enough compared to existing paths"""
        if not existing_paths:
            return True
        
        for existing_path in existing_paths:
            overlap_ratio = self._calculate_path_overlap_ratio(new_path, existing_path)
            if overlap_ratio > (1.0 - threshold):  # If overlap is too high, path is not diverse enough
                return False
        
        return True
    
    def _calculate_path_overlap_ratio(self, path1, path2):
        """Calculate the ratio of overlapping links between two paths"""
        if len(path1) < 2 or len(path2) < 2:
            return 0.0
        
        # Convert paths to sets of links
        links1 = set((path1[i], path1[i+1]) for i in range(len(path1)-1))
        links2 = set((path2[i], path2[i+1]) for i in range(len(path2)-1))
        
        # Also consider reverse direction links
        links1.update((path1[i+1], path1[i]) for i in range(len(path1)-1))
        links2.update((path2[i+1], path2[i]) for i in range(len(path2)-1))
        
        # Calculate overlap
        common_links = links1.intersection(links2)
        total_unique_links = links1.union(links2)
        
        if not total_unique_links:
            return 0.0
        
        return len(common_links) / len(total_unique_links)
    
    def _calculate_path_diversity_score(self, new_path, existing_paths):
        """Calculate diversity score for a path compared to existing paths"""
        if not existing_paths:
            return 1.0
        
        total_diversity = 0.0
        for existing_path in existing_paths:
            overlap_ratio = self._calculate_path_overlap_ratio(new_path, existing_path)
            diversity = 1.0 - overlap_ratio
            total_diversity += diversity
        
        return total_diversity / len(existing_paths)
    
    def _generate_additional_diverse_paths(self, src, dst, cost, existing_paths, needed_paths):
        """Generate additional diverse paths using different strategies"""
        if needed_paths <= 0:
            return
        
        # Strategy 1: Try avoiding high-utilization links
        high_util_threshold = self.parent_app.THRESHOLD_BPS * 0.5
        avoided_links = set()
        
        for (u, v), link_cost in cost.items():
            if link_cost > high_util_threshold:
                avoided_links.add((u, v))
                avoided_links.add((v, u))
        
        if avoided_links:
            modified_cost = dict(cost)
            for link in avoided_links:
                modified_cost[link] = float('inf')
            
            alt_path = self._dijkstra(src, dst, modified_cost, avoid_congested=False)
            if alt_path and self._is_path_diverse_enough(alt_path, existing_paths, 0.2):
                existing_paths.append(alt_path)
                needed_paths -= 1
                self.logger.debug("Added congestion-avoiding diverse path: %s", alt_path)
        
        # Strategy 2: Try intermediate nodes as waypoints
        if needed_paths > 0 and hasattr(self.parent_app, 'dp_set'):
            intermediate_nodes = [dpid for dpid in self.parent_app.dp_set.keys() 
                                if dpid != src and dpid != dst]
            
            for intermediate in intermediate_nodes[:needed_paths]:
                # Route via intermediate node
                path_to_intermediate = self._dijkstra(src, intermediate, cost, avoid_congested=False)
                path_from_intermediate = self._dijkstra(intermediate, dst, cost, avoid_congested=False)
                
                if path_to_intermediate and path_from_intermediate and len(path_to_intermediate) > 1:
                    # Combine paths, avoiding duplicate intermediate node
                    waypoint_path = path_to_intermediate + path_from_intermediate[1:]
                    
                    if self._is_path_diverse_enough(waypoint_path, existing_paths, 0.2):
                        existing_paths.append(waypoint_path)
                        needed_paths -= 1
                        self.logger.debug("Added waypoint diverse path via %s: %s", 
                                        intermediate, waypoint_path)
    
    def _get_current_algorithm_name(self):
        """Get the name of the currently active load balancing algorithm."""
        if not hasattr(self.parent_app, 'load_balancing_mode'):
            return 'unknown'
        
        # Import here to avoid circular imports
        from ..config.constants import LOAD_BALANCING_MODES
        
        # Reverse lookup to get algorithm name from mode value
        mode_to_name = {v: k for k, v in LOAD_BALANCING_MODES.items()}
        return mode_to_name.get(self.parent_app.load_balancing_mode, 'unknown')
    
    def _enhance_paths_for_adaptive(self, paths, cost, src, dst):
        """Enhance paths for adaptive algorithm with congestion-aware alternatives."""
        if not paths:
            return paths
        
        enhanced_paths = paths.copy()
        
        # Add congestion-avoiding paths if baseline is congested
        baseline_path = paths[0]
        baseline_congested = any(cost.get((baseline_path[i], baseline_path[i+1]), 0) > self.parent_app.THRESHOLD_BPS * 0.2 
                               for i in range(len(baseline_path)-1))
        
        if baseline_congested:
            # Try to find paths that completely avoid high-congestion links
            high_congestion_links = set()
            for (u, v), link_cost in cost.items():
                if link_cost > self.parent_app.THRESHOLD_BPS * 0.4:
                    high_congestion_links.add((u, v))
                    high_congestion_links.add((v, u))
            
            if high_congestion_links:
                modified_cost = dict(cost)
                for link in high_congestion_links:
                    modified_cost[link] = float('inf')
                
                congestion_free_path = self._dijkstra(src, dst, modified_cost, avoid_congested=False)
                if congestion_free_path and congestion_free_path not in enhanced_paths:
                    enhanced_paths.append(congestion_free_path)
                    self.logger.debug("Adaptive: Added congestion-free path %s", congestion_free_path)
        
        return enhanced_paths
    
    def _filter_paths_by_utilization_diversity(self, paths, cost):
        """Filter paths to ensure utilization diversity for least loaded algorithm."""
        if len(paths) <= 2:
            return paths
        
        # Calculate utilization for each path
        path_utils = []
        for path in paths:
            max_util = 0
            total_util = 0
            for i in range(len(path) - 1):
                u, v = path[i], path[i + 1]
                util = cost.get((u, v), 0)
                max_util = max(max_util, util)
                total_util += util
            path_utils.append((path, max_util, total_util))
        
        # Sort by total utilization and keep diverse paths
        path_utils.sort(key=lambda x: x[2])  # Sort by total utilization
        
        filtered = [path_utils[0][0]]  # Always keep lowest utilization path
        
        for i in range(1, len(path_utils)):
            path, max_util, total_util = path_utils[i]
            
            # Add path if it's significantly different in utilization
            add_path = True
            for existing_path, existing_max, existing_total in [(p, mu, tu) for p, mu, tu in path_utils if p in filtered]:
                util_diff = abs(total_util - existing_total)
                if util_diff < self.parent_app.THRESHOLD_BPS * 0.1:  # Less than 10% threshold difference
                    add_path = False
                    break
            
            if add_path:
                filtered.append(path)
        
        self.logger.debug("Least Loaded: Filtered %d paths to %d with utilization diversity", 
                         len(paths), len(filtered))
        return filtered[:4]  # Limit to 4 diverse paths
    
    def _ensure_path_cost_variance(self, paths, cost):
        """Ensure meaningful cost variance for weighted ECMP."""
        if len(paths) <= 1:
            return paths
        
        path_costs = [(self._calculate_path_cost(path, cost), path) for path in paths]
        path_costs.sort()  # Sort by cost
        
        # Ensure there's meaningful variance in costs
        min_cost = path_costs[0][0]
        max_cost = path_costs[-1][0]
        
        if max_cost > 0 and (max_cost - min_cost) / max_cost < 0.2:  # Less than 20% variance
            # Try to find more diverse paths by avoiding lowest cost links
            diverse_paths = paths.copy()
            
            # Find lowest cost links and try alternative paths
            low_cost_links = set()
            for (u, v), link_cost in cost.items():
                if link_cost < self.parent_app.THRESHOLD_BPS * 0.1:  # Very low utilization
                    low_cost_links.add((u, v))
            
            if low_cost_links and len(paths) >= 2:
                # Create a path that avoids some low-cost links to increase variance
                modified_cost = dict(cost)
                for link in list(low_cost_links)[:len(low_cost_links)//2]:  # Avoid half of low-cost links
                    modified_cost[link] = cost[link] + self.parent_app.THRESHOLD_BPS * 0.3
                
                alt_path = self._dijkstra(paths[0][0], paths[0][-1], modified_cost, avoid_congested=False)
                if alt_path and alt_path not in diverse_paths:
                    diverse_paths.append(alt_path)
            
            self.logger.debug("Weighted ECMP: Enhanced cost variance from %.1f%% to diverse paths", 
                             ((max_cost - min_cost) / max_cost * 100) if max_cost > 0 else 0)
            return diverse_paths
        
        return paths
    
    def _prioritize_by_latency(self, paths, cost):
        """Prioritize paths by latency characteristics for latency-aware algorithm."""
        if not paths:
            return paths
        
        # Sort paths by hop count (latency proxy) and utilization
        path_scores = []
        for path in paths:
            hop_count = len(path) - 1
            path_cost = self._calculate_path_cost(path, cost)
            
            # Latency score: heavily weight hop count, moderate weight on utilization
            latency_score = hop_count * 10 + (path_cost / 1_000_000) * 0.1
            path_scores.append((latency_score, path))
        
        # Sort by latency score (lower is better)
        path_scores.sort()
        
        # Return paths in latency-optimized order
        prioritized = [path for _, path in path_scores]
        
        self.logger.debug("Latency Aware: Prioritized %d paths by latency (hop counts: %s)", 
                         len(prioritized), [len(p)-1 for p in prioritized])
        return prioritized