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
            
            # Enhanced congestion avoidance bonus (increased from 30% to 50%)
            if baseline_congested and path != baseline_path:
                avoidance_bonus = 1.0 - ADAPTIVE_MODE_PARAMS['congestion_avoidance_bonus']
                
                # Flow-aware adjustment to congestion avoidance bonus
                if flow_type:
                    avoidance_bonus = self._adjust_avoidance_bonus_for_flow(avoidance_bonus, flow_type)
                
                score *= avoidance_bonus
                self.logger.debug("Applying enhanced congestion avoidance bonus to path %s (score reduced by %.0f%%) for flow type %s", 
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
    """Select path with minimum total utilization"""
    
    def __init__(self, parent_app):
        self.parent_app = parent_app
    
    def select_path(self, paths, cost, flow_key=None, **kwargs):
        if not paths:
            return None
        
        best_path = None
        best_cost = float('inf')
        
        for path in paths:
            total_cost = self._calculate_path_cost(path, cost)
            if total_cost < best_cost:
                best_cost = total_cost
                best_path = path
        
        return best_path
    
    def _calculate_path_cost(self, path, cost):
        """Calculate total cost of a path."""
        if len(path) < 2:
            return 0
        return sum(cost.get((path[i], path[i+1]), 0) for i in range(len(path)-1))


class WeightedECMPSelector(PathSelectionStrategy):
    """Weighted ECMP based on utilization"""
    
    def __init__(self, parent_app):
        self.parent_app = parent_app
    
    def select_path(self, paths, cost, flow_key=None, **kwargs):
        if not paths:
            return None
        
        if len(paths) == 1:
            return paths[0]
        
        # Calculate weights based on inverse utilization
        weights = []
        for path in paths:
            total_cost = self._calculate_path_cost(path, cost)
            # Higher cost = lower weight
            weight = 1.0 / (total_cost + 1)
            weights.append(weight)
        
        # Use consistent hashing for flow persistence
        flow_hash = self._hash_flow(flow_key) if flow_key else 0
        
        # Select path based on weighted random selection
        total_weight = sum(weights)
        normalized_hash = (flow_hash % 1000) / 1000.0  # Normalize to [0, 1)
        
        cumulative_weight = 0
        for i, weight in enumerate(weights):
            cumulative_weight += weight / total_weight
            if normalized_hash <= cumulative_weight:
                return paths[i]
        
        return paths[0]  # Fallback
    
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
    """Round-robin path selection"""
    
    def __init__(self, parent_app):
        self.parent_app = parent_app
        self.path_counters = {}
    
    def select_path(self, paths, cost, flow_key=None, **kwargs):
        if not paths:
            return None
        
        if len(paths) == 1:
            return paths[0]
        
        # Use flow key for consistency
        counter_key = flow_key if flow_key else "global"
        
        if counter_key not in self.path_counters:
            self.path_counters[counter_key] = 0
        
        selected_index = self.path_counters[counter_key] % len(paths)
        self.path_counters[counter_key] += 1
        
        return paths[selected_index]


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
        """Enhanced path finding with multiple strategies"""
        flow_key = (src, dst)
        
        # Get all possible paths
        all_paths = self._find_k_shortest_paths(src, dst, cost, k=3)
        if not all_paths:
            return None
        
        # Store alternative paths for ECMP
        self.alternative_paths[flow_key] = all_paths
        
        # Select best path based on current mode
        mode = self.parent_app.load_balancing_mode
        selector = self.selectors.get(mode)
        
        if selector:
            return selector.select_path(all_paths, cost, flow_key)
        else:
            # Fallback to first path
            return all_paths[0]
    
    def _find_k_shortest_paths(self, src, dst, cost, k=3):
        """Find k shortest paths using Yen's algorithm (simplified version)"""
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
                        candidates.append(total_path)
            
            if candidates:
                # Select candidate with lowest utilization-based cost
                best_candidate = min(candidates, key=lambda p: self._calculate_path_cost(p, cost))
                paths.append(best_candidate)
                candidates.remove(best_candidate)
        
        return paths
    
    def _calculate_path_cost(self, path, cost):
        """Calculate total cost of a path."""
        if len(path) < 2:
            return 0
        return sum(cost.get((path[i], path[i+1]), 0) for i in range(len(path)-1))
    
    def _dijkstra(self, src, dst, cost, avoid_congested=True):
        """Dijkstra's shortest path algorithm"""
        return self.parent_app._dijkstra(src, dst, cost, avoid_congested)