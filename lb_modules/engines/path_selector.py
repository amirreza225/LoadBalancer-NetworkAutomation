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

from ..config.constants import LOAD_BALANCING_MODES, EWMA_ALPHA, CONGESTION_PREDICTION_WINDOW


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
        
        for i, path in enumerate(paths):
            score = self._calculate_adaptive_score(path, cost)
            
            # Give significant bonus to paths that avoid congested baseline
            if baseline_congested and path != baseline_path:
                score *= 0.7  # 30% bonus for avoiding congested baseline
                self.logger.debug("Applying congestion avoidance bonus to path %s (score reduced by 30%%)", path)
            
            if score < best_score:
                best_score = score
                best_path = path
        
        # Log decision for debugging
        if baseline_congested and best_path != baseline_path:
            self.logger.info("Adaptive path selection: avoided congested baseline %s, selected %s", 
                           baseline_path, best_path)
        elif baseline_congested and best_path == baseline_path:
            self.logger.warning("Adaptive path selection: baseline %s is congested but no better alternative found", 
                              baseline_path)
        
        return best_path
    
    def _is_path_congested(self, path, cost):
        """Check if a path has any congested links."""
        if len(path) < 2:
            return False
        
        for i in range(len(path) - 1):
            u, v = path[i], path[i + 1]
            link_cost = cost.get((u, v), 0)
            if link_cost > self.parent_app.THRESHOLD_BPS * 0.3:  # Consider 30% threshold as approaching congestion
                return True
        return False
    
    def _calculate_adaptive_score(self, path, cost):
        """Calculate adaptive score considering current load and predicted congestion."""
        if len(path) < 2:
            return 0
        
        score = 0
        now = time.time()
        
        for i in range(len(path) - 1):
            u, v = path[i], path[i + 1]
            
            # Current utilization cost
            current_cost = cost.get((u, v), 0)
            score += current_cost
            
            # Congestion prediction penalty
            if (u, v) in self.parent_app.links:
                pu, pv = self.parent_app.links[(u, v)]
                trend_u = self.parent_app._predict_congestion(u, pu, now)
                trend_v = self.parent_app._predict_congestion(v, pv, now)
                prediction_penalty = max(trend_u, trend_v) * 0.3  # 30% weight for prediction
                score += prediction_penalty
            
            # Avoid already congested links
            if current_cost > self.parent_app.THRESHOLD_BPS:
                score += self.parent_app.THRESHOLD_BPS * 0.5  # Heavy penalty for congested links
        
        return score


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