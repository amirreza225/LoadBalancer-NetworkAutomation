"""
Flow Classifier
===============

Classifies network flows into different categories (elephant, mice, normal)
and provides QoS-aware path selection for optimal resource utilization.
"""

import time
import hashlib
from ..config.constants import (
    ELEPHANT_FLOW_THRESHOLD, MICE_FLOW_THRESHOLD, QOS_CLASSES,
    LATENCY_WEIGHT, QOS_WEIGHT, FLOW_TIMEOUT_SEC
)


class FlowClassifier:
    """
    Classifies flows and provides differentiated path selection
    """
    
    def __init__(self, parent_app):
        self.parent_app = parent_app
        self.logger = parent_app.logger
        
        # Flow tracking
        self.flow_characteristics = {}  # flow_key -> flow_info
        self.flow_qos_classes = {}  # flow_key -> qos_class
        self.flow_priorities = {}  # flow_key -> priority_level
        
        # Path selection for different flow types
        self.path_latency_cache = {}  # (src, dst) -> {path_hash: latency_ms}
        
        # Update parent app references
        self.parent_app.flow_characteristics = self.flow_characteristics
        self.parent_app.flow_qos_classes = self.flow_qos_classes
        self.parent_app.flow_priorities = self.flow_priorities
        self.parent_app.path_latency_cache = self.path_latency_cache
    
    def classify_flow(self, flow_key, packet_size=1500):
        """
        Classify flow as elephant, mice, or normal based on observed characteristics.
        Real network engineers use this for differentiated handling.
        """
        now = time.time()
        
        if flow_key not in self.flow_characteristics:
            # Initialize flow tracking
            self.flow_characteristics[flow_key] = {
                'type': 'unknown',
                'rate': 0,
                'start_time': now,
                'packet_count': 0,
                'byte_count': 0,
                'last_seen': now
            }
        
        flow_info = self.flow_characteristics[flow_key]
        flow_info['packet_count'] += 1
        flow_info['byte_count'] += packet_size
        flow_info['last_seen'] = now
        
        # Calculate flow rate over observation window
        duration = now - flow_info['start_time']
        if duration > 1.0:  # At least 1 second of observation
            flow_info['rate'] = flow_info['byte_count'] / duration
            
            # Classify based on sustained rate
            if flow_info['rate'] > ELEPHANT_FLOW_THRESHOLD:
                flow_info['type'] = 'elephant'
            elif flow_info['rate'] < MICE_FLOW_THRESHOLD:
                flow_info['type'] = 'mice'
            else:
                flow_info['type'] = 'normal'
        
        return flow_info['type']
    
    def classify_qos(self, flow_key, packet_info=None):
        """
        Classify flow into QoS classes based on traffic characteristics.
        """
        # Simple QoS classification based on flow characteristics
        if flow_key not in self.flow_qos_classes:
            flow_type = self.classify_flow(flow_key)
            
            # Default QoS classification
            if flow_type == 'elephant':
                # High-bandwidth flows often need guaranteed bandwidth
                self.flow_qos_classes[flow_key] = 'HIGH'
            elif flow_type == 'mice':
                # Small flows often need low latency
                self.flow_qos_classes[flow_key] = 'CRITICAL'
            else:
                self.flow_qos_classes[flow_key] = 'NORMAL'
        
        return self.flow_qos_classes[flow_key]
    
    def select_flow_aware_path(self, paths, cost, flow_key):
        """
        Flow-aware path selection with differentiated handling for elephants vs mice.
        Production networks require this for optimal resource utilization.
        """
        if not paths:
            return None
            
        flow_type = self.classify_flow(flow_key)
        
        if flow_type == 'elephant':
            # Elephant flows: Use dedicated high-capacity paths, avoid sharing
            return self._select_elephant_flow_path(paths, cost)
        elif flow_type == 'mice':
            # Mice flows: Use any available path, prioritize low latency
            return self._select_mice_flow_path(paths, cost)
        else:
            # Normal flows: Use adaptive selection
            return self._select_adaptive_path(paths, cost)
    
    def _select_elephant_flow_path(self, paths, cost):
        """
        Select optimal path for elephant flows - prioritize high capacity links.
        """
        # Score paths based on capacity and current utilization
        best_path = None
        best_score = float('-inf')
        
        for path in paths:
            # Calculate path capacity and utilization
            path_capacity = self._calculate_path_capacity(path)
            path_utilization = self._calculate_path_cost(path, cost)
            
            # Elephants prefer high capacity, low utilization paths
            # Avoid paths that are already heavily utilized
            utilization_ratio = path_utilization / (path_capacity + 1)
            capacity_score = path_capacity / 1_000_000  # Normalize to Mbps
            utilization_penalty = utilization_ratio * 100
            
            score = capacity_score - utilization_penalty
            
            if score > best_score:
                best_score = score
                best_path = path
                
        return best_path or paths[0]
    
    def _select_mice_flow_path(self, paths, cost):
        """
        Select optimal path for mice flows - prioritize low latency.
        """
        # Mice flows prioritize latency over capacity
        best_path = None
        best_latency = float('inf')
        
        for path in paths:
            path_latency = self._estimate_path_latency(path)
            path_cost = self._calculate_path_cost(path, cost)
            
            # Combine latency and current load (light weight on cost)
            total_score = path_latency + (path_cost / 10_000_000)  # Normalize cost
            
            if total_score < best_latency:
                best_latency = total_score
                best_path = path
        
        return best_path or paths[0]
    
    def select_qos_aware_path(self, paths, cost, flow_key):
        """
        QoS-aware path selection based on service requirements.
        """
        if not paths:
            return None
        
        qos_class = self.classify_qos(flow_key)
        qos_requirements = QOS_CLASSES.get(qos_class, QOS_CLASSES['NORMAL'])
        
        best_path = None
        best_score = float('inf')
        
        for path in paths:
            # Calculate path metrics
            path_latency = self._estimate_path_latency(path)
            path_cost = self._calculate_path_cost(path, cost)
            path_capacity = self._calculate_path_capacity(path)
            
            # Check if path meets QoS requirements
            if (path_latency > qos_requirements['max_latency'] or 
                path_capacity < qos_requirements['min_bw']):
                continue  # Skip paths that don't meet requirements
            
            # Calculate QoS score
            latency_score = path_latency * LATENCY_WEIGHT
            utilization_score = path_cost / 1_000_000  # Normalize to Mbps
            priority_bonus = qos_requirements['priority'] * 10
            
            total_score = latency_score + utilization_score - priority_bonus
            
            if total_score < best_score:
                best_score = total_score
                best_path = path
        
        # Fallback to first path if no path meets QoS requirements
        return best_path or paths[0]
    
    def select_latency_aware_path(self, paths, cost):
        """
        Latency-aware path selection for real-time applications.
        """
        if not paths:
            return None
        
        best_path = None
        best_score = float('inf')
        
        for path in paths:
            path_latency = self._estimate_path_latency(path)
            path_cost = self._calculate_path_cost(path, cost)
            
            # Combine latency and current utilization with weights
            latency_score = path_latency * LATENCY_WEIGHT
            utilization_score = (path_cost / 1_000_000) * (1 - LATENCY_WEIGHT)
            
            total_score = latency_score + utilization_score
            
            if total_score < best_score:
                best_score = total_score
                best_path = path
        
        return best_path or paths[0]
    
    def _select_adaptive_path(self, paths, cost):
        """
        Adaptive path selection (delegated to parent app)
        """
        if hasattr(self.parent_app, '_select_adaptive_path'):
            return self.parent_app._select_adaptive_path(paths, cost)
        return paths[0] if paths else None
    
    def _calculate_path_capacity(self, path):
        """
        Calculate the capacity of a path (minimum link capacity).
        """
        if len(path) < 2:
            return float('inf')
        
        min_capacity = float('inf')
        
        for i in range(len(path) - 1):
            u, v = path[i], path[i + 1]
            
            # Assume 1 Gbps links by default
            # In a real implementation, this would query switch capabilities
            link_capacity = 1_000_000_000  # 1 Gbps in bytes/sec
            
            # Check if we have adaptive thresholds for this link
            if hasattr(self.parent_app, 'adaptive_thresholds'):
                adaptive_threshold = self.parent_app.adaptive_thresholds.get(u, 0)
                if adaptive_threshold > 0:
                    link_capacity = adaptive_threshold * 4  # Assume threshold is 25% of capacity
            
            min_capacity = min(min_capacity, link_capacity)
        
        return min_capacity
    
    def _calculate_path_cost(self, path, cost):
        """Calculate total cost of a path."""
        if len(path) < 2:
            return 0
        return sum(cost.get((path[i], path[i+1]), 0) for i in range(len(path)-1))
    
    def _estimate_path_latency(self, path):
        """
        Estimate path latency based on hop count and link characteristics.
        """
        if len(path) < 2:
            return 0
        
        # Generate path hash for caching
        path_hash = hashlib.md5(str(path).encode()).hexdigest()
        
        # Check cache
        path_key = (path[0], path[-1])
        if path_key in self.path_latency_cache:
            cached_latency = self.path_latency_cache[path_key].get(path_hash)
            if cached_latency is not None:
                return cached_latency
        
        # Calculate latency
        total_latency = 0
        
        for i in range(len(path) - 1):
            u, v = path[i], path[i + 1]
            
            # Check if we have measured latency for this link
            if hasattr(self.parent_app, 'link_latencies'):
                link_latency = self.parent_app.link_latencies.get((u, v), 0)
                if link_latency > 0:
                    total_latency += link_latency
                    continue
            
            # Default latency estimation: 1ms per hop + 0.1ms per switch
            total_latency += 1.0 + 0.1
        
        # Cache the result
        if path_key not in self.path_latency_cache:
            self.path_latency_cache[path_key] = {}
        self.path_latency_cache[path_key][path_hash] = total_latency
        
        return total_latency
    
    def get_flow_statistics(self):
        """Get statistics about classified flows"""
        now = time.time()
        stats = {
            'total_flows': len(self.flow_characteristics),
            'elephant_flows': 0,
            'mice_flows': 0,
            'normal_flows': 0,
            'unknown_flows': 0,
            'qos_distribution': {qos: 0 for qos in QOS_CLASSES.keys()},
            'active_flows': 0
        }
        
        for flow_key, flow_info in self.flow_characteristics.items():
            # Count by flow type
            flow_type = flow_info['type']
            if flow_type == 'elephant':
                stats['elephant_flows'] += 1
            elif flow_type == 'mice':
                stats['mice_flows'] += 1
            elif flow_type == 'normal':
                stats['normal_flows'] += 1
            else:
                stats['unknown_flows'] += 1
            
            # Count active flows (seen in last 5 minutes)
            if now - flow_info['last_seen'] < 300:
                stats['active_flows'] += 1
            
            # Count QoS distribution
            qos_class = self.flow_qos_classes.get(flow_key, 'NORMAL')
            stats['qos_distribution'][qos_class] += 1
        
        return stats
    
    def cleanup_old_flows(self, now):
        """Clean up old flow classifications"""
        flows_to_remove = []
        
        for flow_key, flow_info in self.flow_characteristics.items():
            if now - flow_info['last_seen'] > FLOW_TIMEOUT_SEC:
                flows_to_remove.append(flow_key)
        
        for flow_key in flows_to_remove:
            del self.flow_characteristics[flow_key]
            if flow_key in self.flow_qos_classes:
                del self.flow_qos_classes[flow_key]
            if flow_key in self.flow_priorities:
                del self.flow_priorities[flow_key]
        
        if flows_to_remove:
            self.logger.debug("Cleaned up %d old flow classifications", len(flows_to_remove))
    
    def get_flow_info(self, flow_key):
        """Get information about a specific flow"""
        flow_info = self.flow_characteristics.get(flow_key, {})
        qos_class = self.flow_qos_classes.get(flow_key, 'NORMAL')
        priority = self.flow_priorities.get(flow_key, 0)
        
        return {
            'flow_key': flow_key,
            'characteristics': flow_info,
            'qos_class': qos_class,
            'priority': priority,
            'qos_requirements': QOS_CLASSES.get(qos_class, QOS_CLASSES['NORMAL'])
        }
    
    def set_flow_qos(self, flow_key, qos_class):
        """Manually set QoS class for a flow"""
        if qos_class in QOS_CLASSES:
            self.flow_qos_classes[flow_key] = qos_class
            self.flow_priorities[flow_key] = QOS_CLASSES[qos_class]['priority']
            return True
        return False
    
    def reset_flow_classifications(self):
        """Reset all flow classifications"""
        self.flow_characteristics.clear()
        self.flow_qos_classes.clear()
        self.flow_priorities.clear()
        self.path_latency_cache.clear()
        
        self.logger.info("Flow classifications reset")