"""
Path-Based Congestion Avoidance Calculator
==========================================

Implements enhanced path-based congestion avoidance calculation with 
gradient scoring, temporal analysis, and multi-objective optimization.
"""
import math
import time

class PathCongestionCalculator:
    """
    Calculates enhanced path-based congestion avoidance metrics using
    gradient scoring and multi-factor analysis.
    """
    
    def __init__(self, parent_app):
        self.parent_app = parent_app
        self.logger = parent_app.logger
        
        # Configuration for enhanced calculations
        self.config = {
            'utilization_weight': 0.35,
            'trend_weight': 0.20,
            'capacity_weight': 0.20,
            'latency_weight': 0.15,
            'reliability_weight': 0.10,
            'gradient_steepness': 2.0,  # Controls how steep the utilization curve is
            'trend_window': 30,  # seconds for trend analysis
            'baseline_threshold': 0.7,  # 70% baseline threshold
            'critical_threshold': 0.9   # 90% critical threshold
        }
    
    def calculate_path_congestion_avoidance(self, now):
        """Enhanced path-based congestion avoidance using gradient scoring.
        
        Enhanced_Avoidance_% = (Σ(Path_Quality_Score × Flow_Weight) / Σ(Max_Score × Flow_Weight)) × 100
        Where Path_Quality_Score uses gradient functions instead of binary thresholds.
        """
        if not hasattr(self.parent_app, 'alternative_paths'):
            return 0
        
        total_weighted_score = 0
        max_possible_score = 0
        
        # Get current link costs and trends
        cost = self.parent_app._calculate_link_costs(now) if hasattr(self.parent_app, '_calculate_link_costs') else {}
        
        # Analyze all stored alternative paths with enhanced scoring
        for flow_key, paths in self.parent_app.alternative_paths.items():
            flow_weight = self._get_flow_weight(flow_key)
            
            for path in paths:
                # Calculate multi-factor path quality score
                utilization_score = self._calculate_utilization_score(path, cost, now)
                trend_score = self._calculate_trend_score(path, now)
                capacity_score = self._calculate_capacity_score(path, cost, flow_key)
                latency_score = self._calculate_latency_score(path, flow_key)
                reliability_score = self._calculate_reliability_score(path, now)
                
                # Weighted combination of all factors
                path_quality_score = (
                    self.config['utilization_weight'] * utilization_score +
                    self.config['trend_weight'] * trend_score +
                    self.config['capacity_weight'] * capacity_score +
                    self.config['latency_weight'] * latency_score +
                    self.config['reliability_weight'] * reliability_score
                )
                
                total_weighted_score += path_quality_score * flow_weight
                max_possible_score += 1.0 * flow_weight
                
                self.logger.debug("Path %s scores: util=%.3f, trend=%.3f, capacity=%.3f, latency=%.3f, reliability=%.3f, total=%.3f", 
                                path, utilization_score, trend_score, capacity_score, latency_score, reliability_score, path_quality_score)
        
        # Calculate enhanced avoidance percentage
        if max_possible_score > 0:
            enhanced_avoidance_percent = (total_weighted_score / max_possible_score) * 100
            self.logger.info("Enhanced path-based congestion avoidance: %.1f%% (weighted score: %.3f/%.3f)",
                           enhanced_avoidance_percent, total_weighted_score, max_possible_score)
            return enhanced_avoidance_percent
        else:
            return 0
    
    def _get_flow_weight(self, flow_key):
        """Calculate flow weight based on application type, QoS class, and flow characteristics."""
        # Import QoS classes and flow classification constants
        try:
            from ..config.constants import QOS_CLASSES, FLOW_CLASSIFICATION
        except ImportError:
            # Fallback values if import fails
            QOS_CLASSES = {
                'CRITICAL': {'priority': 3, 'weight_multiplier': 2.0},
                'HIGH': {'priority': 2, 'weight_multiplier': 1.5},
                'NORMAL': {'priority': 1, 'weight_multiplier': 1.0},
                'BEST_EFFORT': {'priority': 0, 'weight_multiplier': 0.7}
            }
            FLOW_CLASSIFICATION = {
                'elephant_threshold': 10_000_000,
                'mice_threshold': 1_000_000
            }
        
        base_weight = 1.0
        
        # Try to get flow characteristics from the system
        if hasattr(self.parent_app, 'flow_characteristics') and flow_key in self.parent_app.flow_characteristics:
            flow_chars = self.parent_app.flow_characteristics[flow_key]
            
            # Application-aware prioritization
            qos_class = flow_chars.get('qos_class', 'NORMAL')
            if qos_class in QOS_CLASSES:
                qos_weight = QOS_CLASSES[qos_class].get('weight_multiplier', 1.0)
                base_weight *= qos_weight
                self.logger.debug("Flow %s QoS class %s, weight multiplier: %.1f", flow_key, qos_class, qos_weight)
            
            # Application type prioritization
            app_type = flow_chars.get('application_type', 'unknown')
            if app_type in ['voip', 'video_conference', 'real_time']:
                base_weight *= 1.8  # Critical real-time applications
            elif app_type in ['video_streaming', 'gaming']:
                base_weight *= 1.4  # High priority interactive applications
            elif app_type in ['web', 'email', 'messaging']:
                base_weight *= 1.0  # Normal priority applications
            elif app_type in ['backup', 'bulk_transfer', 'background']:
                base_weight *= 0.6  # Lower priority background traffic
            
            # Flow size-based prioritization (refined)
            flow_rate = flow_chars.get('avg_rate', 0)
            if flow_rate > FLOW_CLASSIFICATION['elephant_threshold']:
                # Elephant flows - higher weight due to impact
                base_weight *= 1.3
            elif flow_rate < FLOW_CLASSIFICATION['mice_threshold']:
                # Mice flows - slightly lower weight but still important for latency
                base_weight *= 0.9
            
            # Flow duration and stability
            flow_duration = flow_chars.get('duration', 0)
            flow_stability = flow_chars.get('stability_score', 1.0)  # 0-1 score
            
            if flow_duration > 300:  # > 5 minutes - established long flow
                base_weight *= 1.2
            elif flow_duration > 60:  # > 1 minute - medium duration
                base_weight *= 1.1
            
            # Stability bonus (stable flows get slight priority for predictability)
            base_weight *= (0.9 + 0.2 * flow_stability)
            
            # Latency sensitivity
            latency_sensitive = flow_chars.get('latency_sensitive', False)
            if latency_sensitive:
                base_weight *= 1.3
            
            # Jitter sensitivity (VoIP, real-time video)
            jitter_sensitive = flow_chars.get('jitter_sensitive', False)
            if jitter_sensitive:
                base_weight *= 1.4
            
        else:
            # If no flow characteristics available, try to infer from flow key
            base_weight = self._infer_flow_priority_from_key(flow_key)
        
        # Cap the weight to reasonable bounds
        base_weight = max(0.3, min(3.0, base_weight))
        
        return base_weight
    
    def _infer_flow_priority_from_key(self, flow_key):
        """Infer flow priority from flow key when characteristics are unavailable."""
        # Try to infer from MAC addresses or port patterns if available
        default_weight = 1.0
        
        # Check if this is a management or control traffic flow
        if isinstance(flow_key, tuple) and len(flow_key) == 2:
            src, dst = flow_key
            
            # Management traffic patterns (heuristic-based)
            if isinstance(src, str) and isinstance(dst, str):
                # Look for special MAC address patterns that might indicate management
                if any(mac.startswith('00:00:00') for mac in [src, dst]):
                    # Could be management or infrastructure traffic
                    return 0.8  # Slightly lower priority
        
        return default_weight
    
    def _calculate_utilization_score(self, path, cost, now):
        """Calculate path utilization score using gradient function with adaptive thresholds."""
        if len(path) < 2:
            return 1.0  # Perfect score for single-node path
        
        path_scores = []
        
        for i in range(len(path) - 1):
            u, v = path[i], path[i + 1]
            link_util = cost.get((u, v), 0)
            
            # Use adaptive threshold for this specific link
            adaptive_threshold = self._calculate_adaptive_threshold(u, v, now)
            utilization_ratio = link_util / adaptive_threshold if adaptive_threshold > 0 else 0
            
            # Gradient scoring with adaptive thresholds
            baseline_threshold = self.config['baseline_threshold']
            critical_threshold = self.config['critical_threshold']
            
            if utilization_ratio <= baseline_threshold:
                # Excellent utilization - high score
                link_score = 1.0
            elif utilization_ratio <= critical_threshold:
                # Moderate utilization - gradient decay
                excess_ratio = (utilization_ratio - baseline_threshold) / (critical_threshold - baseline_threshold)
                link_score = math.exp(-self.config['gradient_steepness'] * excess_ratio)
            else:
                # High utilization - very low score but not zero
                excess_ratio = min(utilization_ratio - critical_threshold, 1.0)
                link_score = 0.1 * math.exp(-self.config['gradient_steepness'] * excess_ratio)
            
            path_scores.append(link_score)
            self.logger.debug("Link %d-%d: util=%.1fM, adaptive_thresh=%.1fM (%.1f%%), score=%.3f", 
                            u, v, link_util/1_000_000, adaptive_threshold/1_000_000, utilization_ratio*100, link_score)
        
        # Path score is the minimum link score (weakest link determines path quality)
        path_utilization_score = min(path_scores) if path_scores else 1.0
        return path_utilization_score
    
    def _calculate_trend_score(self, path, now):
        """Calculate trend score based on congestion prediction."""
        if len(path) < 2:
            return 1.0
        
        if not hasattr(self.parent_app, 'congestion_trends'):
            return 0.5  # Neutral score if no trend data
        
        path_trend_scores = []
        
        for i in range(len(path) - 1):
            u, v = path[i], path[i + 1]
            
            # Get trend data for both directions
            trend_key_1 = (u, self.parent_app.links.get((u, v), [None])[0]) if (u, v) in self.parent_app.links else None
            trend_key_2 = (v, self.parent_app.links.get((u, v), [None, None])[1]) if (u, v) in self.parent_app.links else None
            
            trend_score = 1.0  # Default to good trend
            
            for trend_key in [trend_key_1, trend_key_2]:
                if trend_key and trend_key in self.parent_app.congestion_trends:
                    trend_data = self.parent_app.congestion_trends[trend_key]
                    recent_trends = [(t, util) for t, util in trend_data if now - t <= self.config['trend_window']]
                    
                    if len(recent_trends) >= 3:
                        # Calculate trend slope
                        times = [t for t, _ in recent_trends]
                        utils = [util for _, util in recent_trends]
                        
                        # Simple linear regression for trend
                        n = len(recent_trends)
                        sum_t = sum(times)
                        sum_u = sum(utils)
                        sum_tu = sum(t * u for t, u in zip(times, utils))
                        sum_t2 = sum(t * t for t in times)
                        
                        if n * sum_t2 - sum_t * sum_t != 0:
                            slope = (n * sum_tu - sum_t * sum_u) / (n * sum_t2 - sum_t * sum_t)
                            
                            # Convert slope to score (negative slope = improving = higher score)
                            if slope <= 0:
                                link_trend_score = 1.0  # Improving or stable
                            else:
                                # Degrading trend - score decreases with slope magnitude
                                normalized_slope = min(slope / (self.parent_app.THRESHOLD_BPS / self.config['trend_window']), 2.0)
                                link_trend_score = max(0.1, 1.0 - normalized_slope * 0.5)
                            
                            trend_score = min(trend_score, link_trend_score)
            
            path_trend_scores.append(trend_score)
        
        # Path trend score is the minimum link trend score
        path_trend_score = min(path_trend_scores) if path_trend_scores else 1.0
        return path_trend_score
    
    def _calculate_capacity_score(self, path, cost, flow_key):
        """Calculate capacity adequacy score for the path."""
        if len(path) < 2:
            return 1.0
        
        # Try to estimate flow demand
        flow_demand = self._estimate_flow_demand(flow_key)
        
        # Calculate available capacity along the path
        min_available_capacity = float('inf')
        threshold = self.parent_app.THRESHOLD_BPS
        
        for i in range(len(path) - 1):
            u, v = path[i], path[i + 1]
            link_util = cost.get((u, v), 0)
            available_capacity = max(0, threshold - link_util)
            min_available_capacity = min(min_available_capacity, available_capacity)
        
        if min_available_capacity == float('inf'):
            return 1.0
        
        # Score based on capacity adequacy
        if flow_demand > 0:
            capacity_ratio = min_available_capacity / flow_demand
            if capacity_ratio >= 2.0:
                return 1.0  # Excellent capacity
            elif capacity_ratio >= 1.0:
                return 0.7 + 0.3 * (capacity_ratio - 1.0)  # Good capacity
            else:
                return 0.3 * capacity_ratio  # Insufficient capacity
        else:
            # If we can't estimate demand, score based on absolute available capacity
            capacity_ratio = min_available_capacity / threshold
            return min(1.0, capacity_ratio)
    
    def _estimate_flow_demand(self, flow_key):
        """Estimate bandwidth demand for a flow."""
        # Try to get from flow characteristics
        if hasattr(self.parent_app, 'flow_characteristics') and flow_key in self.parent_app.flow_characteristics:
            flow_chars = self.parent_app.flow_characteristics[flow_key]
            return flow_chars.get('avg_rate', 5_000_000)  # Default 5 Mbps
        
        # Fallback estimate based on flow type
        return 5_000_000  # Default 5 Mbps estimate
    
    def calculate_weighted_path_congestion_avoidance(self, now):
        """Enhanced weighted path-based congestion avoidance with gradient scoring.
        
        Enhanced_Weighted_Avoidance_% = (Σ(Path_Score × Path_Capacity × Flow_Weight) / 
                                         Σ(Max_Score × Path_Capacity × Flow_Weight)) × 100
        This accounts for paths with different capacities and uses gradient scoring.
        """
        if not hasattr(self.parent_app, 'alternative_paths'):
            return 0
        
        total_weighted_capacity = 0
        available_weighted_capacity = 0
        
        # Get current link costs for evaluation
        cost = self.parent_app._calculate_link_costs(now) if hasattr(self.parent_app, '_calculate_link_costs') else {}
        
        # Analyze all stored alternative paths with enhanced capacity weighting
        for flow_key, paths in self.parent_app.alternative_paths.items():
            flow_weight = self._get_flow_weight(flow_key)
            
            for path in paths:
                # Calculate path capacity (minimum available capacity along the path)
                path_capacity = self._calculate_path_capacity(path, cost)
                
                # Calculate enhanced path quality score (reuse from main method)
                utilization_score = self._calculate_utilization_score(path, cost, now)
                trend_score = self._calculate_trend_score(path, now)
                capacity_score = self._calculate_capacity_score(path, cost, flow_key)
                latency_score = self._calculate_latency_score(path, flow_key)
                reliability_score = self._calculate_reliability_score(path, now)
                
                # Combined path quality score
                path_quality_score = (
                    self.config['utilization_weight'] * utilization_score +
                    self.config['trend_weight'] * trend_score +
                    self.config['capacity_weight'] * capacity_score +
                    self.config['latency_weight'] * latency_score +
                    self.config['reliability_weight'] * reliability_score
                )
                
                # Weight by capacity and flow importance
                weighted_capacity = path_capacity * flow_weight
                total_weighted_capacity += weighted_capacity
                available_weighted_capacity += path_quality_score * weighted_capacity
                
                self.logger.debug("Path %s: capacity=%.1fM, scores(u=%.3f,t=%.3f,c=%.3f,l=%.3f,r=%.3f), quality=%.3f, flow_weight=%.3f", 
                                path, path_capacity/1_000_000, utilization_score, trend_score, capacity_score, 
                                latency_score, reliability_score, path_quality_score, flow_weight)
        
        # Calculate enhanced weighted avoidance percentage
        if total_weighted_capacity > 0:
            enhanced_weighted_percent = (available_weighted_capacity / total_weighted_capacity) * 100
            self.logger.info("Enhanced weighted congestion avoidance: %.1f%% (%.1fM effective / %.1fM total)",
                           enhanced_weighted_percent, available_weighted_capacity/1_000_000, 
                           total_weighted_capacity/1_000_000)
            return enhanced_weighted_percent
        else:
            return 0
    
    def _calculate_path_capacity(self, path, cost):
        """Calculate effective capacity of a path (minimum available capacity along the path)."""
        if len(path) < 2:
            return self.parent_app.THRESHOLD_BPS
        
        min_available_capacity = float('inf')
        threshold = self.parent_app.THRESHOLD_BPS
        
        for i in range(len(path) - 1):
            u, v = path[i], path[i + 1]
            link_util = cost.get((u, v), 0)
            available_capacity = max(0, threshold - link_util)
            min_available_capacity = min(min_available_capacity, available_capacity)
        
        return min_available_capacity if min_available_capacity != float('inf') else threshold
    
    def calculate_binary_path_state(self, now):
        """Enhanced binary path state with adaptive thresholds.
        
        Enhanced_Binary_% = (Quality_Weighted_Paths / Total_Weighted_Paths) × 100
        Uses adaptive thresholds and considers path quality gradients.
        """
        if not hasattr(self.parent_app, 'alternative_paths'):
            return 0
        
        total_weighted_paths = 0
        quality_weighted_paths = 0
        
        # Get current link costs for evaluation
        cost = self.parent_app._calculate_link_costs(now) if hasattr(self.parent_app, '_calculate_link_costs') else {}
        
        # Enhanced binary classification with quality weighting
        for flow_key, paths in self.parent_app.alternative_paths.items():
            flow_weight = self._get_flow_weight(flow_key)
            
            for path in paths:
                total_weighted_paths += flow_weight
                
                # Calculate path quality score and convert to binary-like decision
                utilization_score = self._calculate_utilization_score(path, cost, now)
                trend_score = self._calculate_trend_score(path, now)
                latency_score = self._calculate_latency_score(path, flow_key)
                reliability_score = self._calculate_reliability_score(path, now)
                
                # Simplified scoring for binary decision (emphasize key factors)
                combined_score = (
                    0.4 * utilization_score +  # Current utilization
                    0.2 * trend_score +        # Trend prediction
                    0.2 * latency_score +      # Latency adequacy
                    0.2 * reliability_score    # Path reliability
                )
                
                # Binary decision with quality gradient
                # Paths with score > 0.6 are considered "good"
                quality_threshold = 0.6
                if combined_score >= quality_threshold:
                    # Full weight for high-quality paths
                    quality_weighted_paths += flow_weight
                elif combined_score >= quality_threshold * 0.5:
                    # Partial weight for marginal paths
                    quality_weighted_paths += flow_weight * combined_score / quality_threshold
                # else: no weight for poor quality paths
                
                self.logger.debug("Path %s: scores(u=%.3f,t=%.3f,l=%.3f,r=%.3f), combined=%.3f, weight=%.3f", 
                                path, utilization_score, trend_score, latency_score, reliability_score, combined_score, flow_weight)
        
        # Calculate enhanced binary avoidance percentage
        if total_weighted_paths > 0:
            enhanced_binary_percent = (quality_weighted_paths / total_weighted_paths) * 100
            self.logger.info("Enhanced binary congestion avoidance: %.1f%% (%.3f quality weighted / %.3f total weighted)",
                           enhanced_binary_percent, quality_weighted_paths, total_weighted_paths)
            return enhanced_binary_percent
        else:
            return 0
    
    def _calculate_latency_score(self, path, flow_key):
        """Calculate latency score for a path based on hop count and flow requirements."""
        if len(path) < 2:
            return 1.0  # Single hop = perfect latency
        
        # Base latency scoring: shorter paths = better latency
        hop_count = len(path) - 1
        
        # Get flow latency requirements if available
        max_acceptable_latency = 100  # Default 100ms max
        if (hasattr(self.parent_app, 'flow_characteristics') and 
            flow_key in self.parent_app.flow_characteristics):
            flow_chars = self.parent_app.flow_characteristics[flow_key]
            
            # Get latency requirements from flow characteristics
            if flow_chars.get('latency_sensitive', False):
                max_acceptable_latency = 20  # 20ms for latency-sensitive flows
            elif flow_chars.get('application_type') in ['voip', 'video_conference']:
                max_acceptable_latency = 30  # 30ms for real-time applications
            elif flow_chars.get('application_type') in ['gaming', 'video_streaming']:
                max_acceptable_latency = 50  # 50ms for interactive applications
        
        # Estimate latency based on hop count (rough approximation)
        # Assume ~5ms per hop (processing + transmission delay)
        estimated_latency_ms = hop_count * 5
        
        # Calculate latency score using sigmoid function
        if estimated_latency_ms <= max_acceptable_latency * 0.5:
            latency_score = 1.0  # Excellent latency
        elif estimated_latency_ms <= max_acceptable_latency:
            # Good latency - gradual decrease
            excess_ratio = (estimated_latency_ms - max_acceptable_latency * 0.5) / (max_acceptable_latency * 0.5)
            latency_score = 1.0 - 0.3 * excess_ratio
        else:
            # Poor latency - steep decrease
            excess_ratio = min((estimated_latency_ms - max_acceptable_latency) / max_acceptable_latency, 2.0)
            latency_score = max(0.1, 0.7 * math.exp(-excess_ratio))
        
        self.logger.debug("Path %s: hops=%d, est_latency=%dms, max_acceptable=%dms, score=%.3f", 
                        path, hop_count, estimated_latency_ms, max_acceptable_latency, latency_score)
        
        return latency_score
    
    def _calculate_reliability_score(self, path, now):
        """Calculate reliability score based on historical path performance."""
        if len(path) < 2:
            return 1.0  # Single hop = perfect reliability
        
        # Track historical reliability if available
        path_key = tuple(path)
        reliability_score = 1.0  # Default to perfect reliability
        
        # Check if we have historical reliability data
        if hasattr(self.parent_app, 'path_reliability_history'):
            if path_key in self.parent_app.path_reliability_history:
                reliability_data = self.parent_app.path_reliability_history[path_key]
                
                # Calculate recent reliability (last hour)
                recent_data = [
                    (timestamp, success) for timestamp, success in reliability_data
                    if now - timestamp <= 3600  # Last hour
                ]
                
                if recent_data:
                    success_rate = sum(success for _, success in recent_data) / len(recent_data)
                    reliability_score = success_rate
        else:
            # Fallback: estimate reliability based on link utilization patterns
            # Heavily utilized links are less reliable
            total_links = len(path) - 1
            reliable_links = 0
            
            # Get current link costs
            cost = self.parent_app._calculate_link_costs(now) if hasattr(self.parent_app, '_calculate_link_costs') else {}
            threshold = self.parent_app.THRESHOLD_BPS
            
            for i in range(len(path) - 1):
                u, v = path[i], path[i + 1]
                link_util = cost.get((u, v), 0)
                utilization_ratio = link_util / threshold if threshold > 0 else 0
                
                # Links under 60% utilization are considered reliable
                if utilization_ratio < 0.6:
                    reliable_links += 1
                elif utilization_ratio < 0.8:
                    # Partially reliable
                    reliable_links += 0.5
                # else: unreliable link (no contribution)
            
            if total_links > 0:
                reliability_score = reliable_links / total_links
        
        # Add path length penalty (longer paths = more failure points)
        path_length_penalty = math.exp(-0.1 * (len(path) - 2))  # Exponential penalty for long paths
        reliability_score *= path_length_penalty
        
        return max(0.1, min(1.0, reliability_score))  # Bound between 0.1 and 1.0
    
    def _calculate_adaptive_threshold(self, dpid1, dpid2, now):
        """Calculate adaptive congestion threshold for a specific link based on historical patterns."""
        base_threshold = self.parent_app.THRESHOLD_BPS
        
        # If we have traffic monitor, use historical data to adapt threshold
        if hasattr(self.parent_app, 'traffic_monitor'):
            traffic_monitor = self.parent_app.traffic_monitor
            
            # Get historical utilization for this link
            if (dpid1, dpid2) in self.parent_app.links:
                port1, port2 = self.parent_app.links[(dpid1, dpid2)]
                
                # Get trend data for both directions
                trend_key_1 = (dpid1, port1)
                trend_key_2 = (dpid2, port2)
                
                utilization_history = []
                
                # Collect recent utilization data (last 5 minutes)
                for trend_key in [trend_key_1, trend_key_2]:
                    if hasattr(self.parent_app, 'congestion_trends') and trend_key in self.parent_app.congestion_trends:
                        trend_data = self.parent_app.congestion_trends[trend_key]
                        recent_data = [(t, util) for t, util in trend_data if now - t <= 300]  # Last 5 minutes
                        utilization_history.extend([util for _, util in recent_data])
                
                if utilization_history:
                    # Calculate statistical measures
                    avg_utilization = sum(utilization_history) / len(utilization_history)
                    max_utilization = max(utilization_history)
                    
                    # Calculate standard deviation
                    variance = sum((x - avg_utilization) ** 2 for x in utilization_history) / len(utilization_history)
                    std_dev = variance ** 0.5
                    
                    # Adaptive threshold calculation
                    if std_dev > 0:
                        # High variability = lower threshold (more sensitive)
                        # Low variability = higher threshold (less sensitive)
                        variability_factor = min(std_dev / avg_utilization, 1.0) if avg_utilization > 0 else 0.5
                        
                        # Time-of-day factor (if we can detect patterns)
                        time_factor = self._calculate_time_factor(utilization_history, now)
                        
                        # Calculate adaptive threshold
                        # Formula: base * (0.7 + 0.3 * (1 - variability_factor)) * time_factor
                        adaptive_threshold = base_threshold * (0.7 + 0.3 * (1 - variability_factor)) * time_factor
                        
                        # Bound the adaptive threshold to reasonable limits
                        adaptive_threshold = max(base_threshold * 0.5, min(base_threshold * 1.5, adaptive_threshold))
                        
                        self.logger.debug("Link %d-%d adaptive threshold: base=%.1fM, avg_util=%.1fM, variability=%.3f, time_factor=%.3f, adaptive=%.1fM", 
                                        dpid1, dpid2, base_threshold/1_000_000, avg_utilization/1_000_000, 
                                        variability_factor, time_factor, adaptive_threshold/1_000_000)
                        
                        return adaptive_threshold
        
        # Fallback to base threshold if no historical data
        return base_threshold
    
    def _calculate_time_factor(self, utilization_history, now):
        """Calculate time-of-day adjustment factor based on historical patterns."""
        # Simple time-based adjustment
        # This could be enhanced with machine learning for pattern recognition
        
        import datetime
        current_hour = datetime.datetime.fromtimestamp(now).hour
        
        # Simple heuristic: assume business hours (9-17) have higher baseline
        if 9 <= current_hour <= 17:
            return 1.1  # Slightly higher threshold during business hours
        elif 22 <= current_hour or current_hour <= 6:
            return 0.9  # Lower threshold during off-hours (more sensitive)
        else:
            return 1.0  # Normal threshold
    
    def get_enhanced_configuration(self):
        """Get current enhanced calculation configuration."""
        return {
            'algorithm': 'Enhanced Multi-Factor Path-Based Congestion Avoidance',
            'version': '2.0',
            'factors': {
                'utilization': {
                    'weight': self.config['utilization_weight'],
                    'method': 'gradient_scoring',
                    'baseline_threshold': self.config['baseline_threshold'],
                    'critical_threshold': self.config['critical_threshold']
                },
                'trend': {
                    'weight': self.config['trend_weight'],
                    'method': 'linear_regression',
                    'window_seconds': self.config['trend_window']
                },
                'capacity': {
                    'weight': self.config['capacity_weight'],
                    'method': 'flow_demand_aware'
                },
                'latency': {
                    'weight': self.config['latency_weight'],
                    'method': 'hop_count_estimation'
                },
                'reliability': {
                    'weight': self.config['reliability_weight'],
                    'method': 'utilization_based_estimation'
                }
            },
            'features': [
                'gradient_based_utilization_scoring',
                'temporal_trend_integration',
                'application_aware_flow_prioritization',
                'multi_objective_path_scoring',
                'adaptive_threshold_calculation'
            ]
        }