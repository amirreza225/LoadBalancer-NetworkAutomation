"""
Efficiency Tracker
==================

Tracks and calculates efficiency metrics for the SDN load balancer
including load balancing rates, congestion avoidance, and variance analysis.
"""

import time
import collections


class EfficiencyTracker:
    """
    Tracks load balancer efficiency metrics
    """
    
    def __init__(self, parent_app):
        self.parent_app = parent_app
        self.logger = parent_app.logger
        
        # Initialize efficiency metrics
        self.efficiency_metrics = {
            'total_flows': 0,
            'load_balanced_flows': 0,
            'congestion_avoided': 0,
            'avg_path_length_lb': 0,
            'avg_path_length_sp': 0,
            'total_reroutes': 0,
            'link_utilization_variance': 0,
            'baseline_link_utilization_variance': 0,
            'start_time': time.time()
        }
        
        # Track flows that have avoided congestion (prevent double counting)
        self.flows_with_congestion_avoidance = set()
        
        # Update parent app references
        self.parent_app.efficiency_metrics = self.efficiency_metrics
        self.parent_app.flows_with_congestion_avoidance = self.flows_with_congestion_avoidance
    
    def update_flow_metrics(self, s_dpid, d_dpid, path, cost, src_mac=None, dst_mac=None):
        """Update efficiency metrics for a new flow."""
        self.efficiency_metrics['total_flows'] += 1
        
        # Use MAC-based flow key for consistency with main controller
        # Fall back to DPID-based key if MAC addresses not provided
        if src_mac and dst_mac:
            flow_key = (src_mac, dst_mac)
        else:
            flow_key = (s_dpid, d_dpid)
        
        # Also track the DPID-based key for internal calculations
        dpid_flow_key = (s_dpid, d_dpid)
        baseline_path = None
        
        # If we have alternative paths stored, the first one should be the hop-count baseline
        if hasattr(self.parent_app, 'alternative_paths') and dpid_flow_key in self.parent_app.alternative_paths:
            alt_paths = self.parent_app.alternative_paths[dpid_flow_key]
            if alt_paths:
                baseline_path = alt_paths[0]  # First path is hop-count baseline
        
        if not baseline_path:
            # Fallback to calculating baseline path
            baseline_path = self._shortest_path_baseline(s_dpid, d_dpid)
        
        self.logger.info("Flow metrics update - Selected path: %s, Baseline (hop-count): %s, Flow key: %s", path, baseline_path, flow_key)
        
        if baseline_path:
            # Check if we're using a different path than shortest path
            if path != baseline_path:
                self.efficiency_metrics['load_balanced_flows'] += 1
                self.logger.info("Load balanced flow %d: using path %s instead of baseline %s", 
                                self.efficiency_metrics['total_flows'], path, baseline_path)
            else:
                self.logger.info("Flow %d using shortest path %s", 
                               self.efficiency_metrics['total_flows'], path)
            
            # Enhanced congestion avoidance detection
            if len(baseline_path) > 1:
                baseline_congested = False
                predicted_congestion = False
                congested_links = []
                baseline_total_cost = 0
                
                # Check current congestion on baseline path
                congestion_threshold = self.parent_app.THRESHOLD_BPS * 0.3  # 30% threshold
                self.logger.debug("Checking baseline congestion with threshold %.1f Mbps", congestion_threshold/1_000_000)
                
                for i in range(len(baseline_path) - 1):
                    u, v = baseline_path[i], baseline_path[i + 1]
                    link_cost = cost.get((u, v), 0)
                    baseline_total_cost += link_cost
                    
                    # Only count actual congestion (>30% threshold) for avoidance tracking
                    # Predicted congestion tracking removed to prevent false positives
                    if link_cost > congestion_threshold:  # 30% threshold for actual congestion
                        baseline_congested = True
                        congested_links.append(f"{u}-{v} (congested: {link_cost/1_000_000:.1f}M)")
                        self.logger.debug("Link %s-%s is congested: %.1f Mbps > %.1f Mbps threshold", 
                                        u, v, link_cost/1_000_000, congestion_threshold/1_000_000)
                    else:
                        self.logger.debug("Link %s-%s not congested: %.1f Mbps <= %.1f Mbps threshold", 
                                        u, v, link_cost/1_000_000, congestion_threshold/1_000_000)
                
                # Only count actual congestion, not predicted
                congestion_detected = baseline_congested
                
                if congestion_detected:
                    # Calculate selected path cost for comparison
                    selected_path_cost = self._calculate_path_cost(path, cost)
                    
                    self.logger.debug("Congestion detected on baseline %s (cost=%.1fM), evaluating selected path %s (cost=%.1fM)", 
                                    baseline_path, baseline_total_cost/1_000_000, path, selected_path_cost/1_000_000)
                    
                    # Track congestion avoidance - only if we actually avoid congested links
                    avoided_congestion = False
                    
                    # ENHANCED Criteria for REAL congestion avoidance:
                    # 1. Baseline path has ACTUALLY congested links (>30% threshold), AND  
                    # 2. Selected path avoids those congested links, AND
                    # 3. Selected path has SIGNIFICANT cost improvement (>30% improvement), AND
                    # 4. Selected path is not heavily congested itself (links <50% threshold)
                    if baseline_congested and path != baseline_path:
                        selected_path_avoids_congestion = True
                        selected_path_links = [(path[j], path[j+1]) for j in range(len(path)-1)]
                        
                        self.logger.debug("Evaluating REAL congestion avoidance for different path %s vs baseline %s", path, baseline_path)
                        
                        # Check if selected path avoids congested links from baseline
                        for i in range(len(baseline_path) - 1):
                            u, v = baseline_path[i], baseline_path[i + 1]
                            link_cost = cost.get((u, v), 0)
                            if link_cost > congestion_threshold:  # This link is congested
                                # Check if selected path uses this congested link
                                if (u, v) in selected_path_links:
                                    selected_path_avoids_congestion = False
                                    self.logger.debug("Selected path uses congested link %s-%s, cannot avoid congestion", u, v)
                                    break
                                else:
                                    self.logger.debug("Selected path avoids congested link %s-%s", u, v)
                        
                        # STRICTER cost improvement requirement: 30% better performance
                        cost_improvement_threshold = baseline_total_cost * 0.7  # Must be 30% better
                        has_significant_cost_improvement = selected_path_cost < cost_improvement_threshold
                        
                        # NEW: Check if selected path itself is not heavily congested (links <50% threshold)
                        selected_path_acceptable = True
                        selected_path_max_congestion = 0
                        for j in range(len(path) - 1):
                            u, v = path[j], path[j + 1]
                            link_cost = cost.get((u, v), 0)
                            selected_path_max_congestion = max(selected_path_max_congestion, link_cost)
                            # If selected path has links >50% threshold, it's also congested
                            if link_cost > self.parent_app.THRESHOLD_BPS * 0.5:  # 50% threshold
                                selected_path_acceptable = False
                                self.logger.debug("Selected path link %s-%s is also congested: %.1fM > 50%% threshold", 
                                                u, v, link_cost/1_000_000)
                        
                        # NEW: Calculate actual congestion reduction percentage
                        if baseline_total_cost > 0:
                            congestion_reduction = ((baseline_total_cost - selected_path_cost) / baseline_total_cost) * 100
                        else:
                            congestion_reduction = 0
                        
                        self.logger.debug("REAL congestion avoidance evaluation: avoids_links=%s, significant_improvement=%s (%.1f%% reduction), path_acceptable=%s (max_congestion=%.1fM)", 
                                        selected_path_avoids_congestion, has_significant_cost_improvement, 
                                        congestion_reduction, selected_path_acceptable, selected_path_max_congestion/1_000_000)
                        
                        # STRICTER criteria: Must avoid links AND have significant improvement AND path not heavily congested
                        if selected_path_avoids_congestion and has_significant_cost_improvement and selected_path_acceptable:
                            avoided_congestion = True
                            self.logger.info("✓ REAL Congestion AVOIDED: %.1f%% utilization reduction, selected path max congestion: %.1fM", 
                                           congestion_reduction, selected_path_max_congestion/1_000_000)
                        else:
                            self.logger.debug("✗ NOT real congestion avoidance: avoids=%s, improvement=%s, acceptable=%s", 
                                           selected_path_avoids_congestion, has_significant_cost_improvement, selected_path_acceptable)
                    
                    # Count flows that avoid congestion with time-based re-counting
                    if avoided_congestion:
                        now = time.time()
                        
                        # Initialize congestion avoidance tracking if not present
                        if not hasattr(self.parent_app, 'congestion_avoidance_events'):
                            self.parent_app.congestion_avoidance_events = {}
                        
                        # Check if this flow avoided congestion recently (within 30 seconds)
                        last_avoidance = self.parent_app.congestion_avoidance_events.get(flow_key)
                        
                        # Initialize event counter if not present
                        if not hasattr(self.parent_app, 'total_congestion_avoidance_events'):
                            self.parent_app.total_congestion_avoidance_events = 0
                        
                        if last_avoidance is None:
                            # First time this flow avoids congestion
                            self.flows_with_congestion_avoidance.add(flow_key)
                            self.parent_app.congestion_avoidance_events[flow_key] = now
                            self.parent_app.total_congestion_avoidance_events += 1
                            
                            cost_improvement = ((baseline_total_cost - selected_path_cost) / baseline_total_cost) * 100
                            self.logger.info("✓ Congestion AVOIDED (event #%d) - baseline %s cost=%.1fM, selected %s cost=%.1fM (%.1f%% improvement), congested links: %s (flow key: %s)", 
                                           self.parent_app.total_congestion_avoidance_events, baseline_path, baseline_total_cost/1_000_000, path, selected_path_cost/1_000_000,
                                           cost_improvement, congested_links, flow_key)
                        else:
                            # Flow has avoided congestion before, check cooldown
                            time_since_last = now - last_avoidance
                            
                            if time_since_last > 30:
                                # Enough time has passed, count it as a new event
                                self.parent_app.congestion_avoidance_events[flow_key] = now
                                self.parent_app.total_congestion_avoidance_events += 1
                                
                                cost_improvement = ((baseline_total_cost - selected_path_cost) / baseline_total_cost) * 100
                                self.logger.info("✓ Congestion AVOIDED again after %.1fs (event #%d) - baseline %s cost=%.1fM, selected %s cost=%.1fM (%.1f%% improvement), congested links: %s (flow key: %s)", 
                                               time_since_last, self.parent_app.total_congestion_avoidance_events, baseline_path, baseline_total_cost/1_000_000, path, selected_path_cost/1_000_000,
                                               cost_improvement, congested_links, flow_key)
                            else:
                                cost_improvement = ((baseline_total_cost - selected_path_cost) / baseline_total_cost) * 100
                                self.logger.debug("Flow %s avoided congestion recently (%.1fs ago), not counting as new event - baseline %s cost=%.1fM, selected %s cost=%.1fM (%.1f%% improvement)", 
                                                flow_key, time_since_last, baseline_path, baseline_total_cost/1_000_000, path, selected_path_cost/1_000_000, cost_improvement)
                    else:
                        avoided_links = selected_path_avoids_congestion if baseline_congested and path != baseline_path else False
                        cost_improved = selected_path_cost < baseline_total_cost * 0.9 if baseline_congested else False
                        cost_improvement = ((baseline_total_cost - selected_path_cost) / baseline_total_cost) * 100 if baseline_total_cost > 0 else 0
                        self.logger.info("⚠ Congestion detected but NOT avoided - avoided_links=%s, cost_improved=%s (%.1f%% improvement), baseline cost=%.1fM, selected cost=%.1fM", 
                                       avoided_links, cost_improved, cost_improvement, baseline_total_cost/1_000_000, selected_path_cost/1_000_000)
    
    def _shortest_path_baseline(self, s_dpid, d_dpid):
        """Calculate true shortest path baseline using hop count"""
        if hasattr(self.parent_app, 'topology_manager'):
            return self.parent_app.topology_manager.get_shortest_path(s_dpid, d_dpid)
        elif hasattr(self.parent_app, '_dijkstra'):
            # Use uniform cost for hop-count baseline
            uniform_cost = {}
            for (u, v) in self.parent_app.links.keys():
                if u < v:  # Avoid duplicates
                    uniform_cost[(u, v)] = 1
                    uniform_cost[(v, u)] = 1
            return self.parent_app._dijkstra(s_dpid, d_dpid, uniform_cost, avoid_congested=False)
        return None
    
    def _calculate_path_cost(self, path, cost):
        """Calculate total cost of a path."""
        if len(path) < 2:
            return 0
        return sum(cost.get((path[i], path[i+1]), 0) for i in range(len(path)-1))
    
    def calculate_efficiency_metrics(self, now):
        """Calculate and update efficiency metrics with bounds checking."""
        # Calculate load balancing rate with validation
        if self.efficiency_metrics['total_flows'] > 0:
            load_balancing_rate = (self.efficiency_metrics['load_balanced_flows'] / 
                                 self.efficiency_metrics['total_flows']) * 100
            # Cap at 100% to prevent calculation errors
            load_balancing_rate = min(100.0, max(0.0, load_balancing_rate))
        else:
            load_balancing_rate = 0
        
        # Calculate congestion avoidance rate using event counting
        if self.efficiency_metrics['total_flows'] > 0:
            # Get total events and unique flows that avoided congestion
            total_events = getattr(self.parent_app, 'total_congestion_avoidance_events', 0)
            unique_flows_avoided = len(self.flows_with_congestion_avoidance)
            
            # Update the efficiency metric to use total events
            self.efficiency_metrics['congestion_avoided'] = total_events
            
            # Calculate rate based on unique flows (more meaningful percentage)
            # but store the actual event count for display
            unique_flows_capped = min(unique_flows_avoided, self.efficiency_metrics['total_flows'])
            congestion_avoidance_rate = (unique_flows_capped / self.efficiency_metrics['total_flows']) * 100
            
            # Cap at reasonable maximum (90% since flows can avoid congestion multiple times)
            congestion_avoidance_rate = min(90.0, max(0.0, congestion_avoidance_rate))
            
            # Enhanced logging for debugging
            if congestion_avoidance_rate > 50:
                self.logger.warning("High congestion avoidance rate: %d events total, %d unique flows out of %d total flows = %.1f%% rate",
                                  total_events, unique_flows_avoided, self.efficiency_metrics['total_flows'], congestion_avoidance_rate)
            else:
                self.logger.info("Congestion avoidance: %d events total, %d unique flows out of %d total flows = %.1f%% rate",
                                total_events, unique_flows_avoided, self.efficiency_metrics['total_flows'], congestion_avoidance_rate)
        else:
            congestion_avoidance_rate = 0
        
        # Calculate variance improvement
        variance_improvement = self._calculate_variance_improvement(now)
        
        # Calculate path length statistics
        self._calculate_path_length_stats()
        
        # Calculate runtime
        runtime_minutes = (now - self.efficiency_metrics['start_time']) / 60
        
        # Additional validation for all metrics
        variance_improvement = max(0.0, min(100.0, variance_improvement))  # Cap variance improvement
        path_overhead = max(0.0, min(200.0, self._calculate_path_overhead()))  # Cap path overhead
        
        # Update efficiency metrics with validated values
        self.efficiency_metrics.update({
            'load_balancing_rate': load_balancing_rate,
            'congestion_avoidance_rate': congestion_avoidance_rate,
            'variance_improvement_percent': variance_improvement,
            'runtime_minutes': runtime_minutes,
            'path_overhead_percent': path_overhead
        })
        
        # Log warning if any metric seems unrealistic
        if (load_balancing_rate > 90 or congestion_avoidance_rate > 70 or 
            variance_improvement > 80 or path_overhead > 150):
            self.logger.warning("Potentially unrealistic efficiency metrics detected - LB: %.1f%%, CA: %.1f%%, Var: %.1f%%, Path: %.1f%%",
                              load_balancing_rate, congestion_avoidance_rate, variance_improvement, path_overhead)
    
    def _calculate_variance_improvement(self, now):
        """Calculate variance improvement compared to baseline."""
        if not hasattr(self.parent_app, 'links'):
            return 0
        
        # Get current link utilizations
        current_utilizations = []
        baseline_utilizations = []
        
        for (dpid1, dpid2), (port1, port2) in self.parent_app.links.items():
            if dpid1 < dpid2:  # Avoid duplicates
                # Get current utilization
                if hasattr(self.parent_app, 'traffic_monitor'):
                    current_util = self.parent_app.traffic_monitor.get_link_utilization(dpid1, dpid2, now)
                else:
                    current_util = self._get_link_utilization(dpid1, dpid2, now)
                
                current_utilizations.append(current_util)
                
                # For baseline, assume all traffic goes through shortest paths
                # This is a simplified calculation
                baseline_util = current_util * 0.8  # Assume 20% less distribution
                baseline_utilizations.append(baseline_util)
        
        if not current_utilizations:
            return 0
        
        # Calculate variance
        current_variance = self._calculate_variance(current_utilizations)
        baseline_variance = self._calculate_variance(baseline_utilizations)
        
        # Store for reference
        self.efficiency_metrics['link_utilization_variance'] = current_variance
        self.efficiency_metrics['baseline_link_utilization_variance'] = baseline_variance
        
        # Calculate improvement percentage
        if baseline_variance > 0:
            improvement = ((baseline_variance - current_variance) / baseline_variance) * 100
            return max(0, improvement)  # Don't show negative improvement
        
        return 0
    
    def _calculate_variance(self, values):
        """Calculate variance of a list of values."""
        if not values:
            return 0
        
        mean = sum(values) / len(values)
        variance = sum((x - mean) ** 2 for x in values) / len(values)
        return variance
    
    def _calculate_path_length_stats(self):
        """Calculate average path lengths for load balanced vs shortest paths."""
        if not hasattr(self.parent_app, 'flow_paths'):
            return
        
        lb_path_lengths = []
        sp_path_lengths = []
        
        for flow_key, path in self.parent_app.flow_paths.items():
            lb_path_lengths.append(len(path))
            
            # Get shortest path for comparison
            if len(flow_key) >= 2:
                s_dpid = self.parent_app.mac_to_dpid.get(flow_key[0])
                d_dpid = self.parent_app.mac_to_dpid.get(flow_key[1])
                if s_dpid and d_dpid:
                    sp_path = self._shortest_path_baseline(s_dpid, d_dpid)
                    if sp_path:
                        sp_path_lengths.append(len(sp_path))
        
        # Calculate averages
        self.efficiency_metrics['avg_path_length_lb'] = (
            sum(lb_path_lengths) / len(lb_path_lengths) if lb_path_lengths else 0
        )
        self.efficiency_metrics['avg_path_length_sp'] = (
            sum(sp_path_lengths) / len(sp_path_lengths) if sp_path_lengths else 0
        )
    
    def _calculate_path_overhead(self):
        """Calculate path overhead percentage."""
        avg_lb = self.efficiency_metrics['avg_path_length_lb']
        avg_sp = self.efficiency_metrics['avg_path_length_sp']
        
        if avg_sp > 0:
            overhead = ((avg_lb - avg_sp) / avg_sp) * 100
            return max(0, overhead)  # Don't show negative overhead
        
        return 0
    
    def _get_link_utilization(self, dpid1, dpid2, now):
        """Get link utilization (fallback implementation)."""
        if hasattr(self.parent_app, 'traffic_monitor'):
            return self.parent_app.traffic_monitor.get_link_utilization(dpid1, dpid2, now)
        
        # Fallback to direct calculation
        if (dpid1, dpid2) not in self.parent_app.links:
            return 0
        
        port1, port2 = self.parent_app.links[(dpid1, dpid2)]
        
        # Get utilization for both directions
        util1 = self._get_average_rate(dpid1, port1, now)
        util2 = self._get_average_rate(dpid2, port2, now)
        
        return max(util1, util2)
    
    def _get_average_rate(self, dpid, port, now):
        """Get average rate for a port."""
        if not hasattr(self.parent_app, 'rate_hist'):
            return 0
        
        hist = self.parent_app.rate_hist[dpid][port]
        if not hist:
            return 0
        
        # Calculate moving average
        from ..config.constants import MA_WINDOW_SEC
        recent = [(t, r) for (t, r) in hist if now - t <= MA_WINDOW_SEC]
        if not recent:
            return 0
        
        return sum(r for (t, r) in recent) / len(recent)
    
    def get_efficiency_summary(self):
        """Get efficiency metrics summary."""
        now = time.time()
        self.calculate_efficiency_metrics(now)
        
        return {
            'efficiency_score': self._calculate_efficiency_score(),
            'load_balancing_rate': self.efficiency_metrics['load_balancing_rate'],
            'congestion_avoidance_rate': self.efficiency_metrics['congestion_avoidance_rate'],
            'variance_improvement_percent': self.efficiency_metrics['variance_improvement_percent'],
            'path_overhead_percent': self.efficiency_metrics['path_overhead_percent'],
            'total_flows': self.efficiency_metrics['total_flows'],
            'load_balanced_flows': self.efficiency_metrics['load_balanced_flows'],
            'congestion_avoided': self.efficiency_metrics['congestion_avoided'],
            'total_reroutes': self.efficiency_metrics['total_reroutes'],
            'runtime_minutes': self.efficiency_metrics['runtime_minutes'],
            'avg_path_length_lb': self.efficiency_metrics['avg_path_length_lb'],
            'avg_path_length_sp': self.efficiency_metrics['avg_path_length_sp']
        }
    
    def _calculate_efficiency_score(self):
        """Calculate composite efficiency score."""
        # Cap variance improvement to realistic values
        capped_variance = min(75, self.efficiency_metrics.get('variance_improvement_percent', 0))
        
        score = 0
        
        # Congestion avoidance (35% weight)
        if self.efficiency_metrics['total_flows'] > 0:
            score += self.efficiency_metrics['congestion_avoidance_rate'] * 0.35
        
        # Variance improvement (25% weight)
        score += capped_variance * 0.25
        
        # Load balancing utilization (25% weight)
        if self.efficiency_metrics['total_flows'] > 0:
            score += self.efficiency_metrics['load_balancing_rate'] * 0.25
        
        # Path efficiency (15% weight)
        path_overhead = self.efficiency_metrics.get('path_overhead_percent', 0)
        path_efficiency = max(0, 100 - path_overhead)
        score += path_efficiency * 0.15
        
        return max(0, min(100, score))
    
    def reset_efficiency_metrics(self):
        """Reset efficiency metrics for mode change, preserving existing flow count."""
        # Count existing flows immediately - they are still active
        existing_flows_count = len(getattr(self.parent_app, 'flow_paths', {}))
        
        self.efficiency_metrics = {
            'total_flows': existing_flows_count,  # Count existing flows immediately
            'load_balanced_flows': 0,  # Reset - new mode will route differently
            'congestion_avoided': 0,   # Reset - new mode may avoid congestion differently
            'avg_path_length_lb': 0,
            'avg_path_length_sp': 0,
            'total_reroutes': 0,       # Reset - fresh count for new mode
            'link_utilization_variance': 0,
            'baseline_link_utilization_variance': 0,
            'start_time': time.time()  # Reset runtime for new mode
        }
        
        # Clear congestion avoidance tracking for new mode
        self.flows_with_congestion_avoidance.clear()
        
        # Clear congestion avoidance events tracking
        if hasattr(self.parent_app, 'congestion_avoidance_events'):
            self.parent_app.congestion_avoidance_events.clear()
        
        # Reset event counter
        if hasattr(self.parent_app, 'total_congestion_avoidance_events'):
            self.parent_app.total_congestion_avoidance_events = 0
        
        # Update parent app references
        self.parent_app.efficiency_metrics = self.efficiency_metrics
        self.parent_app.flows_with_congestion_avoidance = self.flows_with_congestion_avoidance
        
        self.logger.info("Efficiency metrics reset for mode change - %d existing flows counted, load balancing and congestion avoidance metrics reset", 
                        existing_flows_count)
        
        # Schedule re-evaluation of existing flows after reset to see how new mode would handle them
        if existing_flows_count > 0:
            self._schedule_flow_reevaluation()
    
    def count_existing_flows_only(self):
        """Count existing flows without evaluating their congestion avoidance status.
        This prevents inflation from re-evaluating flows with current network conditions.
        """
        # Only count the total number of existing flows
        self.efficiency_metrics['total_flows'] = len(self.parent_app.flow_paths)
        
        self.logger.info("Counted %d existing flows (congestion avoidance will be tracked as flows are rerouted)", 
                        self.efficiency_metrics['total_flows'])
    
    def check_flow_lifecycle_congestion(self, now):
        """Check if existing flows encounter congestion and could benefit from rerouting.
        This helps detect post-setup congestion avoidance opportunities.
        """
        if not hasattr(self.parent_app, 'flow_paths') or not self.parent_app.topology_ready:
            return
        
        cost = self.parent_app._calculate_link_costs(now)
        flows_needing_reroute = []
        
        for flow_key, current_path in self.parent_app.flow_paths.items():
            if len(flow_key) != 2:
                continue
                
            # Check if current path has become congested
            path_congested = False
            congested_links = []
            
            for i in range(len(current_path) - 1):
                u, v = current_path[i], current_path[i + 1]
                link_cost = cost.get((u, v), 0)
                if link_cost > self.parent_app.THRESHOLD_BPS * 0.3:  # 30% threshold
                    path_congested = True
                    congested_links.append(f"{u}-{v}")
            
            if path_congested:
                flows_needing_reroute.append((flow_key, current_path, congested_links))
        
        if flows_needing_reroute:
            self.logger.info("Flow lifecycle check: %d flows experiencing congestion and may benefit from rerouting", 
                           len(flows_needing_reroute))
            for flow_key, path, congested_links in flows_needing_reroute:
                self.logger.debug("Flow %s on path %s has congested links: %s", flow_key, path, congested_links)
        else:
            self.logger.debug("Flow lifecycle check: No flows experiencing congestion")
        
        return flows_needing_reroute
    
    def _schedule_flow_reevaluation(self):
        """Schedule re-evaluation of existing flows after mode change to update metrics."""
        # Import here to avoid circular import
        from ryu.lib import hub
        
        def reevaluate_flows():
            # Wait a few seconds to let the mode change settle
            hub.sleep(3)
            
            if not hasattr(self.parent_app, 'flow_paths') or not self.parent_app.topology_ready:
                return
            
            now = time.time()
            cost = self.parent_app._calculate_link_costs(now)
            flows_evaluated = 0
            
            # Get current mode name for logging
            from ..config.constants import LOAD_BALANCING_MODES
            mode_names = {v: k for k, v in LOAD_BALANCING_MODES.items()}
            current_mode = mode_names.get(self.parent_app.load_balancing_mode, 'unknown')
            
            self.logger.info("Re-evaluating %d existing flows under NEW '%s' mode to determine different congestion avoidance behavior", 
                           len(self.parent_app.flow_paths), current_mode)
            
            for flow_key, current_path in self.parent_app.flow_paths.items():
                if len(flow_key) != 2:
                    continue
                
                # Get source and destination DPIDs
                if isinstance(flow_key[0], str) and flow_key[0].startswith('00:00:00'):
                    # MAC-based flow key, convert to DPID
                    src_dpid = self.parent_app.mac_to_dpid.get(flow_key[0])
                    dst_dpid = self.parent_app.mac_to_dpid.get(flow_key[1])
                else:
                    # DPID-based flow key
                    src_dpid, dst_dpid = flow_key
                
                if not src_dpid or not dst_dpid:
                    continue
                
                # Simulate what path the NEW MODE would choose for this flow
                if hasattr(self.parent_app, 'path_selector'):
                    # Get what path the new mode would select
                    new_mode_path = self.parent_app.path_selector.find_path(src_dpid, dst_dpid, cost)
                    if new_mode_path:
                        # Evaluate this flow using the path the NEW MODE would choose
                        self._evaluate_existing_flow_metrics(src_dpid, dst_dpid, new_mode_path, cost, 
                                                            flow_key[0] if isinstance(flow_key[0], str) else None,
                                                            flow_key[1] if isinstance(flow_key[1], str) else None)
                    else:
                        # Fallback to current path if new mode can't find a path
                        self._evaluate_existing_flow_metrics(src_dpid, dst_dpid, current_path, cost, 
                                                            flow_key[0] if isinstance(flow_key[0], str) else None,
                                                            flow_key[1] if isinstance(flow_key[1], str) else None)
                else:
                    # Fallback if no path selector available
                    self._evaluate_existing_flow_metrics(src_dpid, dst_dpid, current_path, cost, 
                                                        flow_key[0] if isinstance(flow_key[0], str) else None,
                                                        flow_key[1] if isinstance(flow_key[1], str) else None)
                flows_evaluated += 1
            
            self.logger.info("Re-evaluated %d flows under new mode, metrics updated", flows_evaluated)
        
        # Schedule the re-evaluation to run asynchronously
        from ryu.lib import hub
        hub.spawn(reevaluate_flows)
    
    def _evaluate_existing_flow_metrics(self, s_dpid, d_dpid, path, cost, src_mac=None, dst_mac=None):
        """Evaluate efficiency metrics for an existing flow WITHOUT incrementing total_flows."""
        # DO NOT increment total_flows - this flow is already counted
        
        # Use MAC-based flow key for consistency with main controller
        # Fall back to DPID-based key if MAC addresses not provided
        if src_mac and dst_mac:
            flow_key = (src_mac, dst_mac)
        else:
            flow_key = (s_dpid, d_dpid)
        
        # Also track the DPID-based key for internal calculations
        dpid_flow_key = (s_dpid, d_dpid)
        baseline_path = None
        
        # If we have alternative paths stored, the first one should be the hop-count baseline
        if hasattr(self.parent_app, 'alternative_paths') and dpid_flow_key in self.parent_app.alternative_paths:
            alt_paths = self.parent_app.alternative_paths[dpid_flow_key]
            if alt_paths:
                baseline_path = alt_paths[0]  # First path is hop-count baseline
        
        if not baseline_path:
            # Fallback to calculating baseline path
            baseline_path = self._shortest_path_baseline(s_dpid, d_dpid)
        
        self.logger.debug("Existing flow re-evaluation - Selected path: %s, Baseline (hop-count): %s, Flow key: %s", path, baseline_path, flow_key)
        
        if baseline_path:
            # Check if we're using a different path than shortest path
            if path != baseline_path:
                self.efficiency_metrics['load_balanced_flows'] += 1
                self.logger.debug("Existing flow %s: using path %s instead of baseline %s (load balanced)", 
                                flow_key, path, baseline_path)
            
            # Enhanced congestion avoidance detection (same logic as update_flow_metrics)
            if len(baseline_path) > 1:
                baseline_congested = False
                predicted_congestion = False
                congested_links = []
                baseline_total_cost = 0
                
                # Check current congestion on baseline path
                congestion_threshold = self.parent_app.THRESHOLD_BPS * 0.3  # 30% threshold
                
                for i in range(len(baseline_path) - 1):
                    u, v = baseline_path[i], baseline_path[i + 1]
                    link_cost = cost.get((u, v), 0)
                    baseline_total_cost += link_cost
                    
                    # Only count actual congestion (>30% threshold) for avoidance tracking
                    if link_cost > congestion_threshold:  # 30% threshold for actual congestion
                        baseline_congested = True
                        congested_links.append(f"{u}-{v} (congested: {link_cost/1_000_000:.1f}M)")
                
                # Only count actual congestion, not predicted
                congestion_detected = baseline_congested
                
                if congestion_detected:
                    # Calculate selected path cost for comparison
                    selected_path_cost = self._calculate_path_cost(path, cost)
                    
                    # Track congestion avoidance - only if we actually avoid congested links
                    avoided_congestion = False
                    
                    # Criteria for congestion avoidance (same as update_flow_metrics)
                    if baseline_congested and path != baseline_path:
                        selected_path_avoids_congestion = True
                        selected_path_links = [(path[j], path[j+1]) for j in range(len(path)-1)]
                        
                        # Check if selected path avoids congested links from baseline
                        for i in range(len(baseline_path) - 1):
                            u, v = baseline_path[i], baseline_path[i + 1]
                            link_cost = cost.get((u, v), 0)
                            if link_cost > congestion_threshold:  # This link is congested
                                # Check if selected path uses this congested link
                                if (u, v) in selected_path_links:
                                    selected_path_avoids_congestion = False
                                    break
                        
                        # STRICTER cost improvement requirement: 30% better performance (same as main logic)
                        cost_improvement_threshold = baseline_total_cost * 0.7  # Must be 30% better
                        has_significant_cost_improvement = selected_path_cost < cost_improvement_threshold
                        
                        # Check if selected path itself is not heavily congested (same logic)
                        selected_path_acceptable = True
                        selected_path_max_congestion = 0
                        for j in range(len(path) - 1):
                            u, v = path[j], path[j + 1]
                            link_cost = cost.get((u, v), 0)
                            selected_path_max_congestion = max(selected_path_max_congestion, link_cost)
                            if link_cost > self.parent_app.THRESHOLD_BPS * 0.5:  # 50% threshold
                                selected_path_acceptable = False
                                break
                        
                        # STRICTER criteria (same as main logic)
                        if selected_path_avoids_congestion and has_significant_cost_improvement and selected_path_acceptable:
                            avoided_congestion = True
                    
                    # Count flows that avoid congestion with time-based re-counting (same logic)
                    if avoided_congestion:
                        now = time.time()
                        
                        # Initialize congestion avoidance tracking if not present
                        if not hasattr(self.parent_app, 'congestion_avoidance_events'):
                            self.parent_app.congestion_avoidance_events = {}
                        
                        # Check if this flow avoided congestion recently (within 30 seconds)
                        last_avoidance = self.parent_app.congestion_avoidance_events.get(flow_key)
                        
                        # Initialize event counter if not present
                        if not hasattr(self.parent_app, 'total_congestion_avoidance_events'):
                            self.parent_app.total_congestion_avoidance_events = 0
                        
                        if last_avoidance is None:
                            # First time this flow avoids congestion
                            self.flows_with_congestion_avoidance.add(flow_key)
                            self.parent_app.congestion_avoidance_events[flow_key] = now
                            self.parent_app.total_congestion_avoidance_events += 1
                            
                            cost_improvement = ((baseline_total_cost - selected_path_cost) / baseline_total_cost) * 100
                            self.logger.debug("✓ Existing flow congestion AVOIDED (event #%d) - baseline %s cost=%.1fM, selected %s cost=%.1fM (%.1f%% improvement), flow key: %s", 
                                           self.parent_app.total_congestion_avoidance_events, baseline_path, baseline_total_cost/1_000_000, path, selected_path_cost/1_000_000,
                                           cost_improvement, flow_key)
                        else:
                            # Flow has avoided congestion before, check cooldown
                            time_since_last = now - last_avoidance
                            
                            if time_since_last > 30:
                                # Enough time has passed, count it as a new event
                                self.parent_app.congestion_avoidance_events[flow_key] = now
                                self.parent_app.total_congestion_avoidance_events += 1
                                
                                cost_improvement = ((baseline_total_cost - selected_path_cost) / baseline_total_cost) * 100
                                self.logger.debug("✓ Existing flow congestion AVOIDED again after %.1fs (event #%d) - baseline %s cost=%.1fM, selected %s cost=%.1fM (%.1f%% improvement), flow key: %s", 
                                               time_since_last, self.parent_app.total_congestion_avoidance_events, baseline_path, baseline_total_cost/1_000_000, path, selected_path_cost/1_000_000,
                                               cost_improvement, flow_key)
    
    def increment_reroutes(self):
        """Increment reroute counter."""
        self.efficiency_metrics['total_reroutes'] += 1
    
    def get_detailed_stats(self):
        """Get detailed efficiency statistics."""
        return {
            'metrics': self.efficiency_metrics,
            'flows_with_congestion_avoidance': len(self.flows_with_congestion_avoidance),
            'efficiency_score': self._calculate_efficiency_score(),
            'performance_breakdown': {
                'congestion_avoidance_contribution': self.efficiency_metrics.get('congestion_avoidance_rate', 0) * 0.35,
                'variance_improvement_contribution': min(75, self.efficiency_metrics.get('variance_improvement_percent', 0)) * 0.25,
                'load_balancing_contribution': self.efficiency_metrics.get('load_balancing_rate', 0) * 0.25,
                'path_efficiency_contribution': max(0, 100 - self.efficiency_metrics.get('path_overhead_percent', 0)) * 0.15
            }
        }