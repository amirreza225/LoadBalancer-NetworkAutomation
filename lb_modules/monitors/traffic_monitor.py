"""
Traffic Monitor
===============

Monitors network traffic statistics, calculates utilization metrics,
and maintains historical data for congestion prediction.
"""

import collections
import time
from ryu.controller import ofp_event
from ryu.controller.handler import MAIN_DISPATCHER, set_ev_cls
from ryu.lib import hub

from ..config.constants import POLL_PERIOD, MA_WINDOW_SEC, CONGESTION_PREDICTION_WINDOW


class TrafficMonitor:
    """
    Monitors network traffic and maintains utilization statistics
    """
    
    def __init__(self, parent_app):
        self.parent_app = parent_app
        self.logger = parent_app.logger
        
        # Statistics tracking
        self.last_bytes = collections.defaultdict(lambda: collections.defaultdict(int))
        self.rate_hist = collections.defaultdict(lambda: collections.defaultdict(list))
        self.congestion_trends = collections.defaultdict(list)  # (dpid, port) -> [(time, utilization)]
        
        # Timing
        self.last_calc = 0
        self.last_cleanup_time = 0
        
        # Update parent app references
        self.parent_app.last_bytes = self.last_bytes
        self.parent_app.rate_hist = self.rate_hist
        self.parent_app.congestion_trends = self.congestion_trends
        
        # Start monitoring
        self._start_monitoring()
    
    def _start_monitoring(self):
        """Start the traffic monitoring process"""
        hub.spawn(self._poll_stats)
    
    def _poll_stats(self):
        """
        Periodically polls all datapaths for port statistics.
        """
        while True:
            for dp in self.parent_app.dp_set.values():
                dp.send_msg(dp.ofproto_parser.OFPPortStatsRequest(
                    dp, 0, dp.ofproto.OFPP_ANY))
            hub.sleep(POLL_PERIOD)
    
    def handle_stats_reply(self, ev):
        """
        Handles EventOFPPortStatsReply events.
        """
        now = time.time()
        dp = ev.msg.datapath
        dpid = dp.id
        
        for stat in ev.msg.body:
            if stat.port_no > dp.ofproto.OFPP_MAX:
                continue
                
            cur = stat.tx_bytes + stat.rx_bytes
            prev = self.last_bytes[dpid][stat.port_no]
            self.last_bytes[dpid][stat.port_no] = cur
            
            if prev:
                bps = (cur - prev) / POLL_PERIOD
                hist = self.rate_hist[dpid][stat.port_no]
                hist.append((now, bps))
                self.rate_hist[dpid][stat.port_no] = [
                    (t, r) for (t, r) in hist if now - t <= MA_WINDOW_SEC
                ]
                
                # Update congestion trends for prediction
                trend_key = (dpid, stat.port_no)
                self.congestion_trends[trend_key].append((now, bps))
                self.congestion_trends[trend_key] = [
                    (t, util) for (t, util) in self.congestion_trends[trend_key] 
                    if now - t <= CONGESTION_PREDICTION_WINDOW
                ]
        
        # MAXIMUM FREQUENCY FOR D-ITG COMPATIBILITY (every 0.5 seconds matching poll period)
        if now - self.last_calc >= 0.5:  # Check every 0.5 seconds matching POLL_PERIOD
            self.last_calc = now
            self._trigger_rebalance(now)
            self._calculate_efficiency_metrics(now)
            self._cleanup_old_flows()
            self._check_flow_lifecycle(now)
    
    def _trigger_rebalance(self, now):
        """Trigger rebalancing in parent app"""
        if hasattr(self.parent_app, '_rebalance'):
            self.parent_app._rebalance(now)
    
    def _calculate_efficiency_metrics(self, now):
        """Calculate efficiency metrics in parent app"""
        if hasattr(self.parent_app, '_calculate_efficiency_metrics'):
            self.parent_app._calculate_efficiency_metrics(now)
    
    def _cleanup_old_flows(self):
        """Clean up old flows"""
        if hasattr(self.parent_app, '_cleanup_old_flows'):
            self.parent_app._cleanup_old_flows()
    
    def _check_flow_lifecycle(self, now):
        """Check flow lifecycle for congestion detection"""
        if hasattr(self.parent_app, 'efficiency_tracker'):
            self.parent_app.efficiency_tracker.check_flow_lifecycle_congestion(now)
    
    def get_average_rate(self, dpid, port, now):
        """
        Calculate the moving average rate for a given port.
        """
        hist = self.rate_hist[dpid][port]
        if not hist:
            return 0.0
        
        recent = [(t, r) for (t, r) in hist if now - t <= MA_WINDOW_SEC]
        if not recent:
            return 0.0
        
        return sum(r for (t, r) in recent) / len(recent)
    
    def get_current_rate(self, dpid, port):
        """Get the most recent rate for a port"""
        hist = self.rate_hist[dpid][port]
        if not hist:
            return 0.0
        return hist[-1][1]
    
    def get_link_utilization(self, dpid1, dpid2, now):
        """Get utilization for a link between two switches"""
        if (dpid1, dpid2) not in self.parent_app.links:
            return 0.0
        
        port1, port2 = self.parent_app.links[(dpid1, dpid2)]
        
        # Get utilization for both directions
        util1 = self.get_average_rate(dpid1, port1, now)
        util2 = self.get_average_rate(dpid2, port2, now)
        
        # Return maximum utilization
        return max(util1, util2)
    
    def get_congestion_trend(self, dpid, port):
        """Get congestion trend data for a port"""
        trend_key = (dpid, port)
        return self.congestion_trends.get(trend_key, [])
    
    def is_link_congested(self, dpid1, dpid2, threshold_bps, now):
        """Check if a link is congested"""
        utilization = self.get_link_utilization(dpid1, dpid2, now)
        return utilization > threshold_bps
    
    def get_all_link_loads(self, now):
        """Get current load for all links"""
        loads = {}
        
        for (dpid1, dpid2), (port1, port2) in self.parent_app.links.items():
            if dpid1 < dpid2:  # Avoid duplicates
                link_key = f"{dpid1}-{dpid2}"
                loads[link_key] = self.get_link_utilization(dpid1, dpid2, now)
        
        return loads
    
    def get_port_statistics(self, dpid, port):
        """Get detailed statistics for a specific port"""
        return {
            'current_rate': self.get_current_rate(dpid, port),
            'average_rate': self.get_average_rate(dpid, port, time.time()),
            'history': self.rate_hist[dpid][port][-10:],  # Last 10 samples
            'congestion_trend': self.get_congestion_trend(dpid, port)[-10:]  # Last 10 samples
        }
    
    def get_switch_statistics(self, dpid):
        """Get statistics for all ports on a switch"""
        if dpid not in self.parent_app.dp_set:
            return {}
        
        stats = {}
        for port in self.rate_hist[dpid]:
            stats[port] = self.get_port_statistics(dpid, port)
        
        return stats
    
    def get_network_statistics(self):
        """Get statistics for the entire network"""
        now = time.time()
        
        stats = {
            'total_switches': len(self.parent_app.dp_set),
            'total_links': len(self.parent_app.links) // 2,  # Each link stored bidirectionally
            'link_loads': self.get_all_link_loads(now),
            'congested_links': [],
            'total_traffic': 0
        }
        
        # Calculate congested links and total traffic
        for link_key, load in stats['link_loads'].items():
            if load > self.parent_app.THRESHOLD_BPS:
                stats['congested_links'].append({
                    'link': link_key,
                    'utilization': load,
                    'utilization_mbps': load / 1_000_000
                })
            stats['total_traffic'] += load
        
        stats['total_traffic_mbps'] = stats['total_traffic'] / 1_000_000
        
        return stats
    
    def track_congestion_avoidance_reroute(self, old_path, new_path, cost, flow_key):
        """
        Track congestion avoidance during re-routing operations.
        """
        # Check if old path has actual congestion (consistent with efficiency_tracker)
        old_path_congested = False
        congested_links = []
        
        for i in range(len(old_path) - 1):
            u, v = old_path[i], old_path[i + 1]
            link_cost = cost.get((u, v), 0)
            
            # Only track actual congestion (>30% threshold) for reroute avoidance
            if link_cost > self.parent_app.THRESHOLD_BPS * 0.3:  # 30% threshold for actual congestion
                old_path_congested = True
                congested_links.append(f"{u}-{v} (congested: {link_cost/1_000_000:.1f}M)")
        
        # Only count actual congestion for reroute tracking
        congestion_detected = old_path_congested
        
        if congestion_detected:
            # Track this flow as having encountered a congested baseline path
            if hasattr(self.parent_app, 'flows_with_congested_baseline'):
                # Convert flow_key to consistent format for tracking
                consistent_flow_key = flow_key
                if isinstance(flow_key, tuple) and len(flow_key) == 2:
                    src_id, dst_id = flow_key
                    if not (isinstance(src_id, str) and src_id.startswith('00:00:00')):
                        # Convert DPID to MAC addresses for consistency
                        src_mac = dst_mac = None
                        for mac, dpid in self.parent_app.mac_to_dpid.items():
                            if dpid == src_id:
                                src_mac = mac
                            elif dpid == dst_id:
                                dst_mac = mac
                        
                        if src_mac and dst_mac:
                            consistent_flow_key = (src_mac, dst_mac)
                
                self.parent_app.flows_with_congested_baseline.add(consistent_flow_key)
                self.logger.info("TRACKING (reroute): Flow %s has congested baseline path, total with congested baseline: %d", 
                               consistent_flow_key, len(self.parent_app.flows_with_congested_baseline))
            
            # Calculate path costs
            old_cost = self._calculate_path_cost(old_path, cost)
            new_cost = self._calculate_path_cost(new_path, cost)
            
            # Check if we're avoiding the specific congested links
            selected_path_avoids_congestion = True
            for i in range(len(old_path) - 1):
                u, v = old_path[i], old_path[i + 1]
                link_cost = cost.get((u, v), 0)
                if link_cost > self.parent_app.THRESHOLD_BPS * 0.3:  # This link is congested
                    # Check if new path uses this congested link
                    if (u, v) in [(new_path[j], new_path[j+1]) for j in range(len(new_path)-1)]:
                        selected_path_avoids_congestion = False
                        break
            
            # ENHANCED Criteria for REAL congestion avoidance during rerouting:
            # 1. Avoid congested links, AND
            # 2. Significant cost improvement (30%), AND  
            # 3. New path is not heavily congested itself
            cost_improvement_threshold = old_cost * 0.7  # Must be 30% better
            has_significant_improvement = new_cost < cost_improvement_threshold
            
            # Check if new path itself is not heavily congested
            new_path_acceptable = True
            new_path_max_congestion = 0
            for j in range(len(new_path) - 1):
                u, v = new_path[j], new_path[j + 1]
                link_cost = cost.get((u, v), 0)
                new_path_max_congestion = max(new_path_max_congestion, link_cost)
                if link_cost > self.parent_app.THRESHOLD_BPS * 0.5:  # 50% threshold
                    new_path_acceptable = False
                    break
            
            # STRICTER criteria for rerouting congestion avoidance
            avoided_congestion = selected_path_avoids_congestion and has_significant_improvement and new_path_acceptable
            
            if not avoided_congestion:
                # Log why rerouting was not considered real congestion avoidance
                improvement_pct = ((old_cost - new_cost) / old_cost) * 100 if old_cost > 0 else 0
                self.logger.debug("✗ Rerouting NOT real congestion avoidance: avoids_links=%s, significant_improvement=%s (%.1f%% < 30%%), path_acceptable=%s (max=%.1fM)", 
                               selected_path_avoids_congestion, has_significant_improvement, improvement_pct, 
                               new_path_acceptable, new_path_max_congestion/1_000_000)
            
            if avoided_congestion:
                # Track congestion avoidance for rerouting flows
                # Use consistent flow key format for proper tracking
                if hasattr(self.parent_app, 'efficiency_tracker'):
                    # Flow key should already be MAC-based from rerouting context
                    if isinstance(flow_key, tuple) and len(flow_key) == 2:
                        # Check if flow_key contains MAC addresses (starts with '00:00:00')
                        src_id, dst_id = flow_key
                        if isinstance(src_id, str) and src_id.startswith('00:00:00'):
                            # Already MAC-based, use directly
                            reroute_flow_key = flow_key
                        else:
                            # Convert DPID to MAC addresses
                            src_mac = dst_mac = None
                            for mac, dpid in self.parent_app.mac_to_dpid.items():
                                if dpid == src_id:
                                    src_mac = mac
                                elif dpid == dst_id:
                                    dst_mac = mac
                            
                            if src_mac and dst_mac:
                                reroute_flow_key = (src_mac, dst_mac)
                            else:
                                reroute_flow_key = None
                        
                        # Track rerouting congestion avoidance with time-based re-counting
                        if reroute_flow_key:
                            now = time.time()
                            
                            # (Removed 10-second window tracking - keeping it simple)
                            
                            # Initialize congestion avoidance tracking if not present
                            if not hasattr(self.parent_app, 'congestion_avoidance_events'):
                                self.parent_app.congestion_avoidance_events = {}
                            
                            # Check if this flow avoided congestion recently (within 30 seconds)
                            last_avoidance = self.parent_app.congestion_avoidance_events.get(reroute_flow_key)
                            
                            # Initialize event counter if not present
                            if not hasattr(self.parent_app, 'total_congestion_avoidance_events'):
                                self.parent_app.total_congestion_avoidance_events = 0
                            
                            if last_avoidance is None:
                                # First time this flow avoids congestion
                                self.parent_app.flows_with_congestion_avoidance.add(reroute_flow_key)
                                self.parent_app.congestion_avoidance_events[reroute_flow_key] = now
                                self.parent_app.total_congestion_avoidance_events += 1
                                self.parent_app.efficiency_metrics['congestion_avoided'] = self.parent_app.total_congestion_avoidance_events
                                self.logger.info("✓ Added flow %s to congestion avoidance (event #%d, unique flows: %d)", 
                                               reroute_flow_key, self.parent_app.total_congestion_avoidance_events, len(self.parent_app.flows_with_congestion_avoidance))
                            else:
                                # Flow has avoided congestion before, check cooldown
                                time_since_last = now - last_avoidance
                                
                                if time_since_last > 30:
                                    # Enough time has passed, count it as a new event
                                    self.parent_app.congestion_avoidance_events[reroute_flow_key] = now
                                    self.parent_app.total_congestion_avoidance_events += 1
                                    self.parent_app.efficiency_metrics['congestion_avoided'] = self.parent_app.total_congestion_avoidance_events
                                    self.logger.info("✓ Flow %s avoided congestion again after %.1fs cooldown (event #%d, unique flows: %d)", 
                                                   reroute_flow_key, time_since_last, self.parent_app.total_congestion_avoidance_events, len(self.parent_app.flows_with_congestion_avoidance))
                                else:
                                    self.logger.debug("Flow %s already avoided congestion recently (%.1fs ago), not counting as new event", 
                                                    reroute_flow_key, time_since_last)
                
                cost_improvement = ((old_cost - new_cost) / old_cost) * 100 if old_cost > 0 else 0
                self.logger.info("✓ Congestion AVOIDED during reroute - old path %s cost=%.1fM, new path %s cost=%.1fM (%.1f%% improvement), congested links: %s (flow key: %s)", 
                               old_path, old_cost/1_000_000, new_path, new_cost/1_000_000,
                               cost_improvement, congested_links, flow_key)
            else:
                avoided_links = selected_path_avoids_congestion
                cost_improved = new_cost < old_cost * 0.9
                cost_improvement = ((old_cost - new_cost) / old_cost) * 100 if old_cost > 0 else 0
                self.logger.info("⚠ Congestion detected during reroute but NOT avoided - avoided_links=%s, cost_improved=%s (%.1f%% improvement), old cost=%.1fM, new cost=%.1fM", 
                               avoided_links, cost_improved, cost_improvement, old_cost/1_000_000, new_cost/1_000_000)
    
    def _calculate_path_cost(self, path, cost):
        """Calculate total cost of a path."""
        if len(path) < 2:
            return 0
        return sum(cost.get((path[i], path[i+1]), 0) for i in range(len(path)-1))
    
    def reset_statistics(self):
        """Reset all statistics"""
        self.last_bytes.clear()
        self.rate_hist.clear()
        self.congestion_trends.clear()
        self.last_calc = 0
        self.last_cleanup_time = 0
        
        self.logger.info("Traffic monitor statistics reset")
    
    def cleanup_old_data(self, now):
        """Clean up old historical data"""
        # Clean up rate history
        for dpid in self.rate_hist:
            for port in self.rate_hist[dpid]:
                self.rate_hist[dpid][port] = [
                    (t, r) for (t, r) in self.rate_hist[dpid][port]
                    if now - t <= MA_WINDOW_SEC
                ]
        
        # Clean up congestion trends
        for trend_key in self.congestion_trends:
            self.congestion_trends[trend_key] = [
                (t, util) for (t, util) in self.congestion_trends[trend_key]
                if now - t <= CONGESTION_PREDICTION_WINDOW
            ]