#!/usr/bin/env python3
"""
Dynamic Topology Load Balancer with LLDP-based topology discovery,
loop-free ARP flooding, proactive path installation, dynamic rebalancing,
and congestion-aware path selection.
Works with any OpenFlow topology.
"""
import collections
import heapq
import json
import time
import hashlib
import struct
import socket

from ryu.base import app_manager
from ryu.controller import ofp_event
from ryu.controller.handler import (
    CONFIG_DISPATCHER, MAIN_DISPATCHER,
    DEAD_DISPATCHER, set_ev_cls
)
from ryu.app.wsgi import WSGIApplication, ControllerBase, route, Response
from ryu.lib import hub
from ryu.lib.packet import packet, ethernet, ether_types, lldp
from ryu.ofproto import ofproto_v1_3
from ryu.topology import event, switches
from ryu.topology.api import get_switch, get_link

POLL_PERIOD    = 1         # seconds
MA_WINDOW_SEC  = 5         # seconds
DEFAULT_THRESH = 25_000_000 # bytes/sec
CONGESTION_PREDICTION_WINDOW = 10  # seconds for trend analysis

# Enhanced load balancing constants
ELEPHANT_FLOW_THRESHOLD = 10_000_000  # 10 Mbps threshold for elephant flows
MICE_FLOW_THRESHOLD = 1_000_000       # 1 Mbps threshold for mice flows
EWMA_ALPHA = 0.3                      # Exponential weighted moving average factor
LATENCY_WEIGHT = 0.2                  # Weight for latency in path selection
QOS_WEIGHT = 0.25                     # Weight for QoS in path selection
FLOW_TIMEOUT_SEC = 300                # Flow tracking timeout (5 minutes)

LOAD_BALANCING_MODES = {
    'round_robin': 0,
    'least_loaded': 1,
    'weighted_ecmp': 2,
    'adaptive': 3,
    'latency_aware': 4,
    'qos_aware': 5,
    'flow_aware': 6
}

# QoS Classes
QOS_CLASSES = {
    'CRITICAL': {'priority': 3, 'min_bw': 10_000_000, 'max_latency': 10},     # VoIP, real-time
    'HIGH': {'priority': 2, 'min_bw': 5_000_000, 'max_latency': 50},         # Video streaming
    'NORMAL': {'priority': 1, 'min_bw': 1_000_000, 'max_latency': 200},      # Web browsing
    'BEST_EFFORT': {'priority': 0, 'min_bw': 0, 'max_latency': 1000}         # Background traffic
}

class LoadBalancerREST(app_manager.RyuApp):
    OFP_VERSIONS = [ofproto_v1_3.OFP_VERSION]
    _CONTEXTS = {'wsgi': WSGIApplication}

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # datapaths & MAC learning
        self.dp_set       = {}
        self.mac_to_port  = {}
        self.mac_to_dpid  = {}
        self.active_flows = set()
        self.flow_paths   = {}
        # Dynamic topology discovery
        self.links        = {}  # (dpid1, dpid2) -> (port1, port2)
        self.adj          = collections.defaultdict(dict)  # dpid -> {neighbor_dpid: port}
        self.flood_ports  = collections.defaultdict(set)  # dpid -> {ports for flooding}
        self.hosts        = {}  # mac -> host_name (discovered dynamically)
        self.host_locations = {}  # dpid -> set of host MACs on this switch
        self.host_counter = 0     # monotonic counter for stable host numbering
        self.topology_ready = False
        # Enhanced load balancing
        self.load_balancing_mode = LOAD_BALANCING_MODES['adaptive']
        self.flow_priorities = {}  # (src, dst) -> priority level
        self.congestion_trends = collections.defaultdict(list)  # (dpid, port) -> [(time, utilization)]
        self.alternative_paths = {}  # (src, dst) -> [path1, path2, ...]
        self.path_weights = {}  # path_id -> current weight for ECMP
        
        # Enhanced network engineering features
        self.flow_characteristics = {}  # flow_key -> {'type': 'elephant/mice', 'rate': bps, 'start_time': time}
        self.link_latencies = {}  # (dpid1, dpid2) -> latency_ms
        self.flow_qos_classes = {}  # flow_key -> qos_class
        self.switch_resources = {}  # dpid -> {'cpu': %, 'memory': %, 'flow_table': %}
        self.adaptive_thresholds = {}  # dpid -> dynamic_threshold_bps
        self.ecmp_flow_table = {}  # flow_hash -> path_index for ECMP persistence
        self.congestion_ewma = collections.defaultdict(float)  # (dpid, port) -> EWMA value
        self.path_latency_cache = {}  # (src, dst) -> {path_hash: latency_ms}
        # stats
        self.last_bytes   = collections.defaultdict(lambda: collections.defaultdict(int))
        self.rate_hist    = collections.defaultdict(lambda: collections.defaultdict(list))
        self.last_calc    = 0
        self.THRESHOLD_BPS = DEFAULT_THRESH
        # efficiency tracking
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
        # start topology discovery and stats polling
        hub.spawn(self._discover_topology)
        hub.spawn(self._poll_stats)
        # Clean up any existing host data on startup - DISABLED to prevent interference
        # hub.spawn(self._cleanup_stale_hosts)
        # register REST API
        kwargs['wsgi'].register(LBRestController, {'lbapp': self})

    @set_ev_cls(ofp_event.EventOFPStateChange, [MAIN_DISPATCHER, DEAD_DISPATCHER])
    def _dp_state(self, ev):
        """
        Store datapaths in self.dp_set when they enter MAIN_DISPATCHER
        and remove them when they leave (DEAD_DISPATCHER).
        """
        dp = ev.datapath
        if ev.state == MAIN_DISPATCHER:
            self.dp_set[dp.id] = dp
            self.logger.info("Switch %s connected", dp.id)
        elif ev.state == DEAD_DISPATCHER and dp.id in self.dp_set:
            self.logger.info("Switch %s disconnected", dp.id)
            del self.dp_set[dp.id]
            # Clean up topology
            self._cleanup_switch(dp.id)

    @set_ev_cls(ofp_event.EventOFPSwitchFeatures, CONFIG_DISPATCHER)
    def _features(self, ev):
        """
        Handle OFPSwitchFeatures and install a single flow entry
        that sends all packets to the controller.
        """
        dp, parser, ofp = ev.msg.datapath, ev.msg.datapath.ofproto_parser, ev.msg.datapath.ofproto
        self._add_flow(dp, 0, parser.OFPMatch(),
                       [parser.OFPActionOutput(ofp.OFPP_CONTROLLER, ofp.OFPCML_NO_BUFFER)])

    @set_ev_cls(ofp_event.EventOFPPacketIn, MAIN_DISPATCHER)
    def _packet_in(self, ev):
        """
        Handle packet-in events.
        Learn MAC addresses and install proactive paths when both hosts are known.
        """
        msg, dp = ev.msg, ev.msg.datapath
        dpid = dp.id; parser, ofp = dp.ofproto_parser, dp.ofproto
        in_port = msg.match['in_port']; pkt = packet.Packet(msg.data)
        eth = pkt.get_protocol(ethernet.ethernet)
        if eth.ethertype == ether_types.ETH_TYPE_LLDP: return
        
        # Learn MAC addresses
        self.mac_to_port.setdefault(dpid, {})[eth.src] = in_port
        if eth.src not in self.mac_to_dpid:
            self.mac_to_dpid[eth.src] = dpid
            
        # Simplified host discovery with hexring priority
        if self._is_host_mac(eth.src) and self._is_host_port(dpid, in_port):
            # Always proceed with host discovery - don't block on topology readiness
            if eth.src not in self.hosts:
                host_name = self._get_proper_host_name(eth.src, dpid)
                is_hexring_mac = self._is_hexring_mac(eth.src)
                
                # Handle hexring MACs with priority (can override existing assignments)
                if is_hexring_mac and host_name:
                    host_num = int(host_name[1:])  # Extract number from h1, h2, etc.
                    if 1 <= host_num <= 6:
                        self._handle_hexring_host_discovery(eth.src, host_name, dpid, in_port)
                        return
                        
                # Handle non-hexring MACs (avoid conflicts)
                elif host_name:
                    # Check if this host name is already taken
                    existing_mac = self._find_host_by_name(host_name)
                    if existing_mac:
                        # Don't override existing assignments for non-hexring MACs
                        self.logger.debug("Host name %s already taken by MAC %s, ignoring MAC %s", 
                                        host_name, existing_mac, eth.src)
                        return
                    
                    # Assign new host
                    self.hosts[eth.src] = host_name
                    self.host_locations.setdefault(dpid, set()).add(eth.src)
                    self.logger.info("Discovered host %s (MAC: %s) at switch %s port %s", 
                                   host_name, eth.src, dpid, in_port)
                    
                # Generate sequential host names as fallback
                else:
                    self.host_counter += 1
                    host_name = f"h{self.host_counter}"
                    self.hosts[eth.src] = host_name
                    self.host_locations.setdefault(dpid, set()).add(eth.src)
                    self.logger.info("Discovered host %s (MAC: %s) at switch %s port %s", 
                                   host_name, eth.src, dpid, in_port)
        
        if not self.topology_ready:
            self._flood_packet(dp, msg, in_port)
            return
            
        # Host-to-host routing
        if eth.dst in self.mac_to_dpid:
            fid = (eth.src, eth.dst)
            s_dpid, d_dpid = self.mac_to_dpid[eth.src], self.mac_to_dpid[eth.dst]
            
            if fid not in self.flow_paths:
                cost = self._calculate_link_costs(time.time())
                path = self._find_path(s_dpid, d_dpid, cost)
                if path:
                    self._install_path(path, eth.src, eth.dst)
                    self.flow_paths[fid] = path
                    src_name = self.hosts.get(eth.src, eth.src)
                    dst_name = self.hosts.get(eth.dst, eth.dst)
                    self.logger.info("Installed path %s→%s: %s", src_name, dst_name, path)
                    
                    # Update efficiency metrics for new flows
                    self._update_flow_metrics(s_dpid, d_dpid, path, cost)
            
            path = self.flow_paths.get(fid)
            if path:
                nxt = self._next_hop(path, dpid)
                if nxt is None:
                    # Last hop - use the specific port where destination MAC was learned
                    out_port = self.mac_to_port[dpid].get(eth.dst, self._get_host_port(dpid))
                else:
                    out_port = self.adj[dpid][nxt]
            else:
                out_port = self.mac_to_port[dpid].get(eth.dst, ofp.OFPP_FLOOD)
            
            data = msg.data if msg.buffer_id == ofp.OFP_NO_BUFFER else None
            dp.send_msg(parser.OFPPacketOut(datapath=dp, buffer_id=msg.buffer_id,
                                           in_port=in_port,
                                           actions=[parser.OFPActionOutput(out_port)],
                                           data=data))
            return
        
        # Flood if destination unknown
        self._flood_packet(dp, msg, in_port)

    def _find_path(self, src, dst, cost):
        """
        Enhanced path finding with multiple strategies.
        """
        flow_key = (src, dst)
        
        # Get all possible paths
        all_paths = self._find_k_shortest_paths(src, dst, cost, k=3)
        if not all_paths:
            return None
        
        # Store alternative paths for ECMP
        self.alternative_paths[flow_key] = all_paths
        
        # Select best path based on current mode
        if self.load_balancing_mode == LOAD_BALANCING_MODES['adaptive']:
            return self._select_adaptive_path(all_paths, cost)
        elif self.load_balancing_mode == LOAD_BALANCING_MODES['least_loaded']:
            return self._select_least_loaded_path(all_paths, cost)
        elif self.load_balancing_mode == LOAD_BALANCING_MODES['weighted_ecmp']:
            return self._select_weighted_path(all_paths, cost)
        elif self.load_balancing_mode == LOAD_BALANCING_MODES['round_robin']:
            return self._select_round_robin_path(all_paths, flow_key)
        elif self.load_balancing_mode == LOAD_BALANCING_MODES['latency_aware']:
            return self._select_latency_aware_path(all_paths, cost)
        elif self.load_balancing_mode == LOAD_BALANCING_MODES['qos_aware']:
            return self._select_qos_aware_path(all_paths, cost, flow_key)
        elif self.load_balancing_mode == LOAD_BALANCING_MODES['flow_aware']:
            return self._select_flow_aware_path(all_paths, cost, flow_key)
    
    def _find_k_shortest_paths(self, src, dst, cost, k=3):
        """
        Find k shortest paths using Yen's algorithm (simplified version).
        First path uses hop-count baseline, alternative paths use current utilization costs.
        """
        paths = []
        
        # Find first shortest path using hop-count (true baseline for congestion avoidance comparison)
        uniform_cost = {}
        for (u, v) in self.links.keys():
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
    
    def _select_adaptive_path(self, paths, cost):
        """
        Select path based on current network conditions and congestion prediction.
        Enhanced to actively avoid congested baseline paths.
        """
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
            if link_cost > self.THRESHOLD_BPS * 0.5:  # Consider 50% threshold as approaching congestion
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
            if (u, v) in self.links:
                pu, pv = self.links[(u, v)]
                trend_u = self._predict_congestion(u, pu, now)
                trend_v = self._predict_congestion(v, pv, now)
                prediction_penalty = max(trend_u, trend_v) * 0.3  # 30% weight for prediction
                score += prediction_penalty
            
            # Avoid already congested links
            if current_cost > self.THRESHOLD_BPS:
                score += self.THRESHOLD_BPS * 0.5  # Heavy penalty for congested links
        
        return score
    
    def _predict_congestion(self, dpid, port, now):
        """
        Enhanced congestion prediction using EWMA and multiple algorithms.
        Combines linear regression with exponential weighted moving average for better accuracy.
        """
        trends = self.congestion_trends.get((dpid, port), [])
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
        
        # Method 1: Linear regression prediction (existing)
        linear_prediction = 0
        if len(recent_trends) >= 3:
            times = [t for t, _ in recent_trends]
            utils = [util for _, util in recent_trends]
            
            n = len(recent_trends)
            sum_t = sum(times)
            sum_u = sum(utils)
            sum_tu = sum(t * u for t, u in recent_trends)
            sum_t2 = sum(t * t for t in times)
            
            # Calculate slope
            denominator = n * sum_t2 - sum_t * sum_t
            if denominator != 0:
                slope = (n * sum_tu - sum_t * sum_u) / denominator
                # Predict utilization in 5 seconds
                linear_prediction = max(0, utils[-1] + slope * 5)
        
        # Method 2: EWMA-based prediction
        # If current utilization is increasing above EWMA, predict higher congestion
        ewma_prediction = ewma_value
        if current_util > ewma_value * 1.1:  # 10% above EWMA
            growth_factor = current_util / (ewma_value + 1)
            ewma_prediction = current_util * min(growth_factor, 2.0)  # Cap at 2x growth
        
        # Method 3: Rate of change prediction
        rate_prediction = 0
        if len(recent_trends) >= 2:
            recent_util = recent_trends[-1][1]
            prev_util = recent_trends[-2][1]
            time_diff = recent_trends[-1][0] - recent_trends[-2][0]
            
            if time_diff > 0:
                rate_of_change = (recent_util - prev_util) / time_diff
                # Project 5 seconds ahead
                rate_prediction = max(0, recent_util + rate_of_change * 5)
        
        # Combine predictions with weights
        # Linear regression: 40%, EWMA: 35%, Rate of change: 25%
        if linear_prediction > 0 and rate_prediction > 0:
            combined_prediction = (0.4 * linear_prediction + 
                                 0.35 * ewma_prediction + 
                                 0.25 * rate_prediction)
        elif linear_prediction > 0:
            combined_prediction = (0.6 * linear_prediction + 0.4 * ewma_prediction)
        else:
            combined_prediction = ewma_prediction
        
        # Add safety margin for critical links (those already above 70% threshold)
        if current_util > self.THRESHOLD_BPS * 0.7:
            combined_prediction *= 1.2  # 20% safety margin
        
        return max(0, combined_prediction)
    
    def _update_flow_metrics(self, s_dpid, d_dpid, path, cost):
        """Update efficiency metrics for a new flow."""
        self.efficiency_metrics['total_flows'] += 1
        
        # Get the true baseline path (hop-count shortest path) for comparison
        flow_key = (s_dpid, d_dpid)
        baseline_path = None
        
        # If we have alternative paths stored, the first one should be the hop-count baseline
        if flow_key in self.alternative_paths and self.alternative_paths[flow_key]:
            baseline_path = self.alternative_paths[flow_key][0]  # First path is hop-count baseline
        else:
            # Fallback to calculating baseline path
            baseline_path = self._shortest_path_baseline(s_dpid, d_dpid)
        
        self.logger.info("Flow metrics update - Selected path: %s, Baseline (hop-count): %s", path, baseline_path)
        
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
                for i in range(len(baseline_path) - 1):
                    u, v = baseline_path[i], baseline_path[i + 1]
                    link_cost = cost.get((u, v), 0)
                    baseline_total_cost += link_cost
                    
                    if link_cost > self.THRESHOLD_BPS:
                        baseline_congested = True
                        congested_links.append(f"{u}-{v} (current: {link_cost/1_000_000:.1f}M)")
                
                # Check predicted congestion on baseline path (>70% threshold)
                if not baseline_congested:
                    for i in range(len(baseline_path) - 1):
                        u, v = baseline_path[i], baseline_path[i + 1]
                        link_cost = cost.get((u, v), 0)
                        if link_cost > self.THRESHOLD_BPS * 0.5:  # 50% threshold for prediction
                            predicted_congestion = True
                            congested_links.append(f"{u}-{v} (predicted: {link_cost/1_000_000:.1f}M)")
                            break
                
                # Calculate selected path cost for comparison
                selected_path_cost = self._calculate_path_cost(path, cost)
                
                # Enhanced congestion avoidance criteria:
                # 1. Baseline path has congestion (current or predicted), AND
                # 2. We chose a different path that has lower utilization cost
                congestion_detected = baseline_congested or predicted_congestion
                path_different = path != baseline_path
                avoided_congestion = selected_path_cost < baseline_total_cost * 0.95  # Selected path is better
                
                if congestion_detected and path_different and avoided_congestion:
                    self.efficiency_metrics['congestion_avoided'] += 1
                    reason = "current" if baseline_congested else "predicted"
                    self.logger.info("✓ Congestion AVOIDED (%s) - baseline path %s cost=%.1fM, selected path %s cost=%.1fM, congested links: %s (flow %d, total avoided: %d)", 
                                   reason, baseline_path, baseline_total_cost/1_000_000, path, selected_path_cost/1_000_000,
                                   congested_links, self.efficiency_metrics['total_flows'],
                                   self.efficiency_metrics['congestion_avoided'])
                elif congestion_detected and path_different:
                    self.logger.info("⚠ Congestion detected but NOT avoided - baseline cost=%.1fM, selected cost=%.1fM (ratio=%.2f)", 
                                   baseline_total_cost/1_000_000, selected_path_cost/1_000_000, 
                                   selected_path_cost / (baseline_total_cost + 1))
                elif congestion_detected:
                    self.logger.info("⚠ Congestion detected on baseline %s but same path selected: %s", baseline_path, path)
                else:
                    self.logger.debug("No congestion detected on baseline path %s (max link: %.1fM, threshold: %.1fM)", 
                                    baseline_path, max([cost.get((baseline_path[i], baseline_path[i+1]), 0) 
                                                       for i in range(len(baseline_path)-1)], default=0)/1_000_000,
                                    self.THRESHOLD_BPS/1_000_000)
        else:
            self.logger.warning("No baseline path found for %s → %s", s_dpid, d_dpid)
        
        # Log metrics update with validation
        total_flows = self.efficiency_metrics['total_flows']
        load_balanced = self.efficiency_metrics['load_balanced_flows']
        congestion_avoided = self.efficiency_metrics['congestion_avoided']
        
        self.logger.info("Metrics updated: total=%d, load_balanced=%d, congestion_avoided=%d",
                         total_flows, load_balanced, congestion_avoided)
        
        # Validation checks
        if load_balanced > total_flows:
            self.logger.warning("Load balanced flows (%d) exceeds total flows (%d)", 
                              load_balanced, total_flows)
        if congestion_avoided > total_flows:
            self.logger.warning("Congestion avoided (%d) exceeds total flows (%d)", 
                              congestion_avoided, total_flows)
        
        # Calculate and log rates for validation
        if total_flows > 0:
            lb_rate = (load_balanced / total_flows) * 100
            ca_rate = (congestion_avoided / total_flows) * 100
            self.logger.debug("Current rates: Load Balancing=%.1f%%, Congestion Avoidance=%.1f%%", 
                            lb_rate, ca_rate)
    
    def _select_least_loaded_path(self, paths, cost):
        """Select path with lowest total utilization."""
        if not paths:
            return None
        
        return min(paths, key=lambda p: self._calculate_path_cost(p, cost))
    
    def _select_weighted_path(self, paths, cost):
        """
        Enhanced weighted ECMP with flow hashing for TCP flow stickiness.
        Implements proper ECMP with consistent hashing for flow persistence.
        """
        if not paths:
            return None
            
        if len(paths) == 1:
            return paths[0]
        
        # Calculate path weights based on inverse utilization
        path_weights = []
        total_weight = 0
        
        for path in paths:
            path_cost = self._calculate_path_cost(path, cost)
            # Use inverse cost as weight (lower cost = higher weight)
            # Add small constant to avoid division by zero
            weight = 1.0 / (path_cost + 1000)
            path_weights.append(weight)
            total_weight += weight
        
        # Normalize weights
        normalized_weights = [w / total_weight for w in path_weights]
        
        # Create cumulative distribution for weighted random selection
        cumulative_weights = []
        cumsum = 0
        for weight in normalized_weights:
            cumsum += weight
            cumulative_weights.append(cumsum)
        
        # Generate consistent hash for this flow to ensure stickiness
        flow_key = str(sorted(paths[0][:2]))  # Use src-dst as flow identifier
        flow_hash = int(hashlib.md5(flow_key.encode()).hexdigest()[:8], 16)
        
        # Check if we have a cached path for this flow
        if flow_hash in self.ecmp_flow_table:
            cached_index = self.ecmp_flow_table[flow_hash]
            if cached_index < len(paths):
                return paths[cached_index]
        
        # Select path based on weighted distribution with consistent hashing
        random_value = (flow_hash % 10000) / 10000.0  # Normalize hash to [0,1)
        
        selected_index = 0
        for i, cum_weight in enumerate(cumulative_weights):
            if random_value <= cum_weight:
                selected_index = i
                break
        
        # Cache the selection for flow stickiness
        self.ecmp_flow_table[flow_hash] = selected_index
        
        return paths[selected_index]
    
    def _select_round_robin_path(self, paths, flow_key):
        """Select path using round-robin among available paths."""
        if not paths:
            return None
        
        # Simple round-robin based on flow hash
        path_index = hash(str(flow_key)) % len(paths)
        return paths[path_index]

    def _classify_flow(self, flow_key, packet_size=1500):
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
    
    def _select_flow_aware_path(self, paths, cost, flow_key):
        """
        Flow-aware path selection with differentiated handling for elephants vs mice.
        Production networks require this for optimal resource utilization.
        """
        if not paths:
            return None
            
        flow_type = self._classify_flow(flow_key)
        
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
    
    def _calculate_path_capacity(self, path):
        """
        Estimate path capacity based on minimum link capacity.
        """
        if len(path) < 2:
            return 1_000_000_000  # 1 Gbps default
            
        min_capacity = float('inf')
        for i in range(len(path) - 1):
            # Assume 1 Gbps links by default, could be enhanced with LLDP data
            link_capacity = 1_000_000_000  # 1 Gbps in bps
            min_capacity = min(min_capacity, link_capacity)
            
        return min_capacity
    
    def _estimate_path_latency(self, path):
        """
        Estimate path latency. In production, this would use real RTT measurements.
        """
        if len(path) < 2:
            return 1.0  # 1ms for local delivery
            
        # Calculate cached latency if available
        path_hash = hash(tuple(path))
        if path_hash in self.path_latency_cache.get((path[0], path[-1]), {}):
            return self.path_latency_cache[(path[0], path[-1])][path_hash]
        
        # Estimate: base latency + per-hop latency
        base_latency = 0.1  # 0.1ms base
        per_hop_latency = 0.5  # 0.5ms per hop
        estimated_latency = base_latency + (len(path) - 1) * per_hop_latency
        
        # Cache the estimation
        src_dst = (path[0], path[-1])
        if src_dst not in self.path_latency_cache:
            self.path_latency_cache[src_dst] = {}
        self.path_latency_cache[src_dst][path_hash] = estimated_latency
        
        return estimated_latency
    
    def _cleanup_old_flows(self):
        """
        Clean up old flow tracking data to prevent memory leaks.
        """
        now = time.time()
        flows_to_remove = []
        
        for flow_key, flow_info in self.flow_characteristics.items():
            if now - flow_info['last_seen'] > FLOW_TIMEOUT_SEC:
                flows_to_remove.append(flow_key)
        
        for flow_key in flows_to_remove:
            del self.flow_characteristics[flow_key]
            if flow_key in self.flow_qos_classes:
                del self.flow_qos_classes[flow_key]

    def _select_latency_aware_path(self, paths, cost):
        """
        Latency-aware path selection prioritizing RTT and propagation delay.
        Critical for real-time applications and interactive traffic.
        """
        if not paths:
            return None
            
        best_path = None
        best_score = float('inf')
        
        for path in paths:
            # Get estimated latency for this path
            path_latency = self._estimate_path_latency(path)
            path_utilization = self._calculate_path_cost(path, cost)
            
            # Combine latency (primary) with utilization (secondary)
            # High utilization increases queuing delay
            queuing_delay = path_utilization / 100_000_000  # Normalize to ms
            total_latency = path_latency + queuing_delay
            
            if total_latency < best_score:
                best_score = total_latency
                best_path = path
                
        return best_path or paths[0]
    
    def _select_qos_aware_path(self, paths, cost, flow_key):
        """
        QoS-aware path selection with flow classification and priority handling.
        Production networks require this for SLA guarantees.
        """
        if not paths:
            return None
            
        # Classify flow into QoS class
        qos_class = self._classify_qos(flow_key)
        qos_info = QOS_CLASSES.get(qos_class, QOS_CLASSES['BEST_EFFORT'])
        
        best_path = None
        best_score = float('inf')
        
        for path in paths:
            score = 0
            
            # Calculate path metrics
            path_latency = self._estimate_path_latency(path)
            path_capacity = self._calculate_path_capacity(path)
            path_utilization = self._calculate_path_cost(path, cost)
            available_bw = max(0, path_capacity - path_utilization)
            
            # QoS-specific scoring
            if qos_class == 'CRITICAL':
                # Critical flows: latency is paramount
                score = path_latency * 10 + (1 / (available_bw + 1)) * 1000
            elif qos_class == 'HIGH':
                # High priority: balance latency and bandwidth
                score = path_latency * 5 + (1 / (available_bw + 1)) * 500
            elif qos_class == 'NORMAL':
                # Normal flows: prefer available bandwidth
                score = path_latency * 2 + (1 / (available_bw + 1)) * 100
            else:  # BEST_EFFORT
                # Best effort: use least loaded path
                score = path_utilization / 1_000_000
            
            # Check if path meets QoS requirements
            if (path_latency <= qos_info['max_latency'] and 
                available_bw >= qos_info['min_bw']):
                score *= 0.5  # Prefer paths that meet requirements
            
            if score < best_score:
                best_score = score
                best_path = path
                
        return best_path or paths[0]
    
    def _classify_qos(self, flow_key):
        """
        Classify flows into QoS classes based on heuristics and observed behavior.
        In production, this would integrate with DPI or application-aware classification.
        """
        if flow_key in self.flow_qos_classes:
            return self.flow_qos_classes[flow_key]
        
        # Get flow characteristics
        flow_info = self.flow_characteristics.get(flow_key, {})
        flow_rate = flow_info.get('rate', 0)
        packet_count = flow_info.get('packet_count', 0)
        
        # Heuristic classification (in production, use DPI or port-based classification)
        if packet_count > 0:
            avg_packet_size = flow_info.get('byte_count', 0) / packet_count
            
            # Small packets at high rate = likely real-time (VoIP, gaming)
            if avg_packet_size < 500 and flow_rate > 1_000_000:
                qos_class = 'CRITICAL'
            # Large sustained flows = likely video streaming
            elif flow_rate > 5_000_000 and avg_packet_size > 1000:
                qos_class = 'HIGH'
            # Moderate flows = web browsing, file transfers
            elif flow_rate > 1_000_000:
                qos_class = 'NORMAL'
            else:
                qos_class = 'BEST_EFFORT'
        else:
            qos_class = 'BEST_EFFORT'
        
        # Cache the classification
        self.flow_qos_classes[flow_key] = qos_class
        return qos_class

    def _dijkstra(self, src, dst, cost, avoid_congested):
        """
        Runs Dijkstra's algorithm to find the shortest path.
        """
        pq, seen = [(0, src, [src])], set()
        while pq:
            c, u, path = heapq.heappop(pq)
            if u == dst: return path
            if u in seen: continue
            seen.add(u)
            for v in self.adj[u]:
                if v in seen: continue
                edge_cost = cost.get((u, v), 0)
                if avoid_congested and edge_cost > self.THRESHOLD_BPS:
                    continue
                new_cost = c + edge_cost
                heapq.heappush(pq, (new_cost, v, path + [v]))
        return None

    def _add_flow(self, dp, priority, match, actions, **kwargs):
        """
        Adds a flow entry to the datapath.
        """
        inst = [dp.ofproto_parser.OFPInstructionActions(
            dp.ofproto.OFPIT_APPLY_ACTIONS, actions)]
        dp.send_msg(dp.ofproto_parser.OFPFlowMod(
            datapath=dp, priority=priority,
            match=match, instructions=inst, **kwargs))

    def _poll_stats(self):
        """
        Periodically polls all datapaths for port statistics.
        """
        while True:
            for dp in self.dp_set.values():
                dp.send_msg(dp.ofproto_parser.OFPPortStatsRequest(
                    dp, 0, dp.ofproto.OFPP_ANY))
            hub.sleep(POLL_PERIOD)

    @set_ev_cls(ofp_event.EventOFPPortStatsReply, MAIN_DISPATCHER)
    def _stats_reply(self, ev):
        """
        Handles EventOFPPortStatsReply events.
        """
        now = time.time(); dp = ev.msg.datapath; dpid = dp.id
        for stat in ev.msg.body:
            if stat.port_no > dp.ofproto.OFPP_MAX: continue
            cur = stat.tx_bytes + stat.rx_bytes
            prev = self.last_bytes[dpid][stat.port_no]
            self.last_bytes[dpid][stat.port_no] = cur
            if prev:
                bps = (cur - prev) / POLL_PERIOD
                hist = self.rate_hist[dpid][stat.port_no]; hist.append((now, bps))
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
        if now - self.last_calc >= MA_WINDOW_SEC:
            self.last_calc = now
            self._rebalance(now)
            self._calculate_efficiency_metrics(now)
            self._cleanup_old_flows()  # Periodic cleanup to prevent memory leaks

    def _rebalance(self, now):
        """
        Periodically recalculates paths based on current link utilization.
        """
        if not self.topology_ready:
            return
            
        cost = self._calculate_link_costs(now)
        for fid, old_path in list(self.flow_paths.items()):
            src, dst = fid
            # Less aggressive safety check: verify MACs and hosts exist
            if (src not in self.mac_to_dpid or dst not in self.mac_to_dpid or 
                src not in self.hosts or dst not in self.hosts):
                # Log warning but don't immediately delete - MAC might be temporarily missing
                self.logger.debug("Flow %s->%s has missing MAC/host data, skipping rebalance", 
                                src, dst)
                continue
            s_dpid, d_dpid = self.mac_to_dpid[src], self.mac_to_dpid[dst]
            new_path = self._find_path(s_dpid, d_dpid, cost)
            if new_path and new_path != old_path:
                # Only reroute if there's significant benefit or current path is congested
                old_cost = self._calculate_path_cost(old_path, cost)
                new_cost = self._calculate_path_cost(new_path, cost)
                
                # Check if current path is actually congested
                old_path_congested = any(cost.get((old_path[i], old_path[i+1]), 0) > self.THRESHOLD_BPS 
                                       for i in range(len(old_path)-1))
                
                # Only reroute if: 1) old path is congested, OR 2) new path is significantly better (>20% improvement)
                should_reroute = old_path_congested or (old_cost > 0 and (old_cost - new_cost) / old_cost > 0.2)
                
                if should_reroute:
                    src_name = self.hosts.get(src, src)
                    dst_name = self.hosts.get(dst, dst)
                    self.logger.info("Re-routing %s→%s: %s → %s", src_name, dst_name, old_path, new_path)
                    self._install_path(new_path, src, dst)
                    self.flow_paths[fid] = new_path
                    self.efficiency_metrics['total_reroutes'] += 1

    def _avg_rate(self, dpid, port, now):
        """
        Calculates the moving average of the link utilization rate.
        """
        samp = [r for t, r in self.rate_hist[dpid][port] if now - t <= MA_WINDOW_SEC]
        return sum(samp) / len(samp) if samp else 0

    def _install_path(self, path, src, dst):
        """
        Installs a flow entry on all datapaths in the path.
        """
        out_map = {path[i]: self.adj[path[i]][path[i+1]] for i in range(len(path)-1)}
        # Use the specific port where destination MAC was learned
        dst_dpid = path[-1]
        if dst_dpid in self.mac_to_port and dst in self.mac_to_port[dst_dpid]:
            out_map[path[-1]] = self.mac_to_port[dst_dpid][dst]
        else:
            out_map[path[-1]] = self._get_host_port(path[-1])
        for dpid, dp in self.dp_set.items():
            if dpid not in out_map: continue
            p = dp.ofproto_parser
            match = p.OFPMatch(eth_src=src, eth_dst=dst)
            act   = [p.OFPActionOutput(out_map[dpid])]
            inst  = [p.OFPInstructionActions(dp.ofproto.OFPIT_APPLY_ACTIONS, act)]
            dp.send_msg(p.OFPFlowMod(
                datapath=dp, priority=20, match=match,
                instructions=inst, idle_timeout=30, hard_timeout=120
            ))

    def _next_hop(self, path, dpid):
        """
        Determines the next hop in the given path for the specified datapath identifier.
        """
        if dpid not in path: return None
        idx = path.index(dpid)
        return None if idx == len(path)-1 else path[idx+1]
    
    def _discover_topology(self):
        """
        Periodically discover network topology using OpenFlow topology discovery.
        """
        while True:
            try:
                # Get switches and links from Ryu topology
                switch_list = get_switch(self, None)
                link_list = get_link(self, None)
                
                if switch_list and link_list:
                    self._update_topology(switch_list, link_list)
                    if not self.topology_ready:
                        self.topology_ready = True
                        self.logger.info("Topology discovery complete")
                        
            except Exception as e:
                self.logger.error("Topology discovery error: %s", e)
                
            hub.sleep(5)  # Discover topology every 5 seconds
    
    def _update_topology(self, switch_list, link_list):
        """
        Update internal topology representation from discovered switches and links.
        """
        # Build new topology data structures first (atomic update)
        new_adj = collections.defaultdict(dict)
        new_links = {}
        
        # Build adjacency list from discovered links
        for link in link_list:
            src_dpid = link.src.dpid
            dst_dpid = link.dst.dpid
            src_port = link.src.port_no
            dst_port = link.dst.port_no
            
            new_adj[src_dpid][dst_dpid] = src_port
            new_adj[dst_dpid][src_dpid] = dst_port
            new_links[(src_dpid, dst_dpid)] = (src_port, dst_port)
            new_links[(dst_dpid, src_dpid)] = (dst_port, src_port)
        
        # Atomically replace old topology with new one
        self.adj = new_adj
        self.links = new_links
        
        # Rebuild spanning tree for flooding
        self._build_spanning_tree()
        
        self.logger.debug("Updated topology: %d switches, %d links", 
                         len(switch_list), len(link_list))
    
    def _build_spanning_tree(self):
        """
        Build spanning tree for loop-free flooding using BFS.
        """
        if not self.adj:
            return
            
        # Start BFS from lowest numbered switch
        root = min(self.adj.keys())
        visited = {root}
        queue = [root]
        tree_edges = set()
        
        while queue:
            u = queue.pop(0)
            for v in self.adj[u]:
                if v not in visited:
                    visited.add(v)
                    queue.append(v)
                    tree_edges.update({(u, v), (v, u)})
        
        # Build flood ports: include host ports + spanning tree ports
        self.flood_ports.clear()
        for dpid in self.adj:
            ports = set()
            # Add host port (assume lowest numbered port connects to host)
            host_port = self._get_host_port(dpid)
            if host_port:
                ports.add(host_port)
            # Add spanning tree ports
            for (u, v) in tree_edges:
                if u == dpid:
                    ports.add(self.adj[u][v])
            self.flood_ports[dpid] = ports
    
    def _get_host_port(self, dpid):
        """
        Get the port that connects to a host (usually port 1, but discover dynamically).
        """
        # Find ports that are not inter-switch links
        inter_switch_ports = set(self.adj[dpid].values()) if dpid in self.adj else set()
        all_ports = set()
        
        # Get all ports from MAC learning
        if dpid in self.mac_to_port:
            all_ports.update(self.mac_to_port[dpid].values())
        
        # Host ports are those not used for inter-switch links
        host_ports = all_ports - inter_switch_ports
        return min(host_ports) if host_ports else 1  # Default to port 1
    
    def _cleanup_switch(self, dpid):
        """
        Clean up topology when a switch disconnects.
        """
        # Remove from adjacency list
        if dpid in self.adj:
            for neighbor in list(self.adj[dpid].keys()):
                if neighbor in self.adj and dpid in self.adj[neighbor]:
                    del self.adj[neighbor][dpid]
            del self.adj[dpid]
        
        # Remove links
        for link_key in list(self.links.keys()):
            if dpid in link_key:
                del self.links[link_key]
        
        # Clean up MACs learned on this switch
        macs_to_remove = [mac for mac, switch_id in self.mac_to_dpid.items() if switch_id == dpid]
        for mac in macs_to_remove:
            del self.mac_to_dpid[mac]
            if mac in self.hosts:
                del self.hosts[mac]
        
        # Also clean up any additional MACs in host_locations that weren't in mac_to_dpid
        if dpid in self.host_locations:
            additional_macs = self.host_locations[dpid] - set(macs_to_remove)
            for mac in additional_macs:
                if mac in self.hosts:
                    del self.hosts[mac]
                # Also remove from mac_to_dpid if it exists but wasn't caught above
                if mac in self.mac_to_dpid:
                    del self.mac_to_dpid[mac]
            del self.host_locations[dpid]
        
        if dpid in self.mac_to_port:
            del self.mac_to_port[dpid]
        
        # Rebuild spanning tree
        self._build_spanning_tree()
        
        # Clear affected flow paths
        flows_to_remove = [fid for fid, path in self.flow_paths.items() if dpid in path]
        for fid in flows_to_remove:
            del self.flow_paths[fid]
    
    def _is_host_mac(self, mac):
        """
        Determine if a MAC address belongs to a host (heuristic).
        """
        # Filter out broadcast, multicast, and special protocol MACs
        if mac == "ff:ff:ff:ff:ff:ff":  # Broadcast
            return False
        if mac.startswith("01:80:c2"):  # STP/LLDP
            return False
        if mac.startswith("01:00:5e"):  # IPv4 multicast
            return False
        if mac.startswith("33:33"):     # IPv6 multicast
            return False
        if mac.startswith("01:00:0c"):  # Cisco protocols
            return False
        
        # Check if it's a typical host MAC (not all zeros or ones)
        if mac in ["00:00:00:00:00:00", "11:11:11:11:11:11"]:
            return False
            
        return True
    
    def _get_proper_host_name(self, mac, dpid):
        """
        Map MAC addresses to proper host names for known topologies.
        """
        # Hexring topology MAC to host mapping (from hexring_topo.py)
        hexring_mac_to_host = {
            "00:00:00:00:00:01": "h1",
            "00:00:00:00:00:02": "h2", 
            "00:00:00:00:00:03": "h3",
            "00:00:00:00:00:04": "h4",
            "00:00:00:00:00:05": "h5",
            "00:00:00:00:00:06": "h6"
        }
        
        # Check if this MAC matches hexring topology
        if mac in hexring_mac_to_host:
            return hexring_mac_to_host[mac]
        
        # For generic topologies, use switch-based naming if possible
        # Try to determine topology type based on MAC pattern
        if mac.startswith("00:00:00:00:00:") and len(mac.split(":")[5]) <= 2:
            # Looks like a simple sequential MAC, try to map to switch
            try:
                mac_num = int(mac.split(":")[5], 16)
                # For generic topologies, hosts are typically numbered sequentially
                # Support any number of hosts, not just 1-6
                if mac_num > 0:  # Valid host number
                    return f"h{mac_num}"
            except ValueError:
                pass
        
        # No mapping found, caller will use counter-based naming
        return None
    
    def _is_host_port(self, dpid, port):
        """
        Determine if a port is likely connected to a host (not inter-switch).
        """
        # Check if this port is used for inter-switch links
        if dpid in self.adj:
            inter_switch_ports = set(self.adj[dpid].values())
            return port not in inter_switch_ports
        return True  # If topology not ready, assume it could be a host port
    
    def _is_hexring_mac(self, mac):
        """
        Determine if a MAC address is a valid hexring topology MAC.
        """
        hexring_macs = {
            "00:00:00:00:00:01", "00:00:00:00:00:02", "00:00:00:00:00:03",
            "00:00:00:00:00:04", "00:00:00:00:00:05", "00:00:00:00:00:06"
        }
        return mac in hexring_macs
    
    def _find_host_by_name(self, host_name):
        """
        Find the MAC address of a host by its name.
        """
        for mac, name in self.hosts.items():
            if name == host_name:
                return mac
        return None
    
    def _handle_hexring_host_discovery(self, mac, host_name, dpid, in_port):
        """
        Handle discovery of a hexring MAC with priority override capability.
        """
        existing_mac = self._find_host_by_name(host_name)
        
        if existing_mac and existing_mac != mac:
            # Hexring MAC takes priority - remove the existing assignment
            self.logger.info("🔄 Hexring MAC priority: %s takes over host name %s from %s", 
                           mac, host_name, existing_mac)
            
            # Clean up old host assignment
            old_dpid = self.mac_to_dpid.get(existing_mac)
            if old_dpid and old_dpid in self.host_locations:
                self.host_locations[old_dpid].discard(existing_mac)
            
            # Remove from hosts dictionary
            if existing_mac in self.hosts:
                del self.hosts[existing_mac]
            
            # Clean up flows involving the old MAC
            self._cleanup_flows_for_mac(existing_mac)
            
            # Remove old MAC from mac_to_dpid if it's not used elsewhere
            if existing_mac in self.mac_to_dpid:
                del self.mac_to_dpid[existing_mac]
            
            self.logger.info("Cleaned up old assignment for %s (MAC: %s)", host_name, existing_mac)
        
        # Assign or reassign the hexring MAC
        self.hosts[mac] = host_name
        self.host_locations.setdefault(dpid, set()).add(mac)
        
        self.logger.info("✅ Hexring host %s (MAC: %s) discovered at switch %s port %s", 
                       host_name, mac, dpid, in_port)
        
        # Trigger flow revalidation if topology is ready
        if self.topology_ready:
            self._revalidate_flows_after_host_change()
    
    def _cleanup_flows_for_mac(self, mac):
        """
        Clean up flows that involve a specific MAC address.
        """
        flows_to_remove = []
        for (src, dst), path in self.flow_paths.items():
            if src == mac or dst == mac:
                flows_to_remove.append((src, dst))
        
        for flow_id in flows_to_remove:
            del self.flow_paths[flow_id]
            self.logger.debug("Removed flow for old MAC: %s → %s", flow_id[0], flow_id[1])
        
        return len(flows_to_remove)
    
    def _revalidate_flows_after_host_change(self):
        """
        Enhanced flow revalidation with automatic recovery for proper hexring hosts.
        """
        flows_cleaned = len(self.flow_paths)
        
        if flows_cleaned > 0:
            self.logger.info("🔄 Revalidating %d flows after host discovery changes", flows_cleaned)
            
            # Reset efficiency metrics since topology changed
            self.efficiency_metrics['total_flows'] = max(0, self.efficiency_metrics['total_flows'] - flows_cleaned)
            self.efficiency_metrics['load_balanced_flows'] = max(0, self.efficiency_metrics['load_balanced_flows'])
            self.efficiency_metrics['congestion_avoided'] = max(0, self.efficiency_metrics['congestion_avoided'])
        
        # Trigger automatic flow recovery if we have proper hexring topology
        if self.topology_ready:
            self._trigger_flow_recovery()
    
    def _trigger_flow_recovery(self):
        """
        Automatically recreate flows between discovered hexring hosts.
        """
        hexring_macs = {"00:00:00:00:00:01", "00:00:00:00:00:02", "00:00:00:00:00:03", 
                       "00:00:00:00:00:04", "00:00:00:00:00:05", "00:00:00:00:00:06"}
        
        # Get current hexring hosts
        hexring_hosts = {}
        for mac, host_name in self.hosts.items():
            if mac in hexring_macs:
                hexring_hosts[host_name] = mac
        
        if len(hexring_hosts) >= 2:  # Need at least 2 hosts for flow creation
            self.logger.info("🔧 Flow recovery: %d hexring hosts detected, enabling auto-flow creation", 
                           len(hexring_hosts))
            
            # Enable proactive flow creation between hexring hosts
            self._create_hexring_flows(hexring_hosts)
    
    def _create_hexring_flows(self, hexring_hosts):
        """
        Proactively create flows between hexring hosts to enable load balancing.
        """
        host_macs = list(hexring_hosts.values())
        flows_created = 0
        
        if len(host_macs) < 2:
            return flows_created
        
        # Calculate current costs for path selection
        cost = self._calculate_link_costs(time.time())
        
        # Create flows between first few pairs to establish topology understanding
        for i in range(min(3, len(host_macs))):  # Limit to first 3 hosts to avoid flooding
            for j in range(i + 1, min(i + 3, len(host_macs))):  # Create limited connections
                src_mac = host_macs[i]
                dst_mac = host_macs[j]
                
                # Skip if flow already exists
                if (src_mac, dst_mac) in self.flow_paths or (dst_mac, src_mac) in self.flow_paths:
                    continue
                
                # Get switch locations
                src_dpid = self.mac_to_dpid.get(src_mac)
                dst_dpid = self.mac_to_dpid.get(dst_mac)
                
                if src_dpid and dst_dpid and src_dpid != dst_dpid:
                    # Find and install path
                    path = self._find_path(src_dpid, dst_dpid, cost)
                    if path:
                        self._install_path(path, src_mac, dst_mac)
                        self.flow_paths[(src_mac, dst_mac)] = path
                        
                        src_name = self.hosts.get(src_mac, src_mac[-6:])
                        dst_name = self.hosts.get(dst_mac, dst_mac[-6:])
                        self.logger.info("🔧 Flow recovery: created flow %s→%s path %s", 
                                       src_name, dst_name, path)
                        
                        # Update efficiency metrics for recovered flow
                        self._update_flow_metrics(src_dpid, dst_dpid, path, cost)
                        flows_created += 1
        
        if flows_created > 0:
            self.logger.info("✅ Flow recovery complete: created %d flows between hexring hosts", 
                           flows_created)
        
        return flows_created
    
    def _force_hexring_discovery(self):
        """
        Force discovery and validation of hexring topology (for admin use).
        """
        hexring_macs = {"00:00:00:00:00:01", "00:00:00:00:00:02", "00:00:00:00:00:03", 
                       "00:00:00:00:00:04", "00:00:00:00:00:05", "00:00:00:00:00:06"}
        
        # Clear any non-hexring hosts
        non_hexring_macs = set(self.hosts.keys()) - hexring_macs
        
        for mac in list(non_hexring_macs):
            if mac in self.hosts:
                host_name = self.hosts[mac]
                self.logger.info("🧹 Force cleanup: removing non-hexring host %s (MAC: %s)", 
                               host_name, mac)
                
                # Clean up flows and data structures
                self._cleanup_flows_for_mac(mac)
                
                old_dpid = self.mac_to_dpid.get(mac)
                if old_dpid and old_dpid in self.host_locations:
                    self.host_locations[old_dpid].discard(mac)
                
                if mac in self.hosts:
                    del self.hosts[mac]
                if mac in self.mac_to_dpid:
                    del self.mac_to_dpid[mac]
        
        # Reset counter for clean state
        self.host_counter = 0
        
        # Trigger flow recovery if topology is ready
        if self.topology_ready:
            self._trigger_flow_recovery()
        
        return len(non_hexring_macs)
    
    def _cleanup_stale_hosts(self):
        """
        Enhanced cleanup with periodic conflict detection.
        """
        hub.sleep(10)  # Wait for topology to be established
        
        # Perform initial cleanup
        self._resolve_host_conflicts()
        
        # Periodic cleanup every 30 seconds
        while True:
            hub.sleep(30)
            if self.topology_ready:
                conflicts_resolved = self._resolve_host_conflicts()
                if conflicts_resolved > 0:
                    self.logger.info("Periodic cleanup resolved %d host conflicts", conflicts_resolved)
    
    def _resolve_host_conflicts(self):
        """
        Detect and resolve MAC assignment conflicts, prioritizing hexring MACs.
        """
        hexring_macs = {"00:00:00:00:00:01", "00:00:00:00:00:02", "00:00:00:00:00:03", 
                       "00:00:00:00:00:04", "00:00:00:00:00:05", "00:00:00:00:00:06"}
        
        conflicts_resolved = 0
        current_macs = set(self.hosts.keys())
        hexring_macs_present = current_macs.intersection(hexring_macs)
        non_hexring_macs = current_macs - hexring_macs
        
        # If we have hexring MACs but also non-hexring MACs with conflicting names
        if hexring_macs_present and non_hexring_macs:
            for hexring_mac in hexring_macs_present:
                if hexring_mac in self.hosts:
                    expected_host_name = self._get_proper_host_name(hexring_mac, None)
                    if expected_host_name:
                        # Check if any non-hexring MAC has this host name
                        for non_hexring_mac in list(non_hexring_macs):
                            if (non_hexring_mac in self.hosts and 
                                self.hosts[non_hexring_mac] == expected_host_name and
                                non_hexring_mac != hexring_mac):
                                
                                self.logger.info("🧹 Resolving conflict: removing non-hexring MAC %s using host name %s (hexring MAC %s takes priority)",
                                               non_hexring_mac, expected_host_name, hexring_mac)
                                
                                # Clean up the conflicting non-hexring MAC
                                self._cleanup_flows_for_mac(non_hexring_mac)
                                
                                # Remove from tracking structures
                                old_dpid = self.mac_to_dpid.get(non_hexring_mac)
                                if old_dpid and old_dpid in self.host_locations:
                                    self.host_locations[old_dpid].discard(non_hexring_mac)
                                
                                if non_hexring_mac in self.hosts:
                                    del self.hosts[non_hexring_mac]
                                if non_hexring_mac in self.mac_to_dpid:
                                    del self.mac_to_dpid[non_hexring_mac]
                                
                                conflicts_resolved += 1
        
        # Clean up completely if no hexring MACs but we have random MACs
        elif not hexring_macs_present and non_hexring_macs and not self.topology_ready:
            self.logger.info("🧹 Pre-topology cleanup: removing %d non-hexring host entries", len(non_hexring_macs))
            
            for mac in list(non_hexring_macs):
                self._cleanup_flows_for_mac(mac)
                
                old_dpid = self.mac_to_dpid.get(mac)
                if old_dpid and old_dpid in self.host_locations:
                    self.host_locations[old_dpid].discard(mac)
                
                if mac in self.hosts:
                    del self.hosts[mac]
                if mac in self.mac_to_dpid:
                    del self.mac_to_dpid[mac]
            
            self.host_counter = 0
            conflicts_resolved = len(non_hexring_macs)
            
        return conflicts_resolved
    
    def _validate_hexring_topology(self):
        """
        Validate that we have proper hexring topology with correct host assignments.
        """
        hexring_macs = {"00:00:00:00:00:01", "00:00:00:00:00:02", "00:00:00:00:00:03", 
                       "00:00:00:00:00:04", "00:00:00:00:00:05", "00:00:00:00:00:06"}
        
        validation_results = {
            'hexring_hosts_detected': 0,
            'proper_mappings': 0,
            'conflicts': 0,
            'missing_hosts': []
        }
        
        # Check current host assignments
        for mac, host_name in self.hosts.items():
            if mac in hexring_macs:
                validation_results['hexring_hosts_detected'] += 1
                expected_name = self._get_proper_host_name(mac, None)
                if expected_name == host_name:
                    validation_results['proper_mappings'] += 1
                else:
                    validation_results['conflicts'] += 1
        
        # Check for missing hexring hosts
        for mac in hexring_macs:
            if mac not in self.hosts:
                expected_name = self._get_proper_host_name(mac, None)
                if expected_name:
                    validation_results['missing_hosts'].append(expected_name)
        
        return validation_results
    
    def _flood_packet(self, dp, msg, in_port):
        """
        Flood a packet using spanning tree ports.
        """
        parser, ofp = dp.ofproto_parser, dp.ofproto
        ports = self.flood_ports.get(dp.id, {1}) - {in_port}
        actions = [parser.OFPActionOutput(p) for p in ports]
        dp.send_msg(parser.OFPPacketOut(
            datapath=dp, buffer_id=msg.buffer_id,
            in_port=in_port, actions=actions, data=msg.data))
    
    def _calculate_link_costs(self, now):
        """
        Calculate current link costs based on utilization.
        """
        cost = {}
        for (u, v), (pu, pv) in self.links.items():
            if u < v:  # Avoid duplicate calculations
                rate_u = self._avg_rate(u, pu, now)
                rate_v = self._avg_rate(v, pv, now)
                link_cost = max(rate_u, rate_v)
                cost[(u, v)] = link_cost
                cost[(v, u)] = link_cost
        return cost
    
    def _shortest_path_baseline(self, src, dst):
        """
        Calculate shortest path without considering congestion (baseline comparison).
        """
        # Use hop count as cost (traditional shortest path)
        uniform_cost = {}
        for (u, v) in self.links.keys():
            if u < v:  # Avoid duplicates - only add each link once
                uniform_cost[(u, v)] = 1
                uniform_cost[(v, u)] = 1
        
        baseline_path = self._dijkstra(src, dst, uniform_cost, avoid_congested=False)
        self.logger.debug("Baseline path from %s to %s: %s", src, dst, baseline_path)
        return baseline_path
    
    def _calculate_efficiency_metrics(self, now):
        """
        Calculate efficiency metrics comparing load balancing vs baseline routing.
        """
        if not self.topology_ready:
            return
        
        # Calculate current link utilization variance
        link_utils = []
        for (u, v), (pu, pv) in self.links.items():
            if u < v:  # Avoid duplicates
                rate_u = self._avg_rate(u, pu, now)
                rate_v = self._avg_rate(v, pv, now)
                current_util = max(rate_u, rate_v)
                link_utils.append(current_util)
        
        # Calculate variance of current utilization
        if len(link_utils) > 1:
            mean_util = sum(link_utils) / len(link_utils)
            variance = sum((x - mean_util) ** 2 for x in link_utils) / len(link_utils)
            self.efficiency_metrics['link_utilization_variance'] = variance
            
            # For baseline variance, simulate what would happen with shortest path routing
            # Use uniform traffic distribution for true baseline comparison
            if self.flow_paths:
                baseline_utils = collections.defaultdict(float)
                
                # Use current average utilization for realistic baseline comparison
                # This represents actual network load distributed via shortest paths
                total_current_util = sum(link_utils)
                flows_count = len(self.flow_paths)
                if flows_count > 0 and total_current_util > 0:
                    avg_flow_traffic = total_current_util / flows_count
                else:
                    avg_flow_traffic = 1_000_000  # 1 Mbps default for low-traffic scenarios
                
                for (src, dst), current_path in self.flow_paths.items():
                    if src in self.mac_to_dpid and dst in self.mac_to_dpid:
                        s_dpid = self.mac_to_dpid[src]
                        d_dpid = self.mac_to_dpid[dst]
                        baseline_path = self._shortest_path_baseline(s_dpid, d_dpid)
                        
                        if baseline_path and len(baseline_path) > 1:
                            # Add proportional traffic to baseline path links
                            for i in range(len(baseline_path) - 1):
                                u, v = baseline_path[i], baseline_path[i + 1]
                                key = f"{min(u,v)}-{max(u,v)}"
                                baseline_utils[key] += avg_flow_traffic
                
                # Calculate baseline variance
                baseline_util_list = list(baseline_utils.values())
                # Pad with zeros for links not used in baseline routing
                num_links = len([1 for (u, v) in self.links.keys() if u < v])
                while len(baseline_util_list) < num_links:
                    baseline_util_list.append(0)
                
                if len(baseline_util_list) > 1:
                    baseline_mean = sum(baseline_util_list) / len(baseline_util_list)
                    baseline_variance = sum((x - baseline_mean) ** 2 for x in baseline_util_list) / len(baseline_util_list)
                    self.efficiency_metrics['baseline_link_utilization_variance'] = baseline_variance
                    
                    # Enhanced debug logging with validation
                    variance_improvement = 0
                    if baseline_variance > 0:
                        variance_improvement = ((baseline_variance - variance) / baseline_variance) * 100
                    
                    self.logger.debug("Variance calculation: current=%.2f, baseline=%.2f, improvement=%.1f%%", 
                                    variance, baseline_variance, variance_improvement)
                    
                    # Validation checks
                    if variance < 0 or baseline_variance < 0:
                        self.logger.warning("Negative variance detected: current=%.2f, baseline=%.2f", 
                                          variance, baseline_variance)
                    if variance_improvement > 100:
                        self.logger.warning("Unrealistic variance improvement: %.1f%% (may indicate calculation error)", 
                                          variance_improvement)
        
        # Calculate average path lengths
        if self.flow_paths:
            lb_paths = [len(path) for path in self.flow_paths.values() if path]
            sp_paths = []
            
            for (src, dst) in self.flow_paths.keys():
                if src in self.mac_to_dpid and dst in self.mac_to_dpid:
                    s_dpid = self.mac_to_dpid[src]
                    d_dpid = self.mac_to_dpid[dst]
                    sp_path = self._shortest_path_baseline(s_dpid, d_dpid)
                    if sp_path:
                        sp_paths.append(len(sp_path))
            
            if lb_paths:
                self.efficiency_metrics['avg_path_length_lb'] = sum(lb_paths) / len(lb_paths)
            if sp_paths:
                self.efficiency_metrics['avg_path_length_sp'] = sum(sp_paths) / len(sp_paths)
        
        # Calculate runtime
        runtime = now - self.efficiency_metrics['start_time']
        if runtime > 0:
            self.efficiency_metrics['runtime_minutes'] = runtime / 60

class LBRestController(ControllerBase):
    def __init__(self, req, link, data, **cfg):
        super().__init__(req, link, data, **cfg)
        self.lb = data['lbapp']
    
    def _cors(self, body, status=200):
        return Response(
            body=body, status=status,
            content_type='application/json',
            headers={
                "Access-Control-Allow-Origin": "*",
                "Access-Control-Allow-Methods": "GET, POST, OPTIONS",
                "Access-Control-Allow-Headers": "Content-Type"
            }
        )

    @route('path', '/load/path', methods=['GET'])
    def get_paths(self, req, **_):
        paths = {}
        for (src, dst), path in self.lb.flow_paths.items():
            src_name = self.lb.hosts.get(src, src)
            dst_name = self.lb.hosts.get(dst, dst)
            label = f"{src_name}→{dst_name}"
            paths[label] = path
        return self._cors(json.dumps(paths))

    @route('links', '/load/links', methods=['GET'])
    def get_links(self, req, **_):
        now = time.time()
        data = {}
        for (u, v), (pu, pv) in self.lb.links.items():
            if u < v:  # Avoid duplicates
                key = f"{u}-{v}"
                rate_u = self.lb._avg_rate(u, pu, now)
                rate_v = self.lb._avg_rate(v, pv, now)
                data[key] = max(rate_u, rate_v)
        return self._cors(json.dumps(data))

    @route('topology', '/topology', methods=['GET'])
    def get_topology(self, req, **_):
        """Return current network topology for dynamic visualization."""
        nodes = []
        links = []
        
        # Debug logging for host tracking consistency
        total_hosts_in_hosts_dict = len(self.lb.hosts)
        total_hosts_in_locations = sum(len(macs) for macs in self.lb.host_locations.values())
        
        if total_hosts_in_hosts_dict != total_hosts_in_locations:
            self.lb.logger.warning("Host tracking inconsistency: hosts dict=%d, host_locations=%d", 
                                 total_hosts_in_hosts_dict, total_hosts_in_locations)
            # Debug: show the actual data
            self.lb.logger.debug("hosts dict: %s", list(self.lb.hosts.keys()))
            self.lb.logger.debug("host_locations: %s", dict(self.lb.host_locations))
        
        # Add switch nodes with host count information
        for dpid in self.lb.dp_set.keys():
            host_count = len(self.lb.host_locations.get(dpid, set()))
            nodes.append({
                "id": f"s{dpid}", 
                "type": "switch", 
                "host_count": host_count
            })
        
        # Add host nodes (only properly discovered hosts, deduplicated by name)
        valid_hosts = {}
        added_host_names = set()
        
        for mac, host_name in self.lb.hosts.items():
            # Simplified verification - just check if MAC is known and connected
            if mac in self.lb.mac_to_dpid:
                dpid = self.lb.mac_to_dpid[mac]
                # Additional check: ensure the switch exists
                if dpid in self.lb.dp_set:
                    # Only add if we haven't seen this host name before
                    if host_name not in added_host_names:
                        nodes.append({"id": host_name, "type": "host", "mac": mac})
                        added_host_names.add(host_name)
                    valid_hosts[mac] = (host_name, dpid)
                else:
                    self.lb.logger.warning("Host %s (MAC: %s) references non-existent switch %d", 
                                         host_name, mac, dpid)
            else:
                self.lb.logger.warning("Host %s (MAC: %s) not found in mac_to_dpid", 
                                     host_name, mac)
        
        # Add switch-to-switch links
        added_links = set()
        for (u, v) in self.lb.links.keys():
            if u < v and (u, v) not in added_links:
                links.append({"source": f"s{u}", "target": f"s{v}", "type": "switch-switch"})
                added_links.add((u, v))
        
        # Add host-to-switch links (only for verified hosts)
        for mac, (host_name, dpid) in valid_hosts.items():
            links.append({"source": host_name, "target": f"s{dpid}", "type": "host-switch"})
        
        return self._cors(json.dumps({"nodes": nodes, "links": links}))

    @route('ports', '/load/ports/{dpid}/{port}', methods=['GET'])
    def get_port(self, req, dpid, port, **_):
        hist = self.lb.rate_hist.get(int(dpid), {}).get(int(port), [])
        return self._cors(json.dumps(hist))

    @route('threshold', '/config/threshold', methods=['GET', 'POST', 'OPTIONS'])
    def cfg_threshold(self, req, **_):
        if req.method == 'OPTIONS':
            return self._cors('', 200)
        if req.method == 'POST':
            try:
                new = int(req.json.get('threshold', 0))
                if new <= 0:
                    raise ValueError
                self.lb.THRESHOLD_BPS = new
            except Exception:
                return self._cors(json.dumps({"error": "invalid threshold"}), 400)
        return self._cors(json.dumps({"threshold": self.lb.THRESHOLD_BPS}))

    @route('efficiency', '/stats/efficiency', methods=['GET'])
    def get_efficiency(self, req, **_):
        """Return load balancer efficiency metrics."""
        metrics = dict(self.lb.efficiency_metrics)
        
        # Add debug logging
        self.lb.logger.debug("Raw efficiency metrics: %s", metrics)
        
        # Calculate derived metrics
        total_flows = metrics.get('total_flows', 0)
        load_balanced_flows = metrics.get('load_balanced_flows', 0) 
        congestion_avoided = metrics.get('congestion_avoided', 0)
        
        if total_flows > 0:
            metrics['load_balancing_rate'] = min(100, (load_balanced_flows / total_flows) * 100)
            metrics['congestion_avoidance_rate'] = min(100, (congestion_avoided / total_flows) * 100)
        else:
            metrics['load_balancing_rate'] = 0
            metrics['congestion_avoidance_rate'] = 0
        
        # Enhanced validation and logging
        self.lb.logger.debug("Calculated rates: LB=%.1f%%, CA=%.1f%%", 
                           metrics['load_balancing_rate'], metrics['congestion_avoidance_rate'])
        
        # Validation checks for API response
        if metrics['load_balancing_rate'] > 100:
            self.lb.logger.warning("Load balancing rate exceeds 100%%: %.1f%%", metrics['load_balancing_rate'])
            metrics['load_balancing_rate'] = 100
        
        if metrics['congestion_avoidance_rate'] > 100:
            self.lb.logger.warning("Congestion avoidance rate exceeds 100%%: %.1f%%", metrics['congestion_avoidance_rate'])
            metrics['congestion_avoidance_rate'] = 100
        
        # Calculate efficiency improvement with validation
        if metrics.get('baseline_link_utilization_variance', 0) > 0:
            variance_improvement = ((metrics['baseline_link_utilization_variance'] - metrics.get('link_utilization_variance', 0)) / 
                                  metrics['baseline_link_utilization_variance']) * 100
            metrics['variance_improvement_percent'] = max(0, min(200, variance_improvement))  # Cap at 200% improvement
            
            # Log warning for extreme values
            if variance_improvement > 150:
                self.lb.logger.warning("Very high variance improvement: %.1f%% (may indicate calculation issue)", 
                                     variance_improvement)
        else:
            metrics['variance_improvement_percent'] = 0
        
        # Path length efficiency
        if metrics['avg_path_length_sp'] > 0:
            path_overhead = ((metrics['avg_path_length_lb'] - metrics['avg_path_length_sp']) / 
                           metrics['avg_path_length_sp']) * 100
            metrics['path_overhead_percent'] = path_overhead
        else:
            metrics['path_overhead_percent'] = 0
        
        return self._cors(json.dumps(metrics))

    @route('algorithm', '/stats/algorithm', methods=['GET'])
    def get_algorithm_info(self, req, **_):
        """Return current load balancing algorithm information."""
        mode_names = {v: k for k, v in LOAD_BALANCING_MODES.items()}
        current_mode = mode_names.get(self.lb.load_balancing_mode, 'UNKNOWN')
        
        info = {
            'current_mode': current_mode,
            'available_modes': list(LOAD_BALANCING_MODES.keys()),
            'features': {
                'multi_path_support': True,
                'congestion_prediction': True,
                'adaptive_routing': True,
                'ecmp_support': True
            },
            'algorithm_stats': {
                'alternative_paths_stored': len(self.lb.alternative_paths),
                'congestion_trends_tracked': len(self.lb.congestion_trends),
                'topology_ready': self.lb.topology_ready
            }
        }
        
        return self._cors(json.dumps(info))

    @route('mode', '/config/mode', methods=['GET', 'POST', 'OPTIONS'])
    def mode_config(self, req, **_):
        """Get or set load balancing mode."""
        if req.method == 'OPTIONS':
            return self._cors('', 200)
        if req.method == 'POST':
            try:
                mode_name = req.json.get('mode', '')
                if mode_name not in LOAD_BALANCING_MODES:
                    return self._cors(json.dumps({"error": "invalid mode"}), 400)
                self.lb.load_balancing_mode = LOAD_BALANCING_MODES[mode_name]
                self.lb.logger.info("Load balancing mode changed to: %s", mode_name)
            except Exception as e:
                self.lb.logger.error("Error setting mode: %s", e)
                return self._cors(json.dumps({"error": "invalid mode"}), 400)
        
        # Return current mode
        mode_names = {v: k for k, v in LOAD_BALANCING_MODES.items()}
        current_mode = mode_names.get(self.lb.load_balancing_mode, 'adaptive')
        return self._cors(json.dumps({"mode": current_mode}))

    @route('debug', '/debug/metrics', methods=['GET'])
    def debug_metrics(self, req, **_):
        """Debug endpoint to check raw metrics."""
        debug_info = {
            'raw_efficiency_metrics': dict(self.lb.efficiency_metrics),
            'flow_paths_count': len(self.lb.flow_paths),
            'flow_paths': {f"{src}→{dst}": path for (src, dst), path in self.lb.flow_paths.items()},
            'hosts_discovered': dict(self.lb.hosts),
            'topology_ready': self.lb.topology_ready,
            'links_count': len(self.lb.links),
            'current_threshold': self.lb.THRESHOLD_BPS
        }
        return self._cors(json.dumps(debug_info))
    
    @route('debug_congestion', '/debug/congestion-avoidance', methods=['GET'])
    def debug_congestion_avoidance(self, req, **_):
        """Debug endpoint to analyze congestion avoidance decisions."""
        now = time.time()
        cost = self.lb._calculate_link_costs(now)
        
        debug_info = {
            'current_threshold_mbps': self.lb.THRESHOLD_BPS / 1_000_000,
            'load_balancing_mode': {v: k for k, v in LOAD_BALANCING_MODES.items()}.get(self.lb.load_balancing_mode, 'unknown'),
            'congestion_analysis': {},
            'path_analysis': {},
            'link_utilization': {}
        }
        
        # Analyze current link utilization vs threshold
        for (u, v), (pu, pv) in self.lb.links.items():
            if u < v:  # Avoid duplicates
                rate_u = self.lb._avg_rate(u, pu, now)
                rate_v = self.lb._avg_rate(v, pv, now)
                max_util = max(rate_u, rate_v)
                link_key = f"{u}-{v}"
                
                debug_info['link_utilization'][link_key] = {
                    'utilization_mbps': max_util / 1_000_000,
                    'threshold_mbps': self.lb.THRESHOLD_BPS / 1_000_000,
                    'congestion_ratio': max_util / self.lb.THRESHOLD_BPS,
                    'is_congested': max_util > self.lb.THRESHOLD_BPS,
                    'approaching_congestion': max_util > self.lb.THRESHOLD_BPS * 0.7
                }
        
        # Analyze each active flow's path selection
        for (src, dst), current_path in self.lb.flow_paths.items():
            if src in self.lb.mac_to_dpid and dst in self.lb.mac_to_dpid:
                s_dpid = self.lb.mac_to_dpid[src]
                d_dpid = self.lb.mac_to_dpid[dst]
                flow_key = (s_dpid, d_dpid)
                
                # Get baseline path
                baseline_path = None
                if flow_key in self.lb.alternative_paths and self.lb.alternative_paths[flow_key]:
                    baseline_path = self.lb.alternative_paths[flow_key][0]
                else:
                    baseline_path = self.lb._shortest_path_baseline(s_dpid, d_dpid)
                
                if baseline_path:
                    # Calculate costs
                    baseline_cost = self.lb._calculate_path_cost(baseline_path, cost)
                    current_cost = self.lb._calculate_path_cost(current_path, cost)
                    
                    # Check for congestion on baseline
                    baseline_congested_links = []
                    for i in range(len(baseline_path) - 1):
                        u, v = baseline_path[i], baseline_path[i + 1]
                        link_cost = cost.get((u, v), 0)
                        if link_cost > self.lb.THRESHOLD_BPS:
                            baseline_congested_links.append(f"{u}-{v}")
                    
                    src_name = self.lb.hosts.get(src, src[-6:])
                    dst_name = self.lb.hosts.get(dst, dst[-6:])
                    flow_label = f"{src_name}→{dst_name}"
                    
                    debug_info['path_analysis'][flow_label] = {
                        'current_path': current_path,
                        'baseline_path': baseline_path,
                        'path_different': current_path != baseline_path,
                        'baseline_cost_mbps': baseline_cost / 1_000_000,
                        'current_cost_mbps': current_cost / 1_000_000,
                        'cost_improvement': (baseline_cost - current_cost) / 1_000_000 if baseline_cost > 0 else 0,
                        'baseline_congested_links': baseline_congested_links,
                        'would_avoid_congestion': len(baseline_congested_links) > 0 and current_path != baseline_path,
                        'alternative_paths_available': len(self.lb.alternative_paths.get(flow_key, []))
                    }
        
        return self._cors(json.dumps(debug_info))

    @route('cleanup', '/admin/cleanup-hosts', methods=['POST', 'OPTIONS'])
    def cleanup_hosts(self, req, **_):
        """Manual cleanup endpoint to reset host discovery."""
        if req.method == 'OPTIONS':
            return self._cors('', 200)
            
        try:
            # Store counts before cleanup
            old_host_count = len(self.lb.hosts)
            old_location_count = sum(len(macs) for macs in self.lb.host_locations.values())
            
            # Store list of host MACs before clearing
            host_macs = list(self.lb.hosts.keys())
            
            # Clear all host data
            self.lb.hosts.clear()
            self.lb.host_locations.clear()
            self.lb.host_counter = 0
            
            # Clear related data structures - remove the stored host MACs from mac_to_dpid
            for mac in host_macs:
                if mac in self.lb.mac_to_dpid:
                    del self.lb.mac_to_dpid[mac]
                # Also clear from mac_to_port if present
                for dpid in self.lb.mac_to_port:
                    if mac in self.lb.mac_to_port[dpid]:
                        del self.lb.mac_to_port[dpid][mac]
            
            # Clear flow paths that involve the cleaned up hosts
            flows_to_remove = []
            for fid in self.lb.flow_paths:
                src, dst = fid
                if src in host_macs or dst in host_macs:
                    flows_to_remove.append(fid)
            
            for fid in flows_to_remove:
                del self.lb.flow_paths[fid]
            
            self.lb.logger.info("Manual host cleanup: removed %d hosts, %d locations", 
                              old_host_count, old_location_count)
            
            return self._cors(json.dumps({
                "success": True, 
                "message": f"Cleaned up {old_host_count} hosts",
                "old_count": old_host_count
            }))
        except Exception as e:
            self.lb.logger.error("Host cleanup error: %s", e)
            return self._cors(json.dumps({"error": str(e)}), 500)
    
    @route('validate_topology', '/debug/validate-hexring', methods=['GET'])
    def validate_hexring_topology(self, req, **_):
        """Validate hexring topology and host assignments."""
        validation_results = self.lb._validate_hexring_topology()
        
        # Add current host status
        current_hosts = {}
        hexring_macs = {"00:00:00:00:00:01", "00:00:00:00:00:02", "00:00:00:00:00:03", 
                       "00:00:00:00:00:04", "00:00:00:00:00:05", "00:00:00:00:00:06"}
        
        for mac, host_name in self.lb.hosts.items():
            dpid = self.lb.mac_to_dpid.get(mac, 'unknown')
            current_hosts[host_name] = {
                'mac': mac,
                'switch': dpid,
                'is_hexring_mac': mac in hexring_macs,
                'expected_name': self.lb._get_proper_host_name(mac, dpid)
            }
        
        validation_results['current_hosts'] = current_hosts
        validation_results['topology_ready'] = self.lb.topology_ready
        validation_results['total_hosts'] = len(self.lb.hosts)
        
        return self._cors(json.dumps(validation_results))
    
    @route('resolve_conflicts', '/admin/resolve-conflicts', methods=['POST', 'OPTIONS'])
    def resolve_host_conflicts(self, req, **_):
        """Manually trigger host conflict resolution."""
        if req.method == 'OPTIONS':
            return self._cors('', 200)
            
        try:
            conflicts_resolved = self.lb._resolve_host_conflicts()
            
            # Get updated validation results
            validation_results = self.lb._validate_hexring_topology()
            
            result = {
                'conflicts_resolved': conflicts_resolved,
                'validation_results': validation_results,
                'message': f"Resolved {conflicts_resolved} host conflicts"
            }
            
            self.lb.logger.info("Manual conflict resolution: resolved %d conflicts", conflicts_resolved)
            
            return self._cors(json.dumps(result))
        except Exception as e:
            self.lb.logger.error("Conflict resolution error: %s", e)
            return self._cors(json.dumps({"error": str(e)}), 500)
    
    @route('force_hexring', '/admin/force-hexring-discovery', methods=['POST', 'OPTIONS'])
    def force_hexring_discovery(self, req, **_):
        """Force hexring topology discovery and cleanup."""
        if req.method == 'OPTIONS':
            return self._cors('', 200)
            
        try:
            # Force hexring discovery
            non_hexring_removed = self.lb._force_hexring_discovery()
            
            # Get current status
            validation_results = self.lb._validate_hexring_topology()
            
            result = {
                'non_hexring_hosts_removed': non_hexring_removed,
                'validation_results': validation_results,
                'message': f"Forced hexring discovery: removed {non_hexring_removed} non-hexring hosts",
                'topology_ready': self.lb.topology_ready,
                'flow_count': len(self.lb.flow_paths)
            }
            
            self.lb.logger.info("Forced hexring discovery: removed %d non-hexring hosts", 
                              non_hexring_removed)
            
            return self._cors(json.dumps(result))
        except Exception as e:
            self.lb.logger.error("Force hexring discovery error: %s", e)
            return self._cors(json.dumps({"error": str(e)}), 500)