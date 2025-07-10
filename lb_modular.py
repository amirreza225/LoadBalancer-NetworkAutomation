#!/usr/bin/env python3
"""
Modular SDN Load Balancer
==========================

A refactored version of the monolithic SDN load balancer using modular architecture
for better maintainability, testability, and extensibility.
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
from ryu.app.wsgi import WSGIApplication
from ryu.lib import hub
from ryu.lib.packet import packet, ethernet, ether_types, lldp
from ryu.ofproto import ofproto_v1_3
from ryu.topology import event, switches
from ryu.topology.api import get_switch, get_link

# Import modular components
from lb_modules.config.constants import *
from lb_modules.core.base_controller import SDNController
from lb_modules.engines.path_selector import PathSelectionEngine
from lb_modules.engines.congestion_predictor import CongestionPredictor
from lb_modules.engines.flow_classifier import FlowClassifier
from lb_modules.managers.topology_manager import TopologyManager
from lb_modules.managers.host_manager import HostManager
from lb_modules.monitors.traffic_monitor import TrafficMonitor
from lb_modules.monitors.efficiency_tracker import EfficiencyTracker
from lb_modules.api.rest_server import LBRestController


class ModularLoadBalancer(app_manager.RyuApp):
    """
    Modular SDN Load Balancer using component-based architecture
    """
    OFP_VERSIONS = [ofproto_v1_3.OFP_VERSION]
    _CONTEXTS = {'wsgi': WSGIApplication}

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        
        # Core data structures
        self.dp_set = {}
        self.mac_to_port = {}
        self.mac_to_dpid = {}
        self.active_flows = set()
        self.flow_paths = {}
        
        # Initialize configuration
        self.THRESHOLD_BPS = DEFAULT_THRESH
        self.load_balancing_mode = LOAD_BALANCING_MODES['adaptive']
        
        # Initialize modular components
        self._initialize_components()
        
        # Register REST API
        kwargs['wsgi'].register(LBRestController, {'lbapp': self})
        
        self.logger.info("Modular SDN Load Balancer initialized with %d components", 
                        len(self._get_component_list()))

    def _initialize_components(self):
        """Initialize all modular components"""
        # Core controller
        self.sdn_controller = SDNController(self)
        
        # Managers
        self.topology_manager = TopologyManager(self)
        self.host_manager = HostManager(self)
        
        # Engines
        self.path_selector = PathSelectionEngine(self)
        self.congestion_predictor = CongestionPredictor(self)
        self.flow_classifier = FlowClassifier(self)
        
        # Monitors
        self.traffic_monitor = TrafficMonitor(self)
        self.efficiency_tracker = EfficiencyTracker(self)
        
        # Set up component references
        self._setup_component_references()
        
        self.logger.info("All modular components initialized successfully")

    def _setup_component_references(self):
        """Set up references between components"""
        # These properties are accessed by the original code and REST API
        self.links = self.topology_manager.links
        self.adj = self.topology_manager.adj
        self.flood_ports = self.topology_manager.flood_ports
        self.topology_ready = self.topology_manager.topology_ready
        
        # Make sure dp_set is accessible to REST API (it's already initialized in constructor)
        # self.dp_set is already available, no need to reference it
        
        self.hosts = self.host_manager.hosts
        self.host_locations = self.host_manager.host_locations
        self.host_counter = self.host_manager.host_counter
        
        self.alternative_paths = self.path_selector.alternative_paths
        self.congestion_trends = self.traffic_monitor.congestion_trends
        self.congestion_ewma = self.congestion_predictor.congestion_ewma
        
        self.flow_characteristics = self.flow_classifier.flow_characteristics
        self.flow_qos_classes = self.flow_classifier.flow_qos_classes
        self.flow_priorities = self.flow_classifier.flow_priorities
        self.path_latency_cache = self.flow_classifier.path_latency_cache
        
        self.last_bytes = self.traffic_monitor.last_bytes
        self.rate_hist = self.traffic_monitor.rate_hist
        
        self.efficiency_metrics = self.efficiency_tracker.efficiency_metrics
        self.flows_with_congestion_avoidance = self.efficiency_tracker.flows_with_congestion_avoidance

    def _get_component_list(self):
        """Get list of all components for monitoring"""
        return [
            'sdn_controller', 'topology_manager', 'host_manager',
            'path_selector', 'congestion_predictor', 'flow_classifier',
            'traffic_monitor', 'efficiency_tracker'
        ]

    # OpenFlow event handlers (delegated to SDN controller)
    @set_ev_cls(ofp_event.EventOFPStateChange, [MAIN_DISPATCHER, DEAD_DISPATCHER])
    def _dp_state(self, ev):
        """Handle datapath state changes"""
        self.sdn_controller.handle_datapath_change(ev)

    @set_ev_cls(ofp_event.EventOFPSwitchFeatures, CONFIG_DISPATCHER)
    def _features(self, ev):
        """Handle switch features"""
        self.sdn_controller.handle_switch_features(ev)

    @set_ev_cls(ofp_event.EventOFPPacketIn, MAIN_DISPATCHER)
    def _packet_in(self, ev):
        """Handle packet-in events"""
        self.sdn_controller.handle_packet_in(ev)

    @set_ev_cls(ofp_event.EventOFPPortStatsReply, MAIN_DISPATCHER)
    def _stats_reply(self, ev):
        """Handle port statistics replies"""
        self.traffic_monitor.handle_stats_reply(ev)

    # Core routing functions
    def _find_path(self, src, dst, cost):
        """Find path using path selection engine"""
        return self.path_selector.find_path(src, dst, cost)

    def _predict_congestion(self, dpid, port, now):
        """Predict congestion using congestion predictor"""
        return self.congestion_predictor.predict_congestion(dpid, port, now)

    def _install_path(self, path, src_mac, dst_mac):
        """Install flow rules along a path"""
        if len(path) < 2:
            return
        
        # Install flows for the entire path
        for i in range(len(path) - 1):
            dpid = path[i]
            next_dpid = path[i + 1]
            
            if dpid not in self.dp_set:
                continue
            
            dp = self.dp_set[dpid]
            parser = dp.ofproto_parser
            
            # Determine output port
            out_port = self.adj[dpid][next_dpid]
            
            # Create match and action
            match = parser.OFPMatch(eth_src=src_mac, eth_dst=dst_mac)
            actions = [parser.OFPActionOutput(out_port)]
            
            # Install flow
            self._add_flow(dp, 10, match, actions)
        
        # Install final flow at destination switch
        if path:
            last_dpid = path[-1]
            if last_dpid in self.dp_set:
                dp = self.dp_set[last_dpid]
                parser = dp.ofproto_parser
                
                # Output to host port
                out_port = self.mac_to_port.get(last_dpid, {}).get(dst_mac, 1)
                match = parser.OFPMatch(eth_src=src_mac, eth_dst=dst_mac)
                actions = [parser.OFPActionOutput(out_port)]
                
                self._add_flow(dp, 10, match, actions)

    def _add_flow(self, dp, priority, match, actions):
        """Add flow entry to switch"""
        parser = dp.ofproto_parser
        ofp = dp.ofproto
        
        inst = [parser.OFPInstructionActions(ofp.OFPIT_APPLY_ACTIONS, actions)]
        mod = parser.OFPFlowMod(
            datapath=dp,
            priority=priority,
            match=match,
            instructions=inst
        )
        dp.send_msg(mod)

    def _flood_packet(self, dp, msg, in_port):
        """Flood packet to all ports except input port"""
        dpid = dp.id
        parser = dp.ofproto_parser
        
        # Use spanning tree ports for flooding
        flood_ports = self.flood_ports.get(dpid, set())
        
        actions = []
        for port in flood_ports:
            if port != in_port:
                actions.append(parser.OFPActionOutput(port))
        
        if actions:
            data = msg.data if msg.buffer_id == dp.ofproto.OFP_NO_BUFFER else None
            out = parser.OFPPacketOut(
                datapath=dp,
                buffer_id=msg.buffer_id,
                in_port=in_port,
                actions=actions,
                data=data
            )
            dp.send_msg(out)

    def _cleanup_switch(self, dpid):
        """Clean up when switch disconnects"""
        self.topology_manager.cleanup_switch(dpid)

    def _is_host_mac(self, mac):
        """Check if MAC is a host"""
        return self.host_manager.is_host_mac(mac)

    def _is_host_port(self, dpid, port):
        """Check if port is a host port"""
        return self.host_manager.is_host_port(dpid, port)
    
    def _get_current_algorithm_name(self):
        """Get the name of the currently active load balancing algorithm."""
        if not hasattr(self, 'load_balancing_mode'):
            return 'unknown'
        
        # Import here to avoid circular imports
        from lb_modules.config.constants import LOAD_BALANCING_MODES
        
        # Reverse lookup to get algorithm name from mode value
        mode_to_name = {v: k for k, v in LOAD_BALANCING_MODES.items()}
        return mode_to_name.get(self.load_balancing_mode, 'unknown')
    
    def _select_flows_for_rebalancing(self, algorithm_name, cost):
        """Select flows for rebalancing based on algorithm-specific criteria."""
        all_flows = list(self.flow_paths.items())
        
        if algorithm_name == 'adaptive':
            # Adaptive: Prioritize flows on congested paths
            congested_flows = []
            normal_flows = []
            
            for fid, path in all_flows:
                path_congested = any(cost.get((path[i], path[i+1]), 0) > self.THRESHOLD_BPS * 0.3 
                                   for i in range(len(path)-1))
                if path_congested:
                    congested_flows.append((fid, path))
                else:
                    normal_flows.append((fid, path))
            
            # Return congested flows first, then some normal flows for proactive balancing
            return congested_flows + normal_flows[:len(congested_flows)]
        
        elif algorithm_name == 'least_loaded':
            # Least Loaded: Focus on highest utilization flows
            flow_costs = []
            for fid, path in all_flows:
                path_cost = sum(cost.get((path[i], path[i+1]), 0) for i in range(len(path)-1))
                flow_costs.append((path_cost, fid, path))
            
            # Sort by cost (highest first) and return top 60%
            flow_costs.sort(reverse=True)
            return [(fid, path) for _, fid, path in flow_costs[:int(len(flow_costs) * 0.6)]]
        
        elif algorithm_name == 'weighted_ecmp':
            # Weighted ECMP: Spread evaluation across all flows for load distribution
            return all_flows[::2]  # Every other flow
        
        elif algorithm_name == 'round_robin':
            # Round Robin: Rotate through flows systematically
            if not hasattr(self, '_rr_flow_index'):
                self._rr_flow_index = 0
            
            flows_to_check = 3  # Check 3 flows at a time
            start_idx = self._rr_flow_index % len(all_flows)
            selected_flows = []
            
            for i in range(flows_to_check):
                idx = (start_idx + i) % len(all_flows)
                selected_flows.append(all_flows[idx])
            
            self._rr_flow_index += flows_to_check
            return selected_flows
        
        elif algorithm_name == 'latency_aware':
            # Latency Aware: Focus on longer paths (higher latency)
            path_lengths = [(len(path), fid, path) for fid, path in all_flows]
            path_lengths.sort(reverse=True)  # Longest paths first
            return [(fid, path) for _, fid, path in path_lengths[:int(len(path_lengths) * 0.5)]]
        
        elif algorithm_name == 'qos_aware':
            # QoS Aware: Balanced approach with preference for degraded flows
            moderate_threshold = self.THRESHOLD_BPS * 0.5
            degraded_flows = []
            good_flows = []
            
            for fid, path in all_flows:
                max_link_util = max((cost.get((path[i], path[i+1]), 0) for i in range(len(path)-1)), default=0)
                if max_link_util > moderate_threshold:
                    degraded_flows.append((fid, path))
                else:
                    good_flows.append((fid, path))
            
            # Mix degraded and good flows for balanced QoS
            return degraded_flows + good_flows[:len(degraded_flows)]
        
        elif algorithm_name == 'flow_aware':
            # Flow Aware: Alternate focus between different flow types
            return all_flows[:int(len(all_flows) * 0.7)]  # Evaluate 70% of flows
        
        else:
            # Default: Check all flows
            return all_flows
    
    def _should_reroute_for_algorithm(self, algorithm_name, old_path, new_path, cost):
        """Determine if rerouting should occur based on algorithm-specific criteria."""
        old_cost = sum(cost.get((old_path[i], old_path[i+1]), 0) for i in range(len(old_path)-1))
        new_cost = sum(cost.get((new_path[i], new_path[i+1]), 0) for i in range(len(new_path)-1))
        
        if algorithm_name == 'adaptive':
            # Adaptive: Aggressive rerouting on any congestion or 10% improvement
            old_path_congested = any(cost.get((old_path[i], old_path[i+1]), 0) > self.THRESHOLD_BPS * 0.2 
                                   for i in range(len(old_path)-1))
            improvement_threshold = 0.1  # 10% improvement
            return old_path_congested or (old_cost > 0 and (old_cost - new_cost) / old_cost > improvement_threshold)
        
        elif algorithm_name == 'least_loaded':
            # Least Loaded: Focus on significant load reduction (15% improvement)
            improvement_threshold = 0.15
            return old_cost > 0 and (old_cost - new_cost) / old_cost > improvement_threshold
        
        elif algorithm_name == 'weighted_ecmp':
            # Weighted ECMP: Moderate rerouting for load distribution (20% improvement)
            improvement_threshold = 0.2
            return old_cost > 0 and (old_cost - new_cost) / old_cost > improvement_threshold
        
        elif algorithm_name == 'round_robin':
            # Round Robin: Conservative rerouting (25% improvement or clear congestion)
            old_path_congested = any(cost.get((old_path[i], old_path[i+1]), 0) > self.THRESHOLD_BPS * 0.5 
                                   for i in range(len(old_path)-1))
            improvement_threshold = 0.25
            return old_path_congested or (old_cost > 0 and (old_cost - new_cost) / old_cost > improvement_threshold)
        
        elif algorithm_name == 'latency_aware':
            # Latency Aware: Prefer shorter paths and moderate improvement (18% improvement)
            shorter_path = len(new_path) < len(old_path)
            improvement_threshold = 0.18
            significant_improvement = old_cost > 0 and (old_cost - new_cost) / old_cost > improvement_threshold
            return shorter_path or significant_improvement
        
        elif algorithm_name == 'qos_aware':
            # QoS Aware: Balanced rerouting with service level consideration (20% improvement)
            moderate_congestion = any(cost.get((old_path[i], old_path[i+1]), 0) > self.THRESHOLD_BPS * 0.3 
                                    for i in range(len(old_path)-1))
            improvement_threshold = 0.2
            return moderate_congestion or (old_cost > 0 and (old_cost - new_cost) / old_cost > improvement_threshold)
        
        elif algorithm_name == 'flow_aware':
            # Flow Aware: Adaptive based on flow characteristics (15% improvement)
            improvement_threshold = 0.15
            return old_cost > 0 and (old_cost - new_cost) / old_cost > improvement_threshold
        
        else:
            # Default: Standard rerouting (20% improvement)
            old_path_congested = any(cost.get((old_path[i], old_path[i+1]), 0) > self.THRESHOLD_BPS 
                                   for i in range(len(old_path)-1))
            improvement_threshold = 0.2
            return old_path_congested or (old_cost > 0 and (old_cost - new_cost) / old_cost > improvement_threshold)

    def _calculate_link_costs(self, now):
        """Calculate link costs based on current utilization"""
        costs = {}
        
        for (dpid1, dpid2), (port1, port2) in self.links.items():
            # Get utilization for both directions
            util1 = self.traffic_monitor.get_average_rate(dpid1, port1, now)
            util2 = self.traffic_monitor.get_average_rate(dpid2, port2, now)
            
            # Use maximum utilization as cost
            cost = max(util1, util2)
            costs[(dpid1, dpid2)] = cost
            costs[(dpid2, dpid1)] = cost
        
        return costs

    def _dijkstra(self, src, dst, cost, avoid_congested=True):
        """Dijkstra's shortest path algorithm"""
        if src == dst:
            return [src]
        
        if src not in self.adj or dst not in self.adj:
            return None
        
        # Priority queue: (cost, path)
        pq = [(0, [src])]
        visited = set()
        
        while pq:
            current_cost, path = heapq.heappop(pq)
            current = path[-1]
            
            if current in visited:
                continue
                
            visited.add(current)
            
            if current == dst:
                return path
            
            for neighbor in self.adj[current]:
                if neighbor not in visited:
                    edge_cost = cost.get((current, neighbor), 1)
                    
                    # Skip congested links if avoiding them
                    if avoid_congested and edge_cost > self.THRESHOLD_BPS:
                        continue
                    
                    new_cost = current_cost + edge_cost
                    new_path = path + [neighbor]
                    heapq.heappush(pq, (new_cost, new_path))
        
        return None

    def _avg_rate(self, dpid, port, now):
        """Get average rate for a port"""
        return self.traffic_monitor.get_average_rate(dpid, port, now)

    def _rebalance(self, now):
        """Algorithm-aware rebalancing with differentiated timing"""
        if not self.topology_ready:
            return
        
        # Algorithm-specific rebalancing intervals
        algorithm_intervals = {
            'adaptive': 3,        # Most aggressive - every 3 seconds
            'least_loaded': 4,    # Aggressive - every 4 seconds
            'weighted_ecmp': 6,   # Moderate - every 6 seconds  
            'round_robin': 8,     # Conservative - every 8 seconds
            'latency_aware': 5,   # Latency-focused - every 5 seconds
            'qos_aware': 7,       # QoS-focused - every 7 seconds
            'flow_aware': 5       # Flow-focused - every 5 seconds
        }
        
        current_algorithm = self._get_current_algorithm_name()
        rebalance_interval = algorithm_intervals.get(current_algorithm, 5)
        
        # Check if it's time to rebalance for this algorithm
        last_rebalance_key = f'_last_rebalance_{current_algorithm}'
        if hasattr(self, last_rebalance_key):
            if now - getattr(self, last_rebalance_key) < rebalance_interval:
                return
        
        setattr(self, last_rebalance_key, now)
        
        # Periodic cleanup
        if hasattr(self, '_last_cleanup_time'):
            if now - self._last_cleanup_time > 30:
                self._cleanup_old_flows()
                self._last_cleanup_time = now
        else:
            self._last_cleanup_time = now
        
        cost = self._calculate_link_costs(now)
        
        # Algorithm-specific flow selection strategy
        flows_to_evaluate = self._select_flows_for_rebalancing(current_algorithm, cost)
        
        for fid, old_path in flows_to_evaluate:
            src, dst = fid
            
            # Verify flow is still valid
            if (src not in self.mac_to_dpid or dst not in self.mac_to_dpid or 
                src not in self.hosts or dst not in self.hosts):
                continue
            
            s_dpid, d_dpid = self.mac_to_dpid[src], self.mac_to_dpid[dst]
            new_path = self._find_path(s_dpid, d_dpid, cost)
            
            if new_path and new_path != old_path:
                # Algorithm-specific rerouting criteria
                should_reroute = self._should_reroute_for_algorithm(current_algorithm, old_path, new_path, cost)
                
                if should_reroute:
                    # Track congestion avoidance
                    self.traffic_monitor.track_congestion_avoidance_reroute(old_path, new_path, cost, fid)
                    
                    src_name = self.hosts.get(src, src)
                    dst_name = self.hosts.get(dst, dst)
                    self.logger.info("Re-routing %s→%s: %s → %s", src_name, dst_name, old_path, new_path)
                    
                    self._install_path(new_path, src, dst)
                    self.flow_paths[fid] = new_path
                    
                    self.efficiency_tracker.increment_reroutes()

    def _calculate_efficiency_metrics(self, now):
        """Calculate efficiency metrics"""
        self.efficiency_tracker.calculate_efficiency_metrics(now)

    def _cleanup_old_flows(self):
        """Clean up old flow data"""
        now = time.time()
        
        # Clean up flow classifier data
        self.flow_classifier.cleanup_old_flows(now)
        
        # Clean up traffic monitor data
        self.traffic_monitor.cleanup_old_data(now)
        
        # Clean up stale flow paths
        flows_to_remove = []
        for fid in self.flow_paths:
            src, dst = fid
            if (src not in self.mac_to_dpid or dst not in self.mac_to_dpid or 
                src not in self.hosts or dst not in self.hosts):
                flows_to_remove.append(fid)
        
        for fid in flows_to_remove:
            del self.flow_paths[fid]
        
        if flows_to_remove:
            self.logger.debug("Cleaned up %d stale flow paths", len(flows_to_remove))

    def get_component_status(self):
        """Get status of all components"""
        status = {}
        
        for component_name in self._get_component_list():
            component = getattr(self, component_name, None)
            if component:
                status[component_name] = {
                    'initialized': True,
                    'class': component.__class__.__name__
                }
            else:
                status[component_name] = {
                    'initialized': False,
                    'class': None
                }
        
        return status

    def reset_all_modules(self):
        """Reset all modules to initial state"""
        self.efficiency_tracker.reset_efficiency_metrics()
        self.traffic_monitor.reset_statistics()
        self.congestion_predictor.reset_prediction_state()
        self.flow_classifier.reset_flow_classifications()
        
        # Clear flow paths
        self.flow_paths.clear()
        
        self.logger.info("All modules reset to initial state")


# For backward compatibility, create an alias
LoadBalancerREST = ModularLoadBalancer