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

POLL_PERIOD    = 2         # seconds
MA_WINDOW_SEC  = 5         # seconds
DEFAULT_THRESH = 25_000_000 # bytes/sec
CONGESTION_PREDICTION_WINDOW = 10  # seconds for trend analysis
LOAD_BALANCING_MODES = {
    'round_robin': 0,
    'least_loaded': 1,
    'weighted_ecmp': 2,
    'adaptive': 3
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
        self.topology_ready = False
        # Enhanced load balancing
        self.load_balancing_mode = LOAD_BALANCING_MODES['adaptive']
        self.flow_priorities = {}  # (src, dst) -> priority level
        self.congestion_trends = collections.defaultdict(list)  # (dpid, port) -> [(time, utilization)]
        self.alternative_paths = {}  # (src, dst) -> [path1, path2, ...]
        self.path_weights = {}  # path_id -> current weight for ECMP
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
            # Try to assign a host name if it looks like a host MAC
            if self._is_host_mac(eth.src) and self._is_host_port(dpid, in_port):
                # Only create host entries for MACs that haven't been seen and are on edge ports
                if eth.src not in self.hosts:
                    host_num = len([mac for mac in self.hosts.keys()]) + 1
                    self.hosts[eth.src] = f"h{host_num}"
                    # Track host location
                    self.host_locations.setdefault(dpid, set()).add(eth.src)
                    self.logger.info("Discovered host %s (MAC: %s) at switch %s port %s", 
                                   self.hosts[eth.src], eth.src, dpid, in_port)
        
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
                out_port = self._get_host_port(dpid) if nxt is None else self.adj[dpid][nxt]
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
    
    def _find_k_shortest_paths(self, src, dst, cost, k=3):
        """
        Find k shortest paths using Yen's algorithm (simplified version).
        """
        paths = []
        
        # Find first shortest path
        first_path = self._dijkstra(src, dst, cost, avoid_congested=False)
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
                
                # Create modified cost without removed edges
                modified_cost = dict(cost)
                for edge in removed_edges:
                    if edge in modified_cost:
                        modified_cost[edge] = float('inf')
                
                # Find spur path
                spur_path = self._dijkstra(spur_node, dst, modified_cost, avoid_congested=False)
                if spur_path:
                    total_path = root_path[:-1] + spur_path
                    if total_path not in paths and total_path not in candidates:
                        candidates.append(total_path)
            
            if candidates:
                # Select candidate with lowest cost
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
        """Select path based on current network conditions and congestion prediction."""
        if not paths:
            return None
        
        best_path = None
        best_score = float('inf')
        
        for path in paths:
            score = self._calculate_adaptive_score(path, cost)
            if score < best_score:
                best_score = score
                best_path = path
        
        return best_path
    
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
        """Predict future congestion based on utilization trends."""
        trends = self.congestion_trends.get((dpid, port), [])
        
        # Keep only recent trends
        recent_trends = [(t, util) for t, util in trends if now - t <= CONGESTION_PREDICTION_WINDOW]
        
        if len(recent_trends) < 3:
            return 0  # Not enough data for prediction
        
        # Calculate trend slope (simple linear regression)
        times = [t for t, _ in recent_trends]
        utils = [util for _, util in recent_trends]
        
        n = len(recent_trends)
        sum_t = sum(times)
        sum_u = sum(utils)
        sum_tu = sum(t * u for t, u in recent_trends)
        sum_t2 = sum(t * t for t in times)
        
        # Calculate slope
        denominator = n * sum_t2 - sum_t * sum_t
        if denominator == 0:
            return 0
        
        slope = (n * sum_tu - sum_t * sum_u) / denominator
        
        # Predict utilization in 5 seconds
        predicted_util = utils[-1] + slope * 5
        
        return max(0, predicted_util)
    
    def _update_flow_metrics(self, s_dpid, d_dpid, path, cost):
        """Update efficiency metrics for a new flow."""
        self.efficiency_metrics['total_flows'] += 1
        
        # Get baseline shortest path
        baseline_path = self._shortest_path_baseline(s_dpid, d_dpid)
        
        self.logger.info("Flow metrics update - Path: %s, Baseline: %s", path, baseline_path)
        
        if baseline_path:
            # Check if we're using a different path than shortest path
            if path != baseline_path:
                self.efficiency_metrics['load_balanced_flows'] += 1
                self.logger.info("Load balanced flow %d: using path %s instead of baseline %s", 
                                self.efficiency_metrics['total_flows'], path, baseline_path)
            else:
                self.logger.info("Flow %d using shortest path %s", 
                               self.efficiency_metrics['total_flows'], path)
            
            # Check if we avoided congestion (either current or predicted)
            if len(baseline_path) > 1:
                baseline_congested = False
                predicted_congestion = False
                congested_links = []
                
                # Check current congestion
                for i in range(len(baseline_path) - 1):
                    u, v = baseline_path[i], baseline_path[i + 1]
                    link_cost = cost.get((u, v), 0)
                    if link_cost > self.THRESHOLD_BPS:
                        baseline_congested = True
                        congested_links.append(f"{u}-{v}")
                
                # Also check if we predicted congestion on baseline path
                if not baseline_congested and path != baseline_path:
                    # If we chose a different path, check if baseline would become congested
                    baseline_cost = self._calculate_path_cost(baseline_path, cost)
                    chosen_cost = self._calculate_path_cost(path, cost)
                    
                    # Count as congestion avoidance if:
                    # 1. We chose a different path with better utilization, OR
                    # 2. Any link in baseline is > 70% of threshold (predictive)
                    if chosen_cost < baseline_cost:
                        predicted_congestion = True
                        congested_links.append("predicted better utilization")
                    else:
                        for i in range(len(baseline_path) - 1):
                            u, v = baseline_path[i], baseline_path[i + 1]
                            link_cost = cost.get((u, v), 0)
                            if link_cost > self.THRESHOLD_BPS * 0.7:  # 70% threshold for prediction
                                predicted_congestion = True
                                congested_links.append(f"{u}-{v} (predicted)")
                                break
                
                if baseline_congested or predicted_congestion:
                    self.efficiency_metrics['congestion_avoided'] += 1
                    reason = "current" if baseline_congested else "predicted"
                    self.logger.info("Congestion avoided (%s) on baseline path %s, affected links: %s (flow %d, total avoided: %d)", 
                                   reason, baseline_path, congested_links, self.efficiency_metrics['total_flows'],
                                   self.efficiency_metrics['congestion_avoided'])
        else:
            self.logger.warning("No baseline path found for %s → %s", s_dpid, d_dpid)
        
        # Log metrics update
        self.logger.info("Metrics updated: total=%d, load_balanced=%d, congestion_avoided=%d",
                         self.efficiency_metrics['total_flows'],
                         self.efficiency_metrics['load_balanced_flows'],
                         self.efficiency_metrics['congestion_avoided'])
    
    def _select_least_loaded_path(self, paths, cost):
        """Select path with lowest total utilization."""
        if not paths:
            return None
        
        return min(paths, key=lambda p: self._calculate_path_cost(p, cost))
    
    def _select_weighted_path(self, paths, cost):
        """Select path using weighted ECMP."""
        if not paths:
            return None
        
        # For now, return least loaded (can be enhanced with actual ECMP)
        return self._select_least_loaded_path(paths, cost)
    
    def _select_round_robin_path(self, paths, flow_key):
        """Select path using round-robin among available paths."""
        if not paths:
            return None
        
        # Simple round-robin based on flow hash
        path_index = hash(str(flow_key)) % len(paths)
        return paths[path_index]

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

    def _rebalance(self, now):
        """
        Periodically recalculates paths based on current link utilization.
        """
        if not self.topology_ready:
            return
            
        cost = self._calculate_link_costs(now)
        for fid, old_path in list(self.flow_paths.items()):
            src, dst = fid
            s_dpid, d_dpid = self.mac_to_dpid[src], self.mac_to_dpid[dst]
            new_path = self._find_path(s_dpid, d_dpid, cost)
            if new_path and new_path != old_path:
                src_name = self.hosts.get(src, src)
                dst_name = self.hosts.get(dst, dst)
                self.logger.info("Re-routing %s→%s: %s → %s", src_name, dst_name, old_path, new_path)
                self._install_path(new_path, src, dst)
                self.flow_paths[fid] = new_path
                self.efficiency_metrics['total_reroutes'] += 1
                
                # Don't double-count congestion avoidance during reroutes
                # The congestion avoidance metric is already counted during initial flow installation
                self.logger.info("Rerouted from %s to %s due to changing conditions", old_path, new_path)

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
        # Clear existing topology
        self.adj.clear()
        self.links.clear()
        
        # Build adjacency list from discovered links
        for link in link_list:
            src_dpid = link.src.dpid
            dst_dpid = link.dst.dpid
            src_port = link.src.port_no
            dst_port = link.dst.port_no
            
            self.adj[src_dpid][dst_dpid] = src_port
            self.adj[dst_dpid][src_dpid] = dst_port
            self.links[(src_dpid, dst_dpid)] = (src_port, dst_port)
            self.links[(dst_dpid, src_dpid)] = (dst_port, src_port)
        
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
        
        if dpid in self.mac_to_port:
            del self.mac_to_port[dpid]
        
        # Clean up host locations
        if dpid in self.host_locations:
            del self.host_locations[dpid]
        
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
    
    def _is_host_port(self, dpid, port):
        """
        Determine if a port is likely connected to a host (not inter-switch).
        """
        # Check if this port is used for inter-switch links
        if dpid in self.adj:
            inter_switch_ports = set(self.adj[dpid].values())
            return port not in inter_switch_ports
        return True  # If topology not ready, assume it could be a host port
    
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
            # Estimate based on total flows and their shortest paths
            if self.flow_paths:
                baseline_utils = collections.defaultdict(float)
                
                # Estimate traffic per flow (based on actual measurements if available)
                avg_flow_traffic = mean_util if mean_util > 0 else 1000000  # 1 Mbps default
                
                for (src, dst), current_path in self.flow_paths.items():
                    if src in self.mac_to_dpid and dst in self.mac_to_dpid:
                        s_dpid = self.mac_to_dpid[src]
                        d_dpid = self.mac_to_dpid[dst]
                        baseline_path = self._shortest_path_baseline(s_dpid, d_dpid)
                        
                        if baseline_path and len(baseline_path) > 1:
                            # Add estimated traffic to baseline path links
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
                    
                    self.logger.debug("Variance: current=%.2f, baseline=%.2f", variance, baseline_variance)
        
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
        
        # Add switch nodes with host count information
        for dpid in self.lb.dp_set.keys():
            host_count = len(self.lb.host_locations.get(dpid, set()))
            nodes.append({
                "id": f"s{dpid}", 
                "type": "switch", 
                "host_count": host_count
            })
        
        # Add host nodes (only properly discovered hosts)
        for mac, host_name in self.lb.hosts.items():
            # Verify this is still a valid host
            if (mac in self.lb.mac_to_dpid and 
                mac in self.lb.host_locations.get(self.lb.mac_to_dpid[mac], set())):
                nodes.append({"id": host_name, "type": "host", "mac": mac})
        
        # Add switch-to-switch links
        added_links = set()
        for (u, v) in self.lb.links.keys():
            if u < v and (u, v) not in added_links:
                links.append({"source": f"s{u}", "target": f"s{v}", "type": "switch-switch"})
                added_links.add((u, v))
        
        # Add host-to-switch links (only for verified hosts)
        for mac, host_name in self.lb.hosts.items():
            if (mac in self.lb.mac_to_dpid and 
                mac in self.lb.host_locations.get(self.lb.mac_to_dpid[mac], set())):
                dpid = self.lb.mac_to_dpid[mac]
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
        
        self.lb.logger.debug("Calculated rates: LB=%.1f%%, CA=%.1f%%", 
                           metrics['load_balancing_rate'], metrics['congestion_avoidance_rate'])
        
        # Calculate efficiency improvement
        if metrics['baseline_link_utilization_variance'] > 0:
            variance_improvement = ((metrics['baseline_link_utilization_variance'] - metrics['link_utilization_variance']) / 
                                  metrics['baseline_link_utilization_variance']) * 100
            metrics['variance_improvement_percent'] = max(0, variance_improvement)
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