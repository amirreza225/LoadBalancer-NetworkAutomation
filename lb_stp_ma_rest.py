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
DEFAULT_THRESH = 1_000_000 # bytes/sec

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
        self.topology_ready = False
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
            if self._is_host_mac(eth.src):
                host_num = len([mac for mac in self.mac_to_dpid.keys() if self._is_host_mac(mac)])
                self.hosts[eth.src] = f"h{host_num}"
                self.logger.info("Discovered host %s at switch %s port %s", 
                               self.hosts[eth.src], dpid, in_port)
        
        if not self.topology_ready:
            self._flood_packet(dp, msg, in_port)
            return
            
        # Host-to-host routing
        if eth.dst in self.mac_to_dpid:
            fid = (eth.src, eth.dst)
            if fid not in self.flow_paths:
                s_dpid, d_dpid = self.mac_to_dpid[eth.src], self.mac_to_dpid[eth.dst]
                cost = self._calculate_link_costs(time.time())
                path = self._find_path(s_dpid, d_dpid, cost)
                if path:
                    self._install_path(path, eth.src, eth.dst)
                    self.flow_paths[fid] = path
                    src_name = self.hosts.get(eth.src, eth.src)
                    dst_name = self.hosts.get(eth.dst, eth.dst)
                    self.logger.info("Installed path %s→%s: %s", src_name, dst_name, path)
                    
                    # Update efficiency metrics
                    self.efficiency_metrics['total_flows'] += 1
                    baseline_path = self._shortest_path_baseline(s_dpid, d_dpid)
                    if baseline_path and path != baseline_path:
                        self.efficiency_metrics['load_balanced_flows'] += 1
                        # Check if we avoided congestion on the baseline path
                        if len(baseline_path) > 1:
                            avoided_congestion = any(cost.get((baseline_path[i], baseline_path[i+1]), 0) > self.THRESHOLD_BPS 
                                                   for i in range(len(baseline_path)-1))
                            if avoided_congestion:
                                self.efficiency_metrics['congestion_avoided'] += 1
            
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
        Finds the optimal path from source to destination based on given cost.
        """
        path = self._dijkstra(src, dst, cost, avoid_congested=True)
        if path:
            return path
        # fallback to any path
        return self._dijkstra(src, dst, cost, avoid_congested=False)

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
                self.logger.info("Re-routing %s→%s: %s", src_name, dst_name, new_path)
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
        # Simple heuristic: non-LLDP, non-broadcast MACs learned on edge ports
        return mac != "ff:ff:ff:ff:ff:ff" and not mac.startswith("01:80:c2")
    
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
        uniform_cost = {(u, v): 1 for (u, v) in self.links.keys()}
        return self._dijkstra(src, dst, uniform_cost, avoid_congested=False)
    
    def _calculate_efficiency_metrics(self, now):
        """
        Calculate efficiency metrics comparing load balancing vs baseline routing.
        """
        if not self.topology_ready or not self.flow_paths:
            return
        
        # Calculate current link utilization variance (lower is better)
        link_utils = []
        baseline_utils = collections.defaultdict(float)
        
        for (u, v), (pu, pv) in self.links.items():
            if u < v:  # Avoid duplicates
                rate_u = self._avg_rate(u, pu, now)
                rate_v = self._avg_rate(v, pv, now)
                current_util = max(rate_u, rate_v)
                link_utils.append(current_util)
        
        # Calculate what utilization would be with shortest path routing
        for (src, dst), path in self.flow_paths.items():
            if src in self.mac_to_dpid and dst in self.mac_to_dpid:
                s_dpid = self.mac_to_dpid[src]
                d_dpid = self.mac_to_dpid[dst]
                baseline_path = self._shortest_path_baseline(s_dpid, d_dpid)
                
                if baseline_path:
                    # Estimate traffic for this flow (simplified)
                    flow_traffic = 1000000  # 1 Mbps per flow estimate
                    
                    # Add traffic to baseline path
                    for i in range(len(baseline_path) - 1):
                        u, v = baseline_path[i], baseline_path[i + 1]
                        key = f"{min(u,v)}-{max(u,v)}"
                        baseline_utils[key] += flow_traffic
        
        # Calculate variances
        if link_utils:
            mean_util = sum(link_utils) / len(link_utils)
            variance = sum((x - mean_util) ** 2 for x in link_utils) / len(link_utils)
            self.efficiency_metrics['link_utilization_variance'] = variance
        
        baseline_util_list = list(baseline_utils.values())
        if baseline_util_list:
            baseline_mean = sum(baseline_util_list) / len(baseline_util_list)
            baseline_variance = sum((x - baseline_mean) ** 2 for x in baseline_util_list) / len(baseline_util_list)
            self.efficiency_metrics['baseline_link_utilization_variance'] = baseline_variance
        
        # Calculate average path lengths
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
        
        # Calculate efficiency percentage
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
        
        # Add switch nodes
        for dpid in self.lb.dp_set.keys():
            nodes.append({"id": f"s{dpid}", "type": "switch"})
        
        # Add host nodes
        for mac, host_name in self.lb.hosts.items():
            nodes.append({"id": host_name, "type": "host"})
        
        # Add switch-to-switch links
        added_links = set()
        for (u, v) in self.lb.links.keys():
            if u < v and (u, v) not in added_links:
                links.append({"source": f"s{u}", "target": f"s{v}", "type": "switch-switch"})
                added_links.add((u, v))
        
        # Add host-to-switch links
        for mac, dpid in self.lb.mac_to_dpid.items():
            if mac in self.lb.hosts:
                host_name = self.lb.hosts[mac]
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
        
        # Calculate derived metrics
        if metrics['total_flows'] > 0:
            metrics['load_balancing_rate'] = (metrics['load_balanced_flows'] / metrics['total_flows']) * 100
            metrics['congestion_avoidance_rate'] = (metrics['congestion_avoided'] / metrics['total_flows']) * 100
        else:
            metrics['load_balancing_rate'] = 0
            metrics['congestion_avoidance_rate'] = 0
        
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