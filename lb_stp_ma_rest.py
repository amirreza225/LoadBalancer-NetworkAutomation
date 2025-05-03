#!/usr/bin/env python3
"""
Static PORT_MAP Load Balancer with loop-free ARP flooding,
proactive path installation, dynamic rebalancing,
and congestion-aware path selection.
Topology: six-switch ring+chords, hosts h1–h6 on port 1 of each switch.
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
from ryu.lib.packet import packet, ethernet, ether_types
from ryu.ofproto import ofproto_v1_3

# ─────────── STATIC TOPOLOGY MAP ───────────
# (u, v) -> (port_on_u, port_on_v)
PORT_MAP = {
    (1, 2): (2, 2), (2, 3): (3, 2), (3, 4): (3, 2),
    (4, 5): (3, 2), (5, 6): (3, 2), (6, 1): (3, 3),
    (1, 4): (4, 4), (2, 5): (4, 4), (6, 3): (4, 4),
}

POLL_PERIOD    = 2         # seconds
MA_WINDOW_SEC  = 5         # seconds
DEFAULT_THRESH = 1_000_000 # bytes/sec

HOSTS = {
    '00:00:00:00:00:01': 'h1',
    '00:00:00:00:00:02': 'h2',
    '00:00:00:00:00:03': 'h3',
    '00:00:00:00:00:04': 'h4',
    '00:00:00:00:00:05': 'h5',
    '00:00:00:00:00:06': 'h6',
}
HOST_MAC_TO_DPID = {mac: i for mac, i in zip(HOSTS, range(1, 7))}

class LoadBalancerREST(app_manager.RyuApp):
    OFP_VERSIONS = [ofproto_v1_3.OFP_VERSION]
    _CONTEXTS = {'wsgi': WSGIApplication}

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # datapaths & MAC learning
        self.dp_set       = {}
        self.mac_to_port  = {}
        self.mac_to_dpid  = dict(HOST_MAC_TO_DPID)
        self.active_flows = set()
        self.flow_paths   = {}
        # stats
        self.last_bytes   = collections.defaultdict(lambda: collections.defaultdict(int))
        self.rate_hist    = collections.defaultdict(lambda: collections.defaultdict(list))
        self.last_calc    = 0
        self.THRESHOLD_BPS = DEFAULT_THRESH
        # build adjacency
        self.adj = collections.defaultdict(dict)
        for (u, v), (pu, pv) in PORT_MAP.items():
            self.adj[u][v] = pu
            self.adj[v][u] = pv
        # build loop-free flooding tree (BFS from switch 1)
        visited, queue = {1}, [1]
        tree_edges = set()
        while queue:
            u = queue.pop(0)
            for v in self.adj[u]:
                if v not in visited:
                    visited.add(v); queue.append(v)
                    tree_edges.update({(u, v), (v, u)})
        self.flood_ports = {sw: {1} | {self.adj[u][v] for (u, v) in tree_edges if u==sw}
                            for sw in self.adj}
        self.logger.info("Flood ports: %s", self.flood_ports)
        # start stats polling
        hub.spawn(self._poll_stats)
        # register REST API
        kwargs['wsgi'].register(LBRestController, {'lbapp': self})

    @set_ev_cls(ofp_event.EventOFPStateChange, [MAIN_DISPATCHER, DEAD_DISPATCHER])
    def _dp_state(self, ev):
        """
        Store datapaths in self.dp_set when they enter MAIN_DISPATCHER
        and remove them when they leave (DEAD_DISPATCHER). This is
        necessary because EventOFPStateChange is not datapath-specific
        (i.e., it does not contain the datapath instance).
        """
        dp = ev.datapath
        if ev.state == MAIN_DISPATCHER:
            self.dp_set[dp.id] = dp
        elif ev.state == DEAD_DISPATCHER and dp.id in self.dp_set:
            del self.dp_set[dp.id]

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
        If a path is installed, forward the packet along it; otherwise, flood.
        """
        msg, dp = ev.msg, ev.msg.datapath
        dpid = dp.id; parser, ofp = dp.ofproto_parser, dp.ofproto
        in_port = msg.match['in_port']; pkt = packet.Packet(msg.data)
        eth = pkt.get_protocol(ethernet.ethernet)
        if eth.ethertype == ether_types.ETH_TYPE_LLDP: return
        # learn
        self.mac_to_port.setdefault(dpid, {})[eth.src] = in_port
        if eth.src not in self.mac_to_dpid:
            self.mac_to_dpid[eth.src] = dpid
        # host-to-host
        if eth.dst in self.mac_to_dpid:
            fid = (eth.src, eth.dst)
            if fid not in self.flow_paths:
                s_dpid, d_dpid = self.mac_to_dpid[eth.src], self.mac_to_dpid[eth.dst]
                cost = {(u, v): max(self._avg_rate(u, pu, time.time()),
                                     self._avg_rate(v, pv, time.time()))
                        for (u, v), (pu, pv) in PORT_MAP.items()}
                path = self._find_path(s_dpid, d_dpid, cost)
                if path:
                    self._install_path(path, eth.src, eth.dst)
                    self.flow_paths[fid] = path
                    self.logger.info("Installed path %s→%s: %s",
                                     HOSTS[eth.src], HOSTS[eth.dst], path)
            path = self.flow_paths.get(fid)
            if path:
                nxt = self._next_hop(path, dpid)
                out_port = 1 if nxt is None else self.adj[dpid][nxt]
            else:
                out_port = self.mac_to_port[dpid].get(eth.dst, ofp.OFPP_FLOOD)
            data = msg.data if msg.buffer_id == ofp.OFP_NO_BUFFER else None
            dp.send_msg(parser.OFPPacketOut(datapath=dp, buffer_id=msg.buffer_id,
                                           in_port=in_port,
                                           actions=[parser.OFPActionOutput(out_port)],
                                           data=data))
            return
        # flood
        ports = self.flood_ports.get(dpid, {1})
        actions = [parser.OFPActionOutput(p) for p in ports]
        dp.send_msg(parser.OFPPacketOut(datapath=dp, buffer_id=msg.buffer_id,
                                       in_port=in_port, actions=actions, data=msg.data))

    def _find_path(self, src, dst, cost):
        # try avoiding congested edges first
        """
        Finds the optimal path from the source to the destination based on the given cost.

        Tries to avoid congested edges by first attempting to find a path with the
        `avoid_congested` flag set to True. If no such path is found, it falls back to
        finding any available path without avoiding congestion.

        Args:
            src (int): The source node identifier.
            dst (int): The destination node identifier.
            cost (dict): A dictionary mapping edge tuples to their respective costs.

        Returns:
            list: A list of nodes representing the path from the source to the destination,
                or None if no path is found.
        """

        path = self._dijkstra(src, dst, cost, avoid_congested=True)
        if path:
            return path
        # fallback to any path
        return self._dijkstra(src, dst, cost, avoid_congested=False)

    def _dijkstra(self, src, dst, cost, avoid_congested):
        """
        Runs Dijkstra's algorithm to find the shortest path from the source to the destination.

        Args:
            src (int): The source node identifier.
            dst (int): The destination node identifier.
            cost (dict): A dictionary mapping edge tuples to their respective costs.
            avoid_congested (bool): Whether to avoid edges with costs greater than
                `self.THRESHOLD_BPS`.

        Returns:
            list: A list of nodes representing the shortest path from the source to the
                destination, or None if no path is found.
        """
        pq, seen = [(0, src, [src])], set()
        while pq:
            c, u, path = heapq.heappop(pq)
            if u == dst: return path
            if u in seen: continue
            seen.add(u)
            for v in self.adj[u]:
                if v in seen: continue
                if avoid_congested and cost.get((u, v), 0) > self.THRESHOLD_BPS:
                    continue
                new_cost = c + cost.get((u, v), 0)
                heapq.heappush(pq, (new_cost, v, path + [v]))
        return None

    def _add_flow(self, dp, priority, match, actions, **kwargs):
        """
        Adds a flow entry to the datapath.

        Args:
            dp (Datapath): The datapath to install the flow entry on.
            priority (int): The priority of the flow entry.
            match (OFPMatch): The match fields for the flow entry.
            actions (list): A list of OFPActions for the flow entry.
            **kwargs: Additional keyword arguments to pass to OFPFlowMod.

        Returns:
            None
        """
        inst = [dp.ofproto_parser.OFPInstructionActions(
            dp.ofproto.OFPIT_APPLY_ACTIONS, actions)]
        dp.send_msg(dp.ofproto_parser.OFPFlowMod(
            datapath=dp, priority=priority,
            match=match, instructions=inst, **kwargs))

    def _poll_stats(self):
        """
        Periodically polls all datapaths for port statistics.

        This function runs an infinite loop, sleeping for POLL_PERIOD seconds
        between iterations. In each iteration, it sends an OFPPortStatsRequest
        to all datapaths in dp_set, which triggers an EventOFPPortStatsReply
        event handled by _stats_reply.
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

        Updates the last_bytes and rate_hist dictionaries based on the received
        port statistics. If the time difference between the current time and
        the last calculation is greater than MA_WINDOW_SEC, calls _rebalance
        to recalculate the moving average of link utilization and perform path
        rebalancing if necessary.

        Args:
            ev (EventOFPPortStatsReply): The received event.

        Returns:
            None
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

    def _rebalance(self, now):
        """
        Periodically recalculates the moving average of link utilization
        and updates the current paths for all flows based on the new costs.

        Args:
            now (float): The current time in seconds.

        Returns:
            None
        """
        cost = {(u, v): max(self._avg_rate(u, pu, now),
                             self._avg_rate(v, pv, now))
                for (u, v), (pu, pv) in PORT_MAP.items()}
        for fid, old_path in list(self.flow_paths.items()):
            src, dst = fid
            s_dpid, d_dpid = self.mac_to_dpid[src], self.mac_to_dpid[dst]
            new_path = self._find_path(s_dpid, d_dpid, cost)
            if new_path and new_path != old_path:
                self.logger.info("Re-routing %s→%s: %s",
                                 HOSTS[src], HOSTS[dst], new_path)
                self._install_path(new_path, src, dst)
                self.flow_paths[fid] = new_path

    def _avg_rate(self, dpid, port, now):
        """
        Calculates the moving average of the link utilization rate (in bytes per second) over the last
        MA_WINDOW_SEC seconds.

        Args:
            dpid (int): The datapath identifier.
            port (int): The port number.
            now (float): The current time in seconds.

        Returns:
            float: The moving average rate in bytes per second.
        """
        samp = [r for t, r in self.rate_hist[dpid][port] if now - t <= MA_WINDOW_SEC]
        return sum(samp) / len(samp) if samp else 0

    def _install_path(self, path, src, dst):
        """
        Installs a flow entry on all datapaths in the path from src to dst,
        with an output action on the port that leads to the next hop.

        Args:
            path (list): The list of datapath identifiers along the path.
            src (str): The source MAC address.
            dst (str): The destination MAC address.
        """
        out_map = {path[i]: self.adj[path[i]][path[i+1]] for i in range(len(path)-1)}
        out_map[path[-1]] = 1
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

        Args:
            path (list): The list of datapath identifiers representing the current path.
            dpid (int): The datapath identifier for which to determine the next hop.

        Returns:
            int or None: The datapath identifier of the next hop if it exists, otherwise None.
        """

        if dpid not in path: return None
        idx = path.index(dpid)
        return None if idx == len(path)-1 else path[idx+1]

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

    # @route('path', '/load/path', methods=['GET'])
    # def get_paths(self, req, **_):
    #     paths = {
    #         f"{HOSTS.get(src, src)}→{HOSTS.get(dst, dst)}": path
    #         for (src, dst), path in self.lb.flow_paths.items()
    #     }
    #     return self._cors(json.dumps(paths))
    @route('path', '/load/path', methods=['GET'])
    def get_paths(self, req, **_):
        seen = set()
        paths = {}
        for (src, dst), path in self.lb.flow_paths.items():
            key = tuple(sorted((src, dst)))
            if key in seen:
                continue
            seen.add(key)
            label = f"{HOSTS.get(src, src)}→{HOSTS.get(dst, dst)}"
            paths[label] = path
        return self._cors(json.dumps(paths))

    @route('links', '/load/links', methods=['GET'])
    def get_links(self, req, **_):
        now = time.time()
        data = {
            f"{u}-{v}": max(
                self.lb._avg_rate(u, pu, now),
                self.lb._avg_rate(v, pv, now))
            for (u, v), (pu, pv) in PORT_MAP.items()
        }
        return self._cors(json.dumps(data))

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