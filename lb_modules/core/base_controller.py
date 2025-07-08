"""
Base SDN Controller Module
==========================

Core SDN controller functionality including OpenFlow event handling,
switch management, and basic packet processing.
"""

import collections
import time
from abc import ABC, abstractmethod

from ryu.base import app_manager
from ryu.controller import ofp_event
from ryu.controller.handler import CONFIG_DISPATCHER, MAIN_DISPATCHER, DEAD_DISPATCHER, set_ev_cls
from ryu.lib.packet import packet, ethernet, ether_types, lldp
from ryu.ofproto import ofproto_v1_3


class BaseSDNController(ABC):
    """
    Abstract base class for SDN controllers providing core functionality
    """
    
    def __init__(self):
        self.dp_set = {}  # Connected datapaths
        self.mac_to_port = {}  # MAC learning table
        self.mac_to_dpid = {}  # MAC to switch mapping
        self.active_flows = set()  # Active flows
        self.flow_paths = {}  # Flow routing paths
        
    @abstractmethod
    def _find_path(self, src, dst, cost):
        """Find path between source and destination switches"""
        pass
    
    @abstractmethod
    def _install_path(self, path, src_mac, dst_mac):
        """Install flow rules along a path"""
        pass
    
    @abstractmethod
    def _flood_packet(self, dp, msg, in_port):
        """Flood packet to all ports except input port"""
        pass
    
    @abstractmethod
    def _cleanup_switch(self, dpid):
        """Clean up state when switch disconnects"""
        pass
    
    @abstractmethod
    def _is_host_mac(self, mac):
        """Check if MAC address belongs to a host"""
        pass
    
    @abstractmethod
    def _is_host_port(self, dpid, port):
        """Check if port connects to a host"""
        pass


class SDNController(BaseSDNController):
    """
    Concrete implementation of SDN controller with OpenFlow event handling
    """
    
    def __init__(self, parent_app):
        super().__init__()
        self.parent_app = parent_app
        self.logger = parent_app.logger
        
    def handle_datapath_change(self, ev):
        """Handle datapath state changes"""
        dp = ev.datapath
        if ev.state == MAIN_DISPATCHER:
            self.dp_set[dp.id] = dp
            self.parent_app.dp_set[dp.id] = dp  # Update main app's dp_set too
            self.logger.info("Switch %s connected", dp.id)
        elif ev.state == DEAD_DISPATCHER and dp.id in self.dp_set:
            self.logger.info("Switch %s disconnected", dp.id)
            del self.dp_set[dp.id]
            if dp.id in self.parent_app.dp_set:
                del self.parent_app.dp_set[dp.id]  # Update main app's dp_set too
            self._cleanup_switch(dp.id)
    
    def handle_switch_features(self, ev):
        """Handle switch features and install default flow"""
        dp = ev.msg.datapath
        parser = dp.ofproto_parser
        ofp = dp.ofproto
        
        # Install default flow that sends all packets to controller
        self._add_flow(dp, 0, parser.OFPMatch(),
                      [parser.OFPActionOutput(ofp.OFPP_CONTROLLER, ofp.OFPCML_NO_BUFFER)])
    
    def handle_packet_in(self, ev):
        """Handle packet-in events with MAC learning and routing"""
        msg = ev.msg
        dp = msg.datapath
        dpid = dp.id
        parser = dp.ofproto_parser
        ofp = dp.ofproto
        in_port = msg.match['in_port']
        pkt = packet.Packet(msg.data)
        eth = pkt.get_protocol(ethernet.ethernet)
        
        # Ignore LLDP packets
        if eth.ethertype == ether_types.ETH_TYPE_LLDP:
            return
        
        # Learn MAC addresses
        self.mac_to_port.setdefault(dpid, {})[eth.src] = in_port
        self.parent_app.mac_to_port.setdefault(dpid, {})[eth.src] = in_port  # Sync with main app
        if eth.src not in self.mac_to_dpid:
            self.mac_to_dpid[eth.src] = dpid
            self.parent_app.mac_to_dpid[eth.src] = dpid  # Sync with main app
        
        # Handle host discovery
        if self._is_host_mac(eth.src) and self._is_host_port(dpid, in_port):
            self._handle_host_discovery(eth.src, dpid, in_port)
        
        # Check if topology is ready for routing
        if not self.parent_app.topology_ready:
            self._flood_packet(dp, msg, in_port)
            return
        
        # Handle host-to-host routing
        if eth.dst in self.mac_to_dpid:
            self._handle_host_routing(eth, dp, msg, in_port)
            return
        
        # Flood if destination unknown
        self._flood_packet(dp, msg, in_port)
    
    def _handle_host_discovery(self, src_mac, dpid, in_port):
        """Handle host discovery logic"""
        # Delegate to parent app's host manager
        if hasattr(self.parent_app, 'host_manager'):
            self.parent_app.host_manager.discover_host(src_mac, dpid, in_port)
    
    def _handle_host_routing(self, eth, dp, msg, in_port):
        """Handle routing between known hosts"""
        fid = (eth.src, eth.dst)
        s_dpid = self.mac_to_dpid[eth.src]
        d_dpid = self.mac_to_dpid[eth.dst]
        
        # Find or reuse existing path
        if fid not in self.parent_app.flow_paths:
            cost = self.parent_app._calculate_link_costs(time.time())
            path = self._find_path(s_dpid, d_dpid, cost)
            if path:
                self._install_path(path, eth.src, eth.dst)
                self.flow_paths[fid] = path
                self.parent_app.flow_paths[fid] = path  # Sync with main app
                self.logger.info("Installed path %s→%s: %s", eth.src, eth.dst, path)
                
                # Update efficiency metrics with MAC addresses for consistent flow tracking
                if hasattr(self.parent_app, 'efficiency_tracker'):
                    self.parent_app.efficiency_tracker.update_flow_metrics(s_dpid, d_dpid, path, cost, eth.src, eth.dst)
        
        # Forward packet along path
        path = self.flow_paths.get(fid)
        if path:
            nxt = self._next_hop(path, dp.id)
            if nxt is None:
                # Last hop - use learned MAC port
                out_port = self.mac_to_port[dp.id].get(eth.dst, self._get_host_port(dp.id))
            else:
                out_port = self.parent_app.adj[dp.id][nxt]
        else:
            out_port = self.mac_to_port[dp.id].get(eth.dst, dp.ofproto.OFPP_FLOOD)
        
        # Send packet
        data = msg.data if msg.buffer_id == dp.ofproto.OFP_NO_BUFFER else None
        dp.send_msg(dp.ofproto_parser.OFPPacketOut(
            datapath=dp,
            buffer_id=msg.buffer_id,
            in_port=in_port,
            actions=[dp.ofproto_parser.OFPActionOutput(out_port)],
            data=data
        ))
    
    def _next_hop(self, path, current_dpid):
        """Find next hop in path from current switch"""
        try:
            idx = path.index(current_dpid)
            return path[idx + 1] if idx + 1 < len(path) else None
        except ValueError:
            return None
    
    def _get_host_port(self, dpid):
        """Get default host port for switch"""
        # This would be implemented based on topology
        return 1
    
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
    
    # Abstract methods must be implemented by parent
    def _find_path(self, src, dst, cost):
        return self.parent_app._find_path(src, dst, cost)
    
    def _install_path(self, path, src_mac, dst_mac):
        return self.parent_app._install_path(path, src_mac, dst_mac)
    
    def _flood_packet(self, dp, msg, in_port):
        return self.parent_app._flood_packet(dp, msg, in_port)
    
    def _cleanup_switch(self, dpid):
        return self.parent_app._cleanup_switch(dpid)
    
    def _is_host_mac(self, mac):
        return self.parent_app._is_host_mac(mac)
    
    def _is_host_port(self, dpid, port):
        return self.parent_app._is_host_port(dpid, port)