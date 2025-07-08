"""
Topology Manager
===============

Manages network topology discovery, maintenance, and spanning tree construction
for loop-free flooding in SDN networks.
"""

import collections
import time
from ryu.lib import hub
from ryu.topology.api import get_switch, get_link


class TopologyManager:
    """
    Manages dynamic network topology discovery and maintenance
    """
    
    def __init__(self, parent_app):
        self.parent_app = parent_app
        self.logger = parent_app.logger
        
        # Topology data structures
        self.links = {}  # (dpid1, dpid2) -> (port1, port2)
        self.adj = collections.defaultdict(dict)  # dpid -> {neighbor_dpid: port}
        self.flood_ports = collections.defaultdict(set)  # dpid -> {ports for flooding}
        
        # Topology state
        self.topology_ready = False
        self.discovery_interval = 5  # seconds
        
        # Start topology discovery
        self._start_discovery()
    
    def _start_discovery(self):
        """Start the topology discovery process"""
        hub.spawn(self._discover_topology)
    
    def _discover_topology(self):
        """
        Periodically discover network topology using OpenFlow topology discovery.
        """
        while True:
            try:
                # Get switches and links from Ryu topology
                switch_list = get_switch(self.parent_app, None)
                link_list = get_link(self.parent_app, None)
                
                if switch_list and link_list:
                    self._update_topology(switch_list, link_list)
                    if not self.topology_ready:
                        self.topology_ready = True
                        self.logger.info("Topology discovery complete")
                        
            except Exception as e:
                self.logger.error("Topology discovery error: %s", e)
                
            hub.sleep(self.discovery_interval)
    
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
        
        # Update parent app topology
        self.parent_app.adj = self.adj
        self.parent_app.links = self.links
        self.parent_app.topology_ready = self.topology_ready
        
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
        
        # Update parent app flood ports
        self.parent_app.flood_ports = self.flood_ports
    
    def _get_host_port(self, dpid):
        """
        Get the port that connects to a host (usually port 1, but discover dynamically).
        """
        # Find ports that are not inter-switch links
        inter_switch_ports = set(self.adj[dpid].values()) if dpid in self.adj else set()
        all_ports = set()
        
        # Get all ports from MAC learning
        if dpid in self.parent_app.mac_to_port:
            all_ports.update(self.parent_app.mac_to_port[dpid].values())
        
        # Host ports are those not used for inter-switch links
        host_ports = all_ports - inter_switch_ports
        return min(host_ports) if host_ports else 1  # Default to port 1
    
    def cleanup_switch(self, dpid):
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
        
        # Update parent app topology
        self.parent_app.adj = self.adj
        self.parent_app.links = self.links
        
        # Clean up MACs learned on this switch
        macs_to_remove = [mac for mac, switch_id in self.parent_app.mac_to_dpid.items() if switch_id == dpid]
        for mac in macs_to_remove:
            del self.parent_app.mac_to_dpid[mac]
            if mac in self.parent_app.hosts:
                del self.parent_app.hosts[mac]
        
        # Also clean up any additional MACs in host_locations that weren't in mac_to_dpid
        if dpid in self.parent_app.host_locations:
            additional_macs = self.parent_app.host_locations[dpid] - set(macs_to_remove)
            for mac in additional_macs:
                if mac in self.parent_app.hosts:
                    del self.parent_app.hosts[mac]
                # Also remove from mac_to_dpid if it exists but wasn't caught above
                if mac in self.parent_app.mac_to_dpid:
                    del self.parent_app.mac_to_dpid[mac]
            del self.parent_app.host_locations[dpid]
        
        if dpid in self.parent_app.mac_to_port:
            del self.parent_app.mac_to_port[dpid]
        
        # Rebuild spanning tree
        self._build_spanning_tree()
        
        # Clear affected flow paths
        flows_to_remove = [fid for fid, path in self.parent_app.flow_paths.items() if dpid in path]
        for fid in flows_to_remove:
            del self.parent_app.flow_paths[fid]
    
    def get_topology_info(self):
        """Get current topology information"""
        return {
            'switches': len(self.adj),
            'links': len(self.links),
            'ready': self.topology_ready,
            'adj': dict(self.adj),
            'links': dict(self.links)
        }
    
    def get_shortest_path(self, src, dst):
        """Calculate shortest path between two switches using BFS"""
        if src == dst:
            return [src]
        
        if src not in self.adj or dst not in self.adj:
            return None
        
        # BFS for shortest path
        queue = [(src, [src])]
        visited = {src}
        
        while queue:
            current, path = queue.pop(0)
            
            for neighbor in self.adj[current]:
                if neighbor == dst:
                    return path + [neighbor]
                
                if neighbor not in visited:
                    visited.add(neighbor)
                    queue.append((neighbor, path + [neighbor]))
        
        return None
    
    def is_path_valid(self, path):
        """Check if a path is valid in current topology"""
        if not path or len(path) < 2:
            return True
        
        for i in range(len(path) - 1):
            u, v = path[i], path[i + 1]
            if u not in self.adj or v not in self.adj[u]:
                return False
        
        return True
    
    def get_link_info(self, dpid1, dpid2):
        """Get link information between two switches"""
        link_key = (dpid1, dpid2)
        if link_key in self.links:
            return {
                'src_port': self.links[link_key][0],
                'dst_port': self.links[link_key][1],
                'bidirectional': (dpid2, dpid1) in self.links
            }
        return None
    
    def get_neighbors(self, dpid):
        """Get all neighbors of a switch"""
        return list(self.adj.get(dpid, {}).keys())
    
    def get_switch_count(self):
        """Get total number of switches in topology"""
        return len(self.adj)
    
    def get_link_count(self):
        """Get total number of links in topology"""
        return len(self.links) // 2  # Each link is stored bidirectionally