"""
Host Manager
============

Manages host discovery, naming, and conflict resolution in SDN networks.
Provides consistent host tracking across multiple data structures.
"""

import collections
import time


class HostManager:
    """
    Manages host discovery and naming for SDN networks
    """
    
    def __init__(self, parent_app):
        self.parent_app = parent_app
        self.logger = parent_app.logger
        
        # Host tracking data structures
        self.hosts = {}  # mac -> host_name
        self.host_locations = {}  # dpid -> set of host MACs
        self.host_counter = 0  # monotonic counter for stable numbering
        
        # Known topology mappings
        self.hexring_mac_to_host = {
            "00:00:00:00:00:01": "h1",
            "00:00:00:00:00:02": "h2", 
            "00:00:00:00:00:03": "h3",
            "00:00:00:00:00:04": "h4",
            "00:00:00:00:00:05": "h5",
            "00:00:00:00:00:06": "h6"
        }
        
        # Update parent app references
        self.parent_app.hosts = self.hosts
        self.parent_app.host_locations = self.host_locations
        self.parent_app.host_counter = self.host_counter
    
    def discover_host(self, mac, dpid, port):
        """
        Discover and register a new host
        """
        if not self._is_host_mac(mac) or not self._is_host_port(dpid, port):
            return False
        
        if mac not in self.hosts:
            host_name = self._get_proper_host_name(mac, dpid)
            
            # Use proper naming if available, otherwise use sequential counter
            if host_name:
                # Check if this host name is already taken
                existing_mac = self._find_host_by_name(host_name)
                if existing_mac:
                    # Don't override existing assignments - use sequential naming instead
                    self.logger.debug("Host name %s already taken by MAC %s, using sequential naming for MAC %s", 
                                    host_name, existing_mac, mac)
                    self.host_counter += 1
                    host_name = f"h{self.host_counter}"
                
                # Assign the host (either proper name or sequential)
                self._assign_host_atomically(mac, host_name, dpid)
                self.logger.info("Discovered host %s (MAC: %s) at switch %s port %s", 
                               host_name, mac, dpid, port)
                
            # Generate sequential host names as fallback
            else:
                self.host_counter += 1
                host_name = f"h{self.host_counter}"
                self._assign_host_atomically(mac, host_name, dpid)
                self.logger.info("Discovered host %s (MAC: %s) at switch %s port %s", 
                               host_name, mac, dpid, port)
            
            # Update parent app counter
            self.parent_app.host_counter = self.host_counter
            return True
        
        return False
    
    def _is_host_mac(self, mac):
        """
        Determine if a MAC address belongs to a legitimate host.
        Now more restrictive to prevent spurious host discovery.
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
        
        # Only accept MACs that match known topology patterns
        # This prevents random/temporary MACs from being treated as hosts
        return self._is_known_topology_mac(mac)
    
    def _get_proper_host_name(self, mac, dpid):
        """
        Map MAC addresses to proper host names for known topologies.
        """
        # Check if this MAC matches hexring topology
        if mac in self.hexring_mac_to_host:
            return self.hexring_mac_to_host[mac]
        
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
        More restrictive to prevent spurious host discovery.
        """
        # Check if this port is used for inter-switch links
        if dpid in self.parent_app.adj:
            inter_switch_ports = set(self.parent_app.adj[dpid].values())
            is_host_port = port not in inter_switch_ports
            
            # Additional check: ensure it's a reasonable host port number
            # Most topologies use ports 1-4 for hosts
            if is_host_port and 1 <= port <= 6:
                return True
            elif is_host_port:
                # Log suspicious port numbers but still allow them for now
                self.logger.debug("Host discovered on unusual port: switch %d port %d", dpid, port)
                return True
            else:
                return False
        
        # If topology not ready, be more conservative
        # Only allow typical host ports (1-4)
        return 1 <= port <= 4
    
    def _is_known_topology_mac(self, mac):
        """
        Determine if a MAC address is from a known topology (hexring or generic).
        More restrictive to prevent spurious host discovery.
        """
        # Hexring topology MACs (exact match)
        hexring_macs = {
            "00:00:00:00:00:01", "00:00:00:00:00:02", "00:00:00:00:00:03",
            "00:00:00:00:00:04", "00:00:00:00:00:05", "00:00:00:00:00:06"
        }
        
        # Generic topology MACs (pattern-based, more restrictive)
        if mac in hexring_macs:
            return True
        
        # Check for sequential MAC pattern used in generic topologies
        # Only accept the standard Mininet pattern: 00:00:00:00:00:XX
        if mac.startswith("00:00:00:00:00:") and len(mac) == 17:
            try:
                # Extract the last octet
                last_octet = mac.split(":")[5]
                if len(last_octet) <= 2:  # Valid hex format
                    mac_num = int(last_octet, 16)
                    # Only accept reasonable host numbers (1-50)
                    # This prevents random MACs from being treated as hosts
                    return 1 <= mac_num <= 50
            except ValueError:
                pass
        
        return False
    
    def _find_host_by_name(self, host_name):
        """
        Find the MAC address of a host by its name.
        """
        for mac, name in self.hosts.items():
            if name == host_name:
                return mac
        return None
    
    def _assign_host_atomically(self, mac, host_name, dpid):
        """
        Atomically assign a host to ensure data structure consistency.
        """
        # Update all three data structures atomically
        self.hosts[mac] = host_name
        self.host_locations.setdefault(dpid, set()).add(mac)
        
        # mac_to_dpid should already be set from MAC learning, but ensure consistency
        if mac not in self.parent_app.mac_to_dpid:
            self.parent_app.mac_to_dpid[mac] = dpid
        elif self.parent_app.mac_to_dpid[mac] != dpid:
            # MAC moved to a different switch - update location
            old_dpid = self.parent_app.mac_to_dpid[mac]
            if old_dpid in self.host_locations:
                self.host_locations[old_dpid].discard(mac)
            self.parent_app.mac_to_dpid[mac] = dpid
            self.logger.info("Host %s (MAC: %s) moved from switch %s to switch %s", 
                           host_name, mac, old_dpid, dpid)
    
    def validate_host_consistency(self):
        """
        Validate consistency between host tracking data structures.
        """
        inconsistencies = []
        
        # Check hosts vs host_locations consistency
        for mac, host_name in self.hosts.items():
            dpid = self.parent_app.mac_to_dpid.get(mac)
            if dpid is None:
                inconsistencies.append(f"Host {host_name} (MAC: {mac}) not in mac_to_dpid")
                continue
            
            if dpid not in self.host_locations or mac not in self.host_locations[dpid]:
                inconsistencies.append(f"Host {host_name} (MAC: {mac}) not in host_locations[{dpid}]")
        
        # Check host_locations vs hosts consistency
        for dpid, mac_set in self.host_locations.items():
            for mac in mac_set:
                if mac not in self.hosts:
                    inconsistencies.append(f"MAC {mac} in host_locations[{dpid}] but not in hosts")
        
        if inconsistencies:
            self.logger.warning("Host consistency issues found: %s", inconsistencies)
        
        return inconsistencies
    
    def cleanup_spurious_hosts(self):
        """
        Clean up hosts that don't match known topology patterns.
        """
        hosts_to_remove = []
        
        for mac, host_name in self.hosts.items():
            if not self._is_known_topology_mac(mac):
                hosts_to_remove.append((mac, host_name))
        
        for mac, host_name in hosts_to_remove:
            self.logger.info("Removing spurious host %s (MAC: %s)", host_name, mac)
            
            # Remove from all data structures
            del self.hosts[mac]
            if mac in self.parent_app.mac_to_dpid:
                dpid = self.parent_app.mac_to_dpid[mac]
                if dpid in self.host_locations:
                    self.host_locations[dpid].discard(mac)
                del self.parent_app.mac_to_dpid[mac]
    
    def resolve_host_conflicts(self):
        """
        Resolve conflicts in host naming and location.
        """
        # Check for duplicate host names
        name_to_macs = collections.defaultdict(list)
        for mac, name in self.hosts.items():
            name_to_macs[name].append(mac)
        
        for name, macs in name_to_macs.items():
            if len(macs) > 1:
                self.logger.warning("Duplicate host name %s found for MACs: %s", name, macs)
                # Keep the first one, rename others
                for i, mac in enumerate(macs[1:], 1):
                    self.host_counter += 1
                    new_name = f"h{self.host_counter}"
                    self.hosts[mac] = new_name
                    self.logger.info("Renamed host %s to %s (MAC: %s)", name, new_name, mac)
        
        # Update parent app counter
        self.parent_app.host_counter = self.host_counter
    
    def get_host_info(self):
        """Get current host information"""
        return {
            'total_hosts': len(self.hosts),
            'hosts': dict(self.hosts),
            'host_locations': {dpid: list(macs) for dpid, macs in self.host_locations.items()},
            'host_counter': self.host_counter
        }
    
    def get_host_by_mac(self, mac):
        """Get host name by MAC address"""
        return self.hosts.get(mac)
    
    def get_host_location(self, mac):
        """Get switch location of a host by MAC"""
        return self.parent_app.mac_to_dpid.get(mac)
    
    def get_hosts_on_switch(self, dpid):
        """Get all hosts connected to a specific switch"""
        return list(self.host_locations.get(dpid, set()))
    
    def is_host_mac(self, mac):
        """Public interface to check if MAC is a host"""
        return self._is_host_mac(mac)
    
    def is_host_port(self, dpid, port):
        """Public interface to check if port is a host port"""
        return self._is_host_port(dpid, port)