#!/usr/bin/env python3
"""
Generic topology builder for testing the dynamic load balancer.
Supports various topologies: linear, ring, tree, mesh, custom.
"""

import argparse
from mininet.topo import Topo
from mininet.net import Mininet
from mininet.node import RemoteController, OVSSwitch
from mininet.cli import CLI
from mininet.link import TCLink
from mininet.log import setLogLevel


class GenericTopo(Topo):
    def __init__(self, topology_type='linear', num_switches=4, **opts):
        super(GenericTopo, self).__init__(**opts)
        self.topology_type = topology_type
        self.num_switches = num_switches
        self.build_topology()

    def build_topology(self):
        # Add switches
        switches = {}
        for i in range(1, self.num_switches + 1):
            switches[i] = self.addSwitch(f's{i}')

        # Add hosts (one per switch)
        for i in range(1, self.num_switches + 1):
            ip = f"192.168.1.{i+10}/24"
            mac = f"00:00:00:00:00:{i:02x}"
            host = self.addHost(f'h{i}', ip=ip, mac=mac, defaultRoute=None)
            self.addLink(host, switches[i])

        # Add switch-to-switch links based on topology type
        if self.topology_type == 'linear':
            self._build_linear(switches)
        elif self.topology_type == 'ring':
            self._build_ring(switches)
        elif self.topology_type == 'tree':
            self._build_tree(switches)
        elif self.topology_type == 'mesh':
            self._build_mesh(switches)
        elif self.topology_type == 'hexring':
            self._build_hexring(switches)
        else:
            raise ValueError(f"Unknown topology type: {self.topology_type}")

    def _build_linear(self, switches):
        """Linear topology: s1 -- s2 -- s3 -- s4"""
        for i in range(1, self.num_switches):
            self.addLink(switches[i], switches[i + 1])

    def _build_ring(self, switches):
        """Ring topology: s1 -- s2 -- s3 -- s4 -- s1"""
        # Linear connections
        for i in range(1, self.num_switches):
            self.addLink(switches[i], switches[i + 1])
        # Close the ring
        if self.num_switches > 2:
            self.addLink(switches[self.num_switches], switches[1])

    def _build_tree(self, switches):
        """Binary tree topology"""
        # Connect switches in a binary tree structure
        for i in range(1, self.num_switches // 2 + 1):
            left_child = 2 * i
            right_child = 2 * i + 1
            if left_child <= self.num_switches:
                self.addLink(switches[i], switches[left_child])
            if right_child <= self.num_switches:
                self.addLink(switches[i], switches[right_child])

    def _build_mesh(self, switches):
        """Full mesh topology: every switch connected to every other switch"""
        for i in range(1, self.num_switches + 1):
            for j in range(i + 1, self.num_switches + 1):
                self.addLink(switches[i], switches[j])

    def _build_hexring(self, switches):
        """Hexagonal ring with chords (requires exactly 6 switches)"""
        if self.num_switches != 6:
            raise ValueError("Hexring topology requires exactly 6 switches")
        
        # Ring connections
        ring_edges = [(1, 2), (2, 3), (3, 4), (4, 5), (5, 6), (6, 1)]
        for u, v in ring_edges:
            self.addLink(switches[u], switches[v])
        
        # Chord connections
        chord_edges = [(1, 4), (2, 5), (3, 6)]
        for u, v in chord_edges:
            self.addLink(switches[u], switches[v])


def run_topology(topo_type='linear', num_switches=4):
    setLogLevel('info')
    
    topo = GenericTopo(topology_type=topo_type, num_switches=num_switches)
    
    net = Mininet(
        topo=topo,
        switch=OVSSwitch,
        controller=None,
        link=TCLink,
        build=False,
    )
    
    # Add remote controller
    net.addController('c0', controller=RemoteController,
                      ip='127.0.0.1', port=6653)
    
    net.build()
    net.start()
    
    print(f"\n=== {topo_type.upper()} TOPOLOGY with {num_switches} switches ===")
    print("Hosts:")
    for host in net.hosts:
        print(f"  {host.name}: {host.IP()}")
    
    print("\nSwitches:")
    for switch in net.switches:
        print(f"  {switch.name}")
    
    print(f"\nTotal links: {len(net.links)}")
    print("\nStarting CLI...")
    
    CLI(net)
    net.stop()


def main():
    parser = argparse.ArgumentParser(description='Generic topology for SDN load balancer testing')
    parser.add_argument('--topology', '-t', 
                       choices=['linear', 'ring', 'tree', 'mesh', 'hexring'],
                       default='linear',
                       help='Topology type (default: linear)')
    parser.add_argument('--switches', '-s', type=int, default=4,
                       help='Number of switches (default: 4)')
    
    args = parser.parse_args()
    
    if args.topology == 'hexring' and args.switches != 6:
        print("Warning: Hexring topology requires exactly 6 switches. Setting switches=6.")
        args.switches = 6
    
    run_topology(args.topology, args.switches)


if __name__ == '__main__':
    main()