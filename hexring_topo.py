#!/usr/bin/env python3
"""
Six-switch “hex-ring + chords” testbed.

Hosts:
  h1-h6  →  192.168.8.40-45/24   (one per switch)
  MACs:  00:00:00:00:00:01 to 00:00:00:00:00:06

Inter-switch links:
  (1,2) (2,3) (3,4) (4,5) (5,6) (6,1)
  (6,3) (2,5) (1,4)               ← extra chords
"""

from mininet.topo import Topo
from mininet.net import Mininet
from mininet.node import RemoteController, OVSSwitch
from mininet.cli import CLI
from mininet.link import TCLink
from mininet.log import setLogLevel


class HexRingTopo(Topo):
    def build(self):
        # Add switches s1 to s6
        sw = {i: self.addSwitch(f"s{i}") for i in range(1, 7)}

        # Define fixed IPs and MACs for h1–h6
        ips = [
            "192.168.8.40/24",
            "192.168.8.41/24",
            "192.168.8.42/24",
            "192.168.8.43/24",
            "192.168.8.44/24",
            "192.168.8.45/24",
        ]
        macs = [
            "00:00:00:00:00:01",
            "00:00:00:00:00:02",
            "00:00:00:00:00:03",
            "00:00:00:00:00:04",
            "00:00:00:00:00:05",
            "00:00:00:00:00:06",
        ]

        # Add hosts and link to their corresponding switches
        for i, (ip, mac) in enumerate(zip(ips, macs), start=1):
            h = self.addHost(f"h{i}", ip=ip, mac=mac, defaultRoute=None)
            self.addLink(h, sw[i])

        # Add inter-switch links (ring + chords)
        edges = [
            (1, 2), (2, 3), (3, 4), (4, 5), (5, 6), (6, 1),
            (6, 3), (2, 5), (1, 4)
        ]
        for u, v in edges:
            self.addLink(sw[u], sw[v])


def run():
    setLogLevel("info")
    topo = HexRingTopo()

    net = Mininet(
        topo=topo,
        switch=OVSSwitch,  # Open vSwitch (OF-1.3)
        controller=None,
        link=TCLink,
        build=False,
    )

    # External Ryu controller
    net.addController("c0", controller=RemoteController,
                      ip="127.0.0.1", port=6653)

    net.build()
    net.start()

    # Optional: enable STP at switch level if needed (for OVS) but it can cause multiple problems
    # for sw in net.switches:
    #     sw.cmd(f'ovs-vsctl set Bridge {sw.name} stp_enable=true')

    CLI(net)
    net.stop()


if __name__ == "__main__":
    run()
