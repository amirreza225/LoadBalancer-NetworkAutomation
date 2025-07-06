#!/usr/bin/env python3
"""
Verification script to demonstrate that Mininet Python 3 fix is working
"""

import sys
import subprocess

def main():
    print("Verifying Mininet Python 3 Fix")
    print("=" * 40)
    
    # Check Python version
    print(f"Python version: {sys.version}")
    
    # Test import
    try:
        from mininet.topo import Topo
        from mininet.net import Mininet
        from mininet.node import OVSSwitch
        print("✓ Mininet modules imported successfully with Python 3")
    except ImportError as e:
        print(f"✗ Import failed: {e}")
        return 1
    
    # Test topology creation
    try:
        class TestTopo(Topo):
            def build(self):
                h1 = self.addHost('h1')
                h2 = self.addHost('h2')
                s1 = self.addSwitch('s1')
                self.addLink(h1, s1)
                self.addLink(h2, s1)
        
        topo = TestTopo()
        print("✓ Topology creation successful")
    except Exception as e:
        print(f"✗ Topology creation failed: {e}")
        return 1
    
    # Check mn command
    try:
        result = subprocess.run(['mn', '--version'], capture_output=True, text=True)
        if result.returncode == 0:
            print(f"✓ mn command available: {result.stdout.strip()}")
        else:
            print("✗ mn command not available")
            return 1
    except Exception as e:
        print(f"✗ Error checking mn command: {e}")
        return 1
    
    print("\n🎉 All verifications passed!")
    print("Mininet is now working correctly with Python 3!")
    
    return 0

if __name__ == "__main__":
    sys.exit(main())