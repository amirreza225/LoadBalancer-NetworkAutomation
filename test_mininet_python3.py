#!/usr/bin/env python3
"""
Test script to verify Mininet Python 3 compatibility
"""

import sys
import os

def test_mininet_import():
    """Test if Mininet modules can be imported with Python 3"""
    print(f"Python version: {sys.version}")
    print(f"Python path: {sys.path}")
    
    try:
        from mininet.topo import Topo
        print("✓ Successfully imported mininet.topo")
    except ImportError as e:
        print(f"✗ Failed to import mininet.topo: {e}")
        return False
        
    try:
        from mininet.net import Mininet
        print("✓ Successfully imported mininet.net")
    except ImportError as e:
        print(f"✗ Failed to import mininet.net: {e}")
        return False
        
    try:
        from mininet.node import RemoteController, OVSSwitch
        print("✓ Successfully imported mininet.node")
    except ImportError as e:
        print(f"✗ Failed to import mininet.node: {e}")
        return False
        
    try:
        from mininet.cli import CLI
        print("✓ Successfully imported mininet.cli")
    except ImportError as e:
        print(f"✗ Failed to import mininet.cli: {e}")
        return False
        
    try:
        from mininet.link import TCLink
        print("✓ Successfully imported mininet.link")
    except ImportError as e:
        print(f"✗ Failed to import mininet.link: {e}")
        return False
        
    try:
        from mininet.log import setLogLevel
        print("✓ Successfully imported mininet.log")
    except ImportError as e:
        print(f"✗ Failed to import mininet.log: {e}")
        return False
        
    return True

def test_simple_topology():
    """Test creating a simple topology"""
    try:
        from mininet.topo import Topo
        
        class SimpleTopo(Topo):
            def build(self):
                # Add hosts and switches
                h1 = self.addHost('h1')
                h2 = self.addHost('h2')
                s1 = self.addSwitch('s1')
                
                # Add links
                self.addLink(h1, s1)
                self.addLink(h2, s1)
        
        # Create topology instance
        topo = SimpleTopo()
        print("✓ Successfully created simple topology")
        return True
        
    except Exception as e:
        print(f"✗ Failed to create simple topology: {e}")
        return False

def check_mn_command():
    """Check if mn command is available"""
    try:
        import subprocess
        result = subprocess.run(['which', 'mn'], capture_output=True, text=True)
        if result.returncode == 0:
            print(f"✓ mn command found at: {result.stdout.strip()}")
            return True
        else:
            print("✗ mn command not found")
            return False
    except Exception as e:
        print(f"✗ Error checking mn command: {e}")
        return False

def check_mnexec_command():
    """Check if mnexec command is available"""
    try:
        import subprocess
        result = subprocess.run(['which', 'mnexec'], capture_output=True, text=True)
        if result.returncode == 0:
            print(f"✓ mnexec command found at: {result.stdout.strip()}")
            return True
        else:
            print("✗ mnexec command not found")
            return False
    except Exception as e:
        print(f"✗ Error checking mnexec command: {e}")
        return False

def main():
    print("Testing Mininet Python 3 Compatibility")
    print("=" * 50)
    
    # Test imports
    print("\n1. Testing Mininet imports...")
    imports_ok = test_mininet_import()
    
    # Test topology creation
    print("\n2. Testing topology creation...")
    topology_ok = test_simple_topology()
    
    # Test mn command
    print("\n3. Testing mn command availability...")
    mn_ok = check_mn_command()
    
    # Test mnexec command
    print("\n4. Testing mnexec command availability...")
    mnexec_ok = check_mnexec_command()
    
    # Summary
    print("\n" + "=" * 50)
    print("SUMMARY:")
    print(f"Mininet imports: {'✓ PASS' if imports_ok else '✗ FAIL'}")
    print(f"Topology creation: {'✓ PASS' if topology_ok else '✗ FAIL'}")
    print(f"mn command: {'✓ PASS' if mn_ok else '✗ FAIL'}")
    print(f"mnexec command: {'✓ PASS' if mnexec_ok else '✗ FAIL'}")
    
    if imports_ok and topology_ok and mn_ok and mnexec_ok:
        print("\n🎉 All tests passed! Mininet Python 3 is working correctly.")
        return 0
    else:
        print("\n❌ Some tests failed. Check the output above for details.")
        return 1

if __name__ == "__main__":
    sys.exit(main())