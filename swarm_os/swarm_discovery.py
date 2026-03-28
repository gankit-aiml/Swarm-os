import socket
import time
from zeroconf import ServiceInfo, Zeroconf, ServiceBrowser

class SwarmNodeListener:
    """Listens for other Swarm laptops on the Wi-Fi."""
    def __init__(self):
        self.discovered_nodes = {}

    def remove_service(self, zeroconf, type, name):
        print(f"\n[-] Swarm Node lost: {name}")
        if name in self.discovered_nodes:
            del self.discovered_nodes[name]

    def add_service(self, zeroconf, type, name):
        info = zeroconf.get_service_info(type, name)
        if info:
            # Extract the IP address from the mDNS broadcast
            ip_address = socket.inet_ntoa(info.addresses[0])
            port = info.port
            node_name = name.split('.')[0]
            
            self.discovered_nodes[node_name] = {"ip": ip_address, "port": port}
            print(f"\n[+] 📡 New Swarm Node Discovered! -> {node_name} at {ip_address}:{port}")

    def update_service(self, zeroconf, type, name):
        pass

class SwarmDiscovery:
    def __init__(self, node_id: str, port: int):
        self.node_id = node_id
        self.port = port
        self.zeroconf = Zeroconf()
        self.service_type = "_swarm._tcp.local."
        
        # Get local IP address safely
        s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        try:
            s.connect(('10.255.255.255', 1))
            self.local_ip = s.getsockname()[0]
        except Exception:
            self.local_ip = '127.0.0.1'
        finally:
            s.close()

    def broadcast_presence(self):
        """Node B calls this to shout its IP to the room."""
        service_name = f"SwarmNode_{self.node_id}.{self.service_type}"
        
        # Create the mDNS broadcast packet
        info = ServiceInfo(
            self.service_type,
            service_name,
            addresses=[socket.inet_aton(self.local_ip)],
            port=self.port,
            server=f"swarm_{self.node_id}.local."
        )
        
        self.zeroconf.register_service(info)
        print(f"[Discovery] Broadcasting presence as {service_name} on {self.local_ip}:{self.port}")
        self.info = info

    def search_for_nodes(self, timeout=5) -> dict:
        """Node A calls this to scan the room for Node B."""
        print(f"[Discovery] Scanning local network for Swarm nodes for {timeout} seconds...")
        listener = SwarmNodeListener()
        browser = ServiceBrowser(self.zeroconf, self.service_type, listener)
        
        time.sleep(timeout)
        browser.cancel()
        
        return listener.discovered_nodes

    def cleanup(self):
        if hasattr(self, 'info'):
            self.zeroconf.unregister_service(self.info)
        self.zeroconf.close()
        
# ==========================================
# 🧪 TESTING THE RADAR
# ==========================================
if __name__ == "__main__":
    import sys
    if len(sys.argv) < 2 or sys.argv[1] not in ['broadcaster', 'scanner']:
        print("Usage: python swarm_discovery.py <broadcaster|scanner>")
        sys.exit(1)

    role = sys.argv[1]

    try:
        if role == 'broadcaster':
            # Run this on Laptop B
            discovery = SwarmDiscovery(node_id="Worker_B", port=8888)
            discovery.broadcast_presence()
            print("Broadcasting... Press Ctrl+C to stop.")
            while True:
                time.sleep(1)
                
        elif role == 'scanner':
            # Run this on Laptop A
            discovery = SwarmDiscovery(node_id="Master_A", port=7777)
            found_nodes = discovery.search_for_nodes(timeout=5)
            
            print("\n--- Scan Complete ---")
            if not found_nodes:
                print("No nodes found. Are you on the same Wi-Fi?")
            else:
                for name, data in found_nodes.items():
                    print(f"✅ Ready to connect to {name} at {data['ip']}:{data['port']}")
                    
    except KeyboardInterrupt:
        print("\nStopping discovery...")
    finally:
        if 'discovery' in locals():
            discovery.cleanup()