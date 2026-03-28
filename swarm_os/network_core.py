import zmq
import json
import numpy as np
import time
from typing import Optional

class SwarmNetworkError(Exception): pass
class SwarmProtocolError(SwarmNetworkError): pass
class SwarmSecurityError(SwarmNetworkError): pass

class SwarmCommRouter:
    def __init__(
        self, 
        node_id: str, 
        listen_port: int, 
        bind_ip: str = "0.0.0.0",
        max_tensor_bytes: int = 1024 * 1024 * 100,
        zmq_context: Optional[zmq.Context] = None
    ):
        self.node_id = node_id
        self.listen_port = listen_port
        self.max_tensor_bytes = max_tensor_bytes
        self.context = zmq_context if zmq_context else zmq.Context()

        # --- 1. Setup the Receiver (PULL socket) ---
        self.receiver = self.context.socket(zmq.PULL)
        self.receiver.setsockopt(zmq.RCVHWM, 2)
        
        try:
            bind_addr = f"tcp://{bind_ip}:{listen_port}"
            self.receiver.bind(bind_addr)
            print(f"[Node {self.node_id}] Network Core Initialized. Listening on {bind_addr}")
        except zmq.ZMQError as e:
            raise SwarmNetworkError(f"Failed to bind to port {listen_port}. Is another Python process running? Error: {e}")

        # --- 2. Setup the Sender (PUSH socket) ---
        self.sender = self.context.socket(zmq.PUSH)
        self.sender.setsockopt(zmq.SNDHWM, 2)
        self.sender.setsockopt(zmq.LINGER, 0)
        self.is_connected_downstream = False

    def connect_to_next_node(self, target_ip: str, target_port: int):
        target_addr = f"tcp://{target_ip}:{target_port}"
        try:
            print(f"[Node {self.node_id}] Connecting downstream to -> {target_addr}...")
            self.sender.connect(target_addr)
            self.is_connected_downstream = True
            print(f"[Node {self.node_id}] Connection established.")
        except zmq.ZMQError as e:
            raise SwarmNetworkError(f"Failed to connect: {e}")

    def send_tensor(self, tensor_np: np.ndarray):
        if not self.is_connected_downstream: return
        if not tensor_np.flags['C_CONTIGUOUS']:
            tensor_np = np.ascontiguousarray(tensor_np)

        header = {"dtype": str(tensor_np.dtype.name), "shape": list(tensor_np.shape)}
        header_bytes = json.dumps(header).encode('utf-8')
        data_bytes = tensor_np.tobytes()
        self.sender.send_multipart([header_bytes, data_bytes], copy=False)

    def recv_tensor(self) -> np.ndarray:
        frames = self.receiver.recv_multipart(copy=False)
        if len(frames) != 2:
            raise SwarmProtocolError("Malformed packet received.")

        header_frame, data_frame = frames
        header_dict = json.loads(header_frame.bytes.decode('utf-8'))
        target_dtype = np.dtype(header_dict["dtype"])
        shape = tuple(header_dict["shape"])
        
        expected_size = np.prod(shape) * target_dtype.itemsize
        if len(data_frame) != expected_size:
             raise SwarmSecurityError("Payload size mismatch.")

        tensor_np = np.frombuffer(data_frame, dtype=target_dtype).reshape(shape)
        return tensor_np

    def close(self):
        """Forcefully close sockets and release ports back to the OS."""
        self.sender.setsockopt(zmq.LINGER, 0)
        self.receiver.setsockopt(zmq.LINGER, 0)
        self.sender.close()
        self.receiver.close()
        if self.context:
             self.context.term()
        print(f"[Node {self.node_id}] Sockets closed and ports released.")


if __name__ == "__main__":
    import sys

    # ==========================================
    # ⚠️ HARDCODE YOUR IP ADDRESSES HERE ⚠️
    # ==========================================
    IP_MACHINE_A = "192.168.31.80"  # Replace with Machine A's actual IP
    IP_MACHINE_B = "192.168.31.179"  # Replace with Machine B's actual IP
    
    if len(sys.argv) < 2 or sys.argv[1] not in ['sender', 'receiver']:
        print("Usage: python network_core.py <sender|receiver>")
        sys.exit(1)

    role = sys.argv[1]
    node = None

    try:
        if role == 'receiver':
            # Node B
            node = SwarmCommRouter(node_id="B", listen_port=8888)
            node.connect_to_next_node(IP_MACHINE_A, 7777)
            
            print("\n[Node B] Waiting for Handshake from Node A...")
            
            test_counter = 0 # Track how many tensors we process
            
            while True:
                received_tensor = node.recv_tensor()
                
                # --- HANDSHAKE LOGIC ---
                if received_tensor.size == 1 and received_tensor[0] == -1.0:
                    print("\n✅ SUCCESS: Connection securely established between Node A and Node B!")
                    print("[Node B] Sending acknowledgment back...")
                    node.send_tensor(np.array([-2.0], dtype=np.float16))
                    print("[Node B] Waiting for main test tensors...\n")
                    continue 
                
                # --- MAIN TEST LOGIC ---
                test_counter += 1
                
                # Print every 10th tensor so we don't spam the console too much
                if test_counter % 10 == 0:
                    print(f"[Node B] Successfully processed and returned tensor {test_counter}/100...")
                
                processed_tensor = received_tensor * 2 
                node.send_tensor(processed_tensor)

        elif role == 'sender':
            # Node A
            node = SwarmCommRouter(node_id="A", listen_port=7777)
            node.connect_to_next_node(IP_MACHINE_B, 8888)
            
            print("\n[Node A] Sending Handshake to Node B...")
            
            # --- HANDSHAKE LOGIC ---
            # Send a tiny 1-element array with a magic number (-1.0)
            handshake_tensor = np.array([-1.0], dtype=np.float16)
            
            # Keep trying to send the handshake until the firewall/network lets it through
            # Keep trying, but don't freeze if the buffer fills up
            handshake_successful = False
            while not handshake_successful:
                try:
                    # Manually construct and send the handshake with NOBLOCK so it doesn't freeze
                    header = {"dtype": str(handshake_tensor.dtype.name), "shape": list(handshake_tensor.shape)}
                    header_bytes = json.dumps(header).encode('utf-8')
                    node.sender.send_multipart([header_bytes, handshake_tensor.tobytes()], flags=zmq.NOBLOCK, copy=False)
                except zmq.Again:
                    pass # Buffer is temporarily full because connection is pending. Ignore and continue.
                
                time.sleep(0.5) # Wait half a second
                
                try:
                    # Check if Node B replied
                    reply = node.receiver.recv_multipart(flags=zmq.NOBLOCK, copy=False)
                    reply_tensor = np.frombuffer(reply[1], dtype=np.float16)
                    
                    if reply_tensor[0] == -2.0:
                        print("\n✅ SUCCESS: Connection securely established between Node A and Node B!")
                        handshake_successful = True
                except zmq.Again:
                    pass # No reply yet, loop again

            # --- MAIN TEST LOGIC ---
            DUMMY_SHAPE = (1, 128, 4096)
            dummy_tensor = np.ones(DUMMY_SHAPE, dtype=np.float16)
            
            print("\n[Node A] Starting Ping-Pong test (100 iterations)...")
            latencies = []

            for i in range(100):
                start_time = time.perf_counter()
                
                node.send_tensor(dummy_tensor)
                returned_tensor = node.recv_tensor()
                
                end_time = time.perf_counter()
                
                rtt_ms = (end_time - start_time) * 1000
                latencies.append(rtt_ms)
                if i % 10 == 0: print(f"Iter {i}: RTT = {rtt_ms:.2f} ms")

            print(f"\nAverage RTT Latency: {np.mean(latencies):.2f} ms")

    except KeyboardInterrupt:
        print("\nStopped by user.")
    except Exception as e:
        print(f"\nError: {e}")
    finally:
        if node:
            node.close()