import torch
import numpy as np
import time
import zmq
from .network_core import SwarmCommRouter
from .model_surgeon import ShardedLlama
from .swarm_discovery import SwarmDiscovery

class SwarmWorker:
    def __init__(self, model_id="TinyLlama/TinyLlama-1.1B-Chat-v1.0", port=8888):
        print(f"🟢 Initializing Swarm Worker (Port {port})...")
        self.brain = ShardedLlama(model_id=model_id, role="B")
        self.net = SwarmCommRouter(node_id="B", listen_port=port)
        self.discovery = SwarmDiscovery(node_id="Worker_B", port=port)
        self.memory_cache = None
        self.seq_len = 0
        self.connected_master_ip = None # Track connection to prevent spam

    def start(self):
        self.discovery.broadcast_presence()
        print("[Node B] Broadcasting... Waiting for Master...")
        
        try:
            while True:
                
                incoming_numpy = self.net.recv_tensor()
                
                # Handshake / Reset Logic
                if incoming_numpy.size == 1 and incoming_numpy[0] == -999.0:
                    self.memory_cache = None
                    self.seq_len = 0
                    self.net.send_tensor(np.array([-2.0], dtype=np.float16))
                    continue
                
                # Connection Logic (Fixed spam)
                if incoming_numpy.size == 5 and incoming_numpy[0] == -99.0:
                    ip_parts = [int(x) for x in incoming_numpy]
                    master_ip = f"{ip_parts[1]}.{ip_parts[2]}.{ip_parts[3]}.{ip_parts[4]}"
                    
                    if self.connected_master_ip != master_ip:
                        print(f"\n[Node B] Master Node A identified at {master_ip}. Connecting back...")
                        self.net.connect_to_next_node(master_ip, 7777)
                        self.connected_master_ip = master_ip
                    
                    self.net.send_tensor(np.array([-2.0], dtype=np.float16))
                    continue

                # Inference Logic (Restored Hacker Logs)
                if incoming_numpy.size > 10:
                    device = self.brain.model.device
                    hidden_states = torch.from_numpy(incoming_numpy).to(dtype=torch.float16, device=device)
                    
                    token_id, self.memory_cache, self.seq_len = self.brain.process_node_B(
                        hidden_states, 
                        past_key_values=self.memory_cache,
                        current_seq_length=self.seq_len
                    )
                    self.net.send_tensor(token_id)
                    
                    # 🌟 THEATRICAL LOG FOR NODE B 🌟
                    predicted_word = self.brain.tokenizer.decode([int(token_id[0])])
                    safe_word = predicted_word.replace('\n', '\\n').replace('\r', '')
                    payload_size = incoming_numpy.nbytes / 1024
                    print(f"⚙️ [Node B Compute] 📥 Rcvd: {payload_size:.2f}KB | 🧠 Executed Layers 11-21 | 🎯 Predicted: '{safe_word}'")
                
        except KeyboardInterrupt:
            print("\n[Node B] Stopping...")
        finally:
            self.net.close()
            self.discovery.cleanup()

class SwarmMaster:
    def __init__(self, model_id="TinyLlama/TinyLlama-1.1B-Chat-v1.0", port=7777):
        print(f"🔵 Initializing Swarm Master (Port {port})...")
        self.brain = ShardedLlama(model_id=model_id, role="A")
        self.net = SwarmCommRouter(node_id="A", listen_port=port)
        self.discovery = SwarmDiscovery(node_id="Master_A", port=port)
        self.connected = False

    def connect(self, timeout=10):
        print("[Node A] Scanning for workers...")
        found_nodes = self.discovery.search_for_nodes(timeout=timeout)
        if not found_nodes:
            raise TimeoutError("No Swarm Workers found.")
        
        target_ip = list(found_nodes.values())[0]['ip']
        target_port = list(found_nodes.values())[0]['port']
        print(f"[Node A] Found worker at {target_ip}:{target_port}")
        self.net.connect_to_next_node(target_ip, target_port)
        
        my_ip = [float(x) for x in self.discovery.local_ip.split('.')]
        handshake = np.array([-99.0, my_ip[0], my_ip[1], my_ip[2], my_ip[3]], dtype=np.float16)
        
        print("[Node A] Sending Dynamic Handshake to Node B...")
        self.net.send_tensor(handshake)
        self.connected = True
        print("✅ Swarm Link Fully Established!")

    def _internal_generator(self, user_prompt, max_tokens):
        # 1. Start UI Logs on Node A
        print(f"\n⚙️ [Node A Compute] Analyzing Prompt & Initializing KV-Cache...")
        print(f"🧠 [Node A Compute] Executing Layers 0-10 locally...")

        # Reset Signal
        self.net.send_tensor(np.array([-999.0], dtype=np.float16))
        self.net.recv_tensor() 

        formatted_prompt = (
            f"<|system|>\nYou are Swarm-OS, an intelligent decentralized AI assistant. Answer accurately.</s>\n"
            f"<|user|>\n{user_prompt}</s>\n<|assistant|>\n"
        )
        
        memory_cache = None
        input_data = formatted_prompt
        seq_len = 0
        generated_ids = []
        prev_text = ""
        self.total_tokens = 0
        self.start_time = time.perf_counter()

        for i in range(max_tokens):
            intermediate, memory_cache, seq_len = self.brain.process_node_A(
                prompt_or_token=input_data, 
                past_key_values=memory_cache, 
                current_seq_length=seq_len
            )
            
            tensor_numpy = intermediate.cpu().numpy().astype(np.float16)
            
            # 🌟 LOG THE INITIAL LARGE PAYLOAD 🌟
            if i == 0:
                payload_size = tensor_numpy.nbytes / 1024
                print(f"📤 [Node A Network] Sent {payload_size:.2f}KB Tensor to Node B over ZeroMQ.\n")
                print(f"💬 Swarm-AI: ", end="", flush=True)

            self.net.send_tensor(tensor_numpy)
            
            token_numpy = self.net.recv_tensor()
            while token_numpy.size > 1 or token_numpy[0] < 0:
                token_numpy = self.net.recv_tensor()
            
            new_id = int(token_numpy[0])
            generated_ids.append(new_id)
            self.total_tokens += 1
            
            full_decoded = self.brain.tokenizer.decode(generated_ids, skip_special_tokens=True)
            new_text = full_decoded[len(prev_text):]
            prev_text = full_decoded
            
            yield new_text
            
            if new_id == self.brain.tokenizer.eos_token_id:
                break
            input_data = new_id

    def generate(self, user_prompt, max_tokens=1024, stream=True):
        if not self.connected:
            raise ConnectionError("Not connected. Call connect() first.")

        gen = self._internal_generator(user_prompt, max_tokens)
        
        if stream:
            return gen
        else:
            return "".join(list(gen))

    def close(self):
        self.net.close()
        self.discovery.cleanup()