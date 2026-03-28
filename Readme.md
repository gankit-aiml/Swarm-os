# 🐝 Swarm-OS: Decentralized Asymmetric Edge Inference
**Built for the AMD Slingshot Hackathon**

[![Demo Video](https://img.shields.io/badge/Watch-Live_Demo_Video-red?style=for-the-badge&logo=youtube)](#) *(<-- REPLACE THIS # WITH YOUR UNLISTED YOUTUBE LINK)*
[![GitHub Package](https://img.shields.io/badge/Install%20with-PIP-blue?style=for-the-badge&logo=pypi)](https://github.com/W-Samik/Swarm_OS)

## 📖 Comprehensive Project Overview

**Swarm-OS** is a decentralized, peer-to-peer AI orchestration protocol designed to run large generative AI models (Large Language Models) natively across a fragmented network of consumer laptops over local Wi-Fi. 

Modern AI models are heavily bottlenecked by the "Memory Wall"—the physical VRAM limits of single GPUs. Swarm-OS completely bypasses this hardware limitation by implementing **Asymmetric Pipeline Parallelism**. It cleanly severs the layers of a neural network (like TinyLlama or Qwen), distributes those layers physically across multiple devices in the same room, and seamlessly routes the hidden state tensors over a zero-latency TCP pipe.

By turning a room full of standard, thin-and-light laptops into a unified distributed supercomputer cluster, Swarm-OS enables near $O(1)$ inference speeds on massive models that would otherwise physically fail to load on any single user's device.

---

## 🚀 How Swarm-OS Operates Under the Hood

To truly understand what Swarm-OS does, you must understand its component lifecycle. Swarm-OS operates through a highly optimized, custom framework spanning local network auto-discovery, surgical model manipulation, and explicit mathematical alignment.

### 1. Zero-Config Swarm Discovery (`swarm_discovery.py`)
Swarm-OS requires **zero manual network configuration**.
- It uses an **mDNS Radar (ZeroConf)** to allow laptops to find each other on the local Wi-Fi.
- **Node B (The Worker)** runs a broadcaster, shouting a Reverse-IP beacon into the room (`_swarm._tcp.local.`).
- **Node A (The Master)** runs a scanner, instantly locking onto the exposed socket of Node B. This makes the system plug-and-play, completely immune to IP changes across different Wi-Fi networks.

### 2. Surgical Asymmetric Sharding (`model_surgeon.py`)
Swarm-OS intercepts Hugging Face models (`AutoModelForCausalLM`) dynamically in RAM:
- It probes the model's configuration to identify the total number of layers.
- It slices the `nn.ModuleList` directly down the middle.
- **Node A** assigns itself Layers `0` to `(N/2 - 1)`, stripping out the final normalization layer to expose raw float states.
- **Node B** assigns itself Layers `(N/2)` to `(N)`, taking over the final normalization and the `lm_head` sampling logits.
This ensures a 140GB model can be split perfectly between two machines with 70GB VRAM each.

### 3. Explicit RoPE Synchronization
When you sever an LLM physically in half, the **Rotary Positional Embeddings (RoPE)** normally desynchronize, utterly destroying the attention matrix because Node B does not naturally know what "word sequence position" it is receiving. Swarm-OS fixes this by tracking the exact `current_seq_length` incrementally across both nodes and injecting it explicitly into `position_ids` during inference forwarding. This guarantees **100% mathematical accuracy** against a monolith execution.

### 4. Zero-Latency Tensor Routing (`network_core.py`)
Instead of bloated REST APIs, Swarm-OS uses a low-level memory-buffer streaming protocol.
- Built atop **PyZMQ with `TCP_NODELAY`**, completely bypassing Nagle's algorithm.
- Node A transmits intermediate Hidden States (Microscopic `4.09KB Float16` Tensors) recursively to Node B via a high-bandwidth `PUSH/PULL` socket architecture.
- Node B receives the tensor, finishes the forward pass, and sends back the final predicted `int32` Token ID.
This yields sub-30ms round-trip latency across standard consumer 5GHz Wi-Fi.

---

## 📦 Quick Install (Universal)

Swarm-OS is packaged as a globally executable Python library. You do not need to clone the repo manually. 

**Run this command on both machines:**
```bash
pip install https://github.com/W-Samik/Swarm_OS/archive/main.zip
```
*(This works on Windows, Linux, and Mac, requiring only a Python environment `torch`, `transformers`, `pyzmq`, and `zeroconf`).*

---

## 🎮 How to Run Swarm-OS (CLI Mode)

Once installed, the `swarm-os` command is available system-wide.

### 1. Start the Worker (Node B)
Run this on the first machine. It loads the **second half** of the model and broadcasts its presence via mDNS.
```bash
swarm-os --role B
```

### 2. Start the Master Orchestrator (Node A)
Run this on the second machine. It loads the **first half** of the model, scans the LAN, locks onto the worker, negotiates a dynamic handshake, and opens the user chat console.
```bash
swarm-os --role A
```

---

## 🛠️ Python SDK Integration (Library Usage)

Swarm-OS is a full-fledged SDK. Developers can integrate the core engine into their own decentralized applications.

```python
import swarm_os
import time

# Initialize Master Orchestrator (Node A)
# Ensure `swarm-os --role B` is already running on the sibling device
bot = swarm_os.SwarmMaster(model_id="TinyLlama/TinyLlama-1.1B-Chat-v1.0")

try:
    # Scan LAN and lock onto worker node
    bot.connect()
    
    question = "What is the future of AMD Architecture?"
    print(f"\n[User]: {question}")
    print("[Swarm-AI]: ", end="", flush=True)
    
    start_time = time.perf_counter()
    
    # Generate tokens via decentralized pipeline streaming
    for chunk in bot.generate(question, stream=True):
        print(chunk, end="", flush=True)
    
    # Analyze Swarm Telemetry
    gen_time = time.perf_counter() - start_time
    tps = bot.total_tokens / gen_time
    
    print(f"\n[⚡ Swarm Telemetry: {bot.total_tokens} Tokens | {tps:.2f} Tokens/Sec]")
    
finally:
    bot.close()
```

---

## 🧠 The Deep Tech Architecture

Modern AI is bottlenecked by the "Memory Wall." Swarm-OS shatters this boundary by physically severing the Transformer block across a local area network (LAN).

1. **The mDNS Radar (`swarm_discovery.py`):** Nodes dynamically discover each other over local Wi-Fi using ZeroConf, negotiating a Reverse-IP Handshake. **No hardcoded IP setup is required.**
2. **Asymmetric Sharding (`model_surgeon.py`):** The `nn.ModuleList` is dynamically sharded in RAM. 
   - **Node A (The Master):** Computes Layers 0-10, emitting a microscopic 4KB Float16 Tensor representing the mathematical hidden states.
   - **Node B (The Worker):** Receives the tensor, computes Layers 11-21 + `lm_head`, and returns the predicted Token ID.
3. **Explicit RoPE Synchronization:** Severing a model normally causes Rotary Positional Embeddings (RoPE) to desynchronize, destroying the attention matrix. Swarm-OS explicitly synchronizes the sequence clock across the TCP pipe, guaranteeing **100% mathematical accuracy**.
4. **Zero-Latency Networking (`network_core.py`):** Utilizes `pyzmq` with `TCP_NODELAY` to stream tensors natively via memory buffers, achieving sub-30ms round-trip latency.

---

## 💻 AMD Hardware Integration Strategy

Swarm-OS aligns perfectly with the **AMD Heterogeneous Compute Ecosystem**. 

In a production deployment, the Swarm Orchestrator tasks (mDNS networking, ZeroMQ routing, and KV-Cache management) run natively at ultra-low wattage on the **AMD Ryzen AI NPU**. Simultaneously, the heavy `Float16` transformer layers are routed to idle **AMD Radeon GPUs** distributed across the local subnet. We turn a room full of thin-and-light laptops into a decentralized supercomputer.

---

## 📊 Live Telemetry Visualization

Swarm-OS proves it is working securely and synchronously through its built-in live diagnostics.

### Node A (Master)
Node A tracks the exact number of tensors transferred across the network pipe, the latency, and the stream speed.
```text
⚙️ [Node A Compute] Analyzing Prompt & Initializing KV-Cache...
🧠 [Node A Compute] Executing Layers 0-10 locally...
📤 [Node A Network] Sent 240.00KB Tensor to Node B over ZeroMQ.

💬 Swarm-AI: The Allies won...
[📉 Network Math: 82 x 4.09KB recursive tensors transmitted to Node B]
[⚡ Swarm Telemetry: 82 tokens generated in 13.48s | Speed: 6.08 Tokens/Sec]
```

### Node B (Worker)
Node B features a real-time monitor, showing exactly when it receives a network payload from Node A and when it predicts the next sequence token.
```text
⚙️ [Node B Compute] 📥 Rcvd: 4.09KB | 🧠 Executed Layers 11-21 | 🎯 Predicted: ' Allied'
⚙️ [Node B Compute] 📥 Rcvd: 4.09KB | 🧠 Executed Layers 11-21 | 🎯 Predicted: ' Powers'
```

---

## 🛠️ Troubleshooting: Connection Issues?

**1. Allow Firewall Access:**
Windows/Mac Firewall will ask for network permission. You **MUST click "Allow"** for both Public and Private networks.

**2. The Ping Test:**
If nodes don't find each other, open **Command Prompt as Administrator** and run this to allow ICMP traffic:
```cmd
netsh advfirewall firewall add rule name="Swarm-OS Allow Ping" protocol=icmpv4:8,any dir=in action=allow
```

**3. The Hotspot Fix:**
If venue Wi-Fi blocks P2P traffic (AP Isolation), connect both laptops to a **Mobile Hotspot**. Swarm-OS uses local LAN traffic only and will **not** consume cellular data.

---

## 📂 Repository Structure

| File | Purpose |
| :--- | :--- |
| `setup.py` | Configures the project as a globally installable `pip` library. |
| `swarm_os/cli.py` | The Terminal entry point with telemetry and model flags. |
| `swarm_os/engine.py` | The Core SDK classes (`SwarmMaster`, `SwarmWorker`). |
| `swarm_os/network_core.py`| The low-latency TCP transport layer for tensors. |
| `swarm_os/model_surgeon.py`| Handles dynamic layer slicing and RoPE synchronization. |
| `swarm_os/swarm_discovery.py`| The mDNS Radar for auto-discovery. |

---

## 🔮 Future Roadmap: Enterprise & Research

### V2.0: Scaling & Efficiency
- **Feature 1: Zero-Waste Safetensor Streaming:** Currently, nodes download the full model. In V2, we will use **HTTP Range Requests** to download *only* the specific byte-ranges from Hugging Face representing a node's assigned layers. This allows a 140GB model to run on a laptop with a 128GB SSD by distributing the storage footprint.
- **Feature 2: N-Node Ring Topology:** Transitioning from 2-node "Ping-Pong" to an $N$-node "Ring" ($A \rightarrow B \rightarrow C \rightarrow A$). This enables nearly infinite scaling of model size by pooling hardware.

### V3.0: Research-Level Innovations
- **Feature 3: Speculative Decoding over LAN:** We will load a tiny "Draft Model" (50M params) on Node A. Node A guesses 5 words locally and sends a "Guess Bundle" to Node B. Node B verifies all 5 words in a single parallel pass, effectively hiding network latency.
- **Feature 4: Dynamic Load Balancing:** Real-time profiling of thermal throttling. If Node B slows down, the Orchestrator dynamically shifts layers back to Node A to maintain maximum throughput.
- **Feature 5: Self-Healing Agentic Redistribution:**
  - **Fault Tolerance:** An RL-based Orchestrator will detect if a node disconnects and instantly redistribute the workload across surviving nodes flawlessly.
  - **Dynamic Joining:** If a new node joins mid-sentence, the system re-profiles the network and redistributes layers to the new device to increase speed without interrupting the user.
