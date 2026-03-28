import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, AutoConfig
import torch.nn as nn
import numpy as np

class ShardedLlama:
    def __init__(self, model_id="TinyLlama/TinyLlama-1.1B-Chat-v1.0", role="A"):
        print(f"Loading {model_id} into memory... (This takes a few seconds)")
        
        self.tokenizer = AutoTokenizer.from_pretrained(model_id)
        
        # 1. AUTO-DETECT GPU
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        print(f"🔥 Hardware Acceleration: {self.device.upper()}")
        
        # 2. LOAD CONFIG TO CALCULATE SPLIT
        config = AutoConfig.from_pretrained(model_id)
        total_layers = config.num_hidden_layers
        split_index = total_layers // 2  # Integer division (e.g., 22 -> 11)
        
        self.model = AutoModelForCausalLM.from_pretrained(
            model_id, 
            torch_dtype=torch.float16,
        ).to(self.device)
        
        self.model.eval() 
        self.role = role
        
        # 3. DYNAMIC PHYSICAL SLICE
        # We access the internal layer list. Note: Different architectures might name this differently.
        # This works for Llama, Mistral, Qwen, etc.
        if hasattr(self.model.model, 'layers'):
            layers = self.model.model.layers
        elif hasattr(self.model.model, 'h'): # For older models like GPT-2/BLOOM
            layers = self.model.model.h
        else:
            raise ValueError("Unknown model architecture: Could not find layer list.")

        if role == "A":
            # Keep first half (0 to split_index)
            self.model.model.layers = layers[:split_index]
            self.model.model.norm = nn.Identity() # Remove final norm from A
            print(f"[Node A] Model sliced. Assigned Layers: 0 to {split_index-1} (of {total_layers})")
        else:
            # Keep second half (split_index to end)
            self.model.model.layers = layers[split_index:]
            print(f"[Node B] Model sliced. Assigned Layers: {split_index} to {total_layers-1} (of {total_layers})")

    def process_node_A(self, prompt_or_token, past_key_values=None, current_seq_length=0):
        """STATEFUL NODE A: O(1) Inference with Explicit RoPE tracking."""
        if self.role != "A": raise ValueError("Only Node A can run this.")
        
        device = self.model.device

        if past_key_values is None:
            inputs = self.tokenizer(prompt_or_token, return_tensors="pt").to(device)
            input_ids = inputs.input_ids
            position_ids = torch.arange(0, input_ids.shape[1], dtype=torch.long, device=device).unsqueeze(0)
        else:
            input_ids = torch.tensor([[prompt_or_token]], device=device)
            position_ids = torch.tensor([[current_seq_length]], dtype=torch.long, device=device)

        with torch.no_grad():
            outputs = self.model.model(
                input_ids=input_ids, 
                position_ids=position_ids, 
                past_key_values=past_key_values, 
                use_cache=True
            )
            
        tensor_to_send = outputs.last_hidden_state
        new_memory_cache = outputs.past_key_values
        new_seq_length = current_seq_length + input_ids.shape[1]
        
        return tensor_to_send, new_memory_cache, new_seq_length

    def process_node_B(self, hidden_states: torch.Tensor, past_key_values=None, current_seq_length=0):
        """STATEFUL NODE B: O(1) Inference using synchronized RoPE."""
        if self.role != "B": raise ValueError("Only Node B can run this.")
        device = self.model.device
        
        seq_len = hidden_states.shape[1]
        position_ids = torch.arange(
            current_seq_length, current_seq_length + seq_len, 
            dtype=torch.long, device=device
        ).unsqueeze(0)
        
        with torch.no_grad():
            outputs = self.model(
                inputs_embeds=hidden_states, 
                position_ids=position_ids,
                past_key_values=past_key_values,
                use_cache=True
            )
            
        logits = outputs.logits[0, -1, :].clone()
        
        # Clean Sampler
        temperature = 0.4  
        top_k = 40         
        scaled_logits = logits / temperature
        top_k_values, _ = torch.topk(scaled_logits, top_k)
        scaled_logits[scaled_logits < top_k_values[-1]] = float('-inf')
        probs = torch.nn.functional.softmax(scaled_logits, dim=-1)
        next_token_id = torch.multinomial(probs, num_samples=1)
        
        new_memory_cache = outputs.past_key_values
        new_seq_length = current_seq_length + seq_len
        
        return next_token_id.cpu().numpy().astype(np.int32), new_memory_cache, new_seq_length