import argparse
import sys
import time
from swarm_os.engine import SwarmMaster, SwarmWorker

def main():
    parser = argparse.ArgumentParser(description="Swarm-OS CLI")
    parser.add_argument('--role', choices=['A', 'B'], required=True)
    parser.add_argument('--model', default='tiny')
    args = parser.parse_args()
    
    # Simple preset mapping
    model_id = args.model
    if args.model == "tiny": model_id = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
    if args.model == "qwen": model_id = "Qwen/Qwen2.5-1.5B-Instruct"

    if args.role == 'B':
        worker = SwarmWorker(model_id=model_id)
        worker.start()
        
    elif args.role == 'A':
        master = SwarmMaster(model_id=model_id)
        try:
            master.connect()
            print("\nType 'exit' to shut down the swarm.\n")
            while True:
                user_input = input("[Judge / User]: ")
                if user_input.lower() == 'exit': break
                
                # Start generation and capture chunks
                for chunk in master.generate(user_input):
                    print(chunk, end="", flush=True)
                
                # 🌟 PRINT THE TELEMETRY SUMMARY 🌟
                end_time = time.perf_counter()
                gen_time = end_time - master.start_time
                tps = master.total_tokens / gen_time
                
                print(f"\n\n[📉 Network Math: {master.total_tokens} x 4.09KB recursive tensors transmitted to Node B]")
                print(f"[⚡ Swarm Telemetry: {master.total_tokens} tokens in {gen_time:.2f}s | Speed: {tps:.2f} Tokens/Sec]")
                print("-" * 30 + "\n")
                
        except KeyboardInterrupt:
            print("\nStopped by user.")
        finally:
            master.close()

if __name__ == "__main__":
    main()