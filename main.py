import torch
import torch.multiprocessing as mp
import os
import argparse
from tqdm import tqdm
from src import LCMWorker, LOSS_DICT  # __init__.py를 활용한 import

# ==========================================
# Worker Process
# ==========================================
def worker_process(worker_id, gpu_pair, task_queue, progress_queue, prompt, seed):
    device_unet = f"cuda:{gpu_pair[0]}"
    device_vae = f"cuda:{gpu_pair[1]}"
    
    try:
        worker = LCMWorker(device_unet, device_vae)
    except Exception as e:
        progress_queue.put((0, f"[W{worker_id}] Init Error: {e}"))
        return

    for task in task_queue:
        loss_name, n_recur, tfg_scale = task
        loss_fn = LOSS_DICT.get(loss_name) 
        
        try:
            image = worker.generate(
                prompt=prompt,
                num_inference_steps=4,
                tfg_scale=float(tfg_scale),
                n_recur=int(n_recur),
                loss_fn=loss_fn,
                seed=seed
            )
            
            # [수정된 부분] Notion 친화적 저장 로직
            # 1. Scale을 4자리 숫자로 패딩 (예: 100 -> 0100) -> 정렬 시 중요!
            scale_str = f"{int(tfg_scale):04d}"
            
            # 2. 폴더는 Loss 이름으로만 구분 (드래그 & 드롭하기 좋게)
            # 예: results/red/
            save_dir = os.path.join("results", loss_name)
            os.makedirs(save_dir, exist_ok=True)
            
            # 3. 파일명에 모든 정보를 담음 (대문자로 통일)
            # 예: RED_R2_S0300.png
            filename = f"{loss_name.upper()}_R{n_recur}_S{scale_str}.png"
            full_path = os.path.join(save_dir, filename)
            
            image.save(full_path)
            progress_queue.put((1, f"[W{worker_id}] Saved {filename}"))

        except Exception as e:
            progress_queue.put((1, f"[W{worker_id}] Task Error ({loss_name}, s{tfg_scale}): {e}"))

# ==========================================
# Argument Parser
# ==========================================
def get_args():
    parser = argparse.ArgumentParser(description="LCM-TFG Experiment Runner")
    
    parser.add_argument("--prompt", type=str, 
                        default="Self-portrait oil painting, a beautiful cyborg with golden hair, 8k",
                        help="Text prompt for generation")
    parser.add_argument("--seed", type=int, default=2, help="Fixed random seed")
    parser.add_argument("--gpus", type=str, default="0,1,2,3,4,5,6,7", 
                        help="Comma-separated list of GPU IDs to use")
    
    return parser.parse_args()

# ==========================================
# Main Execution
# ==========================================
if __name__ == "__main__":
    mp.set_start_method('spawn', force=True)
    
    # 1. Flags 파싱
    args = get_args()
    
    # 결과 폴더 초기화
    os.makedirs("results", exist_ok=True)

    print(f"Experimental Settings:\n - Seed: {args.seed}\n - GPUs: {args.gpus}\n - Prompt: {args.prompt[:50]}...")
    
    # 2. Experiment Plan
    LOSS_NAMES = ["red", "symmetry", "center", "edge"]
    EXPERIMENTS = []
    
    # (1) Baseline (Identity Loss) - Recur 별로 Scale 0 하나씩
    # Scale 0은 Loss 종류와 상관없이 결과가 같으므로 'none' 폴더에 모아둠
    EXPERIMENTS.append(("none", 1, 0))
    EXPERIMENTS.append(("none", 2, 0))
    EXPERIMENTS.append(("none", 4, 0))
    
    # (2) Adaptive Grid Construction
    # Recur 1: Low Scales
    for s in [100, 150, 300]:
        for l in LOSS_NAMES: EXPERIMENTS.append((l, 1, s))
    
    # Recur 2: Mid Scales
    for s in [100, 300, 500, 750]:
        for l in LOSS_NAMES: EXPERIMENTS.append((l, 2, s))

    # Recur 4: High Scales
    for s in [100, 500, 750, 1000]:
        for l in LOSS_NAMES: EXPERIMENTS.append((l, 4, s))
        
    total_tasks = len(EXPERIMENTS)
    print(f"Total tasks: {total_tasks}")

    # 3. Multiprocessing Setup
    manager = mp.Manager()
    progress_queue = manager.Queue()
    
    # GPU Allocation
    available_gpus = [int(x) for x in args.gpus.split(",")]
    # 2 GPUs per worker
    gpu_pairs = [available_gpus[i:i+2] for i in range(0, len(available_gpus), 2)]
    
    # 마지막 짝이 안 맞으면 버림 (안전장치)
    if len(gpu_pairs[-1]) < 2:
        gpu_pairs.pop()
        
    num_workers = len(gpu_pairs)
    print(f"Spawning {num_workers} workers...")

    # Distribute Tasks
    worker_tasks = [[] for _ in range(num_workers)]
    for i, task in enumerate(EXPERIMENTS):
        worker_tasks[i % num_workers].append(task)
    
    processes = []
    for i in range(num_workers):
        p = mp.Process(
            target=worker_process,
            args=(i, gpu_pairs[i], worker_tasks[i], progress_queue, args.prompt, args.seed)
        )
        p.start()
        processes.append(p)
    
    # 4. Monitoring
    completed_count = 0
    with tqdm(total=total_tasks, unit="img", desc="Progress") as pbar:
        while completed_count < total_tasks:
            count, msg = progress_queue.get()
            completed_count += count
            pbar.update(count)
            
            if "Error" in msg:
                tqdm.write(f"⚠️ {msg}") 

    for p in processes:
        p.join()
        
    print("\n✅ All experiments completed.")