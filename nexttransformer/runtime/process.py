from contextlib import nullcontext
import hashlib
import json
import os
from pathlib import Path
import random
import subprocess
import time

import numpy as np
import torch
import torch.distributed as dist
from nexttransformer.runtime.files import sha256


def initialize(seed):
    world = int(os.environ.get('WORLD_SIZE','1'))
    rank = int(os.environ.get('RANK','0'))
    local_rank = int(os.environ.get('LOCAL_RANK','0'))
    device = torch.device('cuda',local_rank) if torch.cuda.is_available() else torch.device('cpu')
    if device.type=='cuda':
        torch.cuda.set_device(device)
        torch.backends.cuda.matmul.allow_tf32=True
        torch.backends.cudnn.allow_tf32=True
    if world>1:
        initialize_process_group(device, world)
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    if device.type=='cuda': torch.cuda.manual_seed_all(seed)
    return rank,world,device


def initialize_process_group(device, world):
    """Initialize distributed communication with an explicit CUDA device mapping."""
    if world <= 1:
        return
    backend = 'nccl' if device.type == 'cuda' else 'gloo'
    if device.type == 'cuda':
        dist.init_process_group(backend, device_id=device)
    else:
        dist.init_process_group(backend)


def autocast(device):
    return torch.autocast('cuda',dtype=torch.bfloat16) if device.type=='cuda' else nullcontext()


def synchronize(device):
    if device.type=='cuda': torch.cuda.synchronize(device)


def reduce_sum(tensor):
    if dist.is_initialized(): dist.all_reduce(tensor)
    return tensor


def source_manifest():
    root=Path(__file__).resolve().parents[2]
    files={str(p.relative_to(root)):sha256(p) for folder in ('nexttransformer','nanochat','configs','launch','requirements')
           for p in sorted((root/folder).rglob('*')) if p.is_file() and p.suffix in ('.py','.sh','.json','.txt') and '__pycache__' not in p.parts and p.name != 'gpu-environment.lock.txt' and not p.name.endswith(('.local.json','.local.toml'))}
    for name in ('pyproject.toml','uv.lock'):
        files[name]=sha256(root/name)
    commit=subprocess.run(['git','rev-parse','HEAD'],cwd=root,capture_output=True,text=True).stdout.strip()
    return {'commit':commit,'files':files,'sha256':hashlib.sha256(json.dumps(files,sort_keys=True).encode()).hexdigest()}


def rng_state():
    return {'python':random.getstate(),'numpy':np.random.get_state(),'torch':torch.get_rng_state(),
            'cuda':torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None}


def restore_rng(state):
    random.setstate(state['python']); np.random.set_state(state['numpy']); torch.set_rng_state(state['torch'])
    if state['cuda'] is not None: torch.cuda.set_rng_state_all(state['cuda'])


def append_event(output, event):
    with (Path(output)/'metrics.jsonl').open('a') as f:
        f.write(json.dumps(event,ensure_ascii=False)+'\n')


def initial_state(manifest_hash):
    return {'step':0,'input_tokens':0,'training_flops_estimate':0.0,'training_seconds':0.0,
            'validation_forward_flops':0.0,'validation_seconds':0.0,'completed_val_steps':[],
            'data_manifest_sha256':manifest_hash,'next_global_sequence':0}


def rewind_events(output, step):
    """Keep one active event history after restarting from a committed checkpoint."""
    output=Path(output)
    metrics=output/'metrics.jsonl'
    if not metrics.exists(): return
    kept, later=[],[]
    for line in metrics.read_text().splitlines():
        try:
            event=json.loads(line)
        except json.JSONDecodeError:
            later.append(line)
            continue
        (kept if event.get('step',0)<=step else later).append(line)
    if not later: return
    archive=output/'replayed_events'/str(time.time_ns())
    archive.mkdir(parents=True,exist_ok=True)
    (archive/'metrics.jsonl').write_text('\n'.join(later)+'\n')
    temporary=metrics.with_suffix('.jsonl.tmp')
    temporary.write_text('\n'.join(kept)+'\n')
    temporary.replace(metrics)
    for path in output.glob('val_*.json'):
        if int(path.stem.split('_')[1])>step: path.rename(archive/path.name)
