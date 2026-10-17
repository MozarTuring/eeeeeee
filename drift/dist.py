"""Minimal torch.distributed helpers: works under torchrun (multi-GPU) and plain python (1 GPU)."""
import os

import torch
import torch.distributed as dist


def setup_dist():
    """Return (rank, world_size, local_rank) and select this process's GPU."""
    if "RANK" in os.environ:
        dist.init_process_group("nccl")
        rank, world = dist.get_rank(), dist.get_world_size()
        local = int(os.environ["LOCAL_RANK"])
    else:
        rank, world, local = 0, 1, 0
    torch.cuda.set_device(local)
    return rank, world, local


def is_dist():
    return dist.is_available() and dist.is_initialized()


def all_reduce_(t, op="sum"):
    if is_dist():
        dist.all_reduce(t, op=dist.ReduceOp.SUM if op == "sum" else dist.ReduceOp.MAX)
    return t


def cleanup():
    if is_dist():
        dist.barrier()
        dist.destroy_process_group()
