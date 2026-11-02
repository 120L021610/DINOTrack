import os
import argparse
import random


def parse_args():
    """
    args for training.
    """
    parser = argparse.ArgumentParser(description='Parse args for training')
    # for train
    parser.add_argument('--script', type=str, help='training script name')
    parser.add_argument('--config', type=str, default='baseline', help='yaml configure file name')
    parser.add_argument('--save_dir', type=str, help='root directory to save checkpoints, logs, and tensorboard')
    parser.add_argument('--mode', type=str, choices=["single", "multiple", "multi_node"], default="multiple",
                        help="train on single gpu or multiple gpus")
    parser.add_argument('--nproc_per_node', type=int, help="number of GPUs per node")  # specify when mode is multiple
    parser.add_argument('--use_lmdb', type=int, choices=[0, 1], default=0)  # whether datasets are in lmdb format
    parser.add_argument('--script_prv', type=str, help='training script name (warm start from a previous experiment)')
    parser.add_argument('--config_prv', type=str, help='yaml configure file name (warm start from a previous experiment)')
    parser.add_argument('--use_wandb', type=int, choices=[0, 1], default=0)  # whether to use wandb
    # for knowledge distillation
    parser.add_argument('--distill', type=int, choices=[0, 1], default=0)  # whether to use knowledge distillation
    parser.add_argument('--script_teacher', type=str, help='teacher script name')
    parser.add_argument('--config_teacher', type=str, help='teacher yaml configure file name')

    # for multiple machines
    parser.add_argument('--rank', type=int, help='Rank of the current process.')
    parser.add_argument('--world-size', type=int, help='Number of processes participating in the job.')
    parser.add_argument('--ip', type=str, default='127.0.0.1', help='IP of the current rank 0.')
    parser.add_argument('--port', type=int, default='20000', help='Port of the current rank 0.')

    args = parser.parse_args()

    return args


def _optional_args(args):
    """Only pass warm-start / distillation flags when they were actually given:
    run_training treats the literal string 'None' as a real experiment name and
    would try to load train/None/baseline as a checkpoint."""
    extra = " --use_lmdb %d --use_wandb %d --distill %d" % (args.use_lmdb, args.use_wandb, args.distill)
    if args.script_prv and args.config_prv:
        extra += " --script_prv %s --config_prv %s" % (args.script_prv, args.config_prv)
    if args.distill:
        extra += " --script_teacher %s --config_teacher %s" % (args.script_teacher, args.config_teacher)
    return extra


def main():
    args = parse_args()
    core = "--script %s --config %s --save_dir %s" % (args.script, args.config, args.save_dir)
    if args.mode == "single":
        train_cmd = "python lib/train/run_training.py " + core + _optional_args(args)
    elif args.mode == "multiple":
        train_cmd = ("python -m torch.distributed.launch --nproc_per_node %d --master_port %d "
                     "lib/train/run_training.py " % (args.nproc_per_node, random.randint(10000, 50000))
                     + core + _optional_args(args))
    elif args.mode == "multi_node":
        train_cmd = ("python -m torch.distributed.launch --nproc_per_node %d --master_addr %s --master_port %d "
                     "--nnodes %d --node_rank %d lib/train/run_training.py "
                     % (args.nproc_per_node, args.ip, args.port, args.world_size, args.rank)
                     + core + _optional_args(args))
    else:
        raise ValueError("mode should be 'single', 'multiple' or 'multi_node'.")
    print(train_cmd)
    os.system(train_cmd)


if __name__ == "__main__":
    main()
