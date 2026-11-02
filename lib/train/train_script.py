import os
import torch
# loss function related
from lib.utils.box_ops import giou_loss
from torch.nn.functional import l1_loss
from torch.nn import BCEWithLogitsLoss
# train pipeline related
from lib.train.trainers import LTRTrainer
# distributed training related
from torch.nn.parallel import DistributedDataParallel as DDP
# some more advanced functions
from .base_functions import *
# network related
from lib.models.dinotrack import build_dinotrack
# forward propagation related
from lib.train.actors import DINOTrackActor
# for import modules
import importlib

from ..utils.focal_loss import FocalLoss


def _convert_sync_batchnorm(net, enabled=True):
    """Convert the trainable head BN layers before wrapping the model in DDP."""
    if not enabled:
        return net
    return torch.nn.SyncBatchNorm.convert_sync_batchnorm(net)


def run(settings):
    settings.description = 'Training script for STARK-S, STARK-ST stage1, and STARK-ST stage2'

    # update the default configs with config file
    if not os.path.exists(settings.cfg_file):
        raise ValueError("%s doesn't exist." % settings.cfg_file)
    config_module = importlib.import_module("lib.config.%s.config" % settings.script_name)
    cfg = config_module.cfg
    config_module.update_config_from_file(settings.cfg_file)
    if settings.local_rank in [-1, 0]:
        print("New configuration is shown below.")
        for key in cfg.keys():
            print("%s configuration:" % key, cfg[key])
            print('\n')

    # update settings based on cfg
    update_settings(settings, cfg)

    # Record the training log
    log_dir = os.path.join(settings.save_dir, 'logs')
    if settings.local_rank in [-1, 0]:
        if not os.path.exists(log_dir):
            os.makedirs(log_dir)
    settings.log_file = os.path.join(log_dir, "%s-%s.log" % (settings.script_name, settings.config_name))

    # Build dataloaders
    loader_train, loader_val = build_dataloaders(cfg, settings)

    if "RepVGG" in cfg.MODEL.BACKBONE.TYPE or "swin" in cfg.MODEL.BACKBONE.TYPE or "LightTrack" in cfg.MODEL.BACKBONE.TYPE:
        cfg.ckpt_dir = settings.save_dir

    # Create network
    if settings.script_name == "dinotrack":
        net = build_dinotrack(cfg)
    else:
        raise ValueError("illegal script name")

    # Match the paper's single-GPU batch-128 BatchNorm statistics when the
    # same effective batch is split over DDP ranks.  DDP buffer broadcast is
    # not sufficient: it only copies rank-0 running buffers and does not
    # synchronize the per-step batch mean/variance.
    if settings.local_rank != -1:
        if cfg.TRAIN.SYNC_BN:
            net = _convert_sync_batchnorm(net)
            if settings.local_rank == 0:
                sync_bn_count = sum(isinstance(m, torch.nn.SyncBatchNorm)
                                    for m in net.modules())
                print(f"[dinotrack] SyncBatchNorm enabled: {sync_bn_count} layers")
    net.cuda()
    if settings.local_rank != -1:
        net = DDP(net, device_ids=[settings.local_rank], find_unused_parameters=True)
        settings.device = torch.device("cuda:%d" % settings.local_rank)
    else:
        settings.device = torch.device("cuda:0")
    settings.deep_sup = getattr(cfg.TRAIN, "DEEP_SUPERVISION", False)
    settings.distill = getattr(cfg.TRAIN, "DISTILL", False)
    settings.distill_loss_type = getattr(cfg.TRAIN, "DISTILL_LOSS_TYPE", "KL")
    # DINOTrack: checkpoint cadence from the experiment yaml (100/200/245 gates)
    settings.save_interval = getattr(cfg.TRAIN, "SAVE_INTERVAL", 40)
    # Loss functions and Actors
    if settings.script_name == "dinotrack":
        focal_loss = FocalLoss()
        objective = {'giou': giou_loss, 'l1': l1_loss, 'focal': focal_loss, 'cls': BCEWithLogitsLoss()}
        loss_weight = {'giou': cfg.TRAIN.GIOU_WEIGHT, 'l1': cfg.TRAIN.L1_WEIGHT, 'focal': 1., 'cls': 1.0}
        actor = DINOTrackActor(net=net, objective=objective, loss_weight=loss_weight, settings=settings, cfg=cfg)
    else:
        raise ValueError("illegal script name")

    # if cfg.TRAIN.DEEP_SUPERVISION:
    #     raise ValueError("Deep supervision is not supported now.")

    # Optimizer, parameters, and learning rates
    optimizer, lr_scheduler = get_optimizer_scheduler(net, cfg)
    use_amp = getattr(cfg.TRAIN, "AMP", False)
    trainer = LTRTrainer(actor, [loader_train, loader_val], optimizer, settings, lr_scheduler, use_amp=use_amp)

    # train process. DINOTrack fine-tune: when launched with
    # --script_prv/--config_prv, warm-start weights from the previous
    # experiment's latest checkpoint (net-only, epoch restarts at 1) -- but
    # only while the NEW project dir has no checkpoints yet, so relaunches
    # resume (load_latest) instead of re-clobbering the resumed weights.
    load_previous_ckpt = False
    if getattr(settings, "project_path_prv", None) is not None:
        new_dir = os.path.join(trainer._checkpoint_dir, settings.project_path)
        has_new = os.path.isdir(new_dir) and any(f.endswith(".pth.tar") for f in os.listdir(new_dir))
        load_previous_ckpt = not has_new
        if not load_previous_ckpt:
            print("[dinotrack] project_path_prv given but %s already has checkpoints "
                  "-> skipping load_previous (resume instead)" % new_dir)
    trainer.train(cfg.TRAIN.EPOCH, load_latest=True, fail_safe=True,
                  load_previous_ckpt=load_previous_ckpt)
