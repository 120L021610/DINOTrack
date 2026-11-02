from easydict import EasyDict as edict
import yaml

"""
Add default config for DINOTrack.
"""
cfg = edict()

# MODEL
cfg.MODEL = edict()
cfg.MODEL.PRETRAIN_FILE = "mae_pretrain_vit_base.pth"
cfg.MODEL.EXTRA_MERGER = False

cfg.MODEL.RETURN_INTER = False
cfg.MODEL.RETURN_STAGES = []

# MODEL.BACKBONE
cfg.MODEL.BACKBONE = edict()
cfg.MODEL.BACKBONE.TYPE = "vit_base_patch16_224"
cfg.MODEL.BACKBONE.STRIDE = 16
cfg.MODEL.BACKBONE.MID_PE = False
cfg.MODEL.BACKBONE.SEP_SEG = False
cfg.MODEL.BACKBONE.CAT_MODE = 'direct'
cfg.MODEL.BACKBONE.MERGE_LAYER = 0
cfg.MODEL.BACKBONE.ADD_CLS_TOKEN = False
cfg.MODEL.BACKBONE.CLS_TOKEN_USE_MODE = 'ignore'

cfg.MODEL.BACKBONE.CE_LOC = []
cfg.MODEL.BACKBONE.CE_KEEP_RATIO = []
cfg.MODEL.BACKBONE.CE_TEMPLATE_RANGE = 'ALL'  # choose between ALL, CTR_POINT, CTR_REC, GT_BOX

# MODEL.HEAD
cfg.MODEL.HEAD = edict()
cfg.MODEL.HEAD.TYPE = "CENTER"
cfg.MODEL.HEAD.NUM_CHANNELS = 256

# MODEL.DINOTrack specifics (the stock ViT/CE backbone fields above are unused
# by build_dinotrack; MODEL.BACKBONE.{STRIDE,PATCH_SIZE} stay in sync at 16)
cfg.MODEL.BACKBONE.WEIGHTS = "models/dinov3-vitb16-pretrain-lvd1689m"  # repo-root relative
cfg.MODEL.BACKBONE.NUM_LAYERS = 12          # DINOv3-Base depth (checkpoint config.json)
cfg.MODEL.BACKBONE.EMBED_DIM = 768          # checkpoint hidden_size
cfg.MODEL.BACKBONE.PATCH_SIZE = 16          # §3.2 P = 16
cfg.MODEL.BACKBONE.REGISTER_TOKENS = 4      # checkpoint num_register_tokens

cfg.MODEL.NECK = edict()
cfg.MODEL.NECK.NUM_LAYERS = 6               # §4.1 K = 6 aggregated layers
cfg.MODEL.NECK.PROJ_DIM = 128               # §4.1 D = 128
cfg.MODEL.NECK.PER_LAYER_PROJ = False       # ablation: phi per layer instead of shared
cfg.MODEL.NECK.USE_CLS_TOKEN = True         # §3.2 Eq. 3; False = Table 5 #1 baseline
cfg.MODEL.NECK.USE_CGHM = True              # §3.3; False = Table 5 #1 baseline
cfg.MODEL.NECK.TAU = 0.07                   # §3.3
cfg.MODEL.NECK.CGHM_SIMILARITY = 'dot'      # literal Eq. 4; cosine is legacy/rejected

cfg.MODEL.DIFF = edict()
cfg.MODEL.DIFF.ATTENTION = 'diff'           # 'diff' = §3.4 DDI; 'std' = Table 5 #1 baseline
cfg.MODEL.DIFF.N_BLOCKS = 1                 # §3.4 single Diff Block
cfg.MODEL.DIFF.N_HEADS = 6                  # §4.1
cfg.MODEL.DIFF.HEAD_DIM = 64                # §4.1
cfg.MODEL.DIFF.LAMBDA_INIT = 0.2            # §3.4 Eq. 8 (d=0)
cfg.MODEL.DIFF.LAMBDA_STD = 0.1             # Differential Transformer [42] init std
cfg.MODEL.DIFF.MLP_RATIO = 4                # [UNSPECIFIED]
cfg.MODEL.DIFF.DROP_PATH = 0.0              # [UNSPECIFIED]
cfg.MODEL.DIFF.NORM_EPS = 1e-5              # [UNSPECIFIED]

# TRAIN
cfg.TRAIN = edict()
cfg.TRAIN.LR = 0.0001
cfg.TRAIN.WEIGHT_DECAY = 0.0001
cfg.TRAIN.EPOCH = 500
cfg.TRAIN.LR_DROP_EPOCH = 400
cfg.TRAIN.BATCH_SIZE = 16
cfg.TRAIN.NUM_WORKER = 8
cfg.TRAIN.OPTIMIZER = "ADAMW"
cfg.TRAIN.BACKBONE_MULTIPLIER = 0.1
cfg.TRAIN.GIOU_WEIGHT = 2.0
cfg.TRAIN.L1_WEIGHT = 5.0
cfg.TRAIN.FREEZE_LAYERS = [0, ]
cfg.TRAIN.PRINT_INTERVAL = 50
cfg.TRAIN.VAL_EPOCH_INTERVAL = 20
cfg.TRAIN.GRAD_CLIP_NORM = 0.1
cfg.TRAIN.AMP = False
# The center head contains BatchNorm2d layers.  The paper trained with a
# single batch of 128; distributed reproduction therefore needs synchronized
# statistics so 8 x 16 behaves like that global batch rather than eight
# independent batch-16 heads.  The conversion is only applied under DDP, so
# single-process training is unchanged.
cfg.TRAIN.SYNC_BN = True

cfg.TRAIN.CE_START_EPOCH = 20  # candidate elimination start epoch
cfg.TRAIN.CE_WARM_EPOCH = 80  # candidate elimination warm up epoch
cfg.TRAIN.DROP_PATH_RATE = 0.1  # drop path rate for ViT backbone

cfg.TRAIN.LOSS_FOCAL_W = 1.0   # Eq. 9 focal coefficient (§4.1: alpha=5 L1, beta=2 GIoU)
cfg.TRAIN.FOCAL_ALPHA = 2.0    # CenterNet focal internal alpha (Lin et al. [26])
cfg.TRAIN.FOCAL_BETA = 4.0     # CenterNet focal internal beta
cfg.TRAIN.HEATMAP_MIN_OVERLAP = 0.7  # CornerNet radius formula overlap (v1 default)
cfg.TRAIN.SAVE_INTERVAL = 40   # also save at epoch % SAVE_INTERVAL == 0 (plus final)
cfg.TRAIN.SEED = 42

# TRAIN.SCHEDULER
cfg.TRAIN.SCHEDULER = edict()
cfg.TRAIN.SCHEDULER.TYPE = "step"
cfg.TRAIN.SCHEDULER.DECAY_RATE = 0.1

# DATA
cfg.DATA = edict()
cfg.DATA.SAMPLER_MODE = "causal"  # sampling methods
cfg.DATA.MEAN = [0.485, 0.456, 0.406]
cfg.DATA.STD = [0.229, 0.224, 0.225]
cfg.DATA.MAX_SAMPLE_INTERVAL = 200
# Augmentation scope. Stock OSTrack applies flip+brightness to BOTH sides and
# adds a joint grayscale(0.05) transform. DINOTrack §4.1: "horizontal flipping
# and brightness jitter are applied to the search region" — the dinotrack
# experiment yamls therefore set TEMPLATE_AUG: False and
# JOINT_GRAYSCALE_PROB: 0.0 (search keeps flip + brightness via
# SEARCH_AUG / transform_search).
cfg.DATA.TEMPLATE_AUG = True
cfg.DATA.SEARCH_AUG = True
cfg.DATA.JOINT_GRAYSCALE_PROB = 0.05
# DATA.TRAIN
cfg.DATA.TRAIN = edict()
cfg.DATA.TRAIN.DATASETS_NAME = ["LASOT", "GOT10K_vottrain"]
cfg.DATA.TRAIN.DATASETS_RATIO = [1, 1]
cfg.DATA.TRAIN.SAMPLE_PER_EPOCH = 60000
# DATA.VAL
cfg.DATA.VAL = edict()
cfg.DATA.VAL.DATASETS_NAME = ["GOT10K_votval"]
cfg.DATA.VAL.DATASETS_RATIO = [1]
cfg.DATA.VAL.SAMPLE_PER_EPOCH = 10000
# DATA.SEARCH
cfg.DATA.SEARCH = edict()
cfg.DATA.SEARCH.SIZE = 320
cfg.DATA.SEARCH.FACTOR = 5.0
cfg.DATA.SEARCH.CENTER_JITTER = 4.5
cfg.DATA.SEARCH.SCALE_JITTER = 0.5
cfg.DATA.SEARCH.NUMBER = 1
# DATA.TEMPLATE
cfg.DATA.TEMPLATE = edict()
cfg.DATA.TEMPLATE.NUMBER = 1
cfg.DATA.TEMPLATE.SIZE = 128
cfg.DATA.TEMPLATE.FACTOR = 2.0
cfg.DATA.TEMPLATE.CENTER_JITTER = 0
cfg.DATA.TEMPLATE.SCALE_JITTER = 0

# TEST
cfg.TEST = edict()
cfg.TEST.TEMPLATE_FACTOR = 2.0
cfg.TEST.TEMPLATE_SIZE = 128
cfg.TEST.SEARCH_FACTOR = 5.0
cfg.TEST.SEARCH_SIZE = 320
cfg.TEST.EPOCH = 500
cfg.TEST.HANN = True             # §4.1-style Hann window damping on the score map


def _edict2dict(dest_dict, src_edict):
    if isinstance(dest_dict, dict) and isinstance(src_edict, dict):
        for k, v in src_edict.items():
            if not isinstance(v, edict):
                dest_dict[k] = v
            else:
                dest_dict[k] = {}
                _edict2dict(dest_dict[k], v)
    else:
        return


def gen_config(config_file):
    cfg_dict = {}
    _edict2dict(cfg_dict, cfg)
    with open(config_file, 'w') as f:
        yaml.dump(cfg_dict, f, default_flow_style=False)


def _update_config(base_cfg, exp_cfg):
    if isinstance(base_cfg, dict) and isinstance(exp_cfg, edict):
        for k, v in exp_cfg.items():
            if k in base_cfg:
                if not isinstance(v, dict):
                    base_cfg[k] = v
                else:
                    _update_config(base_cfg[k], v)
            else:
                raise ValueError("{} not exist in config.py".format(k))
    else:
        return


def update_config_from_file(filename, base_cfg=None):
    exp_config = None
    with open(filename) as f:
        exp_config = edict(yaml.safe_load(f))
        if base_cfg is not None:
            _update_config(base_cfg, exp_config)
        else:
            _update_config(cfg, exp_config)
