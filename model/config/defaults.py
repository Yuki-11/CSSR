from yacs.config import CfgNode as CN

_C = CN()
_C.DEVICE = 'cuda'

_C.MODEL = CN()
_C.MODEL.SCALE_FACTOR = 1
_C.MODEL.DETECTOR_TYPE = 'u-net16' # 'PSPNet'
_C.MODEL.SR = 'DBPN'
_C.MODEL.UP_SAMPLE_METHOD = "deconv" # "pixel_shuffle"
_C.MODEL.DETECTOR_DBPN_NUM_STAGES = 4
_C.MODEL.OPTIMIZER = 'Adam' # SGD
_C.MODEL.NUM_CLASSES = 1  
_C.MODEL.NUM_STAGES = 6
_C.MODEL.SR_SEG_INV = False
_C.MODEL.JOINT_LEARNING = True

_C.SOLVER = CN()
_C.SOLVER.MAX_ITER = 100000
_C.SOLVER.TRAIN_DATASET_RATIO = 0.95
_C.SOLVER.SR_PRETRAIN_ITER = 0
_C.SOLVER.SEG_PRETRAIN_ITER = 0
_C.SOLVER.BATCH_SIZE = 8 # default 16

_C.SOLVER.TASK_LOSS_WEIGHT = 0.5
_C.SOLVER.SEG_LOSS_FUNC = "Dice"    # Dice or BCE or WeightedBCE or Boundary or WBCE&Dice or GDC_Boundary
_C.SOLVER.BOUNDARY_DEC_RATIO = 1.0
_C.SOLVER.SR_LOSS_FUNC = "L1"    # L1 or Boundary
_C.SOLVER.WB_AND_D_WEIGHT = [6, 1] # [WBCE ratio, Dice ratio]
_C.SOLVER.BCELOSS_WEIGHT = [20, 1] # [True ratio, False ratio]
_C.SOLVER.ALPHA_MIN = 0.01
_C.SOLVER.DECREASE_RATIO = 1.0
_C.SOLVER.SYNC_BATCHNORM = True
_C.SOLVER.NORM_SR_OUTPUT = "all"
_C.SOLVER.LR = 1e-3
_C.SOLVER.LR_STEPS = []
_C.SOLVER.GAMMA = 0.1
_C.SOLVER.WARMUP_FACTOR = 1.0/6
_C.SOLVER.WARMUP_ITERS = 5000

_C.INPUT = CN()
_C.INPUT.IMAGE_SIZE = [448, 448] # H x W
_C.INPUT.MEAN = [0.4741, 0.4937, 0.5048]
_C.INPUT.STD = [0.1621, 0.1532, 0.1523]

_C.DATASET = CN()
_C.DATASET.TRAIN_IMAGE_DIR = 'datasets/crack_segmentation_dataset/train/images'
_C.DATASET.TRAIN_MASK_DIR = 'datasets/crack_segmentation_dataset/train/masks'
_C.DATASET.TEST_IMAGE_DIR = 'datasets/crack_segmentation_dataset/test/images'
_C.DATASET.TEST_MASK_DIR = 'datasets/crack_segmentation_dataset/test/masks'

_C.OUTPUT_DIR = 'output/CSSR_SR-SS'
_C.SEED = 123

_C.BASE_NET = 'weights/vgg16_reducedfc.pth'
