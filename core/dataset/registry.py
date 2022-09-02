from core.utils.registry import Registry

DATASETS_CLS = Registry('cls_dataset')  # 2D classification
DATASETS_DET = Registry('det_dataset')  # 2D detection
DATASETS_SEG = Registry('seg_dataset')  # 2D segmentation
DATASETS_PCD = Registry('pdc_dataset')  # 3D detection (point cloud)

# Augmentations
PIPELINES = Registry('piplines')
