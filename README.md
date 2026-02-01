# Cat vs Dog CNN with Grad-CAM

## Overview
- Task: Binary image classification (cats vs dogs) using a custom CNN trained on `data/training_set` with validation split and evaluated on `data/test_set`.
- Model: convolutional network with conv-BN-ReLU-MaxPool blocks (up to conv4 considered in analysis) and a dropout-regularized MLP head. Grad-CAM is applied on conv4 for interpretability.
- Robust loading: `SafeImageFolder` skips corrupted images (e.g., PIL `UnidentifiedImageError`).
- Docker: `intel/intel-extension-for-pytorch:2.8.10-xpu`

## Key Artifacts (inline)
- Training curves
  
	![training_history](training_history.png)

- Filters (conv1-conv4)
  
	![conv1_filters](conv1_filters.png)
  
	![conv2_filters](conv2_filters.png)
  
	![conv3_filters](conv3_filters.png)
  
	![conv4_filters](conv4_filters.png)

- Feature maps (conv1-conv4)
  
	![conv1_feature_maps](conv1_feature_maps.png)
  
	![conv2_feature_maps](conv2_feature_maps.png)
  
	![conv3_feature_maps](conv3_feature_maps.png)
  
	![conv4_feature_maps](conv4_feature_maps.png)

- Grad-CAM (conv4 target)
  
	![grad_cam](grad_cam.png)

- Weights: trained parameters saved to cat_dog_cnn_model.pth

## Quick Start
```bash
# (Optional) install deps inside the container
pip install torch torchvision pillow matplotlib

# train, validate, test, and generate plots & CAM
python main.py
```

## Notes & Reflections
- The code was drafted with generative AI, but hands-on tweaking (data loaders, conv stack up to conv4 for analysis, Grad-CAM hooks) was where the learning happened.
- Grad-CAM on conv4 showed attention was not consistently on the animals; performance felt modest, pointing to architectural and data improvements.
- Running in the Intel GPU devcontainer highlighted environment setup as a learning point (device selection via `torch.xpu`, verifying availability, and pip installs inside the container lifecycle).

## How to Read the Plots
- `training_history.png`: convergence and generalization gap; look for divergence between train/val.
- `grad_cam.png`: red regions show where conv4 influences the predicted class most.
- `conv*_filters.png` / `conv*_feature_maps.png`: early layers capture edges/colors; deeper layers capture shapes/parts—useful to sanity-check learning quality.

## Environment
- Container: Ubuntu 22.04 (devcontainer) on Intel GPU-capable stack. Pip installs are fine during a live session; reinstall if the container is rebuilt. Validating `torch.xpu.is_available()` is part of the workflow.

## Credits
- Dataset: folder-based cats vs dogs under `data/`.
- Code: custom PyTorch pipeline with SafeImageFolder, deeper CNN, and Grad-CAM utility.

## TODO / Next Steps
- Revisit CNN design after studying receptive fields, effective stride, feature hierarchy, regularization (aug/label smoothing/mixup-cutmix), schedulers, and CAM variants (Grad-CAM++, Score-CAM) plus attribution caveats.
- Re-implement from scratch post-study; run controlled experiments beyond a plain CNN: depthwise/DSC blocks, residual links, and transfer baselines (ResNet18/34, MobileNetV3, EfficientNet-B0). Compare CAM quality & accuracy.
- Add evaluation hygiene: stratified split with seeds, sanity-check subset, confusion matrix, per-class precision/recall, ROC-AUC, calibration, checkpointing best models, and run logging (TensorBoard/W&B).