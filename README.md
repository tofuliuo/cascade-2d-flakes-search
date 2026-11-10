# Cascade 2D Flake Search

This project identifies and analyzes two-dimensional material flakes from paired bright-field (BF) and dark-field (DF) microscope images. The pipeline has three stages:

1. **Stage 1** separates flake foreground from the substrate background.
2. **Stage 2** classifies foreground pixels as single-layer, contaminant, or multilayer.
3. **Stage 3** combines the segmentation result with the paired DF image for post-processing.

## 3. Execution Environments of Microscopy Image Analysis Programs

### 3.1 Hardware and software environments

All commands in this section should be run from the repository root. An NVIDIA GPU is recommended for training and inference. The commands below install the CUDA 11.8-compatible (`cu118`) build of PyTorch for GPU acceleration.

```bash
conda create -n seg python=3.12.3 -y
conda activate seg

pip install torch==2.6.0 torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
pip install numpy opencv-python pillow matplotlib tqdm scikit-learn
pip install segmentation-models-pytorch
pip install timm albumentations pycocotools openpyxl scikit-image
pip install albumentations
```

Users who wish to run evaluation or inference without training the models can download the complete checkpoint package from Google Drive:

- Google Drive: [Download the complete checkpoint package](https://drive.google.com/drive/folders/1kUTgH-X7X5nQ6GWLQt8ce-uZ3DAcOLuK?usp=sharing)

After downloading:

1. open `DL weights`.
2. Select every folder inside `DL weights`; do not copy the outer `DL weights` itself.
3. Paste the selected folders into the repository root.
4. Allow folders with identical names to merge. If the system reports that files already exist, choose **Skip these files**.

This directory structure places the main-model, comparison-model, ablation-study checkpoints at the paths expected by the corresponding programs.

### 3.2 Quick start: Simply running programs for function verification

The complete pipeline is provided in the [single-sample cascade directory](https://github.com/tofuliuo/cascade-2d-flakes-search/tree/main/cascade%20code%20for%20single%20sample).

1. Put BF images in `cascade code for single sample/BF and DF images for tests/BF/`.
2. Put the matching DF images in `cascade code for single sample/BF and DF images for tests/DF/`.
3. Keep each BF/DF pair under the same base filename, for example `BF/1.jpeg` and `DF/1.jpeg`.

The BF/DF test images in these folders come from `image datasets/test dataset/`; when needed, copy the required images from its `BF/` and `DF/` subfolders into the corresponding test folders above.

4. Run:

```bash
python "cascade code for single sample/cascade_3stages_test.py"
```

Results are written under `cascade code for single sample/output/`. The included sample folders already contain matching BF/DF pairs, so this command can be used immediately after the checkpoint folders are copied into the repository.

## 4. Manual Annotation of Microscopy Image Files

The BF microscopy images used for model training and validation, together with paired BF/DF examples for Stage 3, are included in the repository. The BF images were manually annotated into single-layer, contaminant, and multilayer regions.

The corresponding COCO-format training and validation annotations are stored under `image datasets/annotation COCO dataset/`:

```text
image datasets/
|-- training dataset/                 # Bright-field training images
|-- validation dataset/               # Bright-field validation images
|-- test dataset/
|   |-- BF/                           # Bright-field test images
|   `-- DF/                           # Dark-field test images
|-- manual annotations/
|   |-- training/                      # LabelMe polygon annotations
|   `-- validation/                    # LabelMe polygon annotations
`-- annotation COCO dataset/
    |-- instances_train2019.json
    `-- instances_val2019.json
```

The training code reads COCO annotations from `image datasets/annotation COCO dataset/`. To regenerate both files from the bundled LabelMe polygon annotations in `image datasets/manual annotations/training/` and `validation/`, run:

```bash
python "COCO transform code/coco.py"
```

The command writes `instances_train2019.json` and `instances_val2019.json` directly to `image datasets/annotation COCO dataset/`. LabelMe shapes must use the labels `single layer`, `contaminants`, and `multilayer`; the corresponding images must be present in `image datasets/training dataset/` and `validation dataset/`. 


## 5. Deep Learning Programs for BF-image Analysis (stage 1 and 2)

### 5.1 Default settings for training deep learning models

Our models use the same optimization settings for Stage 1 and Stage 2. The shared training parameters are summarized below.

Our Stage 1 and Stage 2 segmentation models use a U-Net-style encoder-decoder architecture with skip connections, in which ConvNeXt-Tiny serves as the encoder backbone.

| Hyperparameter           | Value                                                        |
| ------------------------ | ------------------------------------------------------------ |
| Optimizer                | AdamW (`beta1 = 0.9`, `beta2 = 0.999`)                       |
| Initial learning rate    | `1 x 10^-4`                                                  |
| Weight decay             | `1 x 10^-4`                                                  |
| Batch size               | `4`                                                          |
| Number of epochs         | `50`                                                         |
| Learning rate scheduling | Warm-up for 5 epochs, followed by cosine annealing           |
| Image data augmentation  | Random horizontal/vertical flips, 90-degree rotation, rotation (+/-15 degrees), scaling (+/-10%), brightness/contrast adjustment, Gaussian blur, elastic deformation, and Gaussian noise |



### 5.2 Stage 1: Binary classification for foreground extraction

Stage 1 performs binary segmentation of all flake foreground pixels from the substrate background. The source files are in [`stage1 (binary classification) training codes`](https://github.com/tofuliuo/cascade-2d-flakes-search/tree/main/stage1%20%28binary%20classification%29%20training%20codes).

#### 5.2.1 Training our stage 1 model

```bash
python "stage1 (binary classification) training codes/stage1_training.py"
```

The training program writes the following files:

```text
stage1 (binary classification) training codes/convnext_tiny_stage1_results/
|-- best_convnext_tiny_stage1.pth      # Best validation checkpoint
|-- final_convnext_tiny_stage1.pth     # Final checkpoint
|-- training_history.png               # Training curves
`-- training_history.xlsx              # Curve values
```

#### 5.2.2 Performance evaluation of our stage 1 model

If training is not required, install the downloaded checkpoint package as described in Section 2.2 and run the evaluation program directly. Stage 1 reports foreground metrics only and does not report a separate background row.

```bash
python "stage1 (binary classification) training codes/stage1_evaluation.py"
```

The numerical results are written to `stage1 (binary classification) training codes/convnext_tiny_stage1_results/evaluation_results.txt`.

#### 5.2.3 Performances of other popular deep learning models

Mask R-CNN, MiT-B0, and MiT-B2 were trained and evaluated on the same dataset. Users who install the supplied checkpoint package may omit each training command and run only the corresponding evaluation command.

```bash
# Mask R-CNN
python "stage1 maskrcnn (binary classification) training codes/stage1_maskrcnn_training.py"
python "stage1 maskrcnn (binary classification) training codes/stage1_maskrcnn_evaluation.py"

# MiT-B0
python "stage1 mit-b0 (binary classification) training codes/stage1_mit-b0_training.py"
python "stage1 mit-b0 (binary classification) training codes/stage1_mit-b0_evaluation.py"

# MiT-B2
python "stage1 mit-b2 (binary classification) training codes/stage1_mit-b2_training.py"
python "stage1 mit-b2 (binary classification) training codes/stage1_mit-b2_evaluation.py"
```

Run the evaluation programs above to calculate pixel-wise precision, recall, F1 score, Dice score, and IoU on the validation dataset using the trained model checkpoints.

#### 5.2.4 Validation results of our stage 1 model and other popular deep learning models

After numerical comparison, run the four visualization programs in the order shown below to inspect the masks generated by our model and the three comparison models.

```bash
python "stage1 validation codes/stage1_visualization.py"
python "stage1 maskrcnn validation codes/stage1_maskrcnn_visualization.py"
python "stage1 mit-b0 validation codes/stage1_mit-b0_visualization.py"
python "stage1 mit-b2 validation codes/stage1_mit-b2_visualization.py"
```

The primary Stage 1 visualization results are written under `stage1 validation codes/output/`.

### 5.3 Stage 2: Delicate ternary classification

Stage 2 classifies the Stage 1 foreground into single-layer, contaminant, and multilayer regions. The source files are in [`stage2 (ternary classification) training codes`](https://github.com/tofuliuo/cascade-2d-flakes-search/tree/main/stage2%20%28ternary%20classification%29%20training%20codes).

#### 5.3 Stage 2: Delicate ternary classification

Run Stage 2 training after Stage 1. During training and validation, each Stage 2 program automatically loads its corresponding frozen Stage 1 checkpoint.

```bash
python "stage2 (ternary classification) training codes/stage2_training.py"
```

The training program writes the following files:

```text
stage2 (ternary classification) training codes/convnext_tiny_stage2_results/
|-- best_convnext_tiny_stage2.pth      # Best validation checkpoint
|-- final_convnext_tiny_stage2.pth     # Final checkpoint
|-- training_history.png               # Training curves
`-- training_history.xlsx              # Curve values
```

#### 5.3.2 Performance evaluation of our stage 2 model

If training is not required, install the downloaded checkpoint package and run the evaluation program directly. The evaluator reports the final Single Layer output of the complete Stage 1-Stage 2 cascade: Stage 1 supplies the foreground mask, and Stage 2 removes pixels classified as contaminant or multilayer.

```bash
python "stage2 (ternary classification) training codes/stage2_evaluation.py"
```

The numerical results are written to `stage2 (ternary classification) training codes/convnext_tiny_stage2_results/evaluation_results.txt`.

#### 5.3.3 Performances of other popular deep learning models

The Mask R-CNN, MiT-B0, and MiT-B2 models were trained and evaluated under the same dataset conditions. Each evaluator loads the matching Stage 1 and Stage 2 checkpoints for that model family and reports the final cascaded Single Layer mask. Supplied checkpoints can again be evaluated without training.

```bash
# Mask R-CNN
python "stage2 maskrcnn (ternary classification) training codes/stage2_maskrcnn_training.py"
python "stage2 maskrcnn (ternary classification) training codes/stage2_maskrcnn_evaluation.py"

# MiT-B0
python "stage2 mit-b0 (ternary classification) training codes/stage2_mit-b0_training.py"
python "stage2 mit-b0 (ternary classification) training codes/stage2_mit-b0_evaluation.py"

# MiT-B2
python "stage2 mit-b2 (ternary classification) training codes/stage2_mit-b2_training.py"
python "stage2 mit-b2 (ternary classification) training codes/stage2_mit-b2_evaluation.py"
```

Run the evaluation programs above to calculate per-image precision, recall, F1 score, Dice score, and IoU for the final cascaded Single Layer mask, averaged over the annotated validation images.

#### 5.3.4 Validation results of our stage 2 model and other popular deep learning models

After numerical comparison, run the four visualization programs in the order shown below.

```bash
python "stage2 validation codes/stage2_visualization.py"
python "stage2 maskrcnn validation codes/stage2_maskrcnn_visualization.py"
python "stage2 mit-b0 validation codes/stage2_mit-b0_visualization.py"
python "stage2 mit-b2 validation codes/stage2_mit-b2_visualization.py"
```

The primary Stage 2 visualization loads both main checkpoints and writes its outputs under `stage2 validation codes/output/`.

### 5.4 Comparison of single-stage quaternary classification

The single-stage comparison uses the original-scale ConvNeXt-Tiny U-Net with skip connections to directly predict background, single layer, contaminant, and multilayer, without a Stage 1 foreground mask. 

#### 5.4.1 Training our single-stage quaternary model

```bash
python "single stage (four-class classification) training codes/single_stage_training.py"
```

Training automatically records loss, IoU, and Dice for both training and validation, and writes the following files:

```text
single stage (four-class classification) training codes/convnext_tiny_single_stage_results/
|-- best_convnext_tiny_single_stage.pth   # Best validation checkpoint
|-- final_convnext_tiny_single_stage.pth  # Final checkpoint
|-- training_history.png                # Four panels: loss, IoU, Dice, validation IoU/Dice
`-- training_history.xlsx               # Seven columns: epoch and six training/validation metrics; best metrics summary
```

#### 5.4.2 Performance evaluation of our single-stage quaternary model

```bash
python "single stage (four-class classification) training codes/single_stage_evaluation.py"
```

The evaluator loads the best checkpoint and writes Single Layer IoU, Dice, precision, recall, and F1 to `single stage (four-class classification) training codes/convnext_tiny_single_stage_results/evaluation_results.txt`.

#### 5.4.3 Validation results of our single-stage quaternary model

```bash
python "single stage validation codes/single_stage_visualization.py"
```

Results are written to `single stage validation codes/output/`. Each input image produces `<name>_overlay.png` (segmentation overlay) and `<name>_mask.png` (binary Single Layer mask), both at 512 x 512 pixels.

### 5.5 Ablation study of skip-connections

The ablation models remove the skip connections from the Stage 1 and Stage 2 networks. The supplied checkpoint package includes both ablation checkpoints, so users who do not need to reproduce training may run only the evaluation commands.

Stage 1 ablation training and evaluation:

```bash
python "ablation study codes/ablation_stage1_training.py"
python "ablation study codes/ablation_stage1_evaluation.py"
```

Stage 2 ablation training and evaluation:

```bash
python "ablation study codes/ablation_stage2_training.py"
python "ablation study codes/ablation_stage2_evaluation.py"
```

Stage 1 must be completed before Stage 2 when the ablation models are trained.


## 6. Stage 3: DF-Image Processing Program

Stage 3 is rule-based DF-image post-processing rather than a learned model; therefore, it has no separate training or evaluation script.

Place BF images in `stage3 DF analysis code/dataset/BF/` and matching DF images in `stage3 DF analysis code/dataset/DF/`. Each pair must share the same base filename. The program matches `.jpg`, `.jpeg`, and `.png` files case-insensitively and reports unmatched files.

```bash
python "stage3 DF analysis code/stage3_DF.py"
```

Existing Stage 2 masks are reused per image. If masks are missing for only part of the dataset, the program generates predictions for those pairs only, using the downloaded Stage 1 and Stage 2 checkpoints. It then processes every matched BF/DF pair and writes outputs under `stage3 DF analysis code/output/`.

The complete three-stage pipeline can also be rerun with the Quick Start command:

```bash
python "cascade code for single sample/cascade_3stages_test.py"
```