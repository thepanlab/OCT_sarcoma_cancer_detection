# Sarcoma OCT Classification

Deep learning models for binary (tumor vs. normal) classification of sarcoma tissue from optical coherence tomography (OCT) data. The repository includes both 2D transfer-learning baselines trained on individual OCT-derived image modalities, and a custom 3D CNN trained directly on volumetric data.

## Repository Structure

```
.
├── sarcoma.py              # 2D CNN training/eval across 4 backbones and 4 OCT modalities
├── 3d_cnn_train_sar.py     # Custom 3D CNN trained on volumetric (D, H, W) TIFF stacks
├── labels_intensity.csv    # Filename/label pairs for the intensity modality
├── labels_opticaxis.csv    # Filename/label pairs for the optic axis modality
├── labels_retardation.csv  # Filename/label pairs for the retardation modality
├── labels_dopu.csv         # Filename/label pairs for the DOPU modality
└── README.md
```

## Data

Each `labels_*.csv` contains two columns:

| Column     | Description                                  |
|------------|-----------------------------------------------|
| `filename` | Image filename, e.g. `649_s1_tumor_intensitytumor.png` |
| `Label`    | Binary label — `0` = normal, `1` = tumor      |

Filenames encode the cross-validation fold via an `_sN_` token (`s1`–`s5`), used to construct 5-fold, patient/fold-stratified train/test splits. Each modality (`intensity`, `opticaxis`, `retardation`, `dopu`) has ~12,000 labeled images.

> **Note:** the raw OCT images themselves are not included in this repository. Image paths in the scripts point to local/HPC scratch storage and will need to be updated to your own data location.

## Models

### 2D image classifiers (`sarcoma.py`)

Fine-tunes ImageNet-pretrained backbones as binary classifiers, one modality at a time:

- **ConvNeXt-Base**
- **Swin-B**

Each model is trained with `BCEWithLogitsLoss` and Adam (or SGD based on our choice), evaluated with 5-fold cross-validation (folds `s1`–`s5`), and for each fold saves:
- model checkpoint (`.pt`)
- per-sample predictions with class probabilities (`.csv`, for ROC analysis)
- confusion matrix (`.json`)
- inference/timing breakdown (`.json`, separating pure model inference time from data-loading overhead)

A final `cross_validation_summary.csv` and `statistics.json` aggregate accuracy and timing across all 5 folds.

For InceptionV3, ResNet50, and Octascope, please refer to our group's NACHOS training pipeline (https://github.com/thepanlab/NACHOS). 

**Usage:**
```bash
python sarcoma.py --model convnext --modality intensity
python sarcoma.py --model swin --modality dopu --epochs 20 --lr 1e-5 --batch_size 64
```

| Argument       | Default    | Description                                          |
|----------------|------------|------------------------------------------------------|
| `--model`      | (required) | `convnext` or `swin`                                 |
| `--modality`   | (required) | `intensity`, `opticaxis`, `retardation`, or `dopu`   |
| `--epochs`     | `10`       | Training epochs per fold                             |
| `--lr`         | `1e-4`     | Learning rate                                        |
| `--batch_size` | `32`       | Batch size for training and testing                  |

Update `label_csv`, `image_dir`, and `base_output_dir` in `run_cross_testing()` to match your environment before running.

### 3D CNN (`3d_cnn_train_sar.py`)

A lightweight custom 3D CNN (3 conv blocks + adaptive pooling + 2 FC layers, ~trained from scratch) operating directly on preprocessed volumetric TIFF stacks, with leave-one-subject-out cross-validation.

- Expects data under `<data_dir>/sarcoma_3d_processed/subject_XX/<class_folder>/processed_*.tiff`, where class folder names contain `tumor` or `normal`/`not_tumor`/`non_tumor`.
- Interactively prompts for which subject(s) to hold out for testing (all subjects, a specific list, or all subjects from a given starting point).
- Saves per-subject training history, pickled results, and an aggregated `all_results_processed_sarcoma.json`/`.pickle` summary with mean/min/max test accuracy across subjects.

**Usage:**
```bash
python 3d_cnn_train_sar.py
```
Update the `data_dir` path in `config` (in `main()`) before running.

## Requirements

- Python 3.8+
- PyTorch, torchvision
- pandas, numpy, scikit-learn
- Pillow
- tifffile
- tqdm

Install with:
```bash
pip install torch torchvision pandas numpy scikit-learn pillow tifffile tqdm
```

## Citation

If you use this code, please cite our manuscript: https://opg.optica.org/boe/fulltext.cfm?uri=boe-17-10-5213##
