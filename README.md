# Recyclable Waste Classification

Image classification of four recyclable drink-container materials (aluminium cans, glass,
HDPE milk bottles, PET bottles) on a deliberately imbalanced dataset, comparing a CNN trained
from scratch with transfer learning on a frozen ImageNet-pretrained ResNet18 using PyTorch.
The rare HDPE class is then strengthened with 150 new photos from a webcam and phone, plus
data augmentation.

## Highlights

- Stratified train / devtest / test split with early stopping on the devtest set
- Custom `Conv → MaxPool → ReLU` CNN with a dropout ablation study
- Frozen ResNet18 backbone with a new classification head (2,052 trainable parameters)
- Error analysis with confusion matrices, macro-F1 and per-class recall, not just accuracy
- Minority-class data collection (live OpenCV webcam capture plus phone photos, including `.HEIC`)
- Class balancing through oversampling with on-the-fly augmentation, then fine-tuning

## Results (test set, 620 images)

| Model | Accuracy | Macro-F1 | HDPE recall |
|---|---|---|---|
| Custom CNN (from scratch, dropout 0.5) | 82.4% | 69.4% | 20% (4/20) |
| ResNet18, frozen backbone | 92.4% | 90.1% | 80% (16/20) |
| ResNet18, fine-tuned on balanced + augmented data | **93.4%** | **91.4%** | **95% (19/20)** |

An earlier attempt that unfroze the whole ResNet18 backbone caused catastrophic forgetting:
HDPE recall dropped to 10%. Keeping the backbone frozen and retraining only the head on the
rebalanced data gave the stable improvement shown above. The test set holds only 20 HDPE
images, so each image moves HDPE recall by 5 points.

## Contents

- `notebook.ipynb`: the full pipeline with saved outputs, plots and written analysis
- `dataset.zip`: the 150 self-collected HDPE images (9 from a webcam, 141 from a phone)

## Data

The base images come from the Kaggle
[Drinking Waste Classification](https://www.kaggle.com/datasets/arkadiyhacks/drinking-waste-classification)
dataset. To re-run the notebook, place the dataset at `Images_of_Waste/rawimgs/` next to the
notebook and extract `dataset.zip` into `Images_of_Waste/rawimgs/HDPEM/`. The notebook also
reads a `waste_dataset.csv` file (`path,label` columns) that lists the 3,100-image subset.
That file isn't included here.

## Tech stack

Python, PyTorch, torchvision, OpenCV, scikit-learn, pandas, matplotlib, seaborn.
Training ran on Apple Silicon (MPS) and falls back to CUDA or the CPU automatically.
