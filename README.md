# Computer Vision Rock-Paper-Scissors Classification
This repository aims at attempting to use transfer learning models to classify images of hands that are Rock, Paper, or Scissors using the dataset "Rock Paper Scissors SXSW: Hand Gesture Detection" from Kaggle.

https://www.kaggle.com/datasets/adilshamim8/rock-paper-scissors

## Overview 
The 

## Summary of Workdone
### Data

Data:
- Type:
  - Input:
  - Input:
- Size:
- Instances:

  **Preprocessing and Clean-up**

  **Data Visualization**


  ### Problem Formulation
**Input/Output:**
- Input: Single RGB image representing a hand gesture
- Output: One of 3 classes, which are Rock, Paper, or Scissors

**Models**
- ResNet50: Deep residual network enabling training of very deep models; robust and widely adopted in transfer learning.

- EfficientNet-B0: Efficient architecture scaling depth, width, and resolution for balanced accuracy and computational cost.

- InceptionV3: Classic model optimized for multi-scale feature detection with auxiliary classifier branch for regularization.

These models were chosen for their:
- Proven transfer learning success on image classification tasks
- Availability of pretrained ImageNet weights enabling fast convergence
- Moderate complexity to balance accuracy and training time

**Loss, Optimizer, Hyperparameters**
- Loss: CrossEntropyLoss (multi-class classification).
- Optimizer: Adam optimizer with learning rate 0.001.
- Batch size: 32.
- Epochs: 20.
- Data Augmentation: Random Horizontal Flip (probability 0.5) applied to training set only.
- Image Normalization: Using ImageNet channel means and standard deviations.
- Input Image Size: 224×224 for ResNet50 and EfficientNet; 299×299 for InceptionV3 (required by architecture).

### Training
**Software**
- Framework: PyTorch (current stable release).
- Libraries: torchvision, efficientnet_pytorch (for EfficientNet), PIL, scikit-learn (for evaluation).

**Training Duration**
- Approximate time per epoch depends on dataset size and GPU power; expect a few minutes per epoch on mid-tier GPU.
- Total training: 20 epochs recommended to ensure convergence.

**Training Curves**
- Training loss decreased consistently over epochs indicating effective learning.
- Validation accuracy improved to plateau, showing good generalization without overfitting.
- Learning curves plotted with Matplotlib demonstrated model progress visually.

ADD GRAPHS HERE

**Difficulties and resolutions**
- Managing dataset splits without explicit validation set was solved by stratified splitting of original training data with scikit-learn.
- ImageFolder could not be used due to data stored in CSVs; custom Dataset class allowed flexible loading from CSV filename-label mapping.
- InceptionV3's updated torchvision API required handling of auxiliary logits output with combined loss for stable training.
- Missing validation folder caused FileNotFoundError; resolved by adapting code to DataFrame-based loading.
- Incorporating learning curves and ROC evaluation required addition of tracking lists and sklearn code, integrated post-training.
