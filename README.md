# Computer Vision Rock-Paper-Scissors Classification
This repository aims at attempting to use transfer learning models to classify images of hands that are Rock, Paper, or Scissors using the dataset "Rock Paper Scissors SXSW: Hand Gesture Detection" from Kaggle.

https://www.kaggle.com/datasets/adilshamim8/rock-paper-scissors

## Overview 
The 

## Summary of Workdone
### Data

Data:
- Type:
  - Input: RGB images (originally 1000x1000 pixel JPEGS, resized to 224x224 for ResNet50 and EfficientNet, or 299x299 for InceptionV3. 
  - Additional Input: CSV files containing iamge filenames mapped to their corresponding labels/classes (rock, paper, scissors)
- Size:
  - Train Shape: (4610, 10)
  - Test Shape: (204, 10) 
- Instances:

  **Preprocessing and Clean-up**

  **Data Visualization**
The graphs below show the ROC curves of the three models used for the classification project including ResNet50, EfficientNet (EfficientNet-B0), and InceptionV3.

<img width="1189" height="490" alt="download" src="https://github.com/user-attachments/assets/af150b3b-0f5a-49c6-ab61-f945c1a73bda" />

Based on the epochs and the visualization for ResNet50 we can see that the model learns over time but validation accuracy remains moderate around 45-50%, which suggests limited generalization. The loss lowering without a strong, consistent improvement in accuracy couold indicare that the model improves confidence in training data but does not fully translate to validation gains. 

<img width="1189" height="490" alt="download" src="https://github.com/user-attachments/assets/9019a701-4549-42ea-9cfb-63256d29dc2c" />

As we can see the training loss decreases nicely, the validation accuracy does not improve much beyong low fifty percents and stabilizes around 46-50%. This could suggest overfitting or a celingt on the model's ability to generalize for this dataset. 

<img width="1189" height="490" alt="download" src="https://github.com/user-attachments/assets/351ef906-29c9-4402-b3a5-39aeb3d1768e" />

The InceptionV3 model, similar to ResNet50, suggests steady learning with reasonable reduction in loss and improvement in accuracy. The validation accuracy near 50% shows moderate generalization, but there is still room for improvements. 

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


**Performance Comparison**
As we can see from the graph below, the different augmentation techniques used show that applying Random Horizontal Flip augmentation improves AUC significantly, especially for Rock (0.66) conpared to the baseline with no augmentation (0.52). This suggests that the RHF helps the model generalize better, meanwhile Random Rotation and Random Zoom have mixed and lower effects. With some AUC's that are lower than the baseline, which indicate that these augmentations may not suit this task well. 

<img width="857" height="701" alt="download" src="https://github.com/user-attachments/assets/b30d6f7f-11f5-4145-8e34-3f03bb3f1bbc" /> 
