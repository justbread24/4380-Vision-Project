# Computer Vision Rock-Paper-Scissors Classification
This repository aims at attempting to use transfer learning models to classify images of hands that are Rock, Paper, or Scissors using the dataset "Rock Paper Scissors SXSW: Hand Gesture Detection" from Kaggle.

https://www.kaggle.com/datasets/adilshamim8/rock-paper-scissors

## Overview 
The assignment for this project is to categorize hand gesture images into a rock, paper, or scissors class. The problem is to create models that can effectively differentiate between these classes based on a moderately large dataset of labeled RGB images. Because the dataset is relatively small and we require good generalization, the project takes advantage of transfer learning with state-of-the-art convolutional neural networks that have been pretrained on ImageNet.

Our method casts this as a multi-class image classification task, with three state-of-the-art deep learning architectures—ResNet50, EfficientNet-B0, and InceptionV3—as the backbone models. Each is fine-tuned by swapping out its last classifier to have three classes for the task. We perform minimal data augmentation (random horizontal flip) on the training images to enhance generalization without adding too much variability. The models are developed and assessed on stratified splits of the data, with learning curves and ROC analyses for monitoring the performance.

The best-performing models achieved validation accuracies in the order of around 75%, indicating an average performance with respect to the classification task. These results show that transfer learning, combined with selective augmentation, offers a start for hand gesture recognition in scenarios where data is limited.

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
  - Total data points: Around 2,892 images of hand gestures 
  - Training: 2,520 images
  - Validation: 372 images
  - 80% training and 20% validation

  **Preprocessing and Clean-up**
  
  The dataset consists of RGB images associates with labels rock, paper, or scissors. The lavels and image file paths are maintained in CSV files, which requires a PyTorchh Dataset class to map filenames to labels because the data is not arranged in class subfolders.
  The images themselves are stored in a common directory, which requires careful management to ensure the CSV files entries match actual image files. Corrupted or missing images where identified and removed to avoid runtime errors. The dataset is split into training and validation sets using stratified splitting to maintain balanced class proportions in both subsets.
Image Transformation:
- Images are resized to match each model's expected input dimensions
  - 224×224 pixels for ResNet50 and EfficientNet-B0
  - 299×299 pixels for InceptionV3
- Random Horizontal Flip Augmentation

Data Loader Setup:
- DataLoaders batch and shuffle the training data (shuffle=True) to provide randomness each epoch, aiding generalization.
- Validation DataLoaders do not shuffle to maintain consistency.
- Multiple worker processes (num_workers=4 or similar) speed up image loading and transformation

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

**Difficulties and resolutions**
- Managing dataset splits without explicit validation set was solved by stratified splitting of original training data with scikit-learn.
- ImageFolder could not be used due to data stored in CSVs; custom Dataset class allowed flexible loading from CSV filename-label mapping.
- InceptionV3's updated torchvision API required handling of auxiliary logits output with combined loss for stable training.
- Missing validation folder caused FileNotFoundError; resolved by adapting code to DataFrame-based loading.
- Incorporating learning curves and ROC evaluation required addition of tracking lists and sklearn code, integrated post-training.


**Performance Comparison**

As we can see from the graph below, the different augmentation techniques used show that applying Random Horizontal Flip augmentation improves AUC significantly, especially for Rock (0.66) conpared to the baseline with no augmentation (0.52). This suggests that the RHF helps the model generalize better, meanwhile Random Rotation and Random Zoom have mixed and lower effects. With some AUC's that are lower than the baseline, which indicate that these augmentations may not suit this task well. 

<img width="857" height="701" alt="download" src="https://github.com/user-attachments/assets/b30d6f7f-11f5-4145-8e34-3f03bb3f1bbc" /> 

<img width="638" height="715" alt="download" src="https://github.com/user-attachments/assets/ab386e43-dec7-42b3-a41c-fe61c44bf15e" />



**Conclusions**

These models don't paticularly achieve high validation accuracy since they are 75% accuracy, which suggests that more tuning and training is needed.
**Future Work**

To build on current results, there are several options that can be explored further:
- Fine-tuning deeper layers: Gradually unfreeze and fine-tune more layers beyond the classifier head to potentially gain better feature adaptation to the specific task.
- Additional and more varied augmentations: Incorporate augmentations such as random rotations, color jitter, random cropping, or mixup to increase dataset diversity and robustness.
- Hyperparameter optimization: Systematic tuning of learning rates, optimizers, batch sizes, and regularization techniques like dropout or weight decay.

### How to reproduce results

**Reproducing Training**

- Clone the repository and ensure project dependencies are installed (see Software Setup below).
- Prepare dataset: Place your images in a directory and obtain CSV files listing image filenames and respective class labels.
- Split your dataset into training and validation sets using stratified splitting (example code provided).
- Run the training scripts for each model (ResNet50, EfficientNet-B0, InceptionV3), which include loading pretrained weights, setting up data loaders with transforms (including random horizontal flip), training loops, and evaluation.
- The scripts save the best checkpoint models alongside learning curves and ROC curve plots for performance visualization.

**Software Setup**
Required Packages
- torch (PyTorch)
- torchvision
- efficientnet_pytorch (for EfficientNet models)
- numpy
- pandas
- Pillow
- scikit-learn
- matplotlib

  **Training**
How to:
- Prepare train and validation datasets as described.
- Run the training script for your chosen model.
- The script handles data loading, applies transforms (random horizontal flip on training set), loads pretrained weights, replaces classifier head, freezes backbone, and trains classifier.

**Training Details**
- Optimizer: Adam with default lr=0.001.
- Loss: CrossEntropyLoss.
- Batch size: 32.
- Number of epochs: 20 (adjustable).
- Validation accuracy and loss tracked each epoch.
- Best model checkpoint saved automatically.
- Learning curves plotted post training.

**Performance Evaluation**
- Post training, run the evaluation script or section that loads validation data and saved model checkpoints.
- Calculate standard metrics including accuracy, per-class precision/recall if desired.
- Generate and visualize ROC curves for each model on validation data to compare performance comprehensively.
- ROC curve plotting code supports multi-class one-vs-rest approach for detailed assessment.
- Use saved plots and metrics to compare and select the best model for deployment or further development.
