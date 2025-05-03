# 📱 Smartphone Image Classification with CNNs, Bayesian Optimization, and Ensemble Learning

This project focuses on classifying smartphone images into **"Original"** or **"Fake"** using deep learning models like **ResNet**, **DenseNet**, and **EfficientNet**. It also compares ensemble methods (Softmax Voting, Stacking) to find the best-performing architecture.

---

## 🔍 Visual Overview

### 🔷 Overall Performance
![Overall performance](https://github.com/nishikanta24/Counterfiet-Smartphone-Detection-Using-Deep-Learning/blob/main/pics/Overall%20test%20set%20%20performance%20.png)

### 🔷 Individual Model Confusion Matrices
- **ResNet50:**
  ![ResNet Confusion](https://github.com/nishikanta24/Counterfiet-Smartphone-Detection-Using-Deep-Learning/blob/main/pics/Resnet50%20confusion%20matrix%20(test%20set).png)

- **DenseNet121:**
  ![DenseNet Confusion](https://github.com/nishikanta24/Counterfiet-Smartphone-Detection-Using-Deep-Learning/blob/main/pics/Densenet121%20confusion%20matrix(.png)

- **EfficientNetB0:**
  ![EfficientNet Confusion](https://github.com/nishikanta24/Counterfiet-Smartphone-Detection-Using-Deep-Learning/blob/main/pics/Efficientnet_b0%20confusion%20matrix(Test%20set).png)

### 🔷 Ensemble Model Confusion Matrices
- **Softmax Voting:**
  ![Soft Voting](https://github.com/nishikanta24/Counterfiet-Smartphone-Detection-Using-Deep-Learning/blob/main/pics/Soft%20voting%20confusion%20matrix%20(test%20set).png)

- **Stacking:**
  ![Stacking](https://github.com/nishikanta24/Counterfiet-Smartphone-Detection-Using-Deep-Learning/blob/main/pics/Stacking%20confusion%20matrix%20(Test%20set).png)

### 🔷 Class-wise Performance
- **Fake Class:**
  ![Fake Class](https://github.com/nishikanta24/Counterfiet-Smartphone-Detection-Using-Deep-Learning/blob/main/pics/Test%20Set%20class%20wise%20performance%20(fake).png)

- **Original Class:**
  ![Original Class](https://github.com/nishikanta24/Counterfiet-Smartphone-Detection-Using-Deep-Learning/blob/main/pics/Test%20Set%20class%20wise%20performance%20(original).png)

### 🔷 Sample Predictions
| Prediction Examples |
|---------------------|
| ![](https://github.com/nishikanta24/Counterfiet-Smartphone-Detection-Using-Deep-Learning/blob/main/pics/prediction%201.png) |
| ![](https://github.com/nishikanta24/Counterfiet-Smartphone-Detection-Using-Deep-Learning/blob/main/pics/prediction%202.png) |
| ![](https://github.com/nishikanta24/Counterfiet-Smartphone-Detection-Using-Deep-Learning/blob/main/pics/prediction%205.png) |

---

## 🧠 Objective

To identify fake vs original smartphone images using image classification models, and select the most optimal one via:
- Bayesian Hyperparameter Optimization
- Ensemble Techniques
- Comparative evaluation

---

## 🗂️ Dataset

Images are stored in:
/image/fake
/image/original

Each folder contains real and augmented images (495 each), resized to `128x128` during preprocessing. The dataset is class-imbalanced, so careful model tuning was applied.

---

## ⚙️ Project Workflow

### 1. 📦 Data Preprocessing
- Loaded image data using PyTorch’s custom `Dataset` class
- Resized to `(128, 128)`
- Normalized with `mean=0.5`, `std=0.5`
- Split into train, validation, and test sets

### 2. 🧪 Model Selection via Bayesian Optimization
Trained three CNN architectures separately:
- ✅ **ResNet**
- ✅ **DenseNet**
- ✅ **EfficientNet**

Each model’s:
- Hyperparameters were optimized using **Bayesian Optimization**
  - Tuned learning rate, batch size, optimizers (Adam/SGD)
- Performance was tracked using validation accuracy and loss

---

## 🧬 Ensemble Learning
To check if combining models boosts performance:
- 🔁 **Softmax Voting:** Averaged predictions of the 3 models  
- 🧠 **Stacking:** Meta-learner (Logistic Regression) trained on model predictions

---

## 🏆 Final Model Selection
- **ResNet50 performed best F1 score**
- Trained final ResNet model on full dataset
- Saved the model using `torch.save()`

---

