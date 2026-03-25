# Automatic Calorie and Macronutrient Estimation from RGB-D Images

This repository contains the final project for the **Machine Learning 2026 Course at Skoltech**. We developed an end-to-end machine learning pipeline to estimate the nutritional content of meals using the Nutrition5k dataset. Our work progresses from classical ML baselines to advanced multimodal multi-task deep learning architectures.

## Project Overview

Visual calorie estimation is a difficult task due to the high variance in portion sizes and the presence of hidden ingredients. We used the **Nutrition5k** dataset, which provides overhead RGB images, depth maps, and laboratory-accurate nutritional data. Our final cleaned dataset includes 2892 synchronized meals.

### Key Components
* **Data Cleaning:** A pipeline that filters physical inconsistencies, such as energy density exceeding 9 kcal/g.
* **Baseline Tournament:** Evaluation of ResNet50, EfficientNet-B0, and ViT embeddings with Ridge, Random Forest, and XGBoost regressors.
* **Deep Learning Fine-Tuning:** Comparative study of EfficientNet-B0 and B4 architectures using frozen and unfrozen backbone strategies.
* **Multimodal Integration:** 4-channel input processing (RGB + Depth) to capture the 3D volume of food.
* **Multi-task Learning:** Simultaneous prediction of Calories, Protein, Fat, and Carbohydrates using a custom Energy-Weighted Loss function.

## Dataset

We utilized the **Nutrition5k** dataset by Google Research. The imagery was processed as follows:
* **RGB:** Resized to 224x224 (B0) and 380x380 (B4).
* **Depth:** Normalized and clipped at 400 mm to focus on the dish volume.
* **Synchronization:** Only dishes with complete metadata and both image modalities were used.

## Experimental Results

Our results show that full backbone training and the addition of depth data significantly improve accuracy.

| Project Stage | Input Data | MAE (kcal) | R2 Score |
| :--- | :--- | :--- | :--- |
| 1. Baseline (ResNet50 + XGBoost) | RGB | 75.67 | 0.70 |
| 2. Fine-Tuning (Frozen Body) | RGB | 97.71 | 0.39 |
| 3. Fine-Tuning (Full Unfreeze) | RGB | 65.79 | 0.71 |
| **4. Bonus Model (Multi-task)** | **RGB + Depth** | **49.83** | **0.83** |

### Model Scaling
We also tested EfficientNet-B4. In the fully unfrozen RGB setup, B4 achieved an MAE of 63.49 kcal, outperforming the B0 model.

## Error Analysis

A visual audit of the largest errors revealed three main challenges:
1. **Visual Occlusion:** High-calorie items hidden under low-calorie garnishes.
2. **Invisible Ingredients:** Difficulty in detecting calorie-dense oils and butter.
3. **Regression to the Mean:** The model tends to underestimate extreme high-calorie portions (outliers).

## System Requirements

The project was implemented in **Google Colab** using an **NVIDIA Tesla T4 GPU**.

### Core Libraries
* `torch`, `torchvision`, `timm`
* `scikit-learn`, `xgboost`
* `pandas`, `numpy`, `matplotlib`, `seaborn`

## Team Members

* **Ruslan Konurin:** Literature review, data cleaning, B4 scaling, and error analysis.
* **Ivan Gryakalov:** Baseline ML tournament, multi-task framework design, and data synchronization.
* **Magomedrashad Ismailov:** Training pipeline implementation, custom loss functions, and EDA.

## License
This project is for educational purposes as part of the Skoltech Machine Learning course.
