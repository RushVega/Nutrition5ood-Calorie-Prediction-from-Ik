This README is written in a professional, technical style suitable for a high-level university project. It covers all the stages you completed, including the multimodal (RGB-D) and multi-task learning bonuses.

As requested, I have used natural English and avoided all long dashes (—).

Automatic Calorie and Macronutrient Estimation from RGB-D Images

This repository contains the final project for the Machine Learning 2026 Course at Skoltech. The project focuses on estimating the nutritional content of meals using the Nutrition5k dataset. We implemented a complete machine learning pipeline, ranging from classical ML baselines to advanced multimodal multi-task deep learning models.

Project Overview

Visual calorie estimation is a challenging task due to portion variance and hidden ingredients. Our approach utilizes the Nutrition5k dataset, which provides overhead RGB images, depth maps, and laboratory-accurate nutritional data for 5000 real world meals.

Key Features

Data Cleaning: Automated pipeline to filter physical inconsistencies (e.g., energy density exceeding 9 kcal/g).

Baseline Tournament: Comparison of ResNet50, EfficientNet-B0, and ViT embeddings with Ridge, Random Forest, and XGBoost regressors.

Deep Learning Fine-Tuning: End-to-end training of EfficientNet-B0 and B4 architectures.

Multimodal Integration: Use of 4-channel input (RGB + Depth) for improved volume estimation.

Multi-task Learning: Simultaneous prediction of Calories, Protein, Fat, and Carbohydrates using a custom Energy-Weighted Loss function.

Dataset

We used the Nutrition5k dataset (Google Research). Our final cleaned and synchronized subset contains 2892 unique dishes with:

RGB overhead photos.

Raw depth maps (normalized and clipped at 400 mm).

Precise mass and macronutrient labels.

System Requirements

The project was developed and tested in Google Colab using an NVIDIA Tesla T4 GPU.

Libraries

PyTorch

torchvision

timm (PyTorch Image Models)

scikit-learn

pandas, numpy

matplotlib, seaborn

Results Summary

Our experiments show that unfreezing the backbone and adding depth information significantly reduces prediction error.

Stage	Input Modality	MAE (kcal)	R2 Score
Baseline (XGBoost)	RGB	75.67	0.70
Fine-tuning (Frozen)	RGB	97.71	0.39
Fine-tuning (Unfrozen)	RGB	65.79	0.71
Final Bonus Model	RGB + Depth	49.83	0.83
Scaling Insights

We also tested EfficientNet-B4 at a higher resolution (380x380). While B4 showed better feature extraction, it required a smaller batch size (8) due to VRAM limits on the Tesla T4.

Error Analysis

Visual inspection of the largest errors revealed that the model primarily struggles with:

Visual Occlusion: High-calorie items hidden under garnishes or rice.

Invisible Fats: Difficulty in detecting calorie-dense oils and butter that do not change dish volume.

Extreme Outliers: Large portions (over 1200 kcal) where the model tends to predict values closer to the mean.

Team Members

Ruslan Konurin: Data cleaning, B4 scaling experiments, error analysis, and documentation.

Ivan Gryakalov: Baseline ML tournament, multi-task framework design, and data synchronization.

Magomedrashad Ismailov: Training pipeline implementation, custom loss functions, and EDA visualizations.

License

This project is for educational purposes as part of the Skoltech Machine Learning course.
