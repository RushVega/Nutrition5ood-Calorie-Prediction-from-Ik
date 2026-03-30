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

## Getting Started (Installation & Usage)

The project was originally implemented and tested in **Google Colab** using an **NVIDIA Tesla T4 GPU**. Due to the deep learning models (EfficientNet, ViT) and multimodal data used in this pipeline, a CUDA-enabled GPU is highly recommended for reasonable training times.

### Option 1: Run in Google Colab (Recommended & Easiest)
1. Open a new notebook in [Google Colab](https://colab.research.google.com/).
2. Change the runtime to use a GPU (`Runtime` -> `Change runtime type` -> select `T4 GPU`).
3. Install the required non-standard libraries by running the following command in the first cell:
   ```bash
   !pip install timm xgboost
   ```
4. Upload the project script or notebook and run it. 
5. *Authentication:* The script streams imagery directly from a Google Cloud Storage (GCS) bucket. When the `auth.authenticate_user()` cell runs, a popup will appear. Log in with a Google account that has access to the `nutrition5k_dataset` project.

### Option 2: Run Locally (Windows / macOS / Linux)
For local execution, ensure you have **Python 3.9+** and the [Google Cloud CLI](https://cloud.google.com/sdk/docs/install) installed on your system.

**1. Clone the repository:**
```bash
git clone https://github.com/your-username/your-repo-name.git
cd your-repo-name
```

**2. Set up a virtual environment and install dependencies:**
```bash
python -m venv venv
source venv/bin/activate  # On Windows use: venv\Scripts\activate
pip install -r requirements.txt
```

**3. Authenticate with Google Cloud:**
Since the pipeline streams images directly from GCS (to save local disk space), you must authenticate your local environment. Run the following command and log in via your browser:
```bash
gcloud auth application-default login
```

**4. Execute the pipeline:**
```bash
python project_ml_final.py
```

## License
This project is for educational purposes as part of the Skoltech Machine Learning course.
