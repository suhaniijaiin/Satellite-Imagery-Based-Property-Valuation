# 🏠 Property Price Prediction (Tabular + Satellite Images)

This repository contains the implementation of a **multimodal property price prediction system** that combines **tabular data** with **satellite imagery** using a **residual learning approach**.
Detailed analysis and results are documented in `23411039_report.pdf`.

---

## 📂 Repository Structure

```text
.
├── train.csv
├── test2.csv
├── 23411039_final.csv
├── 23411039_report.pdf
│
├── data_fetcher.ipynb
├── preprocessing.ipynb
├── model training.ipynb
├── predicting_on_test_dataset_submission.ipynb
│
├── lgb_model.pkl
├── image_residual_model.pkl
│
├── README.md
```

---

## 📁 Notebook Overview

### `data_fetcher.ipynb`

* Uses the **Mapbox Static Images API** to download satellite images
* Fetches images based on property location coordinates
* Stores images with filenames mapped to property IDs

---

### `preprocessing.ipynb`

* Loads and cleans data from `train.csv`
* Performs feature engineering on tabular features
* Prepares datasets for model training and validation

---

### `model training.ipynb`

* Trains a **LightGBM model** using data from `train.csv`
* Computes residuals from tabular predictions
* Extracts image embeddings using a pretrained **ResNet**
* Trains an **ElasticNet** model to predict residual corrections
* Saves trained models as:

  * `lgb_model.pkl`
  * `image_residual_model.pkl`

---

### `predicting_on_test_dataset_submission.ipynb`

* Loads trained `.pkl` models
* Generates predictions on `test2.csv`
* Saves final predictions to `23411039_final.csv`

---

## ⚙️ How to Run

1. Clone the repository
2. Install required Python dependencies
3. Run notebooks in the following order:

   1. `data_fetcher.ipynb`
   2. `preprocessing.ipynb`
   3. `model training.ipynb`
   4. `predicting_on_test_dataset_submission.ipynb`

---

## 🛠️ Tech Stack

* Python
* Pandas, NumPy
* LightGBM
* Scikit-learn
* PyTorch, torchvision
* Mapbox API

---

## 📌 Notes

* Models are trained using **train.csv**
* Test predictions are generated on **test2.csv**
* Final predictions are saved in **23411039_final.csv**
* Trained models are stored as `.pkl` files for reuse

---

## 👤 By:

**Suhani Jain**


