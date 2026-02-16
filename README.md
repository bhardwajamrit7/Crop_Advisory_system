# 🌾 Crop Advisory System | Machine Learning Project

A complete end-to-end Machine Learning based Crop Recommendation System that predicts the most suitable crop based on soil nutrients and climatic conditions.

This project demonstrates a full ML pipeline including preprocessing, model comparison, hyperparameter tuning, cross-validation, interpretability, and probability-based recommendation.

---

## 📌 Problem Statement

Selecting the right crop based on soil and environmental conditions is critical for maximizing agricultural productivity. 

This project builds a **Multi-Class Classification Model** that recommends crops based on:

- Nitrogen (N)
- Phosphorus (P)
- Potassium (K)
- Temperature
- Humidity
- pH
- Rainfall

Instead of predicting only one crop, the system provides **Top 3 recommendations with probability confidence levels** to assist better decision-making.

---

## 🧠 Machine Learning Approach

### 1️⃣ Data Preprocessing
- Data exploration & validation
- Label Encoding of target variable
- Train-Test Split
- Feature Scaling using StandardScaler

### 2️⃣ Model Training & Benchmarking

The following classification models were trained and compared:

- Logistic Regression
- Decision Tree
- Random Forest
- Support Vector Machine (SVM)
- AdaBoost
- Gradient Boosting

Each model was evaluated using test accuracy and performance metrics.

---

### 3️⃣ Hyperparameter Optimization

- GridSearchCV used for tuning
- 5-Fold Cross Validation applied during tuning
- Best performing model selected based on validation accuracy

---

## 🏆 Final Selected Model

### Random Forest Classifier

Selected because:

- Highest Test Accuracy
- Stable Cross-Validation Performance
- Low Variance across folds
- Strong generalization capability

---

## 📊 Cross-Validation Performance

Stratified 5-Fold Cross Validation was used to ensure balanced class distribution across folds.

- Mean CV Accuracy ≈ 99%
- Very Low Standard Deviation
- Indicates high stability and reliability

Why StratifiedKFold?
Since this is a multi-class classification problem (22 crop classes), stratified splitting ensures fair evaluation by preserving class proportions.

---

## 🌱 Top 3 Crop Recommendation System

Instead of returning a single prediction, the system:

- Computes probability distribution using `predict_proba()`
- Selects Top 3 crops
- Displays probability score
- Assigns confidence level (Very High / High / Moderate / Low)

### Example Output

| Rank | Crop Name | Probability | Confidence (%) | Confidence Level |
|------|-----------|------------|---------------|------------------|
| 1 | Rice | 0.99 | 99% | Very High |
| 2 | Jute | 0.01 | 1% | Low |
| 3 | Pomegranate | 0.00 | 0% | Low |

This makes the system more practical and decision-support oriented.

---

## 📈 Model Evaluation Metrics Used

- Accuracy Score
- Confusion Matrix
- Classification Report
- Feature Importance
- Cross-Validation Mean & Standard Deviation

---

## 🔬 Feature Importance Insight

Random Forest feature importance analysis revealed:

- Rainfall
- Humidity
- Potassium (K)

as the most influential factors in crop prediction.

This improves interpretability of the model.

---

## 🛠 Tech Stack

- Python
- Pandas
- NumPy
- Scikit-Learn
- Matplotlib
- Seaborn
- Jupyter Notebook

---

## 📂 Project Structure


Crop_Advisory_System/
│
├── CropAdvisory.ipynb
├── Crop_Model_Tuning.ipynb
├── Crop_recommendation.csv
├── README.md

