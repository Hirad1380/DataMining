# 🩺 Diabetes Prediction — Data Mining & Machine Learning Analysis

A comprehensive Data Mining project applying **7 classification algorithms** and **5 clustering algorithms** to predict diabetes from patient health data. Includes full data preprocessing, 20+ visualizations, and evaluation with multiple metrics including ROC curves, confusion matrices, and silhouette scores.

---

## 📊 Dataset

**Diabetes Prediction Dataset** (`diabetes_prediction_dataset.csv`)

| Feature | Description |
|---|---|
| `gender` | Patient gender (Male / Female) |
| `age` | Patient age |
| `hypertension` | Has hypertension (0/1) |
| `heart_disease` | Has heart disease (0/1) |
| `smoking_history` | Smoking status |
| `bmi` | Body Mass Index |
| `HbA1c_level` | Glycated hemoglobin level |
| `blood_glucose_level` | Blood glucose level |
| `diabetes` | **Target** — Diabetic (1) or Not (0) |

---

## 🔁 Pipeline

```
Load CSV
    ↓
Data Cleaning
(remove 'Other' gender, drop duplicates)
    ↓
Exploratory Data Analysis (20+ charts)
    ↓
Feature Engineering
(smoking recategorization, one-hot encoding, label encoding)
    ↓
Standard Scaling
    ↓
Train / Test Split (70% / 30%)
    ↓
Classification Algorithms ──┐
                             ├── Evaluation (Accuracy, F1, ROC, AUC)
Clustering Algorithms ───────┘
```

---

## 🤖 Algorithms Implemented

### 📌 Classification (Supervised)

| Algorithm | Variants |
|---|---|
| **Decision Tree** | ID3 (default), C4.5 (entropy), CART (gini) |
| **SVM** | Default, Linear, RBF, Polynomial, Sigmoid kernels |
| **KNN** | K-Nearest Neighbors |
| **Naive Bayes** | Gaussian Naive Bayes |
| **Random Forest** | Ensemble classifier |

### 📌 Clustering (Unsupervised)

| Algorithm | Notes |
|---|---|
| **K-Means** | n_clusters=2, SSE & Silhouette evaluated |
| **Fuzzy C-Means** | Soft membership, 10 clusters |
| **DBSCAN** | Density-based, eps=0.5 |
| **Hierarchical Agglomerative** | Ward linkage, dendrogram |
| **Hierarchical Divisive** | Ward linkage, Silhouette score |

---

## 📈 Evaluation Metrics

- ✅ Accuracy, Precision, Recall, F1 Score
- ✅ Confusion Matrix (heatmap)
- ✅ ROC Curve + AUC score
- ✅ Classification Report
- ✅ Silhouette Score (clustering)
- ✅ Adjusted Rand Index (clustering)
- ✅ SSE — Sum of Squared Errors (clustering)

---

## 🖼️ Visualizations (20+ Charts)

| Plot | Description |
|---|---|
| Age Distribution | Histogram of patient ages |
| BMI Distribution | KDE + histogram |
| Gender Distribution | Bar chart |
| Diabetes Distribution | Count plot |
| Smoking History | Category distribution |
| BMI vs Diabetes | Boxplot |
| Age vs Diabetes | Boxplot |
| HbA1c vs Diabetes | Boxplot |
| Blood Glucose vs Diabetes | Boxplot |
| Gender vs Diabetes | Grouped count plot |
| Age vs BMI | Scatter colored by diabetes |
| BMI by Gender & Diabetes | Violin + Box plots |
| Correlation Matrix | Full heatmap (21×21) |
| Correlation with Diabetes | Sorted heatmap |
| Pairplot | All features colored by diabetes |
| Clustering Scatter | Age vs BMI by cluster |
| Dendrograms | Hierarchical clustering tree |
| ROC Curves | Per algorithm |

---

## 🛠️ Tech Stack

| Category | Library |
|---|---|
| Data Processing | Pandas, NumPy |
| Visualization | Matplotlib, Seaborn |
| Classification | Scikit-learn |
| Clustering | Scikit-learn, scikit-fuzzy |
| Scaling | StandardScaler |
| Evaluation | Scikit-learn metrics |

---

## 🚀 Getting Started

### Installation
```bash
# 1. Clone the repository
git clone https://github.com/Hirad1380/DataMining.git
cd DataMining

# 2. Install dependencies
pip install pandas matplotlib seaborn scikit-learn scikit-fuzzy scipy
```

### Run
```bash
python main.py
```

> 💡 Most algorithm blocks are commented out for performance. Uncomment the section you want to run and execute the script.

### Dataset
Place `diabetes_prediction_dataset.csv` in the same directory as `main.py`, or update this line in the code:
```python
df = pd.read_csv("diabetes_prediction_dataset.csv")
```

---

## 🗂️ Project Structure

```
DataMining/
│
└── main.py    # Full pipeline — preprocessing, EDA, classification, clustering
```

---

## 👨‍💻 Author

**Hirad Bayat**  
M.Sc. Applied Computer Science — University of Duisburg-Essen  
📧 Bayathirad7@gmail.com  
🔗 LinkedIn: [Hirad Bayat](https://www.linkedin.com/in/hirad-bayat-911480383)  
🐙 GitHub: [Hirad1380](https://github.com/Hirad1380)
