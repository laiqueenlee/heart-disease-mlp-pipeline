# Heart Disease Computational Diagnostic System (MLP Pipeline)

An end-to-end Computational Intelligence workflow utilizing an artificial feedforward **Multi-Layer Perceptron (MLP)** Neural Network to classify and evaluate pathological risk factors for heart disease. The system is trained using the **UCI Cleveland Heart Disease Dataset**, achieving a testing accuracy of **86.67%**.

---

## 🚀 System Architecture Overview

The neural network architecture consists of the following layers:

| Layer          | Configuration                |
| -------------- | ---------------------------- |
| Input Layer    | 13 clinical features         |
| Hidden Layer 1 | 16 neurons, ReLU activation  |
| Hidden Layer 2 | 8 neurons, ReLU activation   |
| Output Layer   | 1 neuron, Sigmoid activation |

### Input Features

The model utilizes 13 clinical attributes, including:

* Age
* Sex
* Chest Pain Type
* Resting Blood Pressure
* Serum Cholesterol
* Fasting Blood Sugar
* Resting ECG Results
* Maximum Heart Rate Achieved
* Exercise-Induced Angina
* ST Depression (Oldpeak)
* Slope of Peak Exercise ST Segment
* Number of Major Vessels Colored by Fluoroscopy
* Thalassemia Status

The sigmoid output layer generates a probability score between **0.0 and 1.0**, representing the likelihood of heart disease.

---

## 📁 Project Structure

```text
heart_disease_solution/
├── .gitignore
└── src/
    ├── vhdp/                      # Local virtual environment (excluded from Git)
    ├── train.py                   # Data preprocessing and model training
    ├── predict.py                 # Interactive prediction interface
    ├── training_curves.png        # Training and validation graphs
    ├── heart_disease_model.h5     # Trained neural network model
    └── data_scaler.pkl            # Saved feature scaler
```

> **Note:** The `vhdp/` virtual environment contains machine-specific dependencies and is excluded from version control through `.gitignore`.

---

## 🛠️ Installation Guide

### Prerequisites

* Python 3.13 (recommended)
* Visual Studio Code (optional)
* PowerShell (Windows)

> **Important:** TensorFlow currently does not support experimental Python releases such as Python 3.14.

### 1. Create a Virtual Environment

Open a terminal and navigate to the source directory:

```powershell
cd src
```

Create a virtual environment:

```powershell
& "C:\Users\HP\AppData\Local\Programs\Python\Python313\python.exe" -m venv vhdp
```

Activate the environment:

```powershell
.\vhdp\Scripts\Activate.ps1
```

You should see:

```text
(vhdp)
```

at the beginning of your terminal prompt.

---

### 2. Install Required Packages

```powershell
pip install tensorflow scikit-learn pandas numpy matplotlib seaborn joblib
```

---

## 🏋️ Model Training

Run the training pipeline:

```powershell
python train.py
```

### Training Workflow

The training script performs:

1. Data loading and preprocessing
2. Missing value handling
3. Feature scaling using Z-score normalization
4. Neural network training
5. Performance evaluation
6. Model serialization

### Dataset Processing

* Missing values represented by `?` are cleaned automatically.
* Final usable dataset size: **297 patient records**

### Overfitting Prevention

An **EarlyStopping** callback monitors validation loss and stops training when performance no longer improves, preventing overfitting.

### Final Test Performance

| Metric                       | Score  |
| ---------------------------- | ------ |
| Accuracy                     | 86.67% |
| Precision (Disease Presence) | 0.92   |
| F1-Score (No Disease)        | 0.88   |
| F1-Score (Disease Presence)  | 0.85   |

### Generated Files

After successful training, the following files are created:

```text
heart_disease_model.h5
data_scaler.pkl
training_curves.png
```

---

## 🔮 Heart Disease Prediction

After training is complete, launch the prediction interface:

```powershell
python predict.py
```

### Example Console Session

```text
==================================================
   HEART DISEASE COMPUTATIONAL DIAGNOSTIC SYSTEM
==================================================

✅ Core Model and Feature Scaler loaded successfully.

Please input the patient's quantitative clinical data:

1. Age (29.0 - 90.0): 54.0
2. Sex (0.0 = Female, 1.0 = Male): 1.0
3. Chest Pain Type (1.0 = Typical, 2.0 = Atypical,
   3.0 = Non-anginal, 4.0 = Asymptomatic): 4.0
...
13. Thalassemia rating
    (3.0 = Normal, 6.0 = Fixed, 7.0 = Reversible): 7.0

========================================
             DIAGNOSTIC REPORT
========================================
Calculated Pathological Risk Factor: 87.42%
Final Status Evaluation:
[1] PRESENCE OF HEART DISEASE

Recommendation:
Route immediately to cardiology unit for evaluation.
========================================
```

---

## 📊 Output Interpretation

| Probability Score | Interpretation                   |
| ----------------- | -------------------------------- |
| < 0.50            | Low likelihood of heart disease  |
| ≥ 0.50            | High likelihood of heart disease |

The model outputs a probability value indicating the estimated risk of heart disease based on the supplied clinical attributes.

---

## 🎓 Academic Information

### Dataset

**UCI Cleveland Heart Disease Dataset**

* Source: UCI Machine Learning Repository
* Domain: Clinical Heart Disease Classification



---

## 👨‍💻 Authors

Developed as part of the **SECJ3563 Computational Intelligence Mini Project**.
