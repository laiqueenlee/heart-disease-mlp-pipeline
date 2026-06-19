# Heart Disease Computational Diagnostic System (MLP Pipeline)

An end-to-end Computational Intelligence workflow utilizing an artificial feedforward **Multi-Layer Perceptron (MLP)** Neural Network to classify and evaluate pathological risk factors for cardiac disease. This system is trained on the classic clinical data matrix from the **UCI Cleveland Heart Disease dataset**, achieving a optimized testing performance accuracy of **86.67%**.

---

## 🚀 System Architecture Overview

The network architecture is structured sequentially to compute risk vectors through backpropagation over dense layers:

* **Input Dimensions:** 13 clinical quantitative/qualitative vectors (Age, Serum Cholesterol, Chest Pain category, etc.)
* **Hidden Layer 1:** 16 Neurons utilizing **ReLU** activation functions.
* **Hidden Layer 2:** 8 Neurons utilizing **ReLU** activation functions.
* **Output Layer:** 1 Neuron utilizing a **Sigmoid** compression function to output a clean binary risk factor probability scale ($0.0 \rightarrow 1.0$).

---

## 📦 Project Workspace Organization

```text
heart_disease_solution/
└── src/
    ├── vhdp/                  # Isolated localized virtual environment
    ├── train.py               # Phase 1: Data ingestion, pipeline scaling, and network training 
    ├── predict.py             # Phase 2: Live interactive inference interface 
    ├── training_curves.png    # Evaluation loss and accuracy graphs
    ├── heart_disease_model.h5 # Serialized model layer configurations and optimized weights
    └── data_scaler.pkl        # Serialized Z-score standardization normalization mappings

## 🛠️ Step-by-Step Installation & Replicability
Follow this execution loop precisely to configure paths cleanly and prevent package structure mismatches on native Windows systems:

1. Initialize Python Runtime Environment
Ensure you are using a stable production runtime layer (e.g., Python 3.13). (Note: Do not run experimental pre-releases like Python 3.14, as heavy libraries like TensorFlow will throw distribution errors).

2. Configure Localized Virtual Environment
Open a fresh terminal console panel in Visual Studio Code and enter the environment setup sequence:

PowerShell
# Navigate into your source code repository
cd src

# Force build a virtual sandbox explicitly pointing to your Python 3.13 binary path
& "C:\Users\HP\AppData\Local\Programs\Python\Python313\python.exe" -m venv vhdp

# Activate your localized virtual sandbox terminal session
.\vhdp\Scripts\Activate.ps1
Verification: Confirm that a green (vhdp) tag prefix appears prominently on the left-hand boundary of your terminal line prompt before issuing any installation strings.

3. Deploy Integrated Library Requirements
Run the unified package alignment array to load mathematical frameworks into your environment:

PowerShell
pip install tensorflow==2.18.0 scikit-learn pandas numpy matplotlib seaborn joblib

## 🏋️‍♂️ Phase 1: Pipeline Execution & Network Training
To train the structural layer metrics, run the ingestion optimization pipeline script:

PowerShell
python train.py
Convergence Mechanics & Results
The data handler replaces null parameters (?) programmatically, resulting in a clean matrix size of 297 active patient files.

Regularization Circuit Breaker: An EarlyStopping monitoring hook watches validation cross-entropy changes. It dynamically cuts execution loops short at Epoch 19 to mitigate overfitting.

Final Testing Metrics Evaluation Output:

Accuracy: 86.67 %

Precision (Disease Presence): 0.92 (An exceptionally low false-positive rate, crucial for avoiding patient panic).

Balanced F1-Score Matrix: Classes balance effectively with scores reading 0.88 (No Disease) and 0.85 (Disease Presence).

The execution finishes by writing heart_disease_model.h5, data_scaler.pkl, and saving the cross-validation error trend directly to your directory as training_curves.png.

## 🔮 Phase 2: User Diagnostic Deployment (Live Inference)
With the finalized core parameters saved to your local storage, execute the production medical inference command tool console interface:

PowerShell
python predict.py
Operational Console Performance Flow
The application hooks directly into terminal prompts, parsing manual numeric entry values dynamically through your saved standardization rules for real-time risk assessment:

Plaintext
==================================================
   HEART DISEASE COMPUTATIONAL DIAGNOSTIC SYSTEM  
==================================================

✅ Core Model and Feature Scaler loaded successfully.

Please input the patient's quantitative clinical data:
1. Age (29.0 - 90.0): 54.0
2. Sex (0.0 = Female, 1.0 = Male): 1.0
...
13. Thalassemia rating (3.0 = Normal, 6.0 = Fixed, 7.0 = Reversible): 7.0

========================================
             DIAGNOSTIC REPORT             
========================================
Calculated Pathological Risk Factor: 87.42%
Final Status Evaluation: [ 1 ] PRESENCE OF HEART DISEASE
Recommendation: Route immediately to cardiology unit for evaluation.
========================================
🎓 Academic Attributions
Dataset Source: UCI Machine Learning Repository (Cleveland Heart Disease Dataset).

Course Module Allocation: SECJ3563 Computational Intelligence Mini Project Framework.