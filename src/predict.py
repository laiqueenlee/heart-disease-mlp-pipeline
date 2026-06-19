import numpy as np
import tensorflow as tf
import joblib

def get_clean_input(prompt, min_val, max_val):
    """Helper function to ensure user entries fall into logical medical ranges."""
    while True:
        try:
            val = float(input(prompt))
            if min_val <= val <= max_val:
                return val
            print(f"⚠️ Input out of logical bounds ({min_val} - {max_val}). Try again.")
        except ValueError:
            print("⚠️ Invalid numeric character entered. Try again.")

def run_diagnostic_system():
    print("==================================================")
    print("   HEART DISEASE COMPUTATIONAL DIAGNOSTIC SYSTEM  ")
    print("==================================================\n")
    
    try:
        model = tf.keras.models.load_model('heart_disease_model.h5')
        scaler = joblib.load('data_scaler.pkl')
        print("✅ Core Model and Feature Scaler loaded successfully.\n")
    except FileNotFoundError:
        print("❌ Error: Missing system files! Run 'train.py' first to build the model files.")
        return

    print("Please input the patient's quantitative clinical data:")
    age      = get_clean_input("1. Age (29.0 - 90.0): ", 29.0, 90.0)
    sex      = get_clean_input("2. Sex (0.0 = Female, 1.0 = Male): ", 0.0, 1.0)
    cp       = get_clean_input("3. Chest Pain Type (1.0 = Typical, 2.0 = Atypical, 3.0 = Non-anginal, 4.0 = Asymptomatic): ", 1.0, 4.0)
    trestbps = get_clean_input("4. Resting Blood Pressure in mm Hg (80.0 - 200.0): ", 80.0, 200.0)
    chol     = get_clean_input("5. Serum Cholesterol in mg/dl (100.0 - 600.0): ", 100.0, 600.0)
    fbs      = get_clean_input("6. Fasting Blood Sugar > 120 mg/dl (0.0 = False, 1.0 = True): ", 0.0, 1.0)
    restecg  = get_clean_input("7. Resting ECG results (0.0 = Normal, 1.0 = ST-T wave anomaly, 2.0 = Left ventricular hypertrophy): ", 0.0, 2.0)
    thalach  = get_clean_input("8. Maximum Heart Rate Achieved (60.0 - 220.0): ", 60.0, 220.0)
    exang    = get_clean_input("9. Exercise Induced Angina (0.0 = No, 1.0 = Yes): ", 0.0, 1.0)
    oldpeak  = get_clean_input("10. ST depression induced by exercise (0.0 - 6.0): ", 0.0, 6.0)
    slope    = get_clean_input("11. Slope of peak exercise ST segment (1.0 = Upsloping, 2.0 = Flat, 3.0 = Downsloping): ", 1.0, 3.0)
    ca       = get_clean_input("12. Number of major vessels colored by fluoroscopy (0.0 - 3.0): ", 0.0, 3.0)
    thal     = get_clean_input("13. Thalassemia rating (3.0 = Normal, 6.0 = Fixed defect, 7.0 = Reversible defect): ", 3.0, 7.0)

    raw_patient_vector = np.array([[age, sex, cp, trestbps, chol, fbs, restecg, 
                                    thalach, exang, oldpeak, slope, ca, thal]])

    scaled_patient_vector = scaler.transform(raw_patient_vector)

    prediction_probability = model.predict(scaled_patient_vector, verbose=0)[0][0]
    
    print("\n" + "="*40)
    print("             DIAGNOSTIC REPORT             ")
    print("="*40)
    print(f"Calculated Pathological Risk Factor: {round(prediction_probability * 100, 2)}%")
    
    if prediction_probability > 0.5:
        print("Final Status Evaluation: [ 1 ] PRESENCE OF HEART DISEASE")
        print("Recommendation: Route immediately to cardiology unit for evaluation.")
    else:
        print("Final Status Evaluation: [ 0 ] ABSENCE OF HEART DISEASE")
        print("Recommendation: Routine periodic health screening suggested.")
    print("="*40 + "\n")

if __name__ == "__main__":
    run_diagnostic_system()