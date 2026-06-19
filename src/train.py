import pandas as pd
import numpy as np

import matplotlib
matplotlib.use('Agg')  

import matplotlib.pyplot as plt
import seaborn as sns
import tensorflow as tf
import joblib
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.callbacks import EarlyStopping

url = "https://archive.ics.uci.edu/ml/machine-learning-databases/heart-disease/processed.cleveland.data"
columns = ['age','sex','cp','trestbps','chol','fbs','restecg',
           'thalach','exang','oldpeak','slope','ca','thal','target']

df = pd.read_csv(url, names=columns, na_values='?')
df.dropna(inplace=True)

df['target'] = df['target'].apply(lambda x: 1 if x > 0 else 0)

print(f"Dataset Cleaned. Remaining rows: {df.shape[0]}")

X = df.drop('target', axis=1).values
y = df['target'].values

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y, test_size=0.2, random_state=42, stratify=y
)

model = Sequential([
    Dense(16, activation='relu', input_shape=(13,)), 
    Dense(8, activation='relu'),                     
    Dense(1, activation='sigmoid')                   
])

model.compile(
    optimizer='adam',
    loss='binary_crossentropy',
    metrics=['accuracy']
)

early_stop = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)

history = model.fit(
    X_train, y_train,
    epochs=100,
    batch_size=16,
    validation_split=0.2,
    callbacks=[early_stop],
    verbose=1
)

y_pred_prob = model.predict(X_test)
y_pred = (y_pred_prob > 0.5).astype(int).flatten()

print("\n=== FINAL TEST METRICS ===")
print("Accuracy:", round(accuracy_score(y_test, y_pred) * 100, 2), "%")
print("\nClassification Report:\n", classification_report(y_test, y_pred, target_names=['No Disease','Disease']))


plt.figure(figsize=(10, 4))
plt.subplot(1, 2, 1)
plt.plot(history.history['accuracy'], label='Train')
plt.plot(history.history['val_accuracy'], label='Val')
plt.title('Accuracy'); plt.legend()

plt.subplot(1, 2, 2)
plt.plot(history.history['loss'], label='Train')
plt.plot(history.history['val_loss'], label='Val')
plt.title('Loss'); plt.legend()
plt.tight_layout()
plt.savefig('training_curves.png', dpi=150)

model.save('heart_disease_model.h5')
joblib.dump(scaler, 'data_scaler.pkl')
print("\nArtifacts Saved! 'heart_disease_model.h5' and 'data_scaler.pkl' are ready.")