import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import confusion_matrix

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.utils import set_random_seed
import warnings

warnings.filterwarnings("ignore", category=UserWarning, module="keras.src.layers.rnn.rnn")

# Reproducibility
np.random.seed(42)
set_random_seed(42)

# Load and prepare data
df = pd.read_csv("data.csv")
df.drop(columns=['id', 'Unnamed: 32'], inplace=True)
df['diagnosis'] = LabelEncoder().fit_transform(df['diagnosis'])

X = df.drop(columns=['diagnosis'])
y = df['diagnosis']

# Scale features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Reshape for LSTM
X_lstm = X_scaled.reshape((X_scaled.shape[0], X_scaled.shape[1], 1))

# Stratified K-Fold
skf = StratifiedKFold(n_splits=10, shuffle=True, random_state=42)

# Results
results = {
    'Fold': [],
    'Model': [],
    'TP': [],
    'TN': [],
    'FP': [],
    'FN': [],
    'Accuracy': []
}

def extract_metrics(y_true, y_pred):
    cm = confusion_matrix(y_true, y_pred)
    tn, fp, fn, tp = cm.ravel()
    accuracy = (tp + tn) / (tp + tn + fp + fn)
    return tp, tn, fp, fn, accuracy

# Training loop
fold = 1
for train_idx, test_idx in skf.split(X_scaled, y):
    X_train, X_test = X_scaled[train_idx], X_scaled[test_idx]
    y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]
    
    # Random Forest
    rf = RandomForestClassifier(n_estimators=100, random_state=42)
    rf.fit(X_train, y_train)
    rf_pred = rf.predict(X_test)
    tp, tn, fp, fn, acc = extract_metrics(y_test, rf_pred)
    results['Fold'].append(fold)
    results['Model'].append('Random Forest')
    results['TP'].append(tp)
    results['TN'].append(tn)
    results['FP'].append(fp)
    results['FN'].append(fn)
    results['Accuracy'].append(acc)

    # SVM
    svm = SVC(kernel='linear', random_state=42)
    svm.fit(X_train, y_train)
    svm_pred = svm.predict(X_test)
    tp, tn, fp, fn, acc = extract_metrics(y_test, svm_pred)
    results['Fold'].append(fold)
    results['Model'].append('SVM')
    results['TP'].append(tp)
    results['TN'].append(tn)
    results['FP'].append(fp)
    results['FN'].append(fn)
    results['Accuracy'].append(acc)

    # LSTM
    X_train_lstm = X_lstm[train_idx]
    X_test_lstm = X_lstm[test_idx]
    lstm = Sequential()
    lstm.add(LSTM(64, input_shape=(X_train_lstm.shape[1], 1)))
    lstm.add(Dropout(0.2))
    lstm.add(Dense(1, activation='sigmoid'))
    lstm.compile(loss='binary_crossentropy', optimizer=Adam(0.001), metrics=['accuracy'])
    lstm.fit(X_train_lstm, y_train, validation_split=0.1, epochs=20, batch_size=16, verbose=0,
             callbacks=[EarlyStopping(monitor='val_loss', patience=3, restore_best_weights=True)])
    lstm_pred = (lstm.predict(X_test_lstm) > 0.5).astype(int).flatten()
    tp, tn, fp, fn, acc = extract_metrics(y.iloc[test_idx], lstm_pred)
    results['Fold'].append(fold)
    results['Model'].append('LSTM')
    results['TP'].append(tp)
    results['TN'].append(tn)
    results['FP'].append(fp)
    results['FN'].append(fn)
    results['Accuracy'].append(acc)

    fold += 1

# Save and view results
results_df = pd.DataFrame(results)
results_df.to_csv("final_term_results.csv", index=False)
print(results_df.pivot(index='Fold', columns='Model', values='Accuracy'))
