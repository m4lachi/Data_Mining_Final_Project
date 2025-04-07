# **Breast Cancer Classification Project: Random Forest, SVM, and LSTM Models**
This project demonstrates the use of three different classification models to predict whether breast cancer is malignant or benign, using the Breast Cancer Wisconsin (Diagnostic) dataset. The models implemented include:
- Random Forest Classifier
- Support Vector Machine (SVM)
- Long Short-Term Memory (LSTM)

The project applies 10-fold cross-validation to evaluate the performance of each model and calculates various classification metrics including True Positives (TP), True Negatives (TN), False Positives (FP), False Negatives (FN), and Accuracy.

# **Project Setup**
## Prerequisites
Ensure the following Python packages are installed before running the code:
- `pandas`
- `numpy`
- `scikit-learn`
- `tensorflow`
- `keras`

You can install the required libraries via pip:
```bash
pip install pandas numpy scikit-learn tensorflow keras
```

## Data
This project uses the Breast Cancer Wisconsin (Diagnostic) dataset, which can be found in CSV format. The file should be named `data.csv` and placed in the same directory as the Python script.

## Data Preprocessing
The dataset undergoes the following preprocessing steps:
1. **Dropping unnecessary columns:** The id and Unnamed: 32 columns are removed.
2. **Label Encoding:** The target column diagnosis (which contains 'M' for malignant and 'B' for benign) is label-encoded to 1 (malignant) and 0 (benign).
3. **Feature Scaling:** All features are scaled using StandardScaler for optimal performance.
4. **Reshaping for LSTM:** The features are reshaped into sequences to make them compatible with the LSTM model.

## Model Implementation
1. **Random Forest Classifier:** Uses the 'RandomForestClassifier' from 'scikit-learn' with 100 estimators.
2. **Support Vector Machine (SVM):** Uses `SVC` with a linear kernel.
3. **Long Short-Term Memory (LSTM):** A sequential model built using Keras. It includes:
- LSTM layer with 64 units
- Dropout for regularization
- Dense output layer with sigmoid activation

## Cross-Validation
The models are evaluated using **Stratified K-Fold Cross-Validation** (with 10 splits), which ensures that each fold has the same distribution of the target classes (malignant vs benign). For each fold, the following metrics are calculated:
- **True Positives (TP)**
- **True Negatives (TN)**
- **False Positives (FP)**
- **False Negatives (FN)**
- **Accuracy:** The overall classification accuracy for each fold.

## Metrics Extraction
A helper function `extract_metrics` is used to compute confusion matrix values and calculate accuracy for each model and fold.

## Results
The results of each fold for each model are stored in a dictionary and later converted to a pandas DataFrame. The final DataFrame is saved as `final_term_results.csv`. The results are displayed as a pivot table, showing the accuracy for each model across the 10 folds.

## Code Structure
The code is divided into the following sections:
1. **Data Loading and Preprocessing:** Data is read, cleaned, and scaled.
2. **Model Training and Evaluation:** The three models (Random Forest, SVM, LSTM) are trained on the data and evaluated using cross-validation.
3. **Results Saving and Display:** The classification metrics are saved to a CSV file and printed as a pivot table.

## Running the Code
1. Download the **Breast Cancer Wisconsin (Diagnostic) dataset** and save it as `data.csv` in the same directory.
2. Execute the script. The models will train on the dataset, and after the 10-fold cross-validation process, the results will be saved in 'final_term_results.csv'.
3. Review the results in the terminal or open the CSV file to analyze the performance metrics.

## Example Output
After running the script, the results will be printed in the following format:
```plaintext
Model         Fold 1   Fold 2   Fold 3   ...  Average
Random Forest 0.98     0.97     0.99     ...  0.98
SVM           0.96     0.95     0.97     ...  0.96
LSTM          0.94     0.93     0.95     ...  0.94
```

## Results Interpretation
- **Accuracy** values represent the overall percentage of correct classifications for each fold.
- **Random Forest** is expected to perform better due to its robust ensemble nature.
- **SVM** performs well but may require tuning for better results.
- **LSTM** might not perform as well due to the non-sequential nature of the data.

## Limitations
- **Small Dataset:** The relatively small size of the dataset may affect the performance of deep learning models like LSTM.
- **Artificial Sequence for LSTM:** The data was reshaped into sequences for LSTM, which may not have been ideal since the dataset isn’t sequential in nature.
- **Binary Classification:** This dataset only supports binary classification (malignant vs benign).

# Conclusion
This project demonstrates the application of machine learning and deep learning models to solve a real-world binary classification problem. Random Forest and SVM models performed comparably well, while the LSTM model showed limited success due to the data format.
