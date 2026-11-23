# AXD-Fraud: credit card fraud detection with neural networks

My MSc Software Engineering dissertation project at the University of Hertfordshire (2026). It detects fraudulent card transactions with a neural network, explains each prediction with SHAP, and watches for model drift with a Page-Hinkley detector I wrote from scratch. A Flask REST API serves the model.

## Results

The selected model is a neural network. These are its scores on the 20% test set, which keeps the real fraud rate of 0.17%, at the default decision threshold of 0.50:

| Metric | Score |
|---|---|
| F1-score | 0.82 |
| Precision | 0.81 |
| Recall | 0.84 |
| ROC-AUC | 0.97 |
| PR-AUC | 0.86 |

Accuracy is 99.94%, but with only 0.17% fraud that number says little. F1, precision and PR-AUC are the metrics that matter here.

Four models were compared on the same split:

| Model | F1 | Precision | Result |
|---|---|---|---|
| Logistic Regression | 0.11 | 0.06 | Rejected |
| Random Forest | 0.75 | 0.67 | Runner-up |
| XGBoost | 0.53 | 0.38 | Rejected |
| **Neural Network** | **0.82** | **0.81** | **Selected** |

## Dataset

[Credit Card Fraud Detection](https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud) by the Machine Learning Group at ULB: 284,807 transactions, of which 492 are fraud (0.17%). Features V1–V28 are PCA components, plus `Time` and `Amount`.

The dataset isn't included in this repo. Download `creditcard.csv` from Kaggle.

## How it works

1. **Data exploration:** class imbalance, distributions, and feature correlations with fraud.
2. **Preprocessing:** `Time` and `Amount` are scaled with `StandardScaler`. The data is split 80/20 with stratification, so both sets keep the 0.17% fraud rate. SMOTE is then applied to the **training set only**.
3. **Model training:** Logistic Regression, Random Forest, XGBoost, and a neural network (Dense 128 → 64 → 32 → 16 → 1, with dropout and batch normalisation, early stopping and learning-rate reduction).
4. **Model comparison and threshold tuning:** thresholds from 0.10 to 0.90 are tested. The best was 0.25 (F1 0.83), and the API uses it. F1 is 0.82 anywhere from 0.30 to 0.70.
5. **Explainability:** SHAP `DeepExplainer` produces summary plots for the whole model and waterfall plots for individual predictions.
6. **Drift detection:** a Page-Hinkley detector, written from scratch, reads the stream of prediction errors and is designed to raise an alert when the error rate climbs. It raised no alerts on the test set.
7. **Flask API:** serves the model and a drift detector, which reads the predicted fraud probabilities because the true labels aren't known at that point.

## Limitations

- The threshold was chosen on the test set, so the 0.83 score at 0.25 is optimistic. The models were also compared on that test set, with no separate final test set.
- `Time` and `Amount` were scaled before the split, so the scalers saw the test data.
- The printed summary in notebook `06` shows a threshold of 0.35. That text is out of date: the saved threshold is 0.25.

## API

| Method | Endpoint | Returns |
|---|---|---|
| GET | `/` | Status, model name, threshold |
| POST | `/predict` | Fraud probability, prediction and drift alert for one transaction. Body: `{"features": [30 numbers]}` |
| GET | `/all_models` | Scores of all four models |
| GET | `/drift/status` | Drift alerts so far, and `STABLE` or `DRIFT DETECTED` |
| GET | `/model/info` | Model, explainability and drift detector details |
| GET | `/metrics` | Prediction counts |

The API was built for the React dashboard I made for the dissertation. The dashboard code isn't in this repository.

## Run it

The notebooks were written for Google Colab, with the project folder in Google Drive at `MyDrive/FraudDetection`.

1. Put `creditcard.csv` in `MyDrive/FraudDetection/dataset/`.
2. Run the notebooks in order, `01` to `06`. They save the processed data, models and charts to Drive.
3. To expose the API from Colab, add your own ngrok token as a Colab Secret named `NGROK_AUTHTOKEN`, then run `07_flask_api.ipynb`.

| Notebook | What it does |
|---|---|
| `01_data_exploration.ipynb` | Explore the data and the class imbalance |
| `02_preprocessing.ipynb` | Scale, split, and apply SMOTE to the training set |
| `03_model_training.ipynb` | Train the four models |
| `04_model_comparison.ipynb` | Compare the models and tune the threshold |
| `05_shap_explainability.ipynb` | Explain predictions with SHAP |
| `06_drift_detection_and_summary.ipynb` | Page-Hinkley drift detection and final summary |
| `07_flask_api.ipynb` | Flask REST API |

## Stack

Python · TensorFlow / Keras · scikit-learn · XGBoost · imbalanced-learn (SMOTE) · SHAP · Flask · pandas · NumPy · Matplotlib

## Author

Shashintha Edirisooriya, Full Stack Developer (AI & LLM Applications), London. [linkedin.com/in/shashinthael](https://www.linkedin.com/in/shashinthael)
