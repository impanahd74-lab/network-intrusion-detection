# NIDS Flask Demo

A lightweight Flask application that showcases a Network Intrusion Detection System (NIDS) workflow.
It walks through uploading a labeled dataset, training multiple baseline classifiers, reviewing their
metrics, selecting the best model automatically, and finally generating predictions for unseen data.

## Features

- CSV upload workflow with 50&nbsp;MB limit and validation
- Automatic preprocessing (one-hot encoding for categorical columns and scaling for numeric columns)
- Training pipelines for Decision Tree, K-Nearest Neighbors, Logistic Regression, and Random Forest
- Metrics computation (accuracy, precision, recall, F1 score, classification report, confusion matrix)
- Persisted metrics (`models/metrics.json`), best model (`models/best_model.joblib`), and last
  prediction summary (`models/last_predict.json`)
- Chart.js visualizations for training metrics and prediction label distribution
- Friendly Jinja2 templates that guide you through training and prediction steps

## Project Structure

```
nids-flask/
├── app.py
├── data/
├── models/
├── static/
│   ├── css/
│   │   └── style.css
│   └── js/
│       └── charts.js
├── templates/
│   ├── base.html
│   ├── index.html
│   └── results.html
├── uploads/
├── README.md
└── requirements.txt
```

- Place optional reference datasets under `data/` (ignored by the app but useful for samples).
- Uploaded CSVs are stored temporarily in `uploads/` to allow model training and prediction.

## Getting Started

1. **Create a virtual environment** (recommended):

   ```bash
   python -m venv .venv
   source .venv/bin/activate
   ```

2. **Install dependencies**:

   ```bash
   pip install -r requirements.txt
   ```

3. **Run the Flask server**:

   ```bash
   export FLASK_APP=app.py
   flask run --app app --debug
   ```

4. **Visit the dashboard** at <http://127.0.0.1:5000/>.

## Usage Notes

- The application assumes **supervised learning** with the **last column in the uploaded CSV as the
  target label**. For prediction uploads, omit this target column; any column matching the stored
  target name will be dropped automatically.
- Both **binary** and **multi-class** labels are supported. During training, a stratified split is
  attempted when multiple classes are available.
- Keep feature columns consistent between training and prediction datasets for reliable inference.
- Metrics and model artifacts are persisted under `models/` so you can restart the app without
  retraining.

## Prediction Summary

When you upload new data for prediction, the app stores:

- Timestamp of prediction
- Total rows processed
- Counts per predicted label (rendered as a doughnut chart)
- A sample (first five rows) of the uploaded features for quick inspection

## Cleaning Up

To remove stored models, metrics, and uploaded CSVs:

```bash
rm -f models/*.joblib models/*.json uploads/*.csv
```

## License

This project is provided as-is for demonstration purposes.
