from flask import Flask, request, render_template_string, redirect, url_for
import pandas as pd
import io
import base64
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from fairlearn.metrics import MetricFrame, selection_rate, demographic_parity_difference, equalized_odds_difference
from fairlearn.reductions import ExponentiatedGradient, DemographicParity
import matplotlib.pyplot as plt
import numpy as np

app = Flask(__name__)

UPLOAD_HTML = """
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8" />
    <meta name="viewport" content="width=device-width, initial-scale=1" />
    <title>Bias Detection and Mitigation - Upload Dataset</title>
    <style>
        body {
            font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
            background: #f0f4f8;
            color: #333;
            margin: 0;
            padding: 0 20px 40px 20px;
            max-width: 900px;
            margin-left: auto;
            margin-right: auto;
        }
        header {
            background: linear-gradient(90deg, #4b6cb7 0%, #182848 100%);
            color: white;
            padding: 20px 0;
            text-align: center;
            border-radius: 8px;
            margin-top: 20px;
            margin-bottom: 40px;
            box-shadow: 0 4px 15px rgba(27, 31, 35, 0.15);
        }
        h1 {
            margin: 0;
            font-weight: 700;
            letter-spacing: 0.05em;
        }
        form {
            background: white;
            padding: 25px;
            border-radius: 10px;
            box-shadow: 0 8px 20px rgba(27, 31, 35, 0.1);
            text-align: center;
        }
        input[type=file] {
            margin-bottom: 15px;
        }
        button {
            background-color: #4b6cb7;
            color: white;
            border: none;
            padding: 12px 25px;
            font-size: 16px;
            border-radius: 6px;
            cursor: pointer;
            transition: background-color 0.3s ease;
            font-weight: 600;
        }
        button:hover {
            background-color: #182848;
        }
        .error {
            color: red;
            font-weight: 600;
            margin-top: 20px;
        }
        .footer {
            margin-top: 60px;
            text-align: center;
            color: #666;
            font-size: 14px;
        }
        a {
            color: #4b6cb7;
            text-decoration: none;
            font-weight: 600;
        }
    </style>
</head>
<body>
    <header>
        <h1>Bias Detection and Mitigation in AI Training Data</h1>
        <p>Upload your CSV dataset to begin bias analysis.</p>
    </header>
    <form method="POST" enctype="multipart/form-data" action="{{ url_for('upload') }}">
        <label for="file">Select CSV file:</label><br />
        <input type="file" id="file" name="file" accept=".csv" required /><br />
        <button type="submit">Upload and Select Columns</button>
    </form>
    {% if error %}
        <div class="error">{{ error }}</div>
    {% endif %}
    <div class="footer">
        <p>This demo uses <a href="https://fairlearn.org/" target="_blank" rel="noopener">Fairlearn</a> and <a href="https://scikit-learn.org/" target="_blank" rel="noopener">Scikit-learn</a>.</p>
    </div>
</body>
</html>
"""

SELECT_HTML = """
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8" />
    <meta name="viewport" content="width=device-width, initial-scale=1" />
    <title>Select Columns for Bias Detection</title>
    <style>
        body {
            font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
            background: #f0f4f8;
            color: #333;
            margin: 0;
            padding: 0 20px 40px 20px;
            max-width: 900px;
            margin-left: auto;
            margin-right: auto;
        }
        header {
            background: linear-gradient(90deg, #4b6cb7 0%, #182848 100%);
            color: white;
            padding: 20px 0;
            text-align: center;
            border-radius: 8px;
            margin-top: 20px;
            margin-bottom: 40px;
            box-shadow: 0 4px 15px rgba(27, 31, 35, 0.15);
        }
        h1 {
            margin: 0 0 20px 0;
            font-weight: 700;
            letter-spacing: 0.05em;
        }
        form {
            background: white;
            padding: 25px;
            border-radius: 10px;
            box-shadow: 0 8px 20px rgba(27, 31, 35, 0.1);
        }
        label {
            display: block;
            margin: 15px 0 8px;
            font-weight: 600;
        }
        select {
            width: 100%;
            padding: 10px;
            font-size: 16px;
            border-radius: 6px;
            border: 1px solid #ccc;
        }
        button {
            background-color: #4b6cb7;
            color: white;
            border: none;
            padding: 12px 25px;
            font-size: 16px;
            border-radius: 6px;
            cursor: pointer;
            transition: background-color 0.3s ease;
            font-weight: 600;
            margin-top: 25px;
            width: 100%;
        }
        button:hover {
            background-color: #182848;
        }
        .footer {
            margin-top: 60px;
            text-align: center;
            color: #666;
            font-size: 14px;
        }
    </style>
</head>
<body>
    <header>
        <h1>Select Columns for Bias Detection and Mitigation</h1>
        <p>Choose the sensitive attribute and label (target) columns from your dataset.</p>
    </header>
    <form method="POST" action="{{ url_for('analyze') }}">
        <input type="hidden" name="filedata" value="{{ filedata }}" />
        <label for="sensitive_attr">Sensitive Attribute (Protected Feature):</label>
        <select name="sensitive_attr" id="sensitive_attr" required>
            <option value="" disabled selected>-- Select Sensitive Attribute --</option>
            {% for col in columns %}
            <option value="{{ col }}">{{ col }}</option>
            {% endfor %}
        </select>

        <label for="label_col">Label (Target Variable):</label>
        <select name="label_col" id="label_col" required>
            <option value="" disabled selected>-- Select Label Column --</option>
            {% for col in columns %}
            <option value="{{ col }}">{{ col }}</option>
            {% endfor %}
        </select>

        <button type="submit">Analyze Bias</button>
    </form>
    <div class="footer">
        <p>This demo uses <a href="https://fairlearn.org/" target="_blank" rel="noopener">Fairlearn</a> and <a href="https://scikit-learn.org/" target="_blank" rel="noopener">Scikit-learn</a>.</p>
    </div>
</body>
</html>
"""

RESULTS_HTML = """
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8" />
    <meta name="viewport" content="width=device-width, initial-scale=1" />
    <title>Bias Detection and Mitigation Results</title>
    <style>
        body {
            font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
            background: #f0f4f8;
            color: #333;
            margin: 0;
            padding: 0 20px 40px 20px;
            max-width: 900px;
            margin-left: auto;
            margin-right: auto;
        }
        header {
            background: linear-gradient(90deg, #4b6cb7 0%, #182848 100%);
            color: white;
            padding: 20px 0;
            text-align: center;
            border-radius: 8px;
            margin-top: 20px;
            margin-bottom: 40px;
            box-shadow: 0 4px 15px rgba(27, 31, 35, 0.15);
        }
        h1 {
            margin: 0;
            font-weight: 700;
            letter-spacing: 0.05em;
        }
        h2 {
            color: #4b6cb7;
            font-weight: 600;
            margin-top: 40px;
            margin-bottom: 15px;
        }
        .output-section {
            background: white;
            margin-top: 30px;
            padding: 25px;
            border-radius: 10px;
            box-shadow: 0 8px 20px rgba(27, 31, 35, 0.1);
        }
        .metrics {
            display: flex;
            justify-content: space-around;
            flex-wrap: wrap;
        }
        .metric {
            flex: 1 0 250px;
            margin: 10px;
            background: #e9efff;
            padding: 15px;
            border-radius: 8px;
            box-shadow: inset 0 0 6px #a3b0f4;
            text-align: center;
        }
        canvas {
            margin-top: 10px;
            max-width: 100%;
        }
        .footer {
            margin-top: 60px;
            text-align: center;
            color: #666;
            font-size: 14px;
        }
        a {
            color: #4b6cb7;
            text-decoration: none;
            font-weight: 600;
        }
        img {
            max-width: 100%;
            margin-top: 10px;
        }
    </style>
</head>
<body>
    <header>
        <h1>Bias Detection and Mitigation Results</h1>
        <p>Results based on your selected sensitive attribute and label columns.</p>
    </header>
    <div class="output-section">
        <h2>Bias Detection Metrics</h2>
        <div class="metrics">
            <div class="metric">
                <h3>Demographic Parity Difference</h3>
                <p>{{ results.demographic_parity_difference }}</p>
            </div>
            <div class="metric">
                <h3>Equalized Odds Difference</h3>
                <p>{{ results.equalized_odds_difference }}</p>
            </div>
        </div>

        <h2>Model Accuracy</h2>
        <div class="metrics">
            <div class="metric">
                <h3>Original Model</h3>
                <p>{{ results.original_accuracy }}</p>
            </div>
            <div class="metric">
                <h3>Debiased Model</h3>
                <p>{{ results.debiased_accuracy }}</p>
            </div>
        </div>

        <h2>Prediction Rates by Group</h2>
        <img src="data:image/png;base64,{{ results.plot_before }}" alt="Before Mitigation" />
        <p style="text-align:center; font-size: 0.9em; margin-top: -12px; margin-bottom: 20px;">Before Mitigation</p>
        <img src="data:image/png;base64,{{ results.plot_after }}" alt="After Mitigation" />
        <p style="text-align:center; font-size: 0.9em;">After Mitigation</p>
    </div>
    <div style="text-align:center; margin-top: 30px;">
        <a href="{{ url_for('index') }}">&#8592; Analyze Another Dataset</a>
    </div>
    <div class="footer">
        <p>This demo uses <a href="https://fairlearn.org/" target="_blank" rel="noopener">Fairlearn</a> and <a href="https://scikit-learn.org/" target="_blank" rel="noopener">Scikit-learn</a>.</p>
    </div>
</body>
</html>
"""

def create_plot(predictions, sensitive_features, title):
    groups = np.unique(sensitive_features)
    rates = []
    for group in groups:
        group_mask = sensitive_features == group
        rate = predictions[group_mask].mean()
        rates.append(rate)
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(6,3))
    bars = ax.bar([str(g) for g in groups], rates, color=['#4b6cb7', '#182848'])
    ax.set_ylim([0,1])
    ax.set_ylabel("Positive Prediction Rate")
    ax.set_title(title)
    for bar, rate in zip(bars, rates):
        yval = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2.0, yval + 0.02, f"{rate:.2f}", ha='center', va='bottom', fontsize=11)
    plt.tight_layout()
    buf = io.BytesIO()
    plt.savefig(buf, format='png')
    plt.close(fig)
    buf.seek(0)
    img_base64 = base64.b64encode(buf.read()).decode('utf-8')
    return img_base64

@app.route('/', methods=['GET'])
def index():
    return render_template_string(UPLOAD_HTML)

@app.route('/upload', methods=['POST'])
def upload():
    if 'file' not in request.files:
        return render_template_string(UPLOAD_HTML, error="No file part in the request.")
    file = request.files['file']
    if file.filename == '':
        return render_template_string(UPLOAD_HTML, error="No selected file.")
    try:
        content = file.read()
        df = pd.read_csv(io.BytesIO(content))
    except Exception as e:
        return render_template_string(UPLOAD_HTML, error=f"Failed to read CSV file: {str(e)}")

    columns = df.columns.tolist()
    # encode file back to base64 to pass with form
    filedata = base64.b64encode(content).decode('utf-8')

    return render_template_string(SELECT_HTML, columns=columns, filedata=filedata)

@app.route('/analyze', methods=['POST'])
def analyze():
    filedata = request.form.get('filedata')
    sensitive_attr = request.form.get('sensitive_attr')
    label_col = request.form.get('label_col')

    if not filedata or not sensitive_attr or not label_col:
        return render_template_string(UPLOAD_HTML, error="Missing input data or selections.")

    try:
        content = base64.b64decode(filedata)
        df = pd.read_csv(io.BytesIO(content))
    except Exception as e:
        return render_template_string(UPLOAD_HTML, error=f"Failed to load dataset: {str(e)}")

    if sensitive_attr not in df.columns or label_col not in df.columns:
        return render_template_string(UPLOAD_HTML, error="Selected columns not found in dataset.")

    # Prepare data for modeling
    y = df[label_col]
    sensitive_features = df[sensitive_attr]

    # Drop label and sensitive attributes and any non-numeric, or ID-like columns for features input
    # For simplicity: drop label_col, sensitive_attr, and columns with non-numeric data or ID column if named 'ID' or 'id'
    # We allow numeric columns only as features
    drop_cols = [label_col, sensitive_attr]
    for col in df.columns:
        if col.lower() == 'id':
            drop_cols.append(col)

    X = df.drop(columns=drop_cols)

    # Filter numeric columns only for features to avoid errors
    X = X.select_dtypes(include=['number'])

    # Simple check to avoid empty feature set
    if X.shape[1] == 0:
        return render_template_string(UPLOAD_HTML, error="No numeric feature columns available after removing label, sensitive attribute, and ID columns.")

    # Train/test split stratified by sensitive features for fairness
    try:
        X_train, X_test, y_train, y_test, sf_train, sf_test = train_test_split(
            X, y, sensitive_features, test_size=0.3, random_state=42, stratify=sensitive_features)
    except Exception as e:
        return render_template_string(UPLOAD_HTML, error=f"Error splitting data: {str(e)}")
    
    # Train baseline model
    model = LogisticRegression(solver='liblinear')
    try:
        model.fit(X_train, y_train)
    except Exception as e:
        return render_template_string(UPLOAD_HTML, error=f"Model training failed: {str(e)}")
    
    y_pred = model.predict(X_test)
    original_accuracy = accuracy_score(y_test, y_pred)

    # Bias metrics before mitigation
    dp_diff_before = demographic_parity_difference(y_test, y_pred, sensitive_features=sf_test)
    eo_diff_before = equalized_odds_difference(y_test, y_pred, sensitive_features=sf_test)

    # Bias mitigation with in-processing fairness constraint (DemographicParity)
    mitigator = ExponentiatedGradient(LogisticRegression(solver='liblinear'), constraints=DemographicParity())
    try:
        mitigator.fit(X_train, y_train, sensitive_features=sf_train)
    except Exception as e:
        return render_template_string(UPLOAD_HTML, error=f"Mitigation training failed: {str(e)}")

    y_pred_debiased = mitigator.predict(X_test)
    debiased_accuracy = accuracy_score(y_test, y_pred_debiased)

    # Bias metrics after mitigation
    dp_diff_after = demographic_parity_difference(y_test, y_pred_debiased, sensitive_features=sf_test)
    eo_diff_after = equalized_odds_difference(y_test, y_pred_debiased, sensitive_features=sf_test)

    plot_before = create_plot(y_pred, sf_test, "Prediction Rates by Group Before Mitigation")
    plot_after = create_plot(y_pred_debiased, sf_test, "Prediction Rates by Group After Mitigation")

    results = {
        "demographic_parity_difference": f"{dp_diff_before:.3f} (before), {dp_diff_after:.3f} (after)",
        "equalized_odds_difference": f"{eo_diff_before:.3f} (before), {eo_diff_after:.3f} (after)",
        "original_accuracy": f"{original_accuracy:.3f}",
        "debiased_accuracy": f"{debiased_accuracy:.3f}",
        "plot_before": plot_before,
        "plot_after": plot_after,
    }

    return render_template_string(RESULTS_HTML, results=results)

if __name__ == '__main__':
    app.run(debug=True)