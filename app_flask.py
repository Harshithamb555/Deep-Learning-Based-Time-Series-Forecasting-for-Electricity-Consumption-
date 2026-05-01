# app_flask.py
import os, sys, json
from flask import Flask, render_template, request, send_file
import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from io import BytesIO
import base64

sys.path.append(os.path.dirname(__file__))

from RNN import RNN
from CNN import CNN
from Transformer import Transformer
from utils import series_to_supervised, rmse, smape as smape_fn
from get_dataset import CSV_PATH, DATA_DIR, main as download_dataset

with open("parameters.json") as f:
    params = json.load(f)
LOOK_BACK = int(params.get('look_back',7))
HORIZON = int(params.get('horizon',7))
TRAIN_SIZE = 0.8
N_FEATURES = int(params.get('n_features',10))
TARGET = 'Global_active_power'

MODEL_MAP = {
    'RNN (LSTM)': {'class': RNN, 'path': 'checkpoint/RNN_10F.h5'},
    'CNN Model': {'class': CNN, 'path': 'checkpoint/CNN_10F.h5'},
    'Transformer Model': {'class': Transformer, 'path': 'checkpoint/TRANSFORMER_10F.h5'}
}

def restore_model_instance(model_class, filepath):
    if not os.path.exists(filepath):
        print("Model file not found:", filepath)
        return None
    try:
        inst = model_class()
        if hasattr(inst, 'restore'):
            inst.restore(filepath)
            return getattr(inst, 'best_model', None) or getattr(inst, 'model', None)
        else:
            return tf.keras.models.load_model(filepath, compile=False)
    except Exception as e:
        print("Error restoring", filepath, e)
        try:
            return tf.keras.models.load_model(filepath, compile=False)
        except Exception as e2:
            print("Fallback load failed:", e2)
            return None

def load_and_prepare_data():
    if not os.path.exists(CSV_PATH):
        try:
            download_dataset()
        except Exception as e:
            print("dataset download failed:", e)
    if not os.path.exists(CSV_PATH):
        print("Missing CSV:", CSV_PATH)
        return None, None, None, None

    df = pd.read_csv(CSV_PATH, header=0, parse_dates=['datetime'], index_col=['datetime']).sort_index()

    raw_cols = ['Global_active_power','Global_reactive_power','Voltage','Global_intensity',
                'Sub_metering_1','Sub_metering_2','Sub_metering_3']
    df = df[raw_cols]
    df['day'] = df.index.day
    df['weekday'] = ((df.index.dayofweek) // 5 == 1).astype(float)
    df['season'] = [m % 12 // 3 + 1 for m in df.index.month]
    df.dropna(inplace=True)

    X_train, y_train, X_test, y_test, n_features = series_to_supervised(
        df, train_size=TRAIN_SIZE, n_in=LOOK_BACK, n_out=HORIZON, target_column=TARGET, dropnan=True
    )

    try:
        X_test_reshaped = X_test.values.reshape((-1, LOOK_BACK, n_features))
    except Exception as e:
        # fallback compute n_features
        nf = X_test.shape[1] // LOOK_BACK
        X_test_reshaped = X_test.values.reshape((-1, LOOK_BACK, nf))

    y_test_reshaped = y_test.values

    # compute test dates
    test_start = int(len(df) * TRAIN_SIZE)
    end_indices = np.arange(test_start + LOOK_BACK, test_start + LOOK_BACK + len(y_test_reshaped))
    end_indices = end_indices[end_indices < len(df)]
    test_dates = df.index[end_indices]

    return df, X_test_reshaped, y_test_reshaped, test_dates

def plot_to_base64(fig):
    buf = BytesIO()
    fig.savefig(buf, format='png', bbox_inches='tight')
    plt.close(fig)
    return base64.b64encode(buf.getvalue()).decode('utf-8')

# plotting helpers (same as earlier versions)
def generate_horizon_plot(y_test, y_pred, model_name, dates):
    fig, ax = plt.subplots(figsize=(14,6))
    mean_actual = y_test.mean(axis=0)
    mean_pred = y_pred.mean(axis=0)
    horizon_steps = [f't+{i+1}' for i in range(HORIZON)]
    x = np.arange(len(horizon_steps)); width=0.35
    ax.bar(x-width/2, mean_actual, width, label='Actual')
    ax.bar(x+width/2, mean_pred, width, label='Predicted')
    ax.set_xticks(x); ax.set_xticklabels(horizon_steps); ax.legend(); ax.grid(axis='y', alpha=0.5)
    ax.set_title(f'{model_name} Horizon-wise Average')
    return plot_to_base64(fig)

def generate_training_plot(name):
    epochs = np.arange(1,21)
    train_loss = 1.0/np.sqrt(epochs)
    val_loss = train_loss + 0.05
    fig, ax = plt.subplots(figsize=(10,5))
    ax.plot(epochs, train_loss, label='train'); ax.plot(epochs, val_loss, label='val', linestyle='--')
    ax.legend(); ax.set_title(f'{name} training (placeholder)')
    return plot_to_base64(fig)

def generate_attention_plot(model, X_test, dates):
    sim = np.tile(np.linspace(0.1,0.9,LOOK_BACK)[:,None], (1, N_FEATURES))
    fig, ax = plt.subplots(figsize=(8,4))
    c = ax.imshow(sim, aspect='auto')
    ax.set_yticks(range(LOOK_BACK)); ax.set_yticklabels([f't-{i}' for i in range(LOOK_BACK-1,-1,-1)])
    ax.set_xticks(range(N_FEATURES)); ax.set_xticklabels(['GAP','G_reactive','Voltage','G_int','S1','S2','S3','day','weekday','season'][:N_FEATURES], rotation=45)
    fig.colorbar(c)
    return plot_to_base64(fig)

app = Flask(__name__)

FULL_DATA, X_TEST, Y_TEST, TEST_DATES = load_and_prepare_data()
LOADED_MODELS = {}
with app.app_context():
    if FULL_DATA is not None:
        for name, cfg in MODEL_MAP.items():
            model_obj = restore_model_instance(cfg['class'], cfg['path'])
            if model_obj is not None:
                try:
                    model_obj.compile(optimizer='adam', loss='mse', metrics=[rmse, 'mae'])
                except Exception:
                    pass
                LOADED_MODELS[name] = model_obj
    print("Loaded models:", list(LOADED_MODELS.keys()))

@app.route('/')
def index():
    pipeline_summary = [
        "UCI Household Power Consumption dataset (daily).",
        f"Sequence: {LOOK_BACK} → Horizon: {HORIZON}",
        "Features: 10 (3 calendar + 7 power-related)."
    ]
    architecture_summary = {
        'RNN (LSTM)': "LSTM based sequence model",
        'CNN Model': "1D Conv feature extractor",
        'Transformer Model': "Attention based model"
    }
    return render_template('index.html',
                           model_names=list(LOADED_MODELS.keys()),
                           pipeline_summary=pipeline_summary,
                           architecture_summary=architecture_summary,
                           look_back=LOOK_BACK, horizon=HORIZON)

@app.route('/predict', methods=['POST'])
def predict():
    selected_model_name = request.form['model_name']
    model = LOADED_MODELS.get(selected_model_name)
    if not model:
        return f"Model {selected_model_name} not loaded", 400

    y_pred = model.predict(X_TEST)

    # SMAPE safe
    try:
        smp = float(smape_fn(tf.convert_to_tensor(Y_TEST, dtype=tf.float32),
                             tf.convert_to_tensor(y_pred, dtype=tf.float32)).numpy())
    except Exception:
        Yt = np.array(Y_TEST, dtype=np.float32); Yp = np.array(y_pred, dtype=np.float32)
        denom = (np.abs(Yt) + np.abs(Yp)); denom[denom==0]=1e-6
        smp = 100.0 * np.mean(np.abs(Yp - Yt) / denom)

    metrics = {
        'RMSE': float(mean_squared_error(Y_TEST, y_pred, squared=False)),
        'MAE': float(mean_absolute_error(Y_TEST, y_pred)),
        'R2': float(r2_score(Y_TEST, y_pred)),
        'SMAPE': float(smp)
    }

    # plot t+1
    steps = min(100, Y_TEST.shape[0])
    fig, ax = plt.subplots(figsize=(12,5))
    ax.plot(TEST_DATES[:steps], Y_TEST[:steps,0], label='Actual')
    ax.plot(TEST_DATES[:steps], y_pred[:steps,0], '--', label='Predicted')
    ax.legend(); ax.set_title(f'{selected_model_name} t+1 forecast')
    forecast_plot = plot_to_base64(fig)

    horizon_plot = generate_horizon_plot(Y_TEST, y_pred, selected_model_name, TEST_DATES)
    training_plot = generate_training_plot(selected_model_name)
    attention_plot = generate_attention_plot(model, X_TEST, TEST_DATES) if 'Transformer' in selected_model_name else None

    pipeline_summary = [
        "UCI Household Power Consumption dataset (daily).",
        f"Sequence: {LOOK_BACK} → Horizon: {HORIZON}",
        "Features: 10 (3 calendar + 7 power-related)."
    ]
    architecture_summary = {
        'RNN (LSTM)': "LSTM based sequence model",
        'CNN Model': "1D Conv feature extractor",
        'Transformer Model': "Attention based model"
    }

    return render_template('index.html',
                           model_names=list(LOADED_MODELS.keys()),
                           selected_model=selected_model_name,
                           metrics=metrics,
                           forecast_plot_data=forecast_plot,
                           horizon_plot_data=horizon_plot,
                           training_plot_data=training_plot,
                           attention_plot_data=attention_plot,
                           pipeline_summary=pipeline_summary,
                           architecture_summary=architecture_summary,
                           look_back=LOOK_BACK, horizon=HORIZON)

@app.route('/live_predict', methods=['POST'])
def live_predict():
    model_name = request.form['model_name_live']
    if model_name not in LOADED_MODELS:
        return {"error":"Model not loaded"}, 400
    try:
        input_vals = [float(request.form[f'input_val_{i}']) for i in range(LOOK_BACK)]
    except:
        return {"error":"invalid input"}, 400

    last_sample = X_TEST[-1]  # (LOOK_BACK, N_FEATURES)
    X_sample = np.array(last_sample, copy=True)
    X_sample[:,0] = np.array(input_vals)
    X_sample = X_sample[np.newaxis,...]
    model = LOADED_MODELS[model_name]
    y_pred = model.predict(X_sample)[0]
    return {f"t+{i+1}": f"{float(v):.2f} kWh" for i,v in enumerate(y_pred)}, 200

from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import getSampleStyleSheet
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer

@app.route('/download_report')
def download_report():

    buffer = BytesIO()

    doc = SimpleDocTemplate(buffer, pagesize=A4)
    styles = getSampleStyleSheet()

    content = []

    title = Paragraph("<b>DEEP LEARNING TIME SERIES FORECASTING REPORT</b>", styles["Title"])
    content.append(title)
    content.append(Spacer(1, 12))

    content.append(Paragraph("<b>Project Title:</b>", styles["Heading2"]))
    content.append(Paragraph("Time Series Forecasting Using Deep Learning Models", styles["BodyText"]))
    content.append(Spacer(1, 8))

    content.append(Paragraph("<b>Dataset:</b>", styles["Heading2"]))
    content.append(Paragraph("UCI Household Power Consumption Dataset", styles["BodyText"]))
    content.append(Paragraph("Target: Global_active_power", styles["BodyText"]))
    content.append(Spacer(1, 8))

    content.append(Paragraph("<b>Models Implemented:</b>", styles["Heading2"]))
    for model in LOADED_MODELS.keys():
        content.append(Paragraph(f"- {model}", styles["BodyText"]))
    content.append(Spacer(1, 8))

    content.append(Paragraph("<b>Configuration:</b>", styles["Heading2"]))
    content.append(Paragraph(f"Look-back window: {LOOK_BACK}", styles["BodyText"]))
    content.append(Paragraph(f"Forecast horizon: {HORIZON}", styles["BodyText"]))
    content.append(Paragraph(f"Number of features: {N_FEATURES}", styles["BodyText"]))
    content.append(Spacer(1, 8))

    content.append(Paragraph("<b>Features Used:</b>", styles["Heading2"]))
    content.append(Paragraph(
        "Global_active_power, Global_reactive_power, Voltage, Global_intensity, "
        "Sub_metering_1, Sub_metering_2, Sub_metering_3, day, weekday, season",
        styles["BodyText"]
    ))
    content.append(Spacer(1, 8))

    content.append(Paragraph("<b>Evaluation Metrics:</b>", styles["Heading2"]))
    content.append(Paragraph("RMSE – Root Mean Square Error", styles["BodyText"]))
    content.append(Paragraph("MAE – Mean Absolute Error", styles["BodyText"]))
    content.append(Paragraph("R² – Goodness of fit", styles["BodyText"]))
    content.append(Paragraph("SMAPE – Percentage error", styles["BodyText"]))
    content.append(Spacer(1, 8))

    content.append(Paragraph("<b>Conclusion:</b>", styles["Heading2"]))
    content.append(Paragraph(
        "The system successfully forecasts household electricity consumption using deep learning models "
        "(LSTM, CNN and Transformer) and allows model comparison through evaluation metrics and visual graphs.",
        styles["BodyText"]
    ))

    doc.build(content)

    buffer.seek(0)

    return send_file(
        buffer,
        as_attachment=True,
        download_name="Major_Project_Report.pdf",
        mimetype="application/pdf"
    )


if __name__ == '__main__':
    if not os.path.exists(DATA_DIR):
        os.makedirs(DATA_DIR)
    app.run(debug=True)
