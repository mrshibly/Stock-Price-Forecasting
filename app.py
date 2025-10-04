import gradio as gr
import pandas as pd
import numpy as np
import torch
import pickle
import matplotlib.pyplot as plt
import io
from torch import nn
from PIL import Image


# Load ARIMA model
with open("arima.pkl", "rb") as f:
    arima_model = pickle.load(f)


# Define LSTM Model
class LSTMModel(nn.Module):
    def __init__(self, input_size=1, hidden_size=50, num_layers=2, output_size=1):
        super(LSTMModel, self).__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        h0 = torch.zeros(2, x.size(0), 50)
        c0 = torch.zeros(2, x.size(0), 50)
        out, _ = self.lstm(x, (h0, c0))
        out = self.fc(out[:, -1, :])
        return out


# Load trained LSTM
lstm_model = LSTMModel()
lstm_model.load_state_dict(torch.load("lstm.pth", map_location=torch.device('cpu')))
lstm_model.eval()


# ARIMA Prediction
def predict_arima(values, horizon=10):
    forecast = arima_model.forecast(steps=horizon)
    return forecast.tolist()


# LSTM Prediction
def predict_lstm(values, horizon=10):
    seq = torch.tensor(values[-50:], dtype=torch.float32).view(1, -1, 1)
    preds = []
    for _ in range(horizon):
        with torch.no_grad():
            pred = lstm_model(seq).item()
        preds.append(pred)
        seq = torch.cat([seq[:, 1:, :], torch.tensor([[[pred]]])], dim=1)
    return preds


# Forecast Function
def forecast(file, horizon, model_choice):
    df = pd.read_csv(file.name)
    if "Close" not in df.columns:
        return "❌ CSV must contain a 'Close' column", None

    values = df["Close"].values.tolist()

    # Run forecasts
    preds_arima = predict_arima(values, horizon)
    preds_lstm = predict_lstm(values, horizon)

    # Prepare DataFrames
    future_index = [f"t+{i+1}" for i in range(horizon)]
    forecast_df = pd.DataFrame({
        "Future": future_index,
        "ARIMA Forecast": preds_arima,
        "LSTM Forecast": preds_lstm
    })

    # Plot
    plt.figure(figsize=(10,5))
    plt.plot(range(len(values)), values, label="Historical")
    if model_choice in ["ARIMA", "Compare Both"]:
        plt.plot(range(len(values), len(values)+horizon), preds_arima, label="ARIMA Forecast")
    if model_choice in ["LSTM", "Compare Both"]:
        plt.plot(range(len(values), len(values)+horizon), preds_lstm, label="LSTM Forecast")
    
    plt.title(f"{model_choice} Stock Forecast")
    plt.xlabel("Time")
    plt.ylabel("Price")
    plt.legend()

    # Save plot to buffer and convert to PIL
    buf = io.BytesIO()
    plt.savefig(buf, format="png")
    buf.seek(0)
    plt.close()
    img = Image.open(buf)

    return forecast_df, img


# Gradio UI
with gr.Blocks() as demo:
    gr.Markdown("# 📈 Stock Price Forecasting")
    gr.Markdown(
        "Upload a CSV containing stock prices (must have a **'Close'** column). "
        "Choose ARIMA, LSTM, or Compare Both, then set forecast horizon."
    )
    
    with gr.Row():
        file = gr.File(label="Upload CSV", file_types=[".csv"])
        horizon = gr.Slider(5, 30, value=10, step=1, label="Forecast Horizon (days)")
        model_choice = gr.Radio(["ARIMA", "LSTM", "Compare Both"], label="Model", value="Compare Both")

    output_table = gr.DataFrame(label="Forecasted Prices")
    output_plot = gr.Image(type="pil", label="Forecast Plot")

    submit = gr.Button("Run Forecast")
    submit.click(forecast, inputs=[file, horizon, model_choice], outputs=[output_table, output_plot])

demo.launch()
