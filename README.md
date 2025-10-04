# 📈 Stock Price Forecasting – ARIMA & LSTM  

This project demonstrates **time-series forecasting** on stock prices using both:  
- **ARIMA** (classical statistical model)  
- **LSTM** (deep learning model)

Live Link: https://huggingface.co/spaces/mrshibly/DataSynthis_Job_task

The app is deployed with **Gradio on Hugging Face Spaces** so you can upload stock price data and generate **future forecasts interactively**.  

---

## 🚀 Features
✅ Upload your own stock CSV (must include a **`Close`** column).  
✅ Forecast with **ARIMA**, **LSTM**, or **Compare Both**.  
✅ Interactive **forecast horizon slider (5–30 days)**.  
✅ **Forecasted table** + **visual chart** output.  
✅ Clean, modular code for easy extension.  

---

## 🧠 Models Used
- **ARIMA** (`arima.pkl`)  
  - Captures autocorrelation and seasonality in time series.  
- **LSTM** (`lstm.pth`)  
  - Learns nonlinear sequential dependencies in stock prices.  

Both models are pre-trained and loaded inside the app.  

---

## 📊 Input Format
Upload a **CSV file** with at least one column:  

| Date       | Open   | High   | Low    | Close   | Volume  |
|------------|--------|--------|--------|---------|---------|
| 2024-01-01 | 100.25 | 101.20 | 99.80  | 100.90  | 2000000 |
| 2024-01-02 | 100.90 | 102.10 | 100.30 | 101.70  | 1800000 |
| ...        | ...    | ...    | ...    | ...     | ...     |

Only the **`Close`** column is required.  

---

## ⚙️ Installation & Run Locally
Clone this repo and install dependencies:
```bash
git clone https://huggingface.co/spaces/mrshibly/DataSynthis_Job_task
cd DataSynthis_Job_task
pip install -r requirements.txt
````

Run the app locally:

```bash
python app.py
```

The app will run on: [http://localhost:7860](http://localhost:7860)

---

## 🌐 Deployment on Hugging Face Spaces

This project is ready to run on **Hugging Face Spaces (Gradio)**.

* Upload `app.py`, `arima.pkl`, `lstm.pth`, `requirements.txt`, and `README.md`.
* Hugging Face will automatically detect `app.py` and launch the Gradio app.

---

## 📈 Example Output

**Forecast Horizon: 10 days, Compare Both**

* **Forecast Table**
  | Future | ARIMA Forecast | LSTM Forecast |
  |--------|----------------|---------------|
  | t+1    | 101.2          | 101.5         |
  | t+2    | 101.4          | 102.1         |
  | ...    | ...            | ...           |

* **Forecast Plot**
  Historical data (blue) + Predictions (orange/green).

---

## 💡 Why This Project is Interesting

* Combines **traditional stats** and **modern deep learning** in one tool.
* Clear comparison of model performance for real-world time series.
* Deployed in an interactive way → **turns research into a product**.

---

## 👨‍💻 Author

Developed by **Md Mahmudur Rahman**
*Built as part of a job interview task – demonstrating forecasting, ML/DL, and deployment skills.*

