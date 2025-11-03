# 📊 Trading Prediction App  

A simple, interactive **machine learning web app** that predicts whether a stock or crypto asset’s price will go **up or down tomorrow**, based on recent trends and moving averages.  

Built using **Streamlit**, **Scikit-Learn**, and **Yahoo Finance (yfinance)**, this project is perfect for beginners exploring **financial data analysis** and **ML-based trading predictions**.  

---

## 🚀 Features  

- 📈 Real-time stock & crypto data via **yfinance**  
- 🧠 Machine learning model: **RandomForestClassifier**  
- 🎛️ Interactive **Streamlit dashboard** with sidebar controls  
- 🕹️ Choose **asset** and **time frame** dynamically  
- 🔮 Predict next-day price direction  
- 📊 Visualize actual prices and model-predicted buy signals  

---

## 🧩 Tech Stack  

| Component | Technology |
|------------|-------------|
| **Frontend** | Streamlit |
| **Backend / ML** | Scikit-Learn |
| **Data Source** | Yahoo Finance (`yfinance`) |
| **Language** | Python 3.x |
| **Visualization** | Matplotlib |

---

## 📁 Project Structure  

Trading_Prediction/
│
├── app.py # Main Streamlit app
├── requirements.txt # Dependencies list (optional)
├── README.md # Project documentation
└── screenshots/ # (Optional) App screenshots


---

## ⚙️ Installation  

### 🖥️ Run Locally  

1. **Clone this repo**  
   ```bash
   git clone https://github.com/<your-username>/Trading_Prediction.git
   cd Trading_Prediction


##Create virtual environment (optional)

python -m venv venv
source venv/bin/activate       # On Mac/Linux  
venv\Scripts\activate          # On Windows

---

##Install dependencies

pip install -r requirements.txt

---

or manually install:

pip install streamlit scikit-learn yfinance matplotlib


##Run the app

streamlit run app.py


##Open in your browser at
👉 http://localhost:8501

##☁️ Run on Google Colab

If you’re running in Colab:

!streamlit run app.py --server.port 8501 & npx localtunnel --port 8501


Then click the generated public link to open your live dashboard.

---

##🎛️ How It Works

Fetch Data → Uses Yahoo Finance for live stock data

Feature Engineering → Creates moving averages (MA5, MA10, MA20) & returns

Train Model → Random Forest predicts “up” or “down”

Visualize → Plots actual prices and model’s buy signals

Predict → Shows whether next day’s price will rise or fall

---

##📸 Example Output
✅ Model Accuracy: 82.45%
📈 The model predicts: Price will go UP tomorrow.

---

##🧠 Future Enhancements

Add RSI, EMA, MACD indicators

Include more ML models (SVM, LSTM, XGBoost)

Integrate live trading signal APIs

Deploy on Streamlit Cloud or HuggingFace Spaces

