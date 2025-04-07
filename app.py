import numpy as np
import pandas as pd
import yfinance as yf
import streamlit as st
import os
import tensorflow as tf
import plotly.graph_objects as go

from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
from tensorflow.keras.models import Model, load_model
from tensorflow.keras.layers import Input, Dense, GRU, Dropout, BatchNormalization
from tensorflow.keras.callbacks import EarlyStopping

# Force CPU usage
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
tf.config.set_visible_devices([], 'GPU')
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

# Streamlit Page Config
st.set_page_config(page_title="📈 Stock Market Predictor", layout="wide")
st.markdown("<h1 style='text-align: center;'>🚀 Stock Market Prediction</h1>", unsafe_allow_html=True)

# Sidebar
with st.sidebar:
    stock_symbol = st.text_input("Enter Stock Symbol (e.g., AAPL, TSLA, GOOG):", "AAPL")
    start_date = st.date_input("Start Date", pd.to_datetime("2010-01-01"))
    end_date = st.date_input("End Date", pd.to_datetime("2025-01-01"))

# Load Data
@st.cache_data
def load_data(symbol, start, end):
    return yf.download(symbol, start=start, end=end)

data = load_data(stock_symbol, start_date, end_date)
st.subheader(f"📊 Raw Stock Data for {stock_symbol}")
st.dataframe(data.tail(), use_container_width=True)

@st.cache_resource(hash_funcs={pd.DataFrame: lambda _: None})
def train_or_load_model(data, stock_symbol):
    data['Return'] = data['Close'].pct_change()
    data['Volatility'] = data['Return'].rolling(window=10).std()
    data['Volume_Change'] = data['Volume'].pct_change()
    data.dropna(inplace=True)

    features = ['Open', 'High', 'Low', 'Close', 'Volume', 'Return', 'Volatility', 'Volume_Change']
    scaler = MinMaxScaler()
    scaled_data = scaler.fit_transform(data[features])

    def create_sequences(data, seq_len=60):
        X, y = [], []
        for i in range(seq_len, len(data)):
            X.append(data[i - seq_len:i])
            y.append(data[i, 3])  # 'Close' is at index 3
        return np.array(X), np.array(y)

    X, y = create_sequences(scaled_data)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=False)

    os.makedirs("models", exist_ok=True)
    model_path = f"models/{stock_symbol}_gru_model_improved.h5"

    if os.path.exists(model_path):
        model = load_model(model_path)
    else:
        input_layer = Input(shape=(X.shape[1], X.shape[2]))
        x = GRU(128, return_sequences=True, dropout=0.2)(input_layer)
        x = GRU(64, return_sequences=False, dropout=0.2)(x)
        x = BatchNormalization()(x)
        x = Dense(32, activation='relu')(x)
        x = Dropout(0.2)(x)
        output_layer = Dense(1)(x)

        model = Model(inputs=input_layer, outputs=output_layer)
        model.compile(optimizer='adam', loss='mean_squared_error')

        early_stop = EarlyStopping(monitor='val_loss', patience=7, restore_best_weights=True)
        model.fit(X_train, y_train, batch_size=32, epochs=50, validation_data=(X_test, y_test), callbacks=[early_stop], verbose=0)
        model.save(model_path)

    preds = model.predict(X_test)
    preds_inv = scaler.inverse_transform(np.hstack([np.zeros((preds.shape[0], 3)), preds, np.zeros((preds.shape[0], 4))]))[:, 3]
    y_test_inv = scaler.inverse_transform(np.hstack([np.zeros((len(y_test), 3)), y_test.reshape(-1, 1), np.zeros((len(y_test), 4))]))[:, 3]

    mse = mean_squared_error(y_test_inv, preds_inv)
    r2 = r2_score(y_test_inv, preds_inv)

    return preds_inv, y_test_inv, mse, r2

with st.spinner("Training or loading model..."):
    predictions, y_actual, mse, r2 = train_or_load_model(data.copy(), stock_symbol)

# Performance
st.markdown("### Model Performance")
col1, col2 = st.columns(2)
col1.metric("Mean Squared Error", f"{mse:.4f}")
col2.metric("R2 Score", f"{r2:.4f}")

# Plotting
st.markdown("### Actual vs Predicted Prices")
test_dates = data.index[-len(y_actual):]

predictions = predictions.flatten()
y_actual = y_actual.flatten()

fig_pred = go.Figure()
fig_pred.add_trace(go.Scatter(x=test_dates, y=y_actual, mode='lines', name='Actual Prices', line=dict(color='blue')))
fig_pred.add_trace(go.Scatter(x=test_dates, y=predictions, mode='lines', name='Predicted Prices', line=dict(color='red', dash='dash')))

fig_pred.update_layout(
    title=f"Stock Price Prediction for {stock_symbol}",
    xaxis_title="Date",
    yaxis_title="Price",
    template='plotly_white',
    xaxis=dict(showgrid=True, tickformat="%Y-%m", tickangle=45)
)

st.plotly_chart(fig_pred, use_container_width=True)
