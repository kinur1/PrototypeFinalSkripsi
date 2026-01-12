import streamlit as st
import yfinance as yf
import numpy as np
import pandas as pd
import math
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, LSTM
import plotly.express as px
import plotly.graph_objects as go

# Title
st.title("📈 Prediksi Harga Cryptocurrency dengan LSTM")
st.write("Aplikasi ini memprediksi harga penutupan cryptocurrency menggunakan model LSTM.")

# Valid options
valid_time_steps = [25, 50, 75, 100]
valid_epochs = [12, 25, 50, 100]
default_time_step = 100
default_epoch = 25

# Session state
if 'model_ran' not in st.session_state:
    st.session_state.model_ran = False

# Input settings
col1, col2 = st.columns(2)
with col1:
    time_step = st.radio("⏳ Time Step", options=valid_time_steps, index=valid_time_steps.index(default_time_step))
with col2:
    epoch_option = st.radio("🔄 Jumlah Epoch", options=valid_epochs, index=valid_epochs.index(default_epoch))

# Date selection
start_date = st.date_input("📅 Tanggal Mulai", pd.to_datetime("2020-01-01"))
end_date = st.date_input("📅 Tanggal Akhir", pd.to_datetime("2024-01-01"))

# Asset selection
asset_name_display = st.radio("💰 Pilih Aset", options=['BITCOIN', 'ETHEREUM'], index=0)

# Validasi Input
is_valid = (start_date < end_date)

# Run Prediction Button
if st.button("🚀 Jalankan Prediksi", disabled=not is_valid):

    asset_mapping = {'BITCOIN': 'BTC-USD', 'ETHEREUM': 'ETH-USD'}
    asset = asset_mapping[asset_name_display]

    # Fetch data
    st.write(f"📥 Mengambil data harga {asset_name_display} ({asset}) dari Yahoo Finance...")
    df = yf.download(asset, start=start_date, end=end_date, progress=False)
    df = df.reset_index()
    df.columns = [col[0] if isinstance(col, tuple) else col for col in df.columns]

    if df.empty:
        st.error("⚠️ Data kosong. Pastikan koneksi internet aktif.")
        st.stop()

    if len(df) <= time_step + 1:
        st.error(
            f"⚠️ Data terlalu sedikit ({len(df)} baris) untuk time_step={time_step}. "
            f"Perpanjang tanggal atau kecilkan time_step."
        )
        st.stop()

    # Plot harga asli
    st.write(f"### 📊 Histori Harga Penutupan {asset_name_display}")
    fig_hist = px.line(df, x='Date', y='Close', title="Harga Historis")
    st.plotly_chart(fig_hist, use_container_width=True)

    # Preprocessing
    closedf = df[['Close']].copy()
    scaler = MinMaxScaler(feature_range=(0, 1))
    closedf_scaled = scaler.fit_transform(closedf)

    # Split data
    training_size = int(len(closedf_scaled) * 0.90)
    train_data, test_data = closedf_scaled[:training_size], closedf_scaled[training_size:]

    def create_dataset(dataset, time_step=1):
        dataX, dataY = [], []
        for i in range(len(dataset) - time_step - 1):
            dataX.append(dataset[i:(i + time_step), 0])
            dataY.append(dataset[i + time_step, 0])
        return np.array(dataX), np.array(dataY)

    X_train, y_train = create_dataset(train_data, time_step)
    X_test, y_test = create_dataset(test_data, time_step)

    if X_train.shape[0] == 0 or X_test.shape[0] == 0:
        st.error("⚠️ Dataset kosong. Kurangi time_step atau perpanjang tanggal.")
        st.stop()

    # Reshape
    X_train = X_train.reshape(X_train.shape[0], X_train.shape[1], 1)
    X_test = X_test.reshape(X_test.shape[0], X_test.shape[1], 1)

    # Build model
    model = Sequential([
        LSTM(50, return_sequences=True, input_shape=(time_step, 1), activation="relu"),
        LSTM(50, return_sequences=False, activation="relu"),
        Dense(1)
    ])
    model.compile(loss="mean_squared_error", optimizer="adam")

    # Train
    model.fit(X_train, y_train, validation_data=(X_test, y_test),
              epochs=epoch_option, batch_size=32, verbose=1)

    # Predict
    train_predict = model.predict(X_train, verbose=0)
    test_predict = model.predict(X_test, verbose=0)

    # Inverse transform
    train_predict = scaler.inverse_transform(train_predict)
    test_predict = scaler.inverse_transform(test_predict)
    original_ytrain = scaler.inverse_transform(y_train.reshape(-1, 1))
    original_ytest = scaler.inverse_transform(y_test.reshape(-1, 1))

    # Metrics
    train_rmse = math.sqrt(mean_squared_error(original_ytrain, train_predict))
    test_rmse = math.sqrt(mean_squared_error(original_ytest, test_predict))

    st.session_state.update({
        'model_ran': True,
        'df': df,
        'train_predict': train_predict,
        'test_predict': test_predict,
        'original_ytrain': original_ytrain,
        'original_ytest': original_ytest,
        'time_step': time_step,
        'asset_name_display': asset_name_display
    })

    st.write("### 📊 Metrik Evaluasi")
    st.write(f"✅ RMSE Training: {train_rmse:.4f}")
    st.write(f"✅ RMSE Testing: {test_rmse:.4f}")

# ============================
# TAMPILKAN HASIL (CHART FINAL)
# ============================
if st.session_state.get("model_ran", False):

    df = st.session_state["df"]
    train_predict = st.session_state["train_predict"]
    test_predict = st.session_state["test_predict"]
    original_ytrain = st.session_state["original_ytrain"]
    original_ytest = st.session_state["original_ytest"]
    time_step = st.session_state["time_step"]
    asset_name_display = st.session_state["asset_name_display"]

    # Align tanggal dengan data y
    y_total_len = len(original_ytrain) + len(original_ytest)
    date_series = df['Date'].iloc[time_step + 1: time_step + 1 + y_total_len].reset_index(drop=True)

    result_df = pd.DataFrame({
        'Date': date_series,
        'Actual': np.concatenate([original_ytrain.flatten(), original_ytest.flatten()]),
        'Predicted': np.concatenate([train_predict.flatten(), test_predict.flatten()])
    })

    result_df = result_df.sort_values("Date").reset_index(drop=True)

    # Split index
    split_index = len(train_predict)
    split_index = min(split_index, len(result_df) - 1)
    split_date = result_df.loc[split_index, "Date"]

    st.write(f"### 🔮 Prediksi Harga {asset_name_display}")

    fig = go.Figure()

    fig.add_trace(go.Scatter(
        x=result_df["Date"], y=result_df["Actual"],
        mode="lines", name="Actual"
    ))

    fig.add_trace(go.Scatter(
        x=result_df["Date"], y=result_df["Predicted"],
        mode="lines", name="Predicted"
    ))

    # Garis split
    fig.add_vline(
        x=split_date,
        line_width=2,
        line_dash="dash",
        annotation_text="Train/Test",
        annotation_position="top"
    )

    # Shading prediksi
    fig.add_vrect(
        x0=split_date,
        x1=result_df["Date"].iloc[-1],
        fillcolor="rgba(0,0,0,0.08)",
        line_width=0
    )

    fig.update_layout(
        title="Actual vs Predicted (Mobile Friendly)",
        xaxis_title="Tanggal",
        yaxis_title="Harga",
        hovermode="x unified"
    )

    st.plotly_chart(fig, use_container_width=True)

    # Tabel ringkas
    st.write("### 📋 Hasil Prediksi (20 data pertama)")
    st.dataframe(result_df.head(20), use_container_width=True, hide_index=True)
