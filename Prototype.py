import streamlit as st
import yfinance as yf
import tensorflow as tf
import numpy as np
import pandas as pd
import math
import time
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

    # Mapping assets
    asset_mapping = {'BITCOIN': 'BTC-USD', 'ETHEREUM': 'ETH-USD'}
    asset = asset_mapping[asset_name_display]

    # Fetch data
    st.write(f"📥 Mengambil data harga {asset_name_display} ({asset}) dari Yahoo Finance...")
    df = yf.download(asset, start=start_date, end=end_date, progress=False)
    df = df.reset_index()
    df.columns = [col[0] if isinstance(col, tuple) else col for col in df.columns]

    # Validasi data
    if df is None or df.empty:
        st.error("⚠️ Gagal mengambil data. Pastikan koneksi internet stabil dan ticker valid.")
        st.stop()

    # Validasi jumlah data
    if len(df) <= time_step + 1:
        st.error(
            f"⚠️ Data terlalu sedikit ({len(df)} baris) untuk time_step={time_step}. "
            f"Coba perpanjang rentang tanggal atau kurangi time_step."
        )
        st.stop()

    # Plot harga asli
    st.write(f"### 📊 Histori Harga Penutupan {asset_name_display}")
    fig_hist = px.line(df, x='Date', y='Close', title=f'Histori Harga {asset_name_display}')
    st.plotly_chart(fig_hist, use_container_width=True)

    # Preprocessing
    closedf = df[['Close']].copy()
    scaler = MinMaxScaler(feature_range=(0, 1))
    closedf_scaled = scaler.fit_transform(np.array(closedf).reshape(-1, 1))

    # Split data
    training_size = int(len(closedf_scaled) * 0.90)
    train_data, test_data = closedf_scaled[:training_size], closedf_scaled[training_size:]

    # Function to create dataset
    def create_dataset(dataset, time_step=1):
        dataX, dataY = [], []
        for i in range(len(dataset) - time_step - 1):
            a = dataset[i:(i + time_step), 0]
            dataX.append(a)
            dataY.append(dataset[i + time_step, 0])
        return np.array(dataX), np.array(dataY)

    X_train, y_train = create_dataset(train_data, time_step)
    X_test, y_test = create_dataset(test_data, time_step)

    # Cegah error reshape jika data kosong
    if X_train.shape[0] == 0 or X_test.shape[0] == 0:
        st.error("⚠️ Dataset training atau testing kosong. Perbesar rentang tanggal atau kurangi time_step.")
        st.stop()

    # Reshape data
    X_train = X_train.reshape(X_train.shape[0], X_train.shape[1], 1)
    X_test = X_test.reshape(X_test.shape[0], X_test.shape[1], 1)

    # Build LSTM Model
    model = Sequential([
        LSTM(50, return_sequences=True, input_shape=(time_step, 1), activation="relu"),
        LSTM(50, return_sequences=False, activation="relu"),
        Dense(1)
    ])
    model.compile(loss="mean_squared_error", optimizer="adam")

    # Train Model
    history = model.fit(
        X_train, y_train,
        validation_data=(X_test, y_test),
        epochs=epoch_option,
        batch_size=32,
        verbose=1
    )

    # Predictions
    train_predict = model.predict(X_train, verbose=0)
    test_predict = model.predict(X_test, verbose=0)

    # Inverse transform
    train_predict_inv = scaler.inverse_transform(train_predict)
    test_predict_inv = scaler.inverse_transform(test_predict)
    original_ytrain_inv = scaler.inverse_transform(y_train.reshape(-1, 1))
    original_ytest_inv = scaler.inverse_transform(y_test.reshape(-1, 1))

    # Evaluation Metrics
    train_rmse = math.sqrt(mean_squared_error(original_ytrain_inv, train_predict_inv))
    test_rmse = math.sqrt(mean_squared_error(original_ytest_inv, test_predict_inv))

    # MAPE (lebih aman: hindari pembagian 0)
    eps = 1e-9
    train_mape = np.mean(np.abs((original_ytrain_inv - train_predict_inv) / (np.abs(original_ytrain_inv) + eps))) * 100
    test_mape = np.mean(np.abs((original_ytest_inv - test_predict_inv) / (np.abs(original_ytest_inv) + eps))) * 100

    # Save Model State
    st.session_state.update({
        'model_ran': True,
        'df': df,
        'train_predict': train_predict_inv,
        'test_predict': test_predict_inv,
        'original_ytrain': original_ytrain_inv,
        'original_ytest': original_ytest_inv,
        'time_step': time_step,
        'asset_name_display': asset_name_display,
        'test_rmse': float(test_rmse)
    })

    # Display metrics
    st.write("### 📊 Metrik Evaluasi")
    st.write(f"**✅ RMSE (Training):** {train_rmse:.6f}")
    st.write(f"**✅ RMSE (Testing):** {test_rmse:.6f}")
    st.write(f"**📉 MAPE (Training):** {train_mape:.2f}%")
    st.write(f"**📉 MAPE (Testing):** {test_mape:.2f}%")

# Menampilkan hasil prediksi setelah model dijalankan
if st.session_state.get("model_ran", False):
    df = st.session_state.get("df", pd.DataFrame())
    train_predict = st.session_state.get("train_predict", np.array([]))
    test_predict = st.session_state.get("test_predict", np.array([]))
    original_ytrain = st.session_state.get("original_ytrain", np.array([]))
    original_ytest = st.session_state.get("original_ytest", np.array([]))
    time_step = st.session_state.get("time_step", 25)
    asset_name_display = st.session_state.get("asset_name_display", "BITCOIN")
    test_rmse = st.session_state.get("test_rmse", None)

    if len(df) > 0 and len(train_predict) > 0 and len(test_predict) > 0:

        # Buat result_df yang align (Date harus pas dengan y yang terbentuk dari create_dataset)
        # y_train panjang = len(train_data) - time_step - 1
        # y_test panjang  = len(test_data) - time_step - 1
        # Date start untuk y_total adalah index (time_step + 1)
        y_total_len = len(original_ytrain) + len(original_ytest)
        date_series = df['Date'].iloc[time_step + 1: time_step + 1 + y_total_len].reset_index(drop=True)

        result_df = pd.DataFrame({
            'Date': date_series,
            'Original_Close': np.concatenate([original_ytrain.flatten(), original_ytest.flatten()]),
            'Predicted_Close': np.concatenate([train_predict.flatten(), test_predict.flatten()])
        })

        # ===== Upgrade: MA + Split + Shaded + Confidence band =====
        result_df = result_df.sort_values("Date").reset_index(drop=True)
        result_df["MA7"] = result_df["Original_Close"].rolling(7).mean()
        result_df["MA14"] = result_df["Original_Close"].rolling(14).mean()

        split_index = len(train_predict)  # test mulai setelah train_predict
        split_index = min(split_index, len(result_df) - 1)
        split_date = result_df.loc[split_index, "Date"]

        if test_rmse is not None:
            result_df["Upper"] = result_df["Predicted_Close"] + float(test_rmse)
            result_df["Lower"] = result_df["Predicted_Close"] - float(test_rmse)

        st.write(f"### 🔮 Prediksi Harga {asset_name_display} (Upgrade Chart)")

        fig = go.Figure()

        # Actual
        fig.add_trace(go.Scatter(
            x=result_df["Date"], y=result_df["Original_Close"],
            mode="lines", name="Actual (Original Close)"
        ))

        # Predicted
        fig.add_trace(go.Scatter(
            x=result_df["Date"], y=result_df["Predicted_Close"],
            mode="lines", name="Predicted (LSTM)"
        ))

        # Moving average
        fig.add_trace(go.Scatter(
            x=result_df["Date"], y=result_df["MA7"],
            mode="lines", name="MA7 (Actual)", line=dict(dash="dot")
        ))
        fig.add_trace(go.Scatter(
            x=result_df["Date"], y=result_df["MA14"],
            mode="lines", name="MA14 (Actual)", line=dict(dash="dash")
        ))

        # Confidence band (Pred ± RMSE)
        if "Upper" in result_df.columns and "Lower" in result_df.columns:
            fig.add_trace(go.Scatter(
                x=result_df["Date"], y=result_df["Upper"],
                mode="lines", line=dict(width=0),
                showlegend=False, name="Upper"
            ))
            fig.add_trace(go.Scatter(
                x=result_df["Date"], y=result_df["Lower"],
                mode="lines", line=dict(width=0),
                fill="tonexty", fillcolor="rgba(0,0,0,0.10)",
                showlegend=True, name="Predicted ± RMSE"
            ))

        # Garis split train/test
        fig.add_vline(
            x=split_date,
            line_width=2,
            line_dash="dash",
            annotation_text="Split Train/Test",
            annotation_position="top"
        )

        # Shaded area untuk test/prediksi
        fig.add_vrect(
            x0=split_date,
            x1=result_df["Date"].iloc[-1],
            fillcolor="rgba(0,0,0,0.06)",
            line_width=0,
            annotation_text="Testing / Prediksi",
            annotation_position="top left"
        )

        fig.update_layout(
            title=f"Actual vs Predicted ({asset_name_display})",
            xaxis_title="Tanggal",
            yaxis_title="Harga",
            hovermode="x unified"
        )

        st.plotly_chart(fig, use_container_width=True)

        # ===== Residual / Error chart =====
        st.write("### 📉 Error (Residual) = Actual − Predicted")
        result_df["Error"] = result_df["Original_Close"] - result_df["Predicted_Close"]

        fig_err = go.Figure()
        fig_err.add_trace(go.Scatter(
            x=result_df["Date"], y=result_df["Error"],
            mode="lines", name="Residual Error"
        ))
        fig_err.add_hline(y=0, line_dash="dash")
        fig_err.add_vline(x=split_date, line_width=2, line_dash="dash")

        fig_err.update_layout(
            title="Residual Error (Actual − Predicted)",
            xaxis_title="Tanggal",
            yaxis_title="Error",
            hovermode="x unified"
        )

        st.plotly_chart(fig_err, use_container_width=True)

        # ===== Tabel hasil =====
        st.write("### 📊 Hasil Prediksi")
        st.dataframe(result_df, use_container_width=True, hide_index=True)

        # ===== Download CSV =====
        csv = result_df.to_csv(index=False).encode("utf-8")
        st.download_button(
            "⬇️ Download Hasil Prediksi (CSV)",
            data=csv,
            file_name=f"hasil_prediksi_{asset_name_display.lower()}_{start_date}_{end_date}.csv",
            mime="text/csv"
        )

    else:
        st.warning(
            "⚠️ Belum ada hasil prediksi yang bisa ditampilkan. "
            "Silakan jalankan model dengan rentang tanggal lebih panjang atau kurangi time_step."
        )
