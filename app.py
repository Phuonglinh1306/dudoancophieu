import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf

from datetime import date, timedelta
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
from tensorflow.keras.callbacks import EarlyStopping

# =========================
# 1. GIAO DIỆN ỨNG DỤNG
# =========================
st.set_page_config(
    page_title="Dự báo giá cổ phiếu",
    page_icon="📈",
    layout="wide"
)

st.title("📈 Ứng dụng dự báo giá cổ phiếu")
st.write(
    "Dự báo giá cổ phiếu bằng mô hình trí tuệ nhân tạo LSTM."
)
st.info(
    "Nhập mã cổ phiếu, chọn ngày bắt đầu và ngày kết thúc, "
    "sau đó nhấn Dự đoán."
)

# =========================
# 2. NHẬP THÔNG TIN
# =========================
with st.sidebar:
    st.header("Thông tin dự báo")

    ticker_input = st.text_input(
        "Mã cổ phiếu",
        value="VNM",
        placeholder="Ví dụ: VNM, FPT, TCB"
    )

    start_date = st.date_input(
        "Ngày bắt đầu",
        value=date(2018, 1, 1),
        min_value=date(2000, 1, 1),
        max_value=date.today()
    )

    end_date = st.date_input(
        "Ngày kết thúc",
        value=date.today(),
        min_value=date(2000, 1, 1),
        max_value=date.today()
    )

    predict_button = st.button(
        "🔮 Dự đoán",
        type="primary",
        use_container_width=True
    )

st.caption(
    "Mô hình sử dụng dữ liệu 60 phiên trước để dự báo "
    "10 phiên giao dịch tiếp theo."
)

# =========================
# 3. TẠO CHUỖI DỮ LIỆU
# =========================
def create_sequences(data, window_size=60):
    X, y = [], []

    for i in range(window_size, len(data)):
        X.append(data[i - window_size:i])
        y.append(data[i])

    return np.array(X), np.array(y)


# =========================
# 4. TẢI DỮ LIỆU CỔ PHIẾU
# =========================
def get_stock_data(ticker, start, end):
    # Tự thêm hậu tố cho mã cổ phiếu Việt Nam
    if "." not in ticker:
        ticker = ticker + ".VN"

    df = yf.download(
        ticker,
        start=start.isoformat(),
        end=(end + timedelta(days=1)).isoformat(),
        auto_adjust=True,
        progress=False
    )

    if df.empty:
        raise ValueError(
            "Không tìm thấy dữ liệu. "
            "Hãy kiểm tra mã cổ phiếu và khoảng ngày."
        )

    # Xử lý dữ liệu có tên cột nhiều cấp
    if isinstance(df.columns, pd.MultiIndex):
        close = df.xs(
            "Close",
            axis=1,
            level=0
        ).iloc[:, 0]
    else:
        close = df["Close"]

    prices = pd.DataFrame({
        "Close": pd.to_numeric(close, errors="coerce")
    }).dropna()

    prices = prices.sort_index()

    if len(prices) < 100:
        raise ValueError(
            "Cần tối thiểu 100 phiên dữ liệu. "
            "Hãy chọn khoảng thời gian dài hơn."
        )

    return ticker, prices


# =========================
# 5. HUẤN LUYỆN MÔ HÌNH LSTM
# =========================
def train_model(prices):
    values = prices[["Close"]].values

    # Chia dữ liệu theo trình tự thời gian
    split = int(len(values) * 0.8)

    train_data = values[:split]
    scaler = MinMaxScaler()

    # Chỉ fit scaler trên tập huấn luyện
    train_scaled = scaler.fit_transform(train_data)
    all_scaled = scaler.transform(values)

    X_all, y_all = create_sequences(train_scaled, 60)

    # Chia tập huấn luyện và validation theo thời gian
    val_size = max(1, int(len(X_all) * 0.1))

    X_train = X_all[:-val_size]
    y_train = y_all[:-val_size]

    X_val = X_all[-val_size:]
    y_val = y_all[-val_size:]

    tf.keras.backend.clear_session()
    np.random.seed(42)
    tf.random.set_seed(42)

    model = Sequential([
        tf.keras.Input(shape=(60, 1)),
        LSTM(50, return_sequences=True),
        LSTM(50),
        Dense(1)
    ])

    model.compile(
        optimizer="adam",
        loss="mean_squared_error"
    )

    early_stop = EarlyStopping(
        monitor="val_loss",
        patience=5,
        restore_best_weights=True
    )

    history = model.fit(
        X_train,
        y_train,
        validation_data=(X_val, y_val),
        epochs=20,
        batch_size=32,
        shuffle=False,
        callbacks=[early_stop],
        verbose=0
    )

    # Đánh giá trên 20% dữ liệu cuối
    test_data = all_scaled[split:]

    test_combined = np.vstack([
        train_scaled[-60:],
        test_data
    ])

    X_test, y_test = create_sequences(test_combined, 60)

    pred_scaled = model.predict(
        X_test,
        verbose=0
    )

    pred = scaler.inverse_transform(pred_scaled).ravel()
    actual = scaler.inverse_transform(
        y_test.reshape(-1, 1)
    ).ravel()

    rmse = np.sqrt(mean_squared_error(actual, pred))
    mae = mean_absolute_error(actual, pred)
    r2 = r2_score(actual, pred)

    # Dự báo cuốn chiếu 10 phiên
    window = all_scaled[-60:].reshape(1, 60, 1)
    future_scaled = []

    for _ in range(10):
        next_value = model.predict(window, verbose=0)

        future_scaled.append(next_value[0, 0])

        window = np.concatenate([
            window[:, 1:, :],
            next_value.reshape(1, 1, 1)
        ], axis=1)

    future_prices = scaler.inverse_transform(
        np.array(future_scaled).reshape(-1, 1)
    ).ravel()

    return (
        history,
        actual,
        pred,
        rmse,
        mae,
        r2,
        future_prices
    )


# =========================
# 6. THỰC HIỆN KHI BẤM NÚT
# =========================
if predict_button:

    if start_date >= end_date:
        st.error("Ngày bắt đầu phải sớm hơn ngày kết thúc.")

    else:
        try:
            ticker = ticker_input.strip().upper()

            if not ticker:
                raise ValueError("Bạn chưa nhập mã cổ phiếu.")

            with st.spinner("Đang tải dữ liệu cổ phiếu..."):
                ticker, prices = get_stock_data(
                    ticker,
                    start_date,
                    end_date
                )

            st.success(
                f"Đã tải dữ liệu {ticker}: "
                f"{len(prices)} phiên giao dịch."
            )

            with st.spinner(
                "AI đang huấn luyện mô hình LSTM..."
            ):
                (
                    history,
                    actual,
                    pred,
                    rmse,
                    mae,
                    r2,
                    future_prices
                ) = train_model(prices)

            # -------------------------
            # HIỂN THỊ CHỈ SỐ ĐÁNH GIÁ
            # -------------------------
            st.subheader("Đánh giá mô hình")

            col1, col2, col3 = st.columns(3)

            col1.metric("RMSE", f"{rmse:.3f}")
            col2.metric("MAE", f"{mae:.3f}")
            col3.metric("R²", f"{r2:.3f}")

            # -------------------------
            # BIỂU ĐỒ LOSS
            # -------------------------
            st.subheader("Quá trình huấn luyện")

            fig, ax = plt.subplots(figsize=(10, 4))

            ax.plot(
                history.history["loss"],
                label="Train Loss"
            )
            ax.plot(
                history.history["val_loss"],
                label="Validation Loss"
            )

            ax.set_xlabel("Epoch")
            ax.set_ylabel("Loss")
            ax.legend()
            ax.grid(alpha=0.3)

            st.pyplot(fig)
            plt.close(fig)

            # -------------------------
            # GIÁ THỰC TẾ VÀ GIÁ DỰ BÁO
            # -------------------------
            st.subheader("Giá thực tế và giá dự báo")

            test_dates = prices.index[-len(actual):]

            fig, ax = plt.subplots(figsize=(12, 5))

            ax.plot(
                test_dates,
                actual,
                label="Giá thực tế"
            )
            ax.plot(
                test_dates,
                pred,
                label="Giá dự báo"
            )

            ax.set_xlabel("Ngày")
            ax.set_ylabel("Giá đóng cửa")
            ax.legend()
            ax.grid(alpha=0.3)

            st.pyplot(fig)
            plt.close(fig)

            # -------------------------
            # DỰ BÁO 10 PHIÊN TIẾP THEO
            # -------------------------
            st.subheader("Dự báo 10 phiên tiếp theo")

            future_dates = pd.bdate_range(
                start=prices.index[-1] + pd.Timedelta(days=1),
                periods=10
            )

            result = pd.DataFrame({
                "Ngày dự báo": future_dates.strftime("%Y-%m-%d"),
                "Giá dự báo": np.round(future_prices, 3)
            })

            st.dataframe(
                result,
                use_container_width=True,
                hide_index=True
            )

            fig, ax = plt.subplots(figsize=(12, 5))

            recent = prices.tail(60)

            ax.plot(
                recent.index,
                recent["Close"],
                label="Giá đóng cửa gần nhất"
            )

            ax.plot(
                future_dates,
                future_prices,
                marker="o",
                linestyle="--",
                label="Giá dự báo"
            )

            ax.set_xlabel("Ngày")
            ax.set_ylabel("Giá đóng cửa")
            ax.legend()
            ax.grid(alpha=0.3)

            st.pyplot(fig)
            plt.close(fig)

            # -------------------------
            # TẢI KẾT QUẢ
            # -------------------------
            st.download_button(
                "⬇️ Tải kết quả dự báo CSV",
                data=result.to_csv(
                    index=False
                ).encode("utf-8-sig"),
                file_name=f"{ticker}_forecast.csv",
                mime="text/csv"
            )

            st.warning(
                "Đây là kết quả mô hình học máy mang tính tham khảo, "
                "không đảm bảo giá thực tế và không phải lời khuyên đầu tư."
            )

        except Exception as e:
            st.error(f"Đã xảy ra lỗi: {e}")

else:
    st.info(
        "Hãy nhập mã cổ phiếu, chọn ngày bắt đầu và ngày kết thúc "
        "ở thanh bên trái rồi nhấn Dự đoán."
    )
