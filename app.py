import os
from datetime import datetime

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
import tensorflow as tf
import yfinance as yf
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.preprocessing import MinMaxScaler


st.set_page_config(page_title="StockSense AI", page_icon="📈", layout="wide")

MODEL_PATH = "lstm_model.keras"
SEQ_LENGTH = 60


@st.cache_data(show_spinner=False)
def load_stock_data(start_date: str, end_date: str, ticker: str = "TSLA") -> pd.DataFrame:
    data = yf.download(ticker, start=start_date, end=end_date, progress=False)
    if data.empty:
        raise ValueError(f"No data returned for {ticker}")

    if isinstance(data.columns, pd.MultiIndex):
        data.columns = [col[0] if col[0] != "" else col[1] for col in data.columns]

    data = data[["Open", "High", "Low", "Close", "Volume"]].copy()
    data.index = pd.to_datetime(data.index)
    data = data.sort_index()
    data = data.rename_axis("Date")
    data["50_MA"] = data["Close"].rolling(window=50).mean()
    data["200_MA"] = data["Close"].rolling(window=200).mean()
    return data


@st.cache_resource(show_spinner=False)
def load_model():
    if not os.path.exists(MODEL_PATH):
        raise FileNotFoundError(f"Model file not found: {MODEL_PATH}")
    return tf.keras.models.load_model(MODEL_PATH)


def create_sequences(values: np.ndarray, seq_length: int):
    X, y = [], []
    for i in range(seq_length, len(values)):
        X.append(values[i - seq_length:i, 0])
        y.append(values[i, 0])
    return np.array(X), np.array(y)


def prepare_features(data: pd.DataFrame):
    close_series = data["Close"].dropna().values.reshape(-1, 1)
    scaler = MinMaxScaler(feature_range=(0, 1))
    scaled = scaler.fit_transform(close_series)
    X_full, y_full = create_sequences(scaled, SEQ_LENGTH)
    return scaler, X_full, y_full


def make_predictions(model, scaler, data: pd.DataFrame, horizon: int = 30):
    close_series = data["Close"].dropna().values.reshape(-1, 1)
    scaled = scaler.transform(close_series)

    actual_scaled = []
    predicted_scaled = []
    max_points = min(horizon, len(scaled) - SEQ_LENGTH)

    for i in range(SEQ_LENGTH, SEQ_LENGTH + max_points):
        window = scaled[i - SEQ_LENGTH:i].reshape(1, SEQ_LENGTH, 1)
        pred_scaled = model.predict(window, verbose=0)[0, 0]
        actual_scaled.append(scaled[i, 0])
        predicted_scaled.append(pred_scaled)

    actual = scaler.inverse_transform(np.array(actual_scaled).reshape(-1, 1))
    predicted = scaler.inverse_transform(np.array(predicted_scaled).reshape(-1, 1))
    dates = data.index[SEQ_LENGTH:SEQ_LENGTH + max_points]
    return actual, predicted, dates


def get_prediction_metrics(actual: np.ndarray, predicted: np.ndarray):
    mae = mean_absolute_error(actual, predicted)
    mse = mean_squared_error(actual, predicted)
    rmse = np.sqrt(mse)
    return mae, mse, rmse


def as_scalar(value):
    if isinstance(value, (pd.Series, np.ndarray)):
        if hasattr(value, "iloc"):
            return float(value.iloc[0])
        return float(value[0])
    return float(value)


st.set_page_config(page_title="StockSense AI", page_icon="📈", layout="wide")

use_dark_theme = st.sidebar.toggle("🌙 Dark theme", value=True)
plot_template = "plotly_dark" if use_dark_theme else "plotly_white"
plot_bg = "#0f172a" if use_dark_theme else "#ffffff"
paper_bg = "#111827" if use_dark_theme else "#ffffff"
text_color = "#f8fafc" if use_dark_theme else "#0f172a"

if use_dark_theme:
    theme_bg = "#0f172a"
    theme_panel = "#111827"
    theme_text = "#f8fafc"
    theme_muted = "#94a3b8"
    theme_card = "#1f2937"
else:
    theme_bg = "#f8fbff"
    theme_panel = "#eef4ff"
    theme_text = "#0f172a"
    theme_muted = "#475569"
    theme_card = "#ffffff"

st.markdown(
    f"""
    <style>
    .stApp {{
        background: {theme_bg};
        color: {theme_text};
    }}
    [data-testid="stSidebar"] {{
        background: {theme_panel};
        color: {theme_text};
    }}
    div[data-testid="metric-container"] {{
        background: {theme_card};
        border: 1px solid #94a3b8;
        border-radius: 14px;
        padding: 12px;
        box-shadow: 0 4px 14px rgba(15, 23, 42, 0.15);
    }}
    .stTabs [data-baseweb="tab-list"] {{
        gap: 8px;
    }}
    .stTabs [data-baseweb="tab"] {{
        border-radius: 10px;
        padding: 8px 14px;
        background: {theme_card};
    }}
    </style>
    """,
    unsafe_allow_html=True,
)

st.markdown(f"# 📈 StockSense AI")
st.caption("Tesla Stock Price Forecasting Using LSTM")

with st.sidebar:
    st.header("Project Controls")
    st.markdown("### Quick Settings")
    ticker = st.text_input("Ticker Symbol", value="TSLA")
    forecast_horizon = st.slider("Forecast Horizon", min_value=5, max_value=60, value=30, step=5)
    refresh = st.button("Refresh Data", use_container_width=True)
    st.markdown("---")
    st.caption("Pre-trained LSTM model • Live Yahoo Finance data")

section = st.sidebar.radio(
    "Navigation",
    ["Home", "EDA", "Forecast", "Insights"],
    index=0,
    help="Switch between the dashboard sections",
)

if refresh:
    st.cache_data.clear()

end_date = datetime.now().strftime("%Y-%m-%d")
start_date = "2015-01-01"

try:
    data = load_stock_data(start_date, end_date, ticker=ticker)
    model = load_model()
except Exception as exc:
    st.error(f"Unable to load app resources: {exc}")
    st.stop()

latest = data.iloc[-1]
previous = data.iloc[-2] if len(data) > 1 else latest
latest_close = as_scalar(latest["Close"])
previous_close = as_scalar(previous["Close"])
change = latest_close - previous_close
change_pct = (change / previous_close) * 100 if previous_close else 0.0

trend_color = "#22c55e" if change_pct >= 0 else "#ef4444"
trend_label = "Bullish" if change_pct >= 0 else "Bearish"

if section == "Home":
    st.markdown("### 🔎 Market Snapshot")
    st.markdown(
        f"""
        <div style="background:linear-gradient(135deg, #1d4ed8, #0f172a); padding:16px 18px; border-radius:16px; color:white; margin-bottom:12px;">
        <div style="font-size:12px; text-transform:uppercase; letter-spacing:1.2px; opacity:0.8;">Latest Trend</div>
        <div style="font-size:24px; font-weight:700; margin-top:4px;">{trend_label}</div>
        <div style="font-size:18px; margin-top:4px; color:{trend_color};">{change_pct:+.2f}% from the prior close</div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    eda_col1, eda_col2, eda_col3 = st.columns(3)
    eda_col1.markdown(f"<div style='background:#1f2937;padding:14px;border-radius:12px;color:white'><b>Latest Close</b><br><span style='font-size:22px'>${latest_close:.2f}</span></div>", unsafe_allow_html=True)
    eda_col2.markdown(f"<div style='background:#2563eb;padding:14px;border-radius:12px;color:white'><b>Price Range</b><br><span style='font-size:22px'>${float(data['Low'].min()):.2f} – ${float(data['High'].max()):.2f}</span></div>", unsafe_allow_html=True)
    eda_col3.markdown(f"<div style='background:#0f766e;padding:14px;border-radius:12px;color:white'><b>Latest Volume</b><br><span style='font-size:22px'>{int(data['Volume'].iloc[-1]):,}</span></div>", unsafe_allow_html=True)

    st.markdown("---")
    st.subheader("Quick Overview")
    st.write("This dashboard combines live Tesla market data with an LSTM-based forecasting workflow for a polished, portfolio-ready experience.")
    st.info("Use the navigation on the left to explore the data, model predictions, and insights.")

elif section == "EDA":
    st.subheader("Historical Closing Price")
    plot_data = data.reset_index()
    plot_data = plot_data.rename(columns={"index": "Date"}) if "index" in plot_data.columns else plot_data
    plot_data["Year"] = plot_data["Date"].dt.year
    fig1 = px.line(
        plot_data,
        x="Date",
        y="Close",
        animation_frame="Year",
        title="Tesla Closing Price History",
        color_discrete_sequence=["#38bdf8"],
    )
    fig1.update_layout(
        template=plot_template,
        paper_bgcolor=paper_bg,
        plot_bgcolor=plot_bg,
        font=dict(color=text_color),
        height=380,
        margin=dict(l=20, r=20, t=50, b=20),
    )
    fig1.update_traces(mode="lines+markers", marker=dict(size=4))
    st.plotly_chart(fig1, use_container_width=True)

    st.subheader("Daily Returns")
    returns_df = plot_data[["Date", "Close", "Year"]].copy()
    returns_df["Daily Return (%)"] = returns_df["Close"].pct_change() * 100
    fig_returns = px.line(
        returns_df,
        x="Date",
        y="Daily Return (%)",
        animation_frame="Year",
        title="Daily Return Trend",
        color_discrete_sequence=["#a78bfa"],
    )
    fig_returns.add_hline(y=0, line_color="#94a3b8", line_dash="dash")
    fig_returns.update_layout(
        template=plot_template,
        paper_bgcolor=paper_bg,
        plot_bgcolor=plot_bg,
        font=dict(color=text_color),
        height=320,
        margin=dict(l=20, r=20, t=50, b=20),
    )
    st.plotly_chart(fig_returns, use_container_width=True)

    st.subheader("Volume Trend")
    fig_volume = px.bar(
        plot_data,
        x="Date",
        y="Volume",
        title="Tesla Trading Volume",
        color="Volume",
        color_continuous_scale="Blues",
    )
    fig_volume.update_layout(
        template=plot_template,
        paper_bgcolor=paper_bg,
        plot_bgcolor=plot_bg,
        font=dict(color=text_color),
        height=320,
        margin=dict(l=20, r=20, t=50, b=20),
    )
    st.plotly_chart(fig_volume, use_container_width=True)

    st.subheader("Moving Averages")
    fig2 = go.Figure()
    fig2.add_trace(go.Scatter(x=data.index, y=data["Close"], mode="lines", name="Close Price", line=dict(color="#38bdf8", width=2)))
    fig2.add_trace(go.Scatter(x=data.index, y=data["50_MA"], mode="lines", name="50-Day MA", line=dict(color="#f59e0b", width=2)))
    fig2.add_trace(go.Scatter(x=data.index, y=data["200_MA"], mode="lines", name="200-Day MA", line=dict(color="#ef4444", width=2)))
    fig2.update_layout(
        template=plot_template,
        paper_bgcolor=paper_bg,
        plot_bgcolor=plot_bg,
        font=dict(color=text_color),
        title="Tesla Price with Moving Averages",
        height=380,
        margin=dict(l=20, r=20, t=50, b=20),
    )
    st.plotly_chart(fig2, use_container_width=True)

elif section == "Forecast":
    st.subheader("Actual vs Predicted Price")
    scaler, _, _ = prepare_features(data)
    actual, predictions, forecast_dates = make_predictions(model, scaler, data, horizon=min(forecast_horizon, len(data) - SEQ_LENGTH))
    mae, mse, rmse = get_prediction_metrics(actual, predictions)

    pred_df = pd.DataFrame({
        "Date": forecast_dates,
        "Actual": actual.flatten(),
        "Predicted": predictions.flatten(),
    })

    fig3 = px.line(pred_df, x="Date", y=["Actual", "Predicted"], title="Actual vs Predicted Tesla Price")
    fig3.update_layout(
        template=plot_template,
        paper_bgcolor=paper_bg,
        plot_bgcolor=plot_bg,
        font=dict(color=text_color),
        height=380,
        margin=dict(l=20, r=20, t=50, b=20),
    )
    st.plotly_chart(fig3, use_container_width=True)

    st.markdown("### Prediction Error")
    error_df = pd.DataFrame({"Date": pred_df["Date"], "Error": (pred_df["Actual"] - pred_df["Predicted"]).abs()})
    fig_error = px.line(error_df, x="Date", y="Error", title="Absolute Prediction Error")
    fig_error.update_layout(
        template=plot_template,
        paper_bgcolor=paper_bg,
        plot_bgcolor=plot_bg,
        font=dict(color=text_color),
        height=320,
        margin=dict(l=20, r=20, t=50, b=20),
    )
    st.plotly_chart(fig_error, use_container_width=True)

    st.markdown("### Model Performance")
    col_a, col_b, col_c = st.columns(3)
    col_a.metric("MAE", f"{mae:.4f}")
    col_b.metric("MSE", f"{mse:.4f}")
    col_c.metric("RMSE", f"{rmse:.4f}")

    st.markdown("### Prediction Table")
    st.dataframe(pred_df.head(15).round(2), use_container_width=True)

    csv = pred_df.to_csv(index=False).encode("utf-8")
    st.download_button(
        label="Download prediction results as CSV",
        data=csv,
        file_name="tesla_predictions.csv",
        mime="text/csv",
    )

else:
    st.subheader("Market Insights")
    st.write(f"The latest close is ${latest_close:.2f} with a {change_pct:+.2f}% change from the previous day.")
    st.write(f"The latest trading volume is {int(data['Volume'].iloc[-1]):,} shares, and the 50-day moving average is ${float(data['50_MA'].dropna().iloc[-1]):.2f}.")

    eda_summary = pd.DataFrame({
        "Metric": ["Rows", "Start Date", "End Date", "Average Close", "Average Volume"],
        "Value": [
            len(data),
            str(data.index.min().date()),
            str(data.index.max().date()),
            f"${data['Close'].mean():.2f}",
            f"{int(data['Volume'].mean()):,}",
        ],
    })
    st.dataframe(eda_summary, use_container_width=True, hide_index=True)

    stats = pd.DataFrame({
        "Metric": ["Open", "High", "Low", "Close", "Volume"],
        "Latest Value": [
            f"${float(latest['Open']):.2f}",
            f"${float(latest['High']):.2f}",
            f"${float(latest['Low']):.2f}",
            f"${latest_close:.2f}",
            f"{int(latest['Volume']):,}",
        ],
    })
    st.dataframe(stats, use_container_width=True, hide_index=True)

    st.markdown("### Candlestick-style Overview")
    candle = go.Figure(data=[go.Candlestick(
        x=data.index,
        open=data['Open'],
        high=data['High'],
        low=data['Low'],
        close=data['Close'],
        increasing_line_color='#16a34a',
        decreasing_line_color='#dc2626'
    )])
    candle.update_layout(
        template=plot_template,
        paper_bgcolor=paper_bg,
        plot_bgcolor=plot_bg,
        font=dict(color=text_color),
        title="Tesla OHLC Overview",
        height=350,
        margin=dict(l=20, r=20, t=50, b=20),
    )
    st.plotly_chart(candle, use_container_width=True)

st.markdown("---")
st.caption("Built for StockSense AI — a portfolio-ready LSTM forecasting dashboard.")
