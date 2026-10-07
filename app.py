import numpy as np
import pandas as pd
import streamlit as st
import yfinance as yf
import plotly.graph_objects as go

from math import erf, sqrt, exp, log, pi

# ============================================================
# 1. Standard Normal CDF
# ============================================================
def normal_cdf(x):
    return 0.5 * (1 + erf(x / sqrt(2)))

# ============================================================
# 2. Black-Scholes Model
# ============================================================
def black_scholes(S0, K, T, r, sigma, option_type):
    d1 = (
        log(S0 / K)
        + (r + 0.5 * sigma**2) * T
    ) / (sigma * sqrt(T))

    d2 = d1 - sigma * sqrt(T)

    if option_type == "Call":
        price = (
            S0 * normal_cdf(d1)
            - K * exp(-r * T) * normal_cdf(d2)
        )
    else:
        price = (
            K * exp(-r * T) * normal_cdf(-d2)
            - S0 * normal_cdf(-d1)
        )

    return price

# ============================================================
# 3. Monte Carlo Simulation
# ============================================================
def monte_carlo(S0, K, T, r, sigma, N, option_type):
    # Random numbers
    Z = np.random.normal(0, 1, N)

    # Simulated price at expiration
    ST = S0 * np.exp(
        (r - 0.5 * sigma**2) * T
        + sigma * np.sqrt(T) * Z
    )

    # Payoff
    if option_type == "Call":
        payoff = np.maximum(ST - K, 0)
    else:
        payoff = np.maximum(K - ST, 0)

    # Monte Carlo Option Price
    option_price = np.exp(-r * T) * np.mean(payoff)

    # Standard Error
    standard_error = (
        np.exp(-r * T)
        * np.std(payoff, ddof=1)
        / np.sqrt(N)
    )

    return option_price, standard_error, ST, payoff

# ============================================================
# 4. Get Market Data
# ============================================================
def get_market_data(ticker_symbol):
    ticker = yf.Ticker(ticker_symbol)
    data = ticker.history(period="1y")

    if data.empty:
        return None, None, None

    # Current Price
    S0 = float(data["Close"].iloc[-1])

    # Log Return
    log_returns = np.log(data["Close"] / data["Close"].shift(1))

    # Annualized Volatility
    sigma = float(log_returns.std() * np.sqrt(252))

    return data, S0, sigma

# ============================================================
# 5. Streamlit Page
# ============================================================
st.set_page_config(
    page_title="Option Pricing Simulator",
    page_icon="📈",
    layout="wide"
)

# ============================================================
# 6. Title
# ============================================================
st.title("📈 Mathematical Option Pricing Simulator")
st.subheader("Black-Scholes Model และ Monte Carlo Simulation")
st.write(
    """
    โปรแกรมนี้ใช้หลักการทางคณิตศาสตร์ในการประเมิน
    ราคาเหมาะสมทางทฤษฎีของสัญญา Option
    โดยเปรียบเทียบผลลัพธ์จากวิธี Black-Scholes
    และ Monte Carlo Simulation
    """
)

# ============================================================
# 7. Input
# ============================================================
st.sidebar.header("Input Parameters")

ticker_symbol = st.sidebar.text_input("Ticker Symbol", value="AAPL").upper()
K = st.sidebar.number_input("Strike Price (K)", min_value=1.0, value=100.0, step=1.0)
T = st.sidebar.number_input("Time to Expiration (T)", min_value=0.1, max_value=10.0, value=1.0, step=0.1)
r = st.sidebar.number_input("Risk-free Rate (r)", min_value=0.0, max_value=0.30, value=0.05, step=0.01)
N = st.sidebar.number_input("Number of Simulations", min_value=1000, max_value=100000, value=10000, step=1000)
option_type = st.sidebar.selectbox("Option Type", ["Call", "Put"])
run_button = st.sidebar.button("Run Simulation")

# ============================================================
# 8. Run Project
# ============================================================
if run_button:
    # --------------------------------------------------------
    # Market Data
    # --------------------------------------------------------
    try:
        data, S0, sigma = get_market_data(ticker_symbol)
    except Exception as e:
        st.error("ไม่สามารถดึงข้อมูลตลาดได้")
        st.write(e)
        st.stop()

    if data is None:
        st.error("ไม่พบข้อมูลของ Ticker นี้")
        st.stop()

    # --------------------------------------------------------
    # Black-Scholes
    # --------------------------------------------------------
    bs_price = black_scholes(S0, K, T, r, sigma, option_type)

    # --------------------------------------------------------
    # Monte Carlo
    # --------------------------------------------------------
    mc_price, standard_error, ST, payoff = monte_carlo(S0, K, T, r, sigma, N, option_type)

    # --------------------------------------------------------
    # Error
    # --------------------------------------------------------
    error = abs(bs_price - mc_price)

    # --------------------------------------------------------
    # Confidence Interval
    # --------------------------------------------------------
    lower = mc_price - 1.96 * standard_error
    upper = mc_price + 1.96 * standard_error

    # --------------------------------------------------------
    # Probability ITM
    # --------------------------------------------------------
    if option_type == "Call":
        probability_itm = np.mean(ST > K)
    else:
        probability_itm = np.mean(ST < K)

    # ========================================================
    # 9. Market Data
    # ========================================================
    st.header(f"Market Data: {ticker_symbol}")
    col1, col2, col3 = st.columns(3)

    with col1:
        st.metric("Current Price (S₀)", f"{S0:.2f}")
    with col2:
        st.metric("Volatility (σ)", f"{sigma:.2%}")
    with col3:
        st.metric("Option Type", option_type)

    # ========================================================
    # 10. Results
    # ========================================================
    st.header("Option Pricing Results")
    col1, col2, col3, col4 = st.columns(4)

    with col1:
        st.metric("Black-Scholes", f"{bs_price:.4f}")
    with col2:
        st.metric("Monte Carlo", f"{mc_price:.4f}")
    with col3:
        st.metric("Absolute Error", f"{error:.4f}")
    with col4:
        st.metric("Probability ITM", f"{probability_itm:.2%}")

    st.write(
        f"""
        **95% Confidence Interval ของ Monte Carlo**

        [{lower:.4f}, {upper:.4f}]
        """
    )

    # ========================================================
    # 11. Graph 1
    # Terminal Price Distribution
    # ========================================================
    st.header("1. Distribution of Asset Price at Expiration")
    histogram = go.Figure()

    histogram.add_trace(go.Histogram(x=ST, nbinsx=50, name="Simulated Price"))
    histogram.add_vline(x=K, line_dash="dash", annotation_text="Strike Price K")
    histogram.add_vline(x=S0, line_dash="dot", annotation_text="Current Price S₀")
    histogram.update_layout(xaxis_title="Asset Price at Expiration", yaxis_title="Frequency", title="Monte Carlo Distribution")
    
    st.plotly_chart(histogram, use_container_width=True)

    # ========================================================
    # 12. Graph 2
    # Simulated Price Paths
    # ========================================================
    st.header("2. Monte Carlo Simulated Price Paths")
    
    steps = 100
    dt = T / steps
    number_of_paths = 100

    # Random numbers
    Z_paths = np.random.normal(0, 1, (number_of_paths, steps))

    # Array สำหรับเก็บราคา
    price_paths = np.zeros((number_of_paths, steps + 1))
    price_paths[:, 0] = S0

    # Simulate Price Paths
    for t in range(steps):
        price_paths[:, t + 1] = (
            price_paths[:, t]
            * np.exp((r - 0.5 * sigma**2) * dt + sigma * np.sqrt(dt) * Z_paths[:, t])
        )

    time = np.linspace(0, T, steps + 1)
    paths = go.Figure()

    for i in range(number_of_paths):
        paths.add_trace(
            go.Scatter(x=time, y=price_paths[i], mode="lines", showlegend=False, opacity=0.2)
        )

    paths.add_hline(y=K, line_dash="dash", annotation_text="Strike Price K")
    paths.update_layout(xaxis_title="Time (years)", yaxis_title="Asset Price", title="Monte Carlo Simulated Price Paths")
    
    st.plotly_chart(paths, use_container_width=True)

    # ========================================================
    # 13. Graph 3
    # Historical Price
    # ========================================================
    st.header("3. Historical Closing Price")
    historical = go.Figure()

    historical.add_trace(go.Scatter(x=data.index, y=data["Close"], mode="lines", name="Closing Price"))
    historical.update_layout(xaxis_title="Date", yaxis_title="Price", title=f"{ticker_symbol} Historical Closing Price")
    
    st.plotly_chart(historical, use_container_width=True)

    # ========================================================
    # 14. Summary
    # ========================================================
    st.header("Summary")
    st.write(
        f"""
        จากการคำนวณ **{option_type} Option**
        ของ **{ticker_symbol}**

        ราคาปัจจุบันของสินทรัพย์ = **{S0:.4f}**
        Volatility = **{sigma:.2%}**

        ราคาโดย Black-Scholes = **{bs_price:.4f}**
        ราคาโดย Monte Carlo = **{mc_price:.4f}**
        ค่าความคลาดเคลื่อน = **{error:.4f}**

        ผลลัพธ์จาก Monte Carlo ควรมีค่าใกล้เคียงกับ
        ราคาที่คำนวณจาก Black-Scholes โดยค่าความคลาดเคลื่อน
        สามารถเปลี่ยนแปลงได้ตามจำนวนครั้งของการจำลอง
        และการสุ่มตัวอย่าง
        """
    )

else:
    st.info("กรุณากรอกข้อมูลทางด้านซ้าย แล้วกด Run Simulation")