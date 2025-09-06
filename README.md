# Stock Price Prediction App

## Overview
This project introduces a financial forecasting application that predicts prices of **stocks, cryptocurrencies, and commodities**. The app integrates **Prophet** (statistical model) and **Long Short-Term Memory (LSTM)** (deep learning model) to generate accurate forecasts.  

The system is built with **Streamlit** and designed to be interactive and user-friendly. Users can select assets such as **Amazon, Apple, Tesla, Bitcoin, Ethereum, gold, silver, or crude oil**, visualize historical price movements with technical indicators, and generate forecasts for future trends.  

The goal is to provide both investors and analysts with a decision-support tool that combines **interpretable statistical modeling** with **advanced machine learning capabilities**.  

---

## Data
- **Source**: All financial data is retrieved from **Yahoo Finance**.  
- **Assets covered**:  
  - Equities: Amazon (AMZN), Apple (AAPL), Tesla (TSLA), Microsoft (MSFT), etc.  
  - Cryptocurrencies: Bitcoin (BTC-USD), Ethereum (ETH-USD)  
  - Commodities: Gold (GC=F), Silver (SI=F), Crude Oil (CL=F)  
- **Attributes**: Open, High, Low, Close, Volume  
- **Technical Indicators**: EMA, MACD, RSI, Parabolic SAR, Stochastic Oscillator  

---

## Methodology
1. **Data Preprocessing**  
   - Cleaned missing values, normalized features (Min-Max scaling)  
   - Computed technical indicators to enrich input features  
   - Applied an 80/20 train-test split  

2. **Modeling**  
   - **Prophet**: captures long-term trends and seasonality  
   - **LSTM**: trained on time-series sequences to capture short- and long-term dependencies  

3. **Evaluation**  
   - Models assessed with RMSE, MAE, R², MAPE  
   - Performance compared across assets  

---

## Results and Key Findings

### Model Performance
| Asset     | Model   | R²   | RMSE | MAE  | MAPE  |
|-----------|---------|------|------|------|-------|
| Amazon    | LSTM    | 0.98 | 3.26 | 2.60 | 1.76% |
| Apple     | LSTM    | 0.98 | 3.33 | 2.67 | 1.79% |
| Eregli    | LSTM    | 0.82 | 1.46 | 1.20 | 2.82% |

### Business Insights
- **Stocks (e.g., Amazon, Apple)**: The LSTM model demonstrated excellent predictive accuracy, suggesting that deep learning can support portfolio optimization and risk management strategies for equities.  
- **Emerging markets (e.g., Eregli)**: Lower R² indicates more volatility and external macroeconomic influences, highlighting the importance of combining forecasts with domain-specific knowledge.  
- **Cryptocurrencies and commodities**: While not fully reported here, initial tests showed LSTM capturing extreme volatility better than Prophet, making it more suitable for fast-moving markets such as Bitcoin and oil.  
- **Prophet vs LSTM**:  
  - Prophet excels in interpretability, seasonality, and trend analysis → useful for long-term strategy reports.  
  - LSTM provides higher accuracy in volatile datasets → valuable for tactical trading and short-term risk assessment.  
- **Managerial implication**: A hybrid use of Prophet for long-term projections and LSTM for short-term tactical decisions provides a balanced forecasting strategy for businesses and investors.  

---

## Application Features
- Interactive selection of assets (stocks, cryptocurrencies, commodities)  
- Visualization of historical data enriched with EMA, MACD, RSI, and other indicators  
- Future predictions using both Prophet and LSTM models  
- Interactive performance metrics displayed (RMSE, MAE, R², MAPE)  
- Side-by-side comparison of actual vs predicted prices  

---

## Technologies Used
- **Python** for data processing and modeling  
- **Streamlit** for the interactive web interface  
- **Yahoo Finance API** for historical data retrieval  
- **Prophet** for time-series forecasting  
- **TensorFlow/Keras** for LSTM modeling  
- **Pandas, NumPy** for data manipulation  
- **Matplotlib, Plotly** for visualization  
- **TA-Lib / Technical Analysis library** for financial indicators  

