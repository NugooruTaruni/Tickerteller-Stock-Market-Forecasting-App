# Amazon Financial Time Series Analysis and Forecasting Web App

This project explores Amazon's stock performance in 2023 using financial time series analysis techniques, followed by building a web application for forecasting future stock prices. We used a combination of technical analysis and machine learning for prediction, built as a user-friendly web app powered by **Streamlit**.



## Overview

The project consists of two main components:

1. **Technical Analysis**: A detailed financial technical analysis of Amazon's stock performance for 2023, utilizing various libraries like **pandas**, **yfinance**, **ta**, and **matplotlib**.
2. **Web Application**: A Streamlit-based web application that allows users to analyze Amazon's stock using technical analysis indicators, view financial news, and forecast stock prices using **Prophet**.



## Features

### Technical Analysis (Jupyter Notebook)

- **Libraries Used**: `pandas`, `yfinance`, `ta`, `matplotlib`
- The Jupyter notebook explores Amazon’s stock data and answers the following questions:
  - What was Amazon's overall stock performance in 2023 compared to the broader market?
  - What were the most volatile periods for Amazon's stock during the year?
  - How did Amazon's stock price trend throughout the year?
  - Did Amazon's stock outperform or underperform its historical averages?
  - What were the major events or news that impacted Amazon's stock price?

### Web Application (Streamlit)

The web application was built using **Streamlit** to display the results of technical analysis and forecasting. The key features of the app include:

- **Stock Details and News**: Displays current stock information and relevant news powered by **yfinance** and **Finnhub**.
- **Technical Analysis Indicators**:
  - **Trend Analysis**: Simple Moving Average (SMA)
  - **Momentum**: Relative Strength Index (RSI)
  - **Volume**: On Balance Volume (OBV)
  - **Volatility**: Bollinger Bands
- **Forecasting**:
  - Forecast stock prices for up to **4 years** ahead using **Prophet**.
  - Ability to download the forecasted data for further analysis.


## Libraries and Technologies Used

- **Python Libraries**:
  - `pandas`: Data manipulation and analysis
  - `yfinance`: Fetch historical stock data
  - `ta`: Technical analysis indicators
  - `matplotlib`: Plotting and visualization
  - `Prophet`: Time series forecasting
  - `finnhub`: Real-time news and stock details

- **Web Framework**:
  - `Streamlit`: For building the interactive web application


## How to Run

### 1. Clone the Repository:

git clone  https://github.com/NugooruTaruni/Tickerteller-Stock-Market-Forecasting-App.git
cd amazon-financial-analysis

### 2. Install Dependencies:
Make sure you have Python 3.x installed, then create a virtual environment and install the necessary libraries.

### 3. Run the Streamlit App:
streamlit run main.py

This will launch the web application in your default browser.

### 4. Technical Analysis (Jupyter Notebook):
The technical analysis can be run by opening the analysis.ipynb notebook in Jupyter:

jupyter notebook analysis.ipynb

This will display the results of the financial analysis in the notebook.


## Future Enhancements
- Real-time stock data integration (currently using historical data).
- Adding more advanced forecasting models for comparison.
- Expanding news integration to display live events affecting stock prices.
- User customization features like choosing different stocks for analysis.
