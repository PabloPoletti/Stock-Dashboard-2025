import streamlit as st
import yfinance as yf
import pandas as pd
import matplotlib.pyplot as plt
import ta
import seaborn as sns

# Configure the page
st.set_page_config(layout="wide")

# Create a sidebar for navigation
st.sidebar.title('Navigation')
page = st.sidebar.selectbox('Select a page:', ['Stock Analysis', 'Correlation Matrix'])

if page == 'Stock Analysis':
    # Inputs in the sidebar
    st.sidebar.title('Stock Analysis')
    symbol = st.sidebar.text_input('Symbol', value='AAPL')
    period = st.sidebar.selectbox('Period', ['1mo', '3mo', '6mo', '1y', '5y', 'max'], index=3)
    interval = st.sidebar.selectbox('Interval', ['1d', '1wk', '1mo'], index=0)
    chart_type = st.sidebar.selectbox('Chart Type', ['Line', 'Candlestick'], index=0)

    st.sidebar.title('Indicators')
    show_volume = st.sidebar.checkbox('Volume', value=True)
    volume_type = st.sidebar.selectbox('Volume Type', ['Bar', 'Line'], index=0)
    show_ma = st.sidebar.multiselect('Moving Averages (MA)', [5, 10, 20, 50, 100, 200], default=[20])
    show_bbands = st.sidebar.checkbox('Bollinger Bands')
    show_rsi = st.sidebar.checkbox('RSI')
    # Additional indicators
    show_macd = st.sidebar.checkbox('MACD')
    show_ema = st.sidebar.multiselect('Exponential Moving Averages (EMA)', [5, 10, 20, 50, 100, 200])
    show_adx = st.sidebar.checkbox('ADX')
    show_obv = st.sidebar.checkbox('OBV')
    show_stochastic = st.sidebar.checkbox('Stochastic Oscillator')
    show_cci = st.sidebar.checkbox('CCI')
    show_williams = st.sidebar.checkbox('Williams %R')
    show_momentum = st.sidebar.checkbox('Momentum')
    show_parabolic_sar = st.sidebar.checkbox('Parabolic SAR')
    show_roc = st.sidebar.checkbox('Rate of Change (ROC)')

    # Download data
    @st.cache
    def load_data(symbol, period, interval):
        data = yf.download(symbol, period=period, interval=interval)
        return data

    data = load_data(symbol, period, interval)

    if data.empty:
        st.write("No data found for the entered symbol.")
    else:
        # Calculate indicators
        if show_ma:
            for ma in show_ma:
                data[f"MA{ma}"] = data['Close'].rolling(window=ma).mean()

        if show_ema:
            for ema in show_ema:
                data[f"EMA{ema}"] = data['Close'].ewm(span=ema, adjust=False).mean()

        if show_bbands:
            bb_indicator = ta.volatility.BollingerBands(close=data['Close'])
            data['bb_middle'] = bb_indicator.bollinger_mavg()
            data['bb_upper'] = bb_indicator.bollinger_hband()
            data['bb_lower'] = bb_indicator.bollinger_lband()

        if show_rsi:
            data['RSI'] = ta.momentum.RSIIndicator(close=data['Close']).rsi()

        if show_macd:
            macd_indicator = ta.trend.MACD(close=data['Close'])
            data['MACD'] = macd_indicator.macd()
            data['MACD_signal'] = macd_indicator.macd_signal()

        if show_adx:
            adx_indicator = ta.trend.ADXIndicator(high=data['High'], low=data['Low'], close=data['Close'])
            data['ADX'] = adx_indicator.adx()

        if show_obv:
            data['OBV'] = ta.volume.OnBalanceVolumeIndicator(close=data['Close'], volume=data['Volume']).on_balance_volume()

        if show_stochastic:
            stoch_indicator = ta.momentum.StochasticOscillator(high=data['High'], low=data['Low'], close=data['Close'])
            data['Stoch_%K'] = stoch_indicator.stoch()
            data['Stoch_%D'] = stoch_indicator.stoch_signal()

        if show_cci:
            data['CCI'] = ta.trend.CCIIndicator(high=data['High'], low=data['Low'], close=data['Close']).cci()

        if show_williams:
            data['Williams %R'] = ta.momentum.WilliamsRIndicator(high=data['High'], low=data['Low'], close=data['Close']).williams_r()

        if show_momentum:
            data['Momentum'] = ta.momentum.MomentumIndicator(close=data['Close']).momentum()

        if show_parabolic_sar:
            psar_indicator = ta.trend.PSARIndicator(high=data['High'], low=data['Low'], close=data['Close'])
            data['Parabolic_SAR'] = psar_indicator.psar()

        if show_roc:
            data['ROC'] = ta.momentum.ROCIndicator(close=data['Close']).roc()

        # Plotting
        st.title(f"Dashboard for {symbol.upper()}")

        if chart_type == 'Candlestick':
            # For candlestick chart, use mplfinance
            import mplfinance as mpf
            candle_data = data[['Open', 'High', 'Low', 'Close', 'Volume']]
            candle_data.index.name = 'Date'
            mpf_style = mpf.make_mpf_style(base_mpf_style='charles', rc={'font.size': 10})
            add_plots = []

            if show_ma:
                for ma in show_ma:
                    add_plots.append(mpf.make_addplot(data[f"MA{ma}"], color='blue', width=1, panel=0))

            if show_ema:
                for ema in show_ema:
                    add_plots.append(mpf.make_addplot(data[f"EMA{ema}"], color='orange', width=1, panel=0))

            if show_bbands:
                add_plots += [
                    mpf.make_addplot(data['bb_upper'], color='grey', linestyle='--', panel=0),
                    mpf.make_addplot(data['bb_lower'], color='grey', linestyle='--', panel=0),
                ]

            if show_parabolic_sar:
                add_plots.append(mpf.make_addplot(data['Parabolic_SAR'], color='red', linestyle='--', panel=0))

            # Plot using mplfinance with returnfig=True
            fig, axlist = mpf.plot(
                candle_data,
                type='candle',
                style=mpf_style,
                addplot=add_plots,
                volume=False,
                returnfig=True,
                figsize=(14, 7)
            )

            st.pyplot(fig)
        else:
            # Line chart
            fig, ax = plt.subplots(figsize=(14, 7))
            ax.plot(data.index, data['Close'], label='Close Price', color='black')

            if show_ma:
                for ma in show_ma:
                    ax.plot(data.index, data[f"MA{ma}"], label=f"MA({ma})")

            if show_ema:
                for ema in show_ema:
                    ax.plot(data.index, data[f"EMA{ema}"], label=f"EMA({ema})", linestyle='--')

            if show_bbands:
                ax.plot(data.index, data['bb_upper'], label='Upper Bollinger Band', linestyle='--', color='grey')
                ax.plot(data.index, data['bb_middle'], label='Middle Bollinger Band', linestyle='--', color='blue')
                ax.plot(data.index, data['bb_lower'], label='Lower Bollinger Band', linestyle='--', color='grey')

            if show_parabolic_sar:
                ax.plot(data.index, data['Parabolic_SAR'], label='Parabolic SAR', linestyle='--', color='red')

            ax.set_xlabel('Date')
            ax.set_ylabel('Price')
            ax.legend()
            st.pyplot(fig)

        # Volume
        if show_volume:
            st.subheader('Volume')
            fig_vol, ax_vol = plt.subplots(figsize=(14, 3))

            if volume_type == 'Bar':
                # Color bars based on price change
                colors = ['green' if data['Close'].iloc[i] > data['Open'].iloc[i] else 'red' for i in range(len(data))]
                ax_vol.bar(data.index, data['Volume'], color=colors)
            else:
                ax_vol.plot(data.index, data['Volume'], color='blue')

            ax_vol.set_xlabel('Date')
            ax_vol.set_ylabel('Volume')
            st.pyplot(fig_vol)

        # Additional indicators (Same as before)

elif page == 'Correlation Matrix':
    # Correlation Matrix Page (Same as before)
