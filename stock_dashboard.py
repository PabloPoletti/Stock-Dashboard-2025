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
    chart_type = st.sidebar.selectbox('Chart Type', ['Line', 'Candlestick', 'Area'], index=0)
    
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
        fig, ax = plt.subplots(figsize=(14, 7))

        if chart_type == 'Line':
            ax.plot(data.index, data['Close'], label='Close Price', color='black')
        elif chart_type == 'Candlestick':
            # For candlestick chart, use mplfinance
            import mplfinance as mpf
            candle_data = data[['Open', 'High', 'Low', 'Close']]
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

            mpf.plot(candle_data, type='candle', style=mpf_style, ax=ax, addplot=add_plots, volume=False)
        elif chart_type == 'Area':
            ax.fill_between(data.index, data['Close'], color='skyblue', alpha=0.5, label='Close Price')

        if chart_type != 'Candlestick':
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

        # Additional indicators
        if show_rsi:
            st.subheader('RSI')
            fig_rsi, ax_rsi = plt.subplots(figsize=(14, 3))
            ax_rsi.plot(data.index, data['RSI'], color='purple')
            ax_rsi.axhline(70, color='red', linestyle='--')
            ax_rsi.axhline(30, color='green', linestyle='--')
            ax_rsi.set_xlabel('Date')
            ax_rsi.set_ylabel('RSI')
            st.pyplot(fig_rsi)

        if show_macd:
            st.subheader('MACD')
            fig_macd, ax_macd = plt.subplots(figsize=(14, 3))
            ax_macd.plot(data.index, data['MACD'], label='MACD', color='blue')
            ax_macd.plot(data.index, data['MACD_signal'], label='Signal Line', color='red')
            ax_macd.set_xlabel('Date')
            ax_macd.set_ylabel('MACD')
            ax_macd.legend()
            st.pyplot(fig_macd)

        if show_adx:
            st.subheader('ADX')
            fig_adx, ax_adx = plt.subplots(figsize=(14, 3))
            ax_adx.plot(data.index, data['ADX'], color='orange')
            ax_adx.set_xlabel('Date')
            ax_adx.set_ylabel('ADX')
            st.pyplot(fig_adx)

        if show_obv:
            st.subheader('On-Balance Volume (OBV)')
            fig_obv, ax_obv = plt.subplots(figsize=(14, 3))
            ax_obv.plot(data.index, data['OBV'], color='brown')
            ax_obv.set_xlabel('Date')
            ax_obv.set_ylabel('OBV')
            st.pyplot(fig_obv)

        if show_stochastic:
            st.subheader('Stochastic Oscillator')
            fig_stoch, ax_stoch = plt.subplots(figsize=(14, 3))
            ax_stoch.plot(data.index, data['Stoch_%K'], label='%K', color='blue')
            ax_stoch.plot(data.index, data['Stoch_%D'], label='%D', color='red')
            ax_stoch.axhline(80, color='red', linestyle='--')
            ax_stoch.axhline(20, color='green', linestyle='--')
            ax_stoch.set_xlabel('Date')
            ax_stoch.set_ylabel('Stochastic Oscillator')
            ax_stoch.legend()
            st.pyplot(fig_stoch)

        if show_cci:
            st.subheader('Commodity Channel Index (CCI)')
            fig_cci, ax_cci = plt.subplots(figsize=(14, 3))
            ax_cci.plot(data.index, data['CCI'], color='magenta')
            ax_cci.set_xlabel('Date')
            ax_cci.set_ylabel('CCI')
            st.pyplot(fig_cci)

        if show_williams:
            st.subheader('Williams %R')
            fig_williams, ax_williams = plt.subplots(figsize=(14, 3))
            ax_williams.plot(data.index, data['Williams %R'], color='darkgreen')
            ax_williams.axhline(-20, color='red', linestyle='--')
            ax_williams.axhline(-80, color='green', linestyle='--')
            ax_williams.set_xlabel('Date')
            ax_williams.set_ylabel('Williams %R')
            st.pyplot(fig_williams)

        if show_momentum:
            st.subheader('Momentum')
            fig_momentum, ax_momentum = plt.subplots(figsize=(14, 3))
            ax_momentum.plot(data.index, data['Momentum'], color='navy')
            ax_momentum.set_xlabel('Date')
            ax_momentum.set_ylabel('Momentum')
            st.pyplot(fig_momentum)

        if show_roc:
            st.subheader('Rate of Change (ROC)')
            fig_roc, ax_roc = plt.subplots(figsize=(14, 3))
            ax_roc.plot(data.index, data['ROC'], color='teal')
            ax_roc.set_xlabel('Date')
            ax_roc.set_ylabel('ROC')
            st.pyplot(fig_roc)
elif page == 'Correlation Matrix':
    # Correlation Matrix Page
    st.title('Stock Correlation Matrix')
    st.sidebar.title('Correlation Matrix Settings')

    # Select stocks
    num_stocks = st.sidebar.slider('Number of Stocks', min_value=2, max_value=10, value=2)
    stock_symbols = []
    for i in range(num_stocks):
        symbol = st.sidebar.text_input(f'Symbol {i+1}', value='AAPL' if i == 0 else '')
        if symbol:
            stock_symbols.append(symbol.upper())

    # Select time period
    period = st.sidebar.selectbox('Period', ['1mo', '3mo', '6mo', '1y', '5y', 'max'], index=3)
    interval = st.sidebar.selectbox('Interval', ['1d', '1wk', '1mo'], index=0)
    # Select correlation method
    corr_method = st.sidebar.selectbox('Correlation Method', ['Pearson', 'Spearman', 'Kendall'], index=0)

    if len(stock_symbols) >= 2:
        @st.cache
        def load_data(symbols, period, interval):
            df = pd.DataFrame()
            for sym in symbols:
                data = yf.download(sym, period=period, interval=interval)
                data = data['Close'].rename(sym)
                df = pd.concat([df, data], axis=1)
            return df

        data = load_data(stock_symbols, period, interval)

        if data.isnull().values.any():
            data = data.fillna(method='ffill').dropna()

        # Calculate correlation
        if corr_method == 'Pearson':
            corr = data.corr(method='pearson')
        elif corr_method == 'Spearman':
            corr = data.corr(method='spearman')
        elif corr_method == 'Kendall':
            corr = data.corr(method='kendall')

        # Plot correlation matrix
        fig_corr, ax_corr = plt.subplots(figsize=(10, 8))
        sns.heatmap(corr, annot=True, cmap='coolwarm', ax=ax_corr, vmin=-1, vmax=1)
        ax_corr.set_title(f'{corr_method} Correlation Matrix')
        st.pyplot(fig_corr)

        # Explanation legend
        st.markdown("""
        **Correlation Coefficient Interpretation:**
        - **1**: Perfect positive correlation
        - **0.7 to 0.99**: Strong positive correlation
        - **0.4 to 0.69**: Moderate positive correlation
        - **0.1 to 0.39**: Weak positive correlation
        - **0**: No correlation
        - **-0.1 to -0.39**: Weak negative correlation
        - **-0.4 to -0.69**: Moderate negative correlation
        - **-0.7 to -0.99**: Strong negative correlation
        - **-1**: Perfect negative correlation
        """)

    else:
        st.write("Please enter at least two stock symbols.")
