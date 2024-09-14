import streamlit as st
import yfinance as yf
import pandas as pd
import matplotlib.pyplot as plt
import ta
import seaborn as sns
import matplotlib.ticker as ticker
import matplotlib.dates as mdates

# Configure the page
st.set_page_config(layout="wide")

# Create a sidebar for navigation
st.sidebar.title('Navigation')
page = st.sidebar.selectbox('Select a page:', ['Stock Analysis', 'Correlation Matrix', 'Fixed Income'])

# Sample data for countries and indices
country_indices = {
    'USA': ['S&P 500', 'NASDAQ 100'],
    'Argentina': ['Merval'],
    'Brazil': ['Bovespa'],
    # Add more countries as needed
}

# Index constituents (sample stocks)
index_stocks = {
    'S&P 500': ['AAPL', 'MSFT', 'GOOGL', 'AMZN', 'TSLA'],
    'NASDAQ 100': ['AAPL', 'MSFT', 'GOOGL', 'AMZN', 'META'],
    'Merval': ['GGAL.BA', 'YPFD.BA', 'BMA.BA', 'TXAR.BA', 'TECO2.BA'],
    'Bovespa': ['PETR4.SA', 'VALE3.SA', 'ITUB4.SA', 'BBDC4.SA', 'ABEV3.SA'],
    # Add more indices and their stocks as needed
}

# Currency symbols mapping
currency_symbols = {
    'USD': '$',
    'EUR': '€',
    'ARS': '$',
    'BRL': 'R$',
    # Add other currencies as needed
}

if page == 'Stock Analysis':
    # Country and Index Selection
    st.sidebar.title('Country and Index Selection')
    country = st.sidebar.selectbox('Select Country', list(country_indices.keys()))
    indices = country_indices[country]
    index = st.sidebar.selectbox('Select Index', indices)
    stocks = index_stocks.get(index, [])
    if stocks:
        symbol = st.sidebar.selectbox('Select Stock', stocks)
    else:
        st.sidebar.write('No stocks available for the selected index.')
        symbol = None

    if symbol:
        # Get ticker info to retrieve company name and currency
        ticker_data = yf.Ticker(symbol)
        ticker_info = ticker_data.info
        company_name = ticker_info.get('shortName', symbol)
        currency_code = ticker_info.get('currency', 'USD')
        currency_symbol = currency_symbols.get(currency_code, currency_code)

        # Inputs in the sidebar
        st.sidebar.title('Stock Analysis')
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
            st.write("No data found for the selected stock.")
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
                # Calculate Momentum manually
                data['Momentum'] = data['Close'] - data['Close'].shift(1)

            if show_parabolic_sar:
                psar_indicator = ta.trend.PSARIndicator(high=data['High'], low=data['Low'], close=data['Close'])
                data['Parabolic_SAR'] = psar_indicator.psar()

            if show_roc:
                data['ROC'] = ta.momentum.ROCIndicator(close=data['Close']).roc()

            # Plotting
            st.title(f"{company_name} ({symbol.upper()})")

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

                # Adjust axes
                ax = axlist[0]  # Main price axis
                ax.yaxis.set_ticks_position('both')
                ax.tick_params(labelright=True)

                # Set y-axis formatter to display currency and format without decimals
                ax.yaxis.set_major_formatter(ticker.StrMethodFormatter(currency_symbol + '{x:,.0f}'))

                # Increase date labels
                locator = mdates.AutoDateLocator()
                formatter = mdates.ConciseDateFormatter(locator)
                ax.xaxis.set_major_locator(locator)
                ax.xaxis.set_major_formatter(formatter)
                plt.setp(ax.get_xticklabels(), rotation=45, ha='right')

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

                # Adjust y-axis to show ticks on both sides
                ax.yaxis.set_ticks_position('both')
                ax.tick_params(labelright=True)

                # Set y-axis formatter to display currency and format without decimals
                ax.yaxis.set_major_formatter(ticker.StrMethodFormatter(currency_symbol + '{x:,.0f}'))

                # Increase date labels
                locator = mdates.AutoDateLocator()
                formatter = mdates.ConciseDateFormatter(locator)
                ax.xaxis.set_major_locator(locator)
                ax.xaxis.set_major_formatter(formatter)
                plt.setp(ax.get_xticklabels(), rotation=45, ha='right')

                ax.set_xlabel('Date')
                ax.set_ylabel(f'Price ({currency_code})')
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

                # Format y-axis to show units (e.g., in millions)
                formatter_vol = ticker.FuncFormatter(lambda x, pos: '{0:g}'.format(x/1e6))
                ax_vol.yaxis.set_major_formatter(formatter_vol)
                ax_vol.set_ylabel('Volume (Millions)')
                ax_vol.set_xlabel('Date')

                # Increase date labels
                ax_vol.xaxis.set_major_locator(locator)
                ax_vol.xaxis.set_major_formatter(formatter)
                plt.setp(ax_vol.get_xticklabels(), rotation=45, ha='right')

                st.pyplot(fig_vol)

            # Additional indicators plotting
            # (Include plotting code for other indicators as needed)

    else:
        st.write("Please select a stock to proceed.")

elif page == 'Correlation Matrix':
    # Correlation Matrix Page
    st.title('Stock Correlation Matrix')
    st.sidebar.title('Correlation Matrix Settings')

    # Select country and index
    st.sidebar.title('Country and Index Selection')
    country = st.sidebar.selectbox('Select Country', list(country_indices.keys()))
    indices = country_indices[country]
    index = st.sidebar.selectbox('Select Index', indices)
    stocks = index_stocks.get(index, [])

    # Select stocks for correlation matrix
    num_stocks = st.sidebar.slider('Number of Stocks', min_value=2, max_value=min(10, len(stocks)), value=2)
    stock_symbols = st.sidebar.multiselect('Select Stocks', stocks, default=stocks[:num_stocks])

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
        corr = data.corr(method=corr_method.lower())

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
        st.write("Please select at least two stocks to proceed.")

elif page == 'Fixed Income':
    # Fixed Income Page
    st.title('Sovereign Bonds Analysis')

    st.sidebar.title('Fixed Income Settings')

    # Sample data for countries and their bonds
    country_bonds = {
        'USA': ['US10Y', 'US30Y'],
        'Argentina': ['AR10Y', 'AR30Y'],
        'Brazil': ['BR10Y', 'BR30Y'],
        # Add more countries and bonds as needed
    }

    country = st.sidebar.selectbox('Select Country', list(country_bonds.keys()))
    bonds = country_bonds[country]

    bond = st.sidebar.selectbox('Select Bond', bonds)

    # Fetch bond data
    # Note: Accessing bond data may require specific data sources or APIs.
    # For demonstration, we'll use sample data or placeholders.

    # Simulate bond data
    bond_info = {
        'US10Y': {'Name': 'US 10-Year Treasury', 'Maturity': '10 years', 'Coupon': '1.5%', 'Yield': '1.3%'},
        'US30Y': {'Name': 'US 30-Year Treasury', 'Maturity': '30 years', 'Coupon': '2.0%', 'Yield': '2.1%'},
        'AR10Y': {'Name': 'Argentina 10-Year Bond', 'Maturity': '10 years', 'Coupon': '7.5%', 'Yield': '8.0%'},
        'AR30Y': {'Name': 'Argentina 30-Year Bond', 'Maturity': '30 years', 'Coupon': '8.5%', 'Yield': '9.0%'},
        'BR10Y': {'Name': 'Brazil 10-Year Bond', 'Maturity': '10 years', 'Coupon': '5.0%', 'Yield': '5.5%'},
        'BR30Y': {'Name': 'Brazil 30-Year Bond', 'Maturity': '30 years', 'Coupon': '6.0%', 'Yield': '6.5%'},
    }

    bond_data = bond_info.get(bond, {})

    if bond_data:
        st.subheader(f"{bond_data['Name']} ({bond})")
        st.write(f"**Maturity:** {bond_data['Maturity']}")
        st.write(f"**Coupon Rate:** {bond_data['Coupon']}")
        st.write(f"**Current Yield:** {bond_data['Yield']}")

        # Simulate price data
        dates = pd.date_range(end=pd.Timestamp.today(), periods=100)
        prices = pd.Series(100 + pd.np.random.randn(100).cumsum(), index=dates)

        fig_bond, ax_bond = plt.subplots(figsize=(14, 7))
        ax_bond.plot(prices.index, prices.values, label='Price')
        ax_bond.set_xlabel('Date')
        ax_bond.set_ylabel('Price')
        ax_bond.legend()
        st.pyplot(fig_bond)

        # Include relevant indicators or analytics for bonds if desired

    else:
        st.write("No data available for the selected bond.")
