import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np  # Import NumPy directly
import matplotlib.pyplot as plt
import ta
import seaborn as sns
import matplotlib.ticker as ticker
import matplotlib.dates as mdates

# Get the list of S&P 500 companies using yfinance
def get_sp500_tickers():
    sp500_tickers = yf.Ticker('^GSPC').components
    return [(ticker, info['longName']) for ticker, info in sp500_tickers.items()]

sp500_list = get_sp500_tickers()

# Configure the page
st.set_page_config(layout="wide")

# Create a sidebar for navigation
st.sidebar.title('Navigation')
page = st.sidebar.selectbox('Select a page:', ['Stock Analysis', 'Correlation Matrix', 'Fixed Income', 'Corporate Bonds'])


# Sample data for countries and indices
country_indices = {
    'USA': ['S&P 500', 'NASDAQ 100'],
    'Argentina': ['Merval'],
    'Brazil': ['Bovespa'],
    # Add more countries as needed
}

# Index constituents (sample stocks)
index_stocks = {
    'S&P 500': 'S&P 500': get_sp500_tickers(),
        #('AAPL', 'Apple Inc.'), ('MSFT', 'Microsoft Corp.'), ('GOOGL', 'Alphabet Inc.'), 
        #('AMZN', 'Amazon.com Inc.'), ('TSLA', 'Tesla Inc.'), # Add more companies here
    ,
    'NASDAQ 100': [
        ('AAPL', 'Apple Inc.'), ('MSFT', 'Microsoft Corp.'), ('GOOGL', 'Alphabet Inc.'), 
        ('AMZN', 'Amazon.com Inc.'), ('META', 'Meta Platforms Inc.'), # Add more companies here
    ],
    'Merval': [
        ('GGAL.BA', 'Grupo Financiero Galicia S.A.'), ('YPFD.BA', 'YPF S.A.'), 
        ('BMA.BA', 'Banco Macro S.A.'), ('TXAR.BA', 'Ternium Argentina S.A.'), 
        ('TECO2.BA', 'Telecom Argentina S.A.'), # Add more companies here
    ],
    'Bovespa': [
        ('PETR4.SA', 'Petrobras'), ('VALE3.SA', 'Vale S.A.'), ('ITUB4.SA', 'Itaú Unibanco'), 
        ('BBDC4.SA', 'Banco Bradesco S.A.'), ('ABEV3.SA', 'Ambev S.A.'), # Add more companies here
    ],
    # Add more indices and their companies as needed
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
        # Create a dictionary for easy lookup of the stock symbol by company name
        stock_dict = {f"{name} ({symbol})": symbol for symbol, name in stocks}
        selected_stock = st.sidebar.selectbox('Select Stock', list(stock_dict.keys()))
        symbol = stock_dict[selected_stock]
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
                        add_plots.append(mpf.make_addplot(data[f"MA{ma}"], color='blue', width=1, panel=0, ylabel='Price'))

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

                # Note: mplfinance does not support legends directly.
                # You can create custom legends if necessary.

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
            if show_rsi:
                st.subheader('RSI')
                fig_rsi, ax_rsi = plt.subplots(figsize=(14, 3))
                ax_rsi.plot(data.index, data['RSI'], color='purple')
                ax_rsi.axhline(70, color='red', linestyle='--')
                ax_rsi.axhline(30, color='green', linestyle='--')
                ax_rsi.set_xlabel('Date')
                ax_rsi.set_ylabel('RSI')
                ax_rsi.legend(['RSI'])
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
                ax_adx.legend(['ADX'])
                st.pyplot(fig_adx)

            if show_obv:
                st.subheader('On-Balance Volume (OBV)')
                fig_obv, ax_obv = plt.subplots(figsize=(14, 3))
                ax_obv.plot(data.index, data['OBV'], color='brown')
                ax_obv.set_xlabel('Date')
                ax_obv.set_ylabel('OBV')
                ax_obv.legend(['OBV'])
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
                ax_cci.legend(['CCI'])
                st.pyplot(fig_cci)

            if show_williams:
                st.subheader('Williams %R')
                fig_williams, ax_williams = plt.subplots(figsize=(14, 3))
                ax_williams.plot(data.index, data['Williams %R'], color='darkgreen')
                ax_williams.axhline(-20, color='red', linestyle='--')
                ax_williams.axhline(-80, color='green', linestyle='--')
                ax_williams.set_xlabel('Date')
                ax_williams.set_ylabel("Williams %R")
                ax_williams.legend(["Williams %R"])
                st.pyplot(fig_williams)

            if show_momentum:
                st.subheader('Momentum')
                fig_momentum, ax_momentum = plt.subplots(figsize=(14, 3))
                ax_momentum.plot(data.index, data['Momentum'], color='navy')
                ax_momentum.set_xlabel('Date')
                ax_momentum.set_ylabel('Momentum')
                ax_momentum.legend(['Momentum'])
                st.pyplot(fig_momentum)

            if show_roc:
                st.subheader('Rate of Change (ROC)')
                fig_roc, ax_roc = plt.subplots(figsize=(14, 3))
                ax_roc.plot(data.index, data['ROC'], color='teal')
                ax_roc.set_xlabel('Date')
                ax_roc.set_ylabel('ROC')
                ax_roc.legend(['ROC'])
                st.pyplot(fig_roc)

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

        if len(stock_symbols) == 2:
            corr_value = corr.iloc[0,1]
            st.write(f"The {corr_method} correlation coefficient between {stock_symbols[0]} and {stock_symbols[1]} is: {corr_value:.4f}")
        else:
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
    st.title('Sovereign Bonds Analysis')

    st.sidebar.title('Fixed Income Settings')

    # Updated country_bonds with more Argentine bonds
    country_bonds = {
        'USA': ['US10Y', 'US30Y'],
        'Argentina': ['AL29', 'AL30', 'AL35', 'AE38', 'AL41'],
        'Brazil': ['BR10Y', 'BR30Y'],
        # Add more countries and bonds as needed
    }

    country = st.sidebar.selectbox('Select Country', list(country_bonds.keys()))
    bonds = country_bonds[country]

    bond = st.sidebar.selectbox('Select Bond', bonds)

    # Updated bond_info with additional bonds and information
    bond_info = {
        # USA Bonds
        'US10Y': {'Name': 'US 10-Year Treasury', 'Maturity': '2029-07-09', 'Coupon': '1.5%', 'Yield': '1.3%', 'ISIN': 'US10Y',
                  'CurrentYield': '1.4%', 'Duration': '8.5', 'Convexity': '70.2', 'CreditRating': 'AAA'},
        'US30Y': {'Name': 'US 30-Year Treasury', 'Maturity': '2051-07-09', 'Coupon': '2.0%', 'Yield': '2.1%', 'ISIN': 'US30Y',
                  'CurrentYield': '2.05%', 'Duration': '25.0', 'Convexity': '200.5', 'CreditRating': 'AAA'},

        # Argentine Bonds
        'AL29': {'Name': 'Bonar 2029', 'Maturity': '2029-07-09', 'Coupon': '4.625%', 'Yield': '7.5%', 'ISIN': 'ARARGE03E113',
                 'CurrentYield': '6.0%', 'Duration': '7.5', 'Convexity': '60.2', 'CreditRating': 'CCC+'},
        'AL30': {'Name': 'Bonar 2030', 'Maturity': '2030-07-09', 'Coupon': '0.125%', 'Yield': '8.0%', 'ISIN': 'ARARGE03E121',
                 'CurrentYield': '7.0%', 'Duration': '8.0', 'Convexity': '65.0', 'CreditRating': 'CCC+'},
        'AL35': {'Name': 'Bonar 2035', 'Maturity': '2035-07-09', 'Coupon': '1.25%', 'Yield': '8.5%', 'ISIN': 'ARARGE03E139',
                 'CurrentYield': '7.5%', 'Duration': '12.0', 'Convexity': '90.0', 'CreditRating': 'CCC+'},
        'AE38': {'Name': 'Bonar 2038', 'Maturity': '2038-07-09', 'Coupon': '3.5%', 'Yield': '9.0%', 'ISIN': 'ARARGE03G688',
                 'CurrentYield': '8.0%', 'Duration': '15.0', 'Convexity': '110.0', 'CreditRating': 'CCC+'},
        'AL41': {'Name': 'Bonar 2041', 'Maturity': '2041-07-09', 'Coupon': '5.0%', 'Yield': '9.5%', 'ISIN': 'ARARGE03E147',
                 'CurrentYield': '8.5%', 'Duration': '18.0', 'Convexity': '130.0', 'CreditRating': 'CCC+'},

        # Brazilian Bonds
        'BR10Y': {'Name': 'Brazil 10-Year Bond', 'Maturity': '2031-01-01', 'Coupon': '5.0%', 'Yield': '5.5%', 'ISIN': 'BR10Y',
                  'CurrentYield': '5.25%', 'Duration': '9.0', 'Convexity': '75.0', 'CreditRating': 'BB'},
        'BR30Y': {'Name': 'Brazil 30-Year Bond', 'Maturity': '2051-01-01', 'Coupon': '6.0%', 'Yield': '6.5%', 'ISIN': 'BR30Y',
                  'CurrentYield': '6.25%', 'Duration': '22.0', 'Convexity': '190.0', 'CreditRating': 'BB'},
        # Add more bonds as needed
    }

    bond_data = bond_info.get(bond, {})

    if bond_data:
        st.subheader(f"{bond_data['Name']} ({bond})")
        st.write(f"**Maturity Date:** {bond_data['Maturity']}")
        st.write(f"**Coupon Rate:** {bond_data['Coupon']}")

        st.sidebar.title('Bond Analysis Indicators')
        show_ytm = st.sidebar.checkbox('Yield to Maturity (YTM)', value=True)
        show_current_yield = st.sidebar.checkbox('Current Yield')
        show_duration = st.sidebar.checkbox('Duration')
        show_convexity = st.sidebar.checkbox('Convexity')
        show_credit_rating = st.sidebar.checkbox('Credit Rating')

        if show_ytm:
            st.write(f"**Yield to Maturity (YTM):** {bond_data.get('Yield', 'N/A')}")
        if show_current_yield:
            st.write(f"**Current Yield:** {bond_data.get('CurrentYield', 'N/A')}")
        if show_duration:
            st.write(f"**Duration:** {bond_data.get('Duration', 'N/A')}")
        if show_convexity:
            st.write(f"**Convexity:** {bond_data.get('Convexity', 'N/A')}")
        if show_credit_rating:
            st.write(f"**Credit Rating:** {bond_data.get('CreditRating', 'N/A')}")

        # Simulate price data
        dates = pd.date_range(end=pd.Timestamp.today(), periods=100)
        prices = pd.Series(100 + np.random.randn(100).cumsum(), index=dates)

        fig_bond, ax_bond = plt.subplots(figsize=(14, 7))
        ax_bond.plot(prices.index, prices.values, label='Price')
        ax_bond.set_xlabel('Date')
        ax_bond.set_ylabel('Price')
        ax_bond.legend()
        st.pyplot(fig_bond)

    else:
        st.write("No data available for the selected bond.")

elif page == 'Corporate Bonds':
    st.title('Corporate Bonds Analysis')

    st.sidebar.title('Corporate Bonds Settings')

    # Sample data for countries and their corporate bonds
    country_corp_bonds = {
        'USA': ['AAPL2025', 'MSFT2030'],
        'Argentina': ['YPF2025', 'TECO2030'],
        'Brazil': ['PETRO2025', 'VALE2030'],
        # Add more countries and bonds as needed
    }

    country = st.sidebar.selectbox('Select Country', list(country_corp_bonds.keys()))
    corp_bonds = country_corp_bonds[country]

    corp_bond = st.sidebar.selectbox('Select Corporate Bond', corp_bonds)

    # Fetch corporate bond data
    # For demonstration, we'll use sample data

    # Simulate corporate bond data
    corp_bond_info = {
        # USA Corporate Bonds
        'AAPL2025': {
            'Name': 'Apple Inc. 2025 Notes',
            'Maturity': '2025-05-06',
            'Coupon': '2.0%',
            'Yield': '1.8%',
            'ISIN': 'US037833AL45',
            'CurrentYield': '1.9%',
            'Duration': '4.5',
            'Convexity': '20.3',
            'CreditRating': 'AA+'
        },
        'MSFT2030': {
            'Name': 'Microsoft Corp. 2030 Notes',
            'Maturity': '2030-11-15',
            'Coupon': '3.0%',
            'Yield': '2.5%',
            'ISIN': 'US594918BQ79',
            'CurrentYield': '2.8%',
            'Duration': '8.0',
            'Convexity': '50.1',
            'CreditRating': 'AAA'
        },
        # Argentine Corporate Bonds
        'YPF2025': {
            'Name': 'YPF S.A. 2025 Notes',
            'Maturity': '2025-12-15',
            'Coupon': '8.5%',
            'Yield': '9.0%',
            'ISIN': 'XS0501197263',
            'CurrentYield': '8.8%',
            'Duration': '3.8',
            'Convexity': '15.2',
            'CreditRating': 'CCC-'
        },
        'TECO2030': {
            'Name': 'Telecom Argentina S.A. 2030 Notes',
            'Maturity': '2030-03-15',
            'Coupon': '9.0%',
            'Yield': '9.5%',
            'ISIN': 'US879273AA72',
            'CurrentYield': '9.2%',
            'Duration': '7.0',
            'Convexity': '40.5',
            'CreditRating': 'CCC'
        },
        # Brazilian Corporate Bonds
        'PETRO2025': {
            'Name': 'Petrobras Global Finance B.V. 2025 Notes',
            'Maturity': '2025-01-27',
            'Coupon': '6.25%',
            'Yield': '6.0%',
            'ISIN': 'US71647NAN03',
            'CurrentYield': '6.1%',
            'Duration': '4.0',
            'Convexity': '18.0',
            'CreditRating': 'BB-'
        },
        'VALE2030': {
            'Name': 'Vale Overseas Limited 2030 Notes',
            'Maturity': '2030-08-11',
            'Coupon': '5.625%',
            'Yield': '5.8%',
            'ISIN': 'US91911TAK39',
            'CurrentYield': '5.7%',
            'Duration': '7.5',
            'Convexity': '35.0',
            'CreditRating': 'BBB-'
        },
    }

    bond_data = corp_bond_info.get(corp_bond, {})

    if bond_data:
        st.subheader(f"{bond_data['Name']} ({corp_bond})")
        st.write(f"**Maturity Date:** {bond_data['Maturity']}")
        st.write(f"**Coupon Rate:** {bond_data['Coupon']}")

        # Reuse the indicator selections from the Fixed Income page
        st.sidebar.title('Bond Analysis Indicators')
        show_ytm = st.sidebar.checkbox('Yield to Maturity (YTM)', value=True)
        show_current_yield = st.sidebar.checkbox('Current Yield')
        show_duration = st.sidebar.checkbox('Duration')
        show_convexity = st.sidebar.checkbox('Convexity')
        show_credit_rating = st.sidebar.checkbox('Credit Rating')

        if show_ytm:
            st.write(f"**Yield to Maturity (YTM):** {bond_data.get('Yield', 'N/A')}")
        if show_current_yield:
            st.write(f"**Current Yield:** {bond_data.get('CurrentYield', 'N/A')}")
        if show_duration:
            st.write(f"**Duration:** {bond_data.get('Duration', 'N/A')}")
        if show_convexity:
            st.write(f"**Convexity:** {bond_data.get('Convexity', 'N/A')}")
        if show_credit_rating:
            st.write(f"**Credit Rating:** {bond_data.get('CreditRating', 'N/A')}")

        # Simulate price data
        dates = pd.date_range(end=pd.Timestamp.today(), periods=100)
        prices = pd.Series(100 + np.random.randn(100).cumsum(), index=dates)

        fig_bond, ax_bond = plt.subplots(figsize=(14, 7))
        ax_bond.plot(prices.index, prices.values, label='Price')
        ax_bond.set_xlabel('Date')
        ax_bond.set_ylabel('Price')
        ax_bond.legend()
        st.pyplot(fig_bond)

        # Include relevant indicators or analytics for bonds if desired

    else:
        st.write("No data available for the selected corporate bond.")
