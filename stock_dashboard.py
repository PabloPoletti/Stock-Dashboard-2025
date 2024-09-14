import streamlit as st
import yfinance as yf
import pandas as pd
import matplotlib.pyplot as plt
import ta

# Configurar la página
st.set_page_config(layout="wide")

# Entradas en la barra lateral
st.sidebar.title('Dashboard de Acciones')
symbol = st.sidebar.text_input('Símbolo', value='AAPL')
period = st.sidebar.selectbox('Período', ['1mo', '3mo', '6mo', '1y', '5y', 'max'], index=3)
interval = st.sidebar.selectbox('Intervalo', ['1d', '1wk', '1mo'], index=0)

st.sidebar.title('Indicadores')
show_volume = st.sidebar.checkbox('Volumen', value=True)
show_ma = st.sidebar.multiselect('Medias Móviles (MA)', [5, 10, 20, 50, 100, 200], default=[20])
show_bbands = st.sidebar.checkbox('Bandas de Bollinger')
show_rsi = st.sidebar.checkbox('RSI')
# Indicadores adicionales
show_macd = st.sidebar.checkbox('MACD')
show_ema = st.sidebar.multiselect('Medias Móviles Exponenciales (EMA)', [5, 10, 20, 50, 100, 200])
show_adx = st.sidebar.checkbox('ADX')
show_obv = st.sidebar.checkbox('OBV')
show_stochastic = st.sidebar.checkbox('Oscilador Estocástico')
show_cci = st.sidebar.checkbox('CCI')
show_williams = st.sidebar.checkbox('Williams %R')
show_momentum = st.sidebar.checkbox('Momentum')
show_parabolic_sar = st.sidebar.checkbox('Parabólico SAR')
show_roc = st.sidebar.checkbox('Tasa de Cambio (ROC)')

# Descargar datos
@st.cache
def load_data(symbol, period, interval):
    data = yf.download(symbol, period=period, interval=interval)
    return data

data = load_data(symbol, period, interval)

if data.empty:
    st.write("No se encontraron datos para el símbolo ingresado.")
else:
    # Calcular indicadores
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
        data['Parabolic_SAR'] = ta.trend.PSARIndicator(high=data['High'], low=data['Low'], close=data['Close']).psar()

    if show_roc:
        data['ROC'] = ta.momentum.ROCIndicator(close=data['Close']).roc()

    # Graficar
    st.title(f"Dashboard de {symbol.upper()}")
    fig, ax = plt.subplots(figsize=(14, 7))

    ax.plot(data.index, data['Close'], label='Precio de Cierre', color='black')

    if show_ma:
        for ma in show_ma:
            ax.plot(data.index, data[f"MA{ma}"], label=f"MA({ma})")

    if show_ema:
        for ema in show_ema:
            ax.plot(data.index, data[f"EMA{ema}"], label=f"EMA({ema})", linestyle='--')

    if show_bbands:
        ax.plot(data.index, data['bb_upper'], label='Banda Superior Bollinger', linestyle='--', color='grey')
        ax.plot(data.index, data['bb_middle'], label='Media Bollinger', linestyle='--', color='blue')
        ax.plot(data.index, data['bb_lower'], label='Banda Inferior Bollinger', linestyle='--', color='grey')

    ax.set_xlabel('Fecha')
    ax.set_ylabel('Precio')
    ax.legend()
    st.pyplot(fig)

    # Volumen
    if show_volume:
        st.subheader('Volumen')
        fig_vol, ax_vol = plt.subplots(figsize=(14, 3))
        ax_vol.bar(data.index, data['Volume'], color='grey')
        ax_vol.set_xlabel('Fecha')
        ax_vol.set_ylabel('Volumen')
        st.pyplot(fig_vol)

    # Indicadores adicionales
    if show_rsi:
        st.subheader('RSI')
        fig_rsi, ax_rsi = plt.subplots(figsize=(14, 3))
        ax_rsi.plot(data.index, data['RSI'], color='purple')
        ax_rsi.axhline(70, color='red', linestyle='--')
        ax_rsi.axhline(30, color='green', linestyle='--')
        ax_rsi.set_xlabel('Fecha')
        ax_rsi.set_ylabel('RSI')
        st.pyplot(fig_rsi)

    if show_macd:
        st.subheader('MACD')
        fig_macd, ax_macd = plt.subplots(figsize=(14, 3))
        ax_macd.plot(data.index, data['MACD'], label='MACD', color='blue')
        ax_macd.plot(data.index, data['MACD_signal'], label='Línea de Señal', color='red')
        ax_macd.set_xlabel('Fecha')
        ax_macd.set_ylabel('MACD')
        ax_macd.legend()
        st.pyplot(fig_macd)

    if show_adx:
        st.subheader('ADX')
        fig_adx, ax_adx = plt.subplots(figsize=(14, 3))
        ax_adx.plot(data.index, data['ADX'], color='orange')
        ax_adx.set_xlabel('Fecha')
        ax_adx.set_ylabel('ADX')
        st.pyplot(fig_adx)

    if show_obv:
        st.subheader('On-Balance Volume (OBV)')
        fig_obv, ax_obv = plt.subplots(figsize=(14, 3))
        ax_obv.plot(data.index, data['OBV'], color='brown')
        ax_obv.set_xlabel('Fecha')
        ax_obv.set_ylabel('OBV')
        st.pyplot(fig_obv)

    if show_stochastic:
        st.subheader('Oscilador Estocástico')
        fig_stoch, ax_stoch = plt.subplots(figsize=(14, 3))
        ax_stoch.plot(data.index, data['Stoch_%K'], label='%K', color='blue')
        ax_stoch.plot(data.index, data['Stoch_%D'], label='%D', color='red')
        ax_stoch.axhline(80, color='red', linestyle='--')
        ax_stoch.axhline(20, color='green', linestyle='--')
        ax_stoch.set_xlabel('Fecha')
        ax_stoch.set_ylabel('Oscilador Estocástico')
        ax_stoch.legend()
        st.pyplot(fig_stoch)

    if show_cci:
        st.subheader('Índice de Canal de Mercancías (CCI)')
        fig_cci, ax_cci = plt.subplots(figsize=(14, 3))
        ax_cci.plot(data.index, data['CCI'], color='magenta')
        ax_cci.set_xlabel('Fecha')
        ax_cci.set_ylabel('CCI')
        st.pyplot(fig_cci)

    if show_williams:
        st.subheader('Williams %R')
        fig_williams, ax_williams = plt.subplots(figsize=(14, 3))
        ax_williams.plot(data.index, data['Williams %R'], color='darkgreen')
        ax_williams.axhline(-20, color='red', linestyle='--')
        ax_williams.axhline(-80, color='green', linestyle='--')
        ax_williams.set_xlabel('Fecha')
        ax_williams.set_ylabel('Williams %R')
        st.pyplot(fig_williams)

    if show_momentum:
        st.subheader('Momentum')
        fig_momentum, ax_momentum = plt.subplots(figsize=(14, 3))
        ax_momentum.plot(data.index, data['Momentum'], color='navy')
        ax_momentum.set_xlabel('Fecha')
        ax_momentum.set_ylabel('Momentum')
        st.pyplot(fig_momentum)

    if show_parabolic_sar:
        st.subheader('Parabólico SAR')
        fig_sar, ax_sar = plt.subplots(figsize=(14, 7))
        ax_sar.plot(data.index, data['Close'], label='Precio de Cierre', color='black')
        ax_sar.plot(data.index, data['Parabolic_SAR'], label='Parabólico SAR', linestyle='--', color='red')
        ax_sar.set_xlabel('Fecha')
        ax_sar.set_ylabel('Precio')
        ax_sar.legend()
        st.pyplot(fig_sar)

    if show_roc:
        st.subheader('Tasa de Cambio (ROC)')
        fig_roc, ax_roc = plt.subplots(figsize=(14, 3))
        ax_roc.plot(data.index, data['ROC'], color='teal')
        ax_roc.set_xlabel('Fecha')
        ax_roc.set_ylabel('ROC')
        st.pyplot(fig_roc)
