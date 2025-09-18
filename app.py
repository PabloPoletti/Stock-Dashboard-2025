"""
🚀 Modern Stock Analysis Dashboard 2025
Advanced Financial Analytics with AI-Powered Insights

Author: Pablo Poletti
Email: lic.poletti@gmail.com
LinkedIn: https://www.linkedin.com/in/pablom-poletti/
GitHub: https://github.com/PabloPoletti
"""

import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import pandas_ta as ta
import warnings
from datetime import datetime, timedelta
import time
import asyncio
from typing import Dict, List, Optional, Tuple
import io
import base64

# ML and AI imports
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from xgboost import XGBRegressor
import lightgbm as lgb
from prophet import Prophet

# Streamlit components
from streamlit_option_menu import option_menu
from st_aggrid import AgGrid, GridOptionsBuilder, GridUpdateMode, DataReturnMode
import streamlit_lottie as st_lottie

# Suppress warnings
warnings.filterwarnings('ignore')

# Page configuration
st.set_page_config(
    page_title="📈 Stock Analysis Dashboard 2025",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for modern styling
st.markdown("""
<style>
    .main-header {
        font-size: 3rem;
        font-weight: bold;
        text-align: center;
        background: linear-gradient(90deg, #FF6B6B, #4ECDC4, #45B7D1);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin-bottom: 2rem;
    }
    
    .metric-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        padding: 1rem;
        border-radius: 10px;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
        margin: 0.5rem 0;
    }
    
    .sidebar-info {
        background: #f0f2f6;
        padding: 1rem;
        border-radius: 10px;
        margin: 1rem 0;
    }
    
    .prediction-card {
        background: linear-gradient(135deg, #ffeaa7 0%, #fab1a0 100%);
        padding: 1rem;
        border-radius: 10px;
        margin: 1rem 0;
    }
    
    .stAlert {
        border-radius: 10px;
    }
</style>
""", unsafe_allow_html=True)

# Color palette for consistent theming
COLORS = {
    'primary': '#FF6B6B',
    'secondary': '#4ECDC4', 
    'accent': '#45B7D1',
    'success': '#26de81',
    'warning': '#ffa726',
    'error': '#ef5350',
    'background': '#f8f9fa'
}

# Global constants
DEFAULT_STOCKS = {
    'US Tech Giants': ['AAPL', 'MSFT', 'GOOGL', 'AMZN', 'META', 'TSLA', 'NVDA'],
    'US Banks': ['JPM', 'BAC', 'WFC', 'C', 'GS', 'MS'],
    'Argentine Stocks': ['GGAL.BA', 'YPF.BA', 'PAM.BA', 'BMA.BA', 'SUPV.BA'],
    'Crypto': ['BTC-USD', 'ETH-USD', 'ADA-USD', 'DOT-USD'],
    'Commodities': ['GC=F', 'SI=F', 'CL=F', 'NG=F'],
    'Indices': ['^GSPC', '^IXIC', '^DJI', '^MERV', '^BVSP']
}

# Cache configuration
@st.cache_data(ttl=300)  # 5 minutes cache
def load_stock_data(symbol: str, period: str = "1y", interval: str = "1d") -> pd.DataFrame:
    """Load stock data with caching and robust error handling"""
    try:
        # Add retry logic for better reliability
        import time
        max_retries = 3
        
        for attempt in range(max_retries):
            try:
                ticker = yf.Ticker(symbol)
                data = ticker.history(period=period, interval=interval)
                
                # Check if data is None or empty
                if data is None or data.empty:
                    if attempt < max_retries - 1:
                        time.sleep(1)  # Wait 1 second before retry
                        continue
                    else:
                        # Silent fail for better UX in cloud environment
                        return pd.DataFrame()
                
                # Validate data structure
                required_columns = ['Open', 'High', 'Low', 'Close', 'Volume']
                if not all(col in data.columns for col in required_columns):
                    return pd.DataFrame()
                
                # Add basic technical indicators safely
                try:
                    if len(data) >= 20:
                        data['SMA_20'] = data['Close'].rolling(window=20).mean()
                    if len(data) >= 50:
                        data['SMA_50'] = data['Close'].rolling(window=50).mean()
                    if len(data) >= 14:
                        data['RSI'] = ta.rsi(data['Close'], length=14)
                    
                    # MACD and Bollinger Bands with error handling
                    try:
                        macd_data = ta.macd(data['Close'])
                        if macd_data is not None and 'MACD_12_26_9' in macd_data.columns:
                            data['MACD'] = macd_data['MACD_12_26_9']
                    except:
                        pass
                    
                    try:
                        bb_data = ta.bbands(data['Close'])
                        if bb_data is not None and len(bb_data.columns) >= 3:
                            data['BB_upper'] = bb_data.iloc[:, 0]
                            data['BB_middle'] = bb_data.iloc[:, 1] 
                            data['BB_lower'] = bb_data.iloc[:, 2]
                    except:
                        pass
                        
                except Exception:
                    # If technical indicators fail, still return basic data
                    pass
                
                return data
                
            except Exception as e:
                if attempt < max_retries - 1:
                    time.sleep(1)
                    continue
                else:
                    # Silent fail for better UX
                    return pd.DataFrame()
                    
    except Exception as e:
        # Silent fail for production environment
        return pd.DataFrame()

@st.cache_data(ttl=3600)  # 1 hour cache
def get_stock_info(symbol: str) -> Dict:
    """Get stock information with caching"""
    try:
        ticker = yf.Ticker(symbol)
        info = ticker.info
        return {
            'name': info.get('shortName', symbol),
            'sector': info.get('sector', 'Unknown'),
            'industry': info.get('industry', 'Unknown'),
            'market_cap': info.get('marketCap', 0),
            'pe_ratio': info.get('trailingPE', 0),
            'dividend_yield': info.get('dividendYield', 0),
            'beta': info.get('beta', 0),
            'currency': info.get('currency', 'USD')
        }
    except Exception as e:
        return {'name': symbol, 'error': str(e)}

def create_candlestick_chart(data: pd.DataFrame, symbol: str, indicators: List[str] = None) -> go.Figure:
    """Create interactive candlestick chart with technical indicators"""
    
    # Create subplots
    subplot_titles = ["Price", "Volume"]
    if indicators and any(ind in indicators for ind in ['RSI', 'MACD']):
        subplot_titles.append("Technical Indicators")
        rows = 3
        row_heights = [0.6, 0.2, 0.2]
    else:
        rows = 2
        row_heights = [0.8, 0.2]
    
    fig = make_subplots(
        rows=rows, cols=1,
        shared_xaxes=True,
        vertical_spacing=0.05,
        subplot_titles=subplot_titles,
        row_heights=row_heights
    )
    
    # Candlestick chart
    fig.add_trace(
        go.Candlestick(
            x=data.index,
            open=data['Open'],
            high=data['High'],
            low=data['Low'],
            close=data['Close'],
            name="OHLC",
            increasing_line_color=COLORS['success'],
            decreasing_line_color=COLORS['error']
        ), row=1, col=1
    )
    
    # Add moving averages if available
    if 'SMA_20' in data.columns:
        fig.add_trace(
            go.Scatter(x=data.index, y=data['SMA_20'], 
                      line=dict(color=COLORS['primary'], width=2),
                      name="SMA 20"), row=1, col=1
        )
    
    if 'SMA_50' in data.columns:
        fig.add_trace(
            go.Scatter(x=data.index, y=data['SMA_50'], 
                      line=dict(color=COLORS['secondary'], width=2),
                      name="SMA 50"), row=1, col=1
        )
    
    # Add Bollinger Bands if available
    if all(col in data.columns for col in ['BB_upper', 'BB_lower']):
        fig.add_trace(
            go.Scatter(x=data.index, y=data['BB_upper'], 
                      line=dict(color='rgba(0,100,80,0)', width=0),
                      showlegend=False), row=1, col=1
        )
        fig.add_trace(
            go.Scatter(x=data.index, y=data['BB_lower'], 
                      line=dict(color='rgba(0,100,80,0)', width=0),
                      fill='tonexty', fillcolor='rgba(0,100,80,0.2)',
                      name="Bollinger Bands"), row=1, col=1
        )
    
    # Volume chart
    colors = ['green' if close > open else 'red' 
              for close, open in zip(data['Close'], data['Open'])]
    
    fig.add_trace(
        go.Bar(x=data.index, y=data['Volume'], 
               marker_color=colors, name="Volume",
               opacity=0.7), row=2, col=1
    )
    
    # Technical indicators subplot
    if rows == 3:
        if indicators and 'RSI' in indicators and 'RSI' in data.columns:
            fig.add_trace(
                go.Scatter(x=data.index, y=data['RSI'], 
                          line=dict(color=COLORS['accent'], width=2),
                          name="RSI"), row=3, col=1
            )
            fig.add_hline(y=70, line_dash="dash", line_color="red", row=3, col=1)
            fig.add_hline(y=30, line_dash="dash", line_color="green", row=3, col=1)
        
        if indicators and 'MACD' in indicators and 'MACD' in data.columns:
            fig.add_trace(
                go.Scatter(x=data.index, y=data['MACD'], 
                          line=dict(color=COLORS['warning'], width=2),
                          name="MACD"), row=3, col=1
            )
    
    # Update layout
    fig.update_layout(
        title=f"{symbol} - Interactive Price Chart",
        xaxis_rangeslider_visible=False,
        height=700,
        template="plotly_white",
        font=dict(size=12),
        hovermode='x unified'
    )
    
    return fig

def create_correlation_heatmap(data_dict: Dict[str, pd.DataFrame]) -> go.Figure:
    """Create correlation heatmap for multiple stocks"""
    
    # Extract closing prices
    price_data = pd.DataFrame()
    for symbol, data in data_dict.items():
        if not data.empty:
            price_data[symbol] = data['Close']
    
    if price_data.empty:
        return go.Figure()
    
    # Calculate correlation matrix
    correlation_matrix = price_data.corr()
    
    # Create heatmap
    fig = go.Figure(data=go.Heatmap(
        z=correlation_matrix.values,
        x=correlation_matrix.columns,
        y=correlation_matrix.columns,
        colorscale='RdBu',
        zmid=0,
        text=correlation_matrix.round(3).values,
        texttemplate="%{text}",
        textfont={"size": 10},
        hoverongaps=False
    ))
    
    fig.update_layout(
        title="Stock Correlation Matrix",
        xaxis_title="Stocks",
        yaxis_title="Stocks",
        template="plotly_white",
        height=500
    )
    
    return fig

def predict_stock_price(data: pd.DataFrame, days_ahead: int = 30) -> Tuple[np.ndarray, float]:
    """Predict stock price using ensemble ML models"""
    
    if len(data) < 60:  # Need sufficient data
        return np.array([]), 0.0
    
    # Prepare features
    data_ml = data.copy()
    data_ml['Returns'] = data_ml['Close'].pct_change()
    data_ml['Volatility'] = data_ml['Returns'].rolling(window=20).std()
    data_ml['Price_MA_ratio'] = data_ml['Close'] / data_ml['SMA_20']
    
    # Create lagged features
    for lag in [1, 2, 3, 5, 10]:
        data_ml[f'Close_lag_{lag}'] = data_ml['Close'].shift(lag)
        data_ml[f'Volume_lag_{lag}'] = data_ml['Volume'].shift(lag)
    
    # Select features
    feature_cols = ['Open', 'High', 'Low', 'Volume', 'SMA_20', 'SMA_50', 'RSI', 
                   'Returns', 'Volatility', 'Price_MA_ratio'] + \
                   [f'Close_lag_{lag}' for lag in [1, 2, 3, 5, 10]] + \
                   [f'Volume_lag_{lag}' for lag in [1, 2, 3, 5, 10]]
    
    # Filter existing columns
    feature_cols = [col for col in feature_cols if col in data_ml.columns]
    
    # Prepare data
    X = data_ml[feature_cols].dropna()
    y = data_ml['Close'].loc[X.index]
    
    if len(X) < 30:
        return np.array([]), 0.0
    
    # Split data
    train_size = int(len(X) * 0.8)
    X_train, X_test = X[:train_size], X[train_size:]
    y_train, y_test = y[:train_size], y[train_size:]
    
    # Scale features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # Train ensemble models
    models = {
        'rf': RandomForestRegressor(n_estimators=100, random_state=42),
        'xgb': XGBRegressor(n_estimators=100, random_state=42),
        'lgb': lgb.LGBMRegressor(n_estimators=100, random_state=42, verbose=-1)
    }
    
    predictions = []
    scores = []
    
    for name, model in models.items():
        try:
            model.fit(X_train_scaled, y_train)
            pred = model.predict(X_test_scaled)
            score = model.score(X_test_scaled, y_test)
            predictions.append(pred)
            scores.append(score)
        except:
            continue
    
    if not predictions:
        return np.array([]), 0.0
    
    # Ensemble prediction
    ensemble_pred = np.mean(predictions, axis=0)
    avg_score = np.mean(scores)
    
    # Predict future prices
    last_features = X.iloc[-1:].values
    last_features_scaled = scaler.transform(last_features)
    
    future_predictions = []
    current_features = last_features_scaled.copy()
    
    for _ in range(days_ahead):
        day_predictions = []
        for name, model in models.items():
            try:
                pred = model.predict(current_features)[0]
                day_predictions.append(pred)
            except:
                continue
        
        if day_predictions:
            pred = np.mean(day_predictions)
            future_predictions.append(pred)
            
            # Update features for next prediction (simplified)
            # This would need more sophisticated feature engineering in production
            if len(current_features[0]) > 0:
                current_features[0][0] = pred  # Update close price feature
    
    return np.array(future_predictions), avg_score

def main():
    """Main application function"""
    
    # Header
    st.markdown('<h1 class="main-header">📈 Stock Analysis Dashboard 2025</h1>', 
                unsafe_allow_html=True)
    
    # Sidebar navigation
    with st.sidebar:
        st.markdown('<div class="sidebar-info"><h3>🎯 Navigation</h3></div>', 
                   unsafe_allow_html=True)
        
        selected = option_menu(
            menu_title=None,
            options=["📊 Market Overview", "🔍 Stock Analysis", "🤖 AI Predictions", 
                    "📈 Portfolio Tracker", "🔗 Correlation Analysis", "📋 Technical Screener"],
            icons=["graph-up", "search", "robot", "briefcase", "link", "filter"],
            default_index=0,
            styles={
                "container": {"padding": "0!important", "background-color": "#fafafa"},
                "icon": {"color": COLORS['primary'], "font-size": "18px"},
                "nav-link": {"font-size": "16px", "text-align": "left", "margin": "0px", 
                           "--hover-color": "#eee"},
                "nav-link-selected": {"background-color": COLORS['primary']},
            }
        )
    
    # Market Overview Page
    if selected == "📊 Market Overview":
        st.subheader("🌍 Global Market Overview")
        
        # Add info about data loading
        with st.expander("ℹ️ About Market Data", expanded=False):
            st.info("""
            **Market data is fetched in real-time from Yahoo Finance.**
            
            - Data may take a few seconds to load due to API rate limits
            - If you see "Loading..." it means the data is being fetched
            - Refresh the page if data doesn't appear after 30 seconds
            - All times are in UTC
            """)
        
        st.divider()
        
        # Major indices
        indices = ['^GSPC', '^IXIC', '^DJI', '^FTSE', '^N225']
        index_names = ['S&P 500', 'NASDAQ', 'Dow Jones', 'FTSE 100', 'Nikkei 225']
        
        cols = st.columns(len(indices))
        
        for i, (index, name) in enumerate(zip(indices, index_names)):
            with cols[i]:
                data = load_stock_data(index, period="5d", interval="1d")
                if not data.empty and len(data) > 0:
                    try:
                        current_price = data['Close'].iloc[-1]
                        prev_price = data['Close'].iloc[-2] if len(data) > 1 else current_price
                        change = current_price - prev_price
                        change_pct = (change / prev_price) * 100
                        
                        st.metric(
                            label=name,
                            value=f"{current_price:.2f}",
                            delta=f"{change_pct:.2f}%"
                        )
                    except Exception:
                        st.metric(
                            label=name,
                            value="Loading...",
                            delta="--"
                        )
                else:
                    st.metric(
                        label=name,
                        value="Loading...",
                        delta="--"
                    )
        
        # Market heatmap
        st.subheader("🗺️ Market Heatmap")
        
        # Top stocks data
        top_stocks = ['AAPL', 'MSFT', 'GOOGL', 'AMZN', 'TSLA', 'META', 'NVDA', 'JPM']
        heatmap_data = []
        
        for symbol in top_stocks:
            data = load_stock_data(symbol, period="5d", interval="1d")
            if not data.empty and len(data) > 0:
                try:
                    current_price = data['Close'].iloc[-1]
                    prev_price = data['Close'].iloc[-2] if len(data) > 1 else current_price
                    change_pct = ((current_price - prev_price) / prev_price) * 100
                    
                    heatmap_data.append({
                        'Symbol': symbol,
                        'Change %': change_pct,
                        'Price': current_price
                    })
                except Exception:
                    continue
        
        if heatmap_data:
            heatmap_df = pd.DataFrame(heatmap_data)
            
            fig = px.treemap(
                heatmap_df,
                path=['Symbol'],
                values='Price',
                color='Change %',
                color_continuous_scale='RdYlGn',
                title="Market Performance Heatmap"
            )
            fig.update_layout(height=400)
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("📊 Market data is currently loading. This may take a few moments due to API limitations. Please refresh the page in a moment.")
    
    # Stock Analysis Page
    elif selected == "🔍 Stock Analysis":
        st.subheader("🔍 Advanced Stock Analysis")
        
        # Stock selection
        col1, col2, col3 = st.columns([2, 1, 1])
        
        with col1:
            # Stock category selection
            category = st.selectbox("📂 Select Category", list(DEFAULT_STOCKS.keys()))
            
            # Custom stock input
            custom_symbol = st.text_input("🔤 Or enter custom symbol:", placeholder="e.g., AAPL")
            
            # Final symbol selection
            if custom_symbol:
                symbol = custom_symbol.upper()
            else:
                symbol = st.selectbox("📊 Select Stock", DEFAULT_STOCKS[category])
        
        with col2:
            period = st.selectbox("📅 Time Period", 
                                 ["1mo", "3mo", "6mo", "1y", "2y", "5y"], index=3)
            
        with col3:
            interval = st.selectbox("⏱️ Interval", 
                                   ["1d", "1wk", "1mo"], index=0)
        
        if symbol:
            # Load data
            data = load_stock_data(symbol, period, interval)
            stock_info = get_stock_info(symbol)
            
            if not data.empty:
                # Stock information header
                col1, col2, col3, col4 = st.columns(4)
                
                current_price = data['Close'].iloc[-1]
                prev_price = data['Close'].iloc[-2] if len(data) > 1 else current_price
                change = current_price - prev_price
                change_pct = (change / prev_price) * 100
                
                with col1:
                    st.metric("💰 Current Price", f"${current_price:.2f}", f"{change_pct:.2f}%")
                
                with col2:
                    high_52w = data['High'].max()
                    st.metric("📈 52W High", f"${high_52w:.2f}")
                
                with col3:
                    low_52w = data['Low'].min()
                    st.metric("📉 52W Low", f"${low_52w:.2f}")
                
                with col4:
                    avg_volume = data['Volume'].mean()
                    st.metric("📊 Avg Volume", f"{avg_volume/1e6:.1f}M")
                
                # Technical indicators selection
                st.subheader("🛠️ Technical Indicators")
                indicators = st.multiselect(
                    "Select indicators to display:",
                    ["RSI", "MACD", "Bollinger Bands", "Volume"],
                    default=["RSI", "Volume"]
                )
                
                # Create and display chart
                fig = create_candlestick_chart(data, symbol, indicators)
                st.plotly_chart(fig, use_container_width=True)
                
                # Additional analysis
                col1, col2 = st.columns(2)
                
                with col1:
                    st.subheader("📊 Price Statistics")
                    stats_df = pd.DataFrame({
                        'Metric': ['Mean', 'Median', 'Std Dev', 'Min', 'Max'],
                        'Value': [
                            f"${data['Close'].mean():.2f}",
                            f"${data['Close'].median():.2f}",
                            f"${data['Close'].std():.2f}",
                            f"${data['Close'].min():.2f}",
                            f"${data['Close'].max():.2f}"
                        ]
                    })
                    st.dataframe(stats_df, hide_index=True)
                
                with col2:
                    st.subheader("🏢 Company Info")
                    if 'error' not in stock_info:
                        info_df = pd.DataFrame({
                            'Attribute': ['Name', 'Sector', 'Industry', 'Market Cap', 'P/E Ratio'],
                            'Value': [
                                stock_info.get('name', 'N/A'),
                                stock_info.get('sector', 'N/A'),
                                stock_info.get('industry', 'N/A'),
                                f"${stock_info.get('market_cap', 0)/1e9:.1f}B" if stock_info.get('market_cap') else 'N/A',
                                f"{stock_info.get('pe_ratio', 0):.2f}" if stock_info.get('pe_ratio') else 'N/A'
                            ]
                        })
                        st.dataframe(info_df, hide_index=True)
            else:
                st.error(f"❌ No data available for {symbol}")
    
    # AI Predictions Page
    elif selected == "🤖 AI Predictions":
        st.subheader("🤖 AI-Powered Price Predictions")
        
        # Stock selection for prediction
        symbol = st.selectbox("Select stock for prediction:", 
                             ['AAPL', 'MSFT', 'GOOGL', 'AMZN', 'TSLA'])
        
        prediction_days = st.slider("📅 Prediction period (days):", 1, 90, 30)
        
        if st.button("🚀 Generate Prediction"):
            with st.spinner("🧠 AI is analyzing patterns..."):
                # Load data
                data = load_stock_data(symbol, period="2y", interval="1d")
                
                if not data.empty:
                    # Generate predictions
                    predictions, accuracy = predict_stock_price(data, prediction_days)
                    
                    if len(predictions) > 0:
                        # Create prediction chart
                        last_date = data.index[-1]
                        future_dates = pd.date_range(start=last_date + timedelta(days=1), 
                                                   periods=len(predictions), freq='D')
                        
                        fig = go.Figure()
                        
                        # Historical data
                        fig.add_trace(go.Scatter(
                            x=data.index[-60:], 
                            y=data['Close'].iloc[-60:],
                            mode='lines',
                            name='Historical',
                            line=dict(color=COLORS['primary'])
                        ))
                        
                        # Predictions
                        fig.add_trace(go.Scatter(
                            x=future_dates,
                            y=predictions,
                            mode='lines',
                            name='Predicted',
                            line=dict(color=COLORS['warning'], dash='dash')
                        ))
                        
                        fig.update_layout(
                            title=f"{symbol} - AI Price Prediction",
                            xaxis_title="Date",
                            yaxis_title="Price ($)",
                            template="plotly_white",
                            height=500
                        )
                        
                        st.plotly_chart(fig, use_container_width=True)
                        
                        # Prediction summary
                        current_price = data['Close'].iloc[-1]
                        predicted_price = predictions[-1]
                        predicted_change = ((predicted_price - current_price) / current_price) * 100
                        
                        col1, col2, col3 = st.columns(3)
                        with col1:
                            st.metric("🎯 Current Price", f"${current_price:.2f}")
                        with col2:
                            st.metric("🔮 Predicted Price", f"${predicted_price:.2f}", 
                                    f"{predicted_change:.2f}%")
                        with col3:
                            st.metric("🎯 Model Accuracy", f"{accuracy:.1%}")
                        
                        # Disclaimer
                        st.warning("⚠️ This is for educational purposes only. Not financial advice!")
                    else:
                        st.error("❌ Unable to generate predictions. Insufficient data.")
                else:
                    st.error(f"❌ No data available for {symbol}")
    
    # Portfolio Tracker Page
    elif selected == "📈 Portfolio Tracker":
        st.subheader("📈 Portfolio Performance Tracker")
        
        st.info("🚧 Portfolio tracking feature coming soon! This will include:")
        st.markdown("""
        - 📊 Real-time portfolio performance tracking
        - 💰 P&L analysis and attribution
        - 📈 Risk metrics and diversification analysis
        - 🎯 Rebalancing recommendations
        - 📱 Export capabilities to Excel/PDF
        """)
        
        # Demo portfolio
        demo_data = {
            'Symbol': ['AAPL', 'MSFT', 'GOOGL', 'AMZN'],
            'Shares': [10, 15, 5, 8],
            'Avg Cost': [150.0, 280.0, 2500.0, 3200.0],
            'Current Price': [175.0, 310.0, 2650.0, 3100.0],
        }
        
        demo_df = pd.DataFrame(demo_data)
        demo_df['Market Value'] = demo_df['Shares'] * demo_df['Current Price']
        demo_df['Total Cost'] = demo_df['Shares'] * demo_df['Avg Cost']
        demo_df['P&L'] = demo_df['Market Value'] - demo_df['Total Cost']
        demo_df['P&L %'] = (demo_df['P&L'] / demo_df['Total Cost']) * 100
        
        st.subheader("🎭 Demo Portfolio")
        st.dataframe(demo_df, hide_index=True)
        
        # Portfolio pie chart
        fig = px.pie(demo_df, values='Market Value', names='Symbol', 
                    title="Portfolio Allocation")
        st.plotly_chart(fig, use_container_width=True)
    
    # Correlation Analysis Page
    elif selected == "🔗 Correlation Analysis":
        st.subheader("🔗 Stock Correlation Analysis")
        
        # Stock selection for correlation
        selected_stocks = st.multiselect(
            "Select stocks for correlation analysis:",
            ['AAPL', 'MSFT', 'GOOGL', 'AMZN', 'TSLA', 'META', 'NVDA', 'JPM', 'BAC'],
            default=['AAPL', 'MSFT', 'GOOGL', 'AMZN']
        )
        
        if len(selected_stocks) >= 2:
            # Load data for all selected stocks
            data_dict = {}
            for symbol in selected_stocks:
                data = load_stock_data(symbol, period="1y", interval="1d")
                if not data.empty:
                    data_dict[symbol] = data
            
            if data_dict:
                # Create correlation heatmap
                fig = create_correlation_heatmap(data_dict)
                st.plotly_chart(fig, use_container_width=True)
                
                # Calculate and display correlation matrix
                price_data = pd.DataFrame()
                for symbol, data in data_dict.items():
                    price_data[symbol] = data['Close']
                
                correlation_matrix = price_data.corr()
                st.subheader("📊 Correlation Matrix")
                st.dataframe(correlation_matrix.round(3))
                
                # Interpretation guide
                st.subheader("📖 Interpretation Guide")
                st.markdown("""
                - **1.0**: Perfect positive correlation
                - **0.7 to 0.99**: Strong positive correlation  
                - **0.3 to 0.69**: Moderate positive correlation
                - **-0.3 to 0.29**: Weak correlation
                - **-0.69 to -0.3**: Moderate negative correlation
                - **-0.99 to -0.7**: Strong negative correlation
                - **-1.0**: Perfect negative correlation
                """)
        else:
            st.warning("⚠️ Please select at least 2 stocks for correlation analysis.")
    
    # Technical Screener Page
    elif selected == "📋 Technical Screener":
        st.subheader("📋 Technical Stock Screener")
        
        st.info("🚧 Technical screener coming soon! This will include:")
        st.markdown("""
        - 🔍 Multi-criteria stock screening
        - 📊 RSI, MACD, and momentum filters
        - 💹 Volume and price action alerts
        - 🎯 Custom screening criteria
        - 📈 Real-time scanning of 1000+ stocks
        """)
        
        # Demo screener results
        demo_results = {
            'Symbol': ['AAPL', 'MSFT', 'GOOGL', 'AMZN', 'TSLA'],
            'Price': [175.50, 310.25, 2650.00, 3100.00, 185.75],
            'RSI': [65.2, 58.7, 42.1, 71.5, 55.8],
            'MACD Signal': ['Bullish', 'Neutral', 'Bearish', 'Bullish', 'Neutral'],
            'Volume Alert': ['High', 'Normal', 'Normal', 'High', 'Normal']
        }
        
        demo_df = pd.DataFrame(demo_results)
        st.subheader("🎭 Demo Screener Results")
        st.dataframe(demo_df, hide_index=True)
    
    # Footer
    st.markdown("---")
    st.markdown("""
    <div style='text-align: center; color: #666;'>
        <p>🚀 <strong>Stock Analysis Dashboard 2025</strong> - Powered by AI & Modern Analytics</p>
        <p>Created by <a href='https://github.com/PabloPoletti'>Pablo Poletti</a> | 
        <a href='mailto:lic.poletti@gmail.com'>lic.poletti@gmail.com</a> | 
        <a href='https://www.linkedin.com/in/pablom-poletti/'>LinkedIn</a></p>
    </div>
    """, unsafe_allow_html=True)

if __name__ == "__main__":
    main()
