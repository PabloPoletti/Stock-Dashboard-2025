"""
Utility functions for the Modern Stock Dashboard 2025
"""

import pandas as pd
import numpy as np
import plotly.graph_objects as go
import streamlit as st
from typing import Dict, List, Optional, Tuple
import yfinance as yf
from datetime import datetime, timedelta
import io
import base64

def format_number(value: float, format_type: str = "currency") -> str:
    """Format numbers for display"""
    if pd.isna(value) or value == 0:
        return "N/A"
    
    if format_type == "currency":
        if abs(value) >= 1e9:
            return f"${value/1e9:.1f}B"
        elif abs(value) >= 1e6:
            return f"${value/1e6:.1f}M"
        elif abs(value) >= 1e3:
            return f"${value/1e3:.1f}K"
        else:
            return f"${value:.2f}"
    elif format_type == "percentage":
        return f"{value:.2f}%"
    elif format_type == "number":
        if abs(value) >= 1e6:
            return f"{value/1e6:.1f}M"
        elif abs(value) >= 1e3:
            return f"{value/1e3:.1f}K"
        else:
            return f"{value:.2f}"
    
    return str(value)

def calculate_returns(data: pd.DataFrame, period: str = "daily") -> pd.Series:
    """Calculate returns for different periods"""
    if period == "daily":
        return data['Close'].pct_change()
    elif period == "weekly":
        return data['Close'].resample('W').last().pct_change()
    elif period == "monthly":
        return data['Close'].resample('M').last().pct_change()
    else:
        return data['Close'].pct_change()

def calculate_volatility(data: pd.DataFrame, window: int = 20) -> pd.Series:
    """Calculate rolling volatility"""
    returns = calculate_returns(data)
    return returns.rolling(window=window).std() * np.sqrt(252)  # Annualized

def calculate_sharpe_ratio(data: pd.DataFrame, risk_free_rate: float = 0.02) -> float:
    """Calculate Sharpe ratio"""
    returns = calculate_returns(data)
    excess_returns = returns.mean() * 252 - risk_free_rate  # Annualized
    volatility = returns.std() * np.sqrt(252)  # Annualized
    
    if volatility == 0:
        return 0
    return excess_returns / volatility

def detect_support_resistance(data: pd.DataFrame, window: int = 20) -> Dict:
    """Detect support and resistance levels"""
    highs = data['High'].rolling(window=window, center=True).max()
    lows = data['Low'].rolling(window=window, center=True).min()
    
    resistance_levels = data[data['High'] == highs]['High'].dropna().unique()
    support_levels = data[data['Low'] == lows]['Low'].dropna().unique()
    
    # Get recent levels
    current_price = data['Close'].iloc[-1]
    
    resistance = [level for level in resistance_levels if level > current_price]
    support = [level for level in support_levels if level < current_price]
    
    return {
        'resistance': sorted(resistance)[:3],  # Top 3 resistance levels
        'support': sorted(support, reverse=True)[:3]  # Top 3 support levels
    }

def generate_trading_signals(data: pd.DataFrame) -> pd.DataFrame:
    """Generate basic trading signals"""
    signals = pd.DataFrame(index=data.index)
    
    # RSI signals
    if 'RSI' in data.columns:
        signals['RSI_oversold'] = data['RSI'] < 30
        signals['RSI_overbought'] = data['RSI'] > 70
    
    # Moving average crossover
    if 'SMA_20' in data.columns and 'SMA_50' in data.columns:
        signals['MA_bullish'] = data['SMA_20'] > data['SMA_50']
        signals['MA_bearish'] = data['SMA_20'] < data['SMA_50']
    
    # Volume spike
    if 'Volume' in data.columns:
        avg_volume = data['Volume'].rolling(window=20).mean()
        signals['Volume_spike'] = data['Volume'] > (avg_volume * 1.5)
    
    return signals

def create_export_data(data: pd.DataFrame, symbol: str) -> bytes:
    """Create Excel export data"""
    output = io.BytesIO()
    with pd.ExcelWriter(output, engine='xlsxwriter') as writer:
        # Main data
        data.to_excel(writer, sheet_name='Price_Data', index=True)
        
        # Summary statistics
        summary = pd.DataFrame({
            'Metric': ['Mean', 'Median', 'Std Dev', 'Min', 'Max', 'Current'],
            'Value': [
                data['Close'].mean(),
                data['Close'].median(),
                data['Close'].std(),
                data['Close'].min(),
                data['Close'].max(),
                data['Close'].iloc[-1]
            ]
        })
        summary.to_excel(writer, sheet_name='Summary', index=False)
        
        # Trading signals if available
        signals = generate_trading_signals(data)
        if not signals.empty:
            signals.to_excel(writer, sheet_name='Signals', index=True)
    
    output.seek(0)
    return output.getvalue()

def get_market_status() -> Dict:
    """Get current market status"""
    now = datetime.now()
    
    # Simple market hours check (NYSE)
    market_open = now.replace(hour=9, minute=30, second=0, microsecond=0)
    market_close = now.replace(hour=16, minute=0, second=0, microsecond=0)
    
    is_weekday = now.weekday() < 5
    is_market_hours = market_open <= now <= market_close
    
    status = "Open" if (is_weekday and is_market_hours) else "Closed"
    
    return {
        'status': status,
        'is_open': status == "Open",
        'next_open': market_open + timedelta(days=1) if status == "Closed" else None,
        'time_to_close': market_close - now if status == "Open" else None
    }

def create_download_link(data: bytes, filename: str, link_text: str) -> str:
    """Create a download link for data"""
    b64 = base64.b64encode(data).decode()
    href = f'<a href="data:application/vnd.openxmlformats-officedocument.spreadsheetml.sheet;base64,{b64}" download="{filename}">{link_text}</a>'
    return href

def validate_symbol(symbol: str) -> bool:
    """Validate if a stock symbol exists"""
    try:
        ticker = yf.Ticker(symbol)
        info = ticker.info
        return 'regularMarketPrice' in info or 'currentPrice' in info
    except:
        return False

def get_sector_performance() -> pd.DataFrame:
    """Get sector performance data"""
    sectors = {
        'Technology': 'XLK',
        'Healthcare': 'XLV',
        'Financial': 'XLF',
        'Energy': 'XLE',
        'Consumer Discretionary': 'XLY',
        'Consumer Staples': 'XLP',
        'Industrials': 'XLI',
        'Materials': 'XLB',
        'Utilities': 'XLU',
        'Real Estate': 'XLRE'
    }
    
    sector_data = []
    for sector, etf in sectors.items():
        try:
            data = yf.download(etf, period="5d", interval="1d")
            if not data.empty:
                current = data['Close'].iloc[-1]
                previous = data['Close'].iloc[-2] if len(data) > 1 else current
                change_pct = ((current - previous) / previous) * 100
                
                sector_data.append({
                    'Sector': sector,
                    'ETF': etf,
                    'Price': current,
                    'Change %': change_pct
                })
        except:
            continue
    
    return pd.DataFrame(sector_data)

def calculate_portfolio_metrics(portfolio: pd.DataFrame) -> Dict:
    """Calculate portfolio performance metrics"""
    if portfolio.empty:
        return {}
    
    total_value = portfolio['Market Value'].sum()
    total_cost = portfolio['Total Cost'].sum()
    total_pnl = total_value - total_cost
    total_pnl_pct = (total_pnl / total_cost) * 100 if total_cost > 0 else 0
    
    return {
        'total_value': total_value,
        'total_cost': total_cost,
        'total_pnl': total_pnl,
        'total_pnl_pct': total_pnl_pct,
        'num_positions': len(portfolio),
        'largest_position': portfolio.loc[portfolio['Market Value'].idxmax(), 'Symbol'] if not portfolio.empty else None,
        'best_performer': portfolio.loc[portfolio['P&L %'].idxmax(), 'Symbol'] if not portfolio.empty else None,
        'worst_performer': portfolio.loc[portfolio['P&L %'].idxmin(), 'Symbol'] if not portfolio.empty else None
    }

class TechnicalAnalyzer:
    """Advanced technical analysis functions"""
    
    @staticmethod
    def fibonacci_retracement(data: pd.DataFrame, high_idx: int = None, low_idx: int = None) -> Dict:
        """Calculate Fibonacci retracement levels"""
        if high_idx is None:
            high_idx = data['High'].idxmax()
        if low_idx is None:
            low_idx = data['Low'].idxmin()
        
        high_price = data.loc[high_idx, 'High']
        low_price = data.loc[low_idx, 'Low']
        diff = high_price - low_price
        
        levels = {
            '0%': high_price,
            '23.6%': high_price - 0.236 * diff,
            '38.2%': high_price - 0.382 * diff,
            '50%': high_price - 0.5 * diff,
            '61.8%': high_price - 0.618 * diff,
            '78.6%': high_price - 0.786 * diff,
            '100%': low_price
        }
        
        return levels
    
    @staticmethod
    def ichimoku_cloud(data: pd.DataFrame) -> pd.DataFrame:
        """Calculate Ichimoku Cloud components"""
        # Tenkan-sen (Conversion Line): (9-period high + 9-period low)/2
        nine_period_high = data['High'].rolling(window=9).max()
        nine_period_low = data['Low'].rolling(window=9).min()
        data['tenkan_sen'] = (nine_period_high + nine_period_low) / 2
        
        # Kijun-sen (Base Line): (26-period high + 26-period low)/2
        period26_high = data['High'].rolling(window=26).max()
        period26_low = data['Low'].rolling(window=26).min()
        data['kijun_sen'] = (period26_high + period26_low) / 2
        
        # Senkou Span A (Leading Span A): (Conversion Line + Base Line)/2
        data['senkou_span_a'] = ((data['tenkan_sen'] + data['kijun_sen']) / 2).shift(26)
        
        # Senkou Span B (Leading Span B): (52-period high + 52-period low)/2
        period52_high = data['High'].rolling(window=52).max()
        period52_low = data['Low'].rolling(window=52).min()
        data['senkou_span_b'] = ((period52_high + period52_low) / 2).shift(26)
        
        # Chikou Span (Lagging Span): Close price shifted back 26 periods
        data['chikou_span'] = data['Close'].shift(-26)
        
        return data
    
    @staticmethod
    def elliott_wave_count(data: pd.DataFrame) -> List[Dict]:
        """Basic Elliott Wave pattern detection"""
        # This is a simplified version - real Elliott Wave analysis is much more complex
        highs = data['High'].rolling(window=10, center=True).max()
        lows = data['Low'].rolling(window=10, center=True).min()
        
        peaks = data[data['High'] == highs].index.tolist()
        troughs = data[data['Low'] == lows].index.tolist()
        
        # Combine and sort turning points
        turning_points = []
        for idx in peaks:
            turning_points.append({'date': idx, 'price': data.loc[idx, 'High'], 'type': 'peak'})
        for idx in troughs:
            turning_points.append({'date': idx, 'price': data.loc[idx, 'Low'], 'type': 'trough'})
        
        turning_points.sort(key=lambda x: x['date'])
        
        return turning_points[-10:]  # Return last 10 turning points
