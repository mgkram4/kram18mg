import json
import logging
import os
import threading
import time
from collections import deque
from datetime import datetime, timedelta

import numpy as np
import pandas as pd
from alpaca.data.historical import (CryptoHistoricalDataClient,
                                    StockHistoricalDataClient)
from alpaca.data.requests import (CryptoLatestQuoteRequest, StockBarsRequest,
                                  StockLatestQuoteRequest)
from alpaca.data.timeframe import TimeFrame
from alpaca.trading.client import TradingClient
from alpaca.trading.enums import OrderSide, TimeInForce
from alpaca.trading.requests import MarketOrderRequest
from dotenv import load_dotenv
from flask import Flask, jsonify, render_template, request

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('trading_bot.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# Load environment variables
load_dotenv()

app = Flask(__name__)

# Initialize clients
api_key = os.getenv('ALPACA_API_KEY')
secret_key = os.getenv('ALPACA_SECRET_KEY')

stock_data_client = StockHistoricalDataClient(api_key, secret_key)
crypto_data_client = CryptoHistoricalDataClient(api_key, secret_key)
trading_client = TradingClient(api_key, secret_key, paper=True)

# Technical Indicators
def calculate_ema(data, period):
    """Calculate Exponential Moving Average"""
    ema = []
    k = 2 / (period + 1)
    
    # Initialize with SMA
    ema.append(sum(data[:period]) / period)
    
    # Calculate EMA
    for price in data[period:]:
        ema.append(price * k + ema[-1] * (1 - k))
    
    # Pad the beginning with None
    return [None] * (len(data) - len(ema)) + ema

def calculate_rsi(data, period=14):
    """Calculate Relative Strength Index"""
    if len(data) < period + 1:
        return [None] * len(data)
        
    deltas = np.diff(data)
    seed = deltas[:period+1]
    up = seed[seed > 0].sum()/period
    down = -seed[seed < 0].sum()/period
    rs = up/down if down != 0 else 0
    rsi = np.zeros_like(data)
    rsi[period] = 100 - 100/(1+rs)
    
    for i in range(period + 1, len(data)):
        delta = deltas[i-1]
        if delta > 0:
            upval = delta
            downval = 0
        else:
            upval = 0
            downval = -delta
            
        up = (up*(period-1) + upval)/period
        down = (down*(period-1) + downval)/period
        rs = up/down if down != 0 else 0
        rsi[i] = 100 - 100/(1+rs)
    
    return rsi.tolist()

def calculate_macd(data, fast_period=12, slow_period=26, signal_period=9):
    """Calculate MACD, Signal line, and Histogram"""
    ema_fast = calculate_ema(data, fast_period)
    ema_slow = calculate_ema(data, slow_period)
    
    macd_line = []
    for i in range(len(data)):
        if ema_fast[i] is not None and ema_slow[i] is not None:
            macd_line.append(ema_fast[i] - ema_slow[i])
        else:
            macd_line.append(None)
    
    signal_line = calculate_ema([x for x in macd_line if x is not None], signal_period)
    signal_line = [None] * (len(macd_line) - len(signal_line)) + signal_line
    
    histogram = []
    for i in range(len(macd_line)):
        if macd_line[i] is not None and signal_line[i] is not None:
            histogram.append(macd_line[i] - signal_line[i])
        else:
            histogram.append(None)
    
    return macd_line, signal_line, histogram


def macd_strategy(data, current_position=0):
    """Enhanced MACD Strategy with Trend and Volume Analysis"""
    prices = data['close']
    volumes = data['volume']
    
    if len(prices) < 50:
        return []
    
    # Calculate indicators
    macd, signal, hist = calculate_macd(prices)
    ema_50 = calculate_ema(prices, 50)
    volume_ma = pd.Series(volumes).rolling(window=20).mean().tolist()
    
    trades = []
    i = len(prices) - 1
    
    if i >= 49:
        current_price = prices[i]
        
        # Calculate trend strength
        trend_strength = abs(current_price - ema_50[i]) / ema_50[i] * 100
        
        # Enhanced buy conditions
        buy_conditions = (
            hist[i] > 0 and hist[i-1] <= 0 and  # MACD histogram crossover
            macd[i] > signal[i] and  # MACD line above signal
            current_price > ema_50[i] and  # Price above trend
            trend_strength < 5 and  # Not overextended
            volumes[i] > volume_ma[i] * 1.3 and  # Above average volume
            current_position == 0
        )
        
        # Enhanced sell conditions
        sell_conditions = (
            current_position > 0 and (
                (hist[i] < 0 and hist[i-1] >= 0) or  # MACD histogram reversal
                (macd[i] < signal[i] and macd[i-1] >= signal[i-1]) or  # MACD crossover down
                current_price < ema_50[i] or  # Price below trend
                trend_strength > 7  # Too extended
            )
        )
        
        if buy_conditions:
            # Position sizing based on trend strength
            base_position = max(1, int(100000 / current_price))
            position_size = int(base_position * (1 - trend_strength/10))  # Reduce size when extended
            trades.append(("BUY", position_size))
        elif sell_conditions:
            trades.append(("SELL", current_position))
    
    return trades

def calculate_vwap(prices, volumes):
    """Calculate Volume Weighted Average Price"""
    cumulative_volume = np.cumsum(volumes)
    cumulative_pv = np.cumsum(np.multiply(prices, volumes))
    return np.divide(cumulative_pv, cumulative_volume, out=np.zeros_like(cumulative_pv), where=cumulative_volume!=0)

def calculate_bollinger_bands(prices, period=20, num_std=2):
    """Calculate Bollinger Bands"""
    rolling_mean = pd.Series(prices).rolling(window=period).mean()
    rolling_std = pd.Series(prices).rolling(window=period).std()
    upper_band = rolling_mean + (rolling_std * num_std)
    lower_band = rolling_mean - (rolling_std * num_std)
    return upper_band.tolist(), rolling_mean.tolist(), lower_band.tolist()

def calculate_atr(high, low, close, period=14):
    """Calculate Average True Range"""
    high = pd.Series(high)
    low = pd.Series(low)
    close = pd.Series(close)
    
    tr1 = high - low
    tr2 = abs(high - close.shift())
    tr3 = abs(low - close.shift())
    
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    atr = tr.rolling(window=period).mean()
    return atr.tolist()

# Backtesting Engine
class BacktestEngine:
    def __init__(self, initial_capital=100000):
        self.initial_capital = initial_capital
        self.reset()
    
    def reset(self):
        self.cash = self.initial_capital
        self.positions = {}
        self.trades = []
        self.equity_history = []
    
    def calculate_metrics(self, equity_curve):
        try:
            # Convert to pandas Series for easier calculation
            equity_series = pd.Series(equity_curve)
            returns = equity_series.pct_change().dropna()
            
            # Total return
            total_return = ((equity_series.iloc[-1] / equity_series.iloc[0]) - 1) * 100
            
            # Sharpe ratio (assuming risk-free rate = 0)
            if len(returns) > 1:
                sharpe_ratio = np.sqrt(252) * (returns.mean() / returns.std())
            else:
                sharpe_ratio = 0
            
            # Maximum drawdown
            rolling_max = equity_series.expanding().max()
            drawdowns = (equity_series - rolling_max) / rolling_max * 100
            max_drawdown = abs(drawdowns.min())
            
            # Win rate
            profitable_trades = sum(1 for trade in self.trades if trade['pnl'] > 0)
            win_rate = (profitable_trades / len(self.trades) * 100) if self.trades else 0
            
            return {
                'total_return': round(total_return, 2),
                'sharpe_ratio': round(sharpe_ratio, 2),
                'max_drawdown': round(max_drawdown, 2),
                'win_rate': round(win_rate, 2)
            }
        except Exception as e:
            logger.error(f"Error calculating metrics: {str(e)}")
            return {
                'total_return': 0.0,
                'sharpe_ratio': 0.0,
                'max_drawdown': 0.0,
                'win_rate': 0.0
            }
    
    def run_backtest(self, data, strategy_func):
        self.reset()
        
        # Convert data to dictionary if it's a DataFrame
        if isinstance(data, pd.DataFrame):
            data_dict = {
                'close': data['close'].values.tolist(),
                'open': data['open'].values.tolist(),
                'high': data['high'].values.tolist(),
                'low': data['low'].values.tolist(),
                'volume': data['volume'].values.tolist()
            }
        else:
            data_dict = data
        
        # Run backtest
        for i in range(len(data_dict['close'])):
            # Prepare historical data slice
            hist_data = {k: v[:i+1] for k, v in data_dict.items()}
            
            # Get current position
            current_position = self.positions.get('BACKTEST', 0)
            
            # Get trading signals
            signals = strategy_func(hist_data, current_position)
            
            # Execute trades
            current_price = data_dict['close'][i]
            for signal in signals:
                side, quantity = signal
                
                if side == "BUY":
                    cost = quantity * current_price
                    if self.cash >= cost:
                        self.cash -= cost
                        self.positions['BACKTEST'] = self.positions.get('BACKTEST', 0) + quantity
                        self.trades.append({
                            'date': i,
                            'side': 'BUY',
                            'price': current_price,
                            'quantity': quantity,
                            'pnl': 0
                        })
                
                elif side == "SELL":
                    if self.positions.get('BACKTEST', 0) >= quantity:
                        revenue = quantity * current_price
                        self.cash += revenue
                        self.positions['BACKTEST'] -= quantity
                        last_buy = next((t for t in reversed(self.trades) if t['side'] == 'BUY'), None)
                        pnl = (current_price - last_buy['price']) * quantity if last_buy else 0
                        self.trades.append({
                            'date': i,
                            'side': 'SELL',
                            'price': current_price,
                            'quantity': quantity,
                            'pnl': pnl
                        })
            
            # Calculate portfolio value
            portfolio_value = self.cash
            for symbol, qty in self.positions.items():
                portfolio_value += qty * current_price
            
            self.equity_history.append(portfolio_value)
        
        return {
            'trades': self.trades,
            'equity_curve': self.equity_history,
            'metrics': self.calculate_metrics(self.equity_history)
        }
    

# Trading Strategies
def ema_strategy(data, current_position=0):
    """Enhanced EMA Crossover Strategy with ATR for position sizing"""
    prices = data['close']
    if len(prices) < 50:
        return []
    
    # Calculate indicators
    ema_short = calculate_ema(prices, 9)
    ema_long = calculate_ema(prices, 21)
    rsi = calculate_rsi(prices, 14)
    
    # Initialize trades list
    trades = []
    i = len(prices) - 1
    
    # Make sure we have enough data for all indicators
    if all(x is not None for x in [ema_short[i], ema_long[i], rsi[i]]):
        # Calculate position size (simplified for backtest)
        position_size = 100  # Fixed position size for testing
        
        # Buy condition: Short EMA crosses above Long EMA and RSI is not overbought
        if (ema_short[i] > ema_long[i] and 
            ema_short[i-1] <= ema_long[i-1] and 
            rsi[i] < 70 and 
            current_position == 0):
            trades.append(("BUY", position_size))
            
        # Sell condition: Short EMA crosses below Long EMA or RSI is overbought
        elif (current_position > 0 and 
              (ema_short[i] < ema_long[i] or rsi[i] > 70)):
            trades.append(("SELL", current_position))
    
    return trades


def mean_reversion_strategy(data, current_position=0):
    """Mean Reversion Strategy using Bollinger Bands"""
    prices = data['close']
    volumes = data['volume']
    
    if len(prices) < 20:
        return []
    
    upper, middle, lower = calculate_bollinger_bands(prices)
    vwap = calculate_vwap(prices, volumes)
    
    trades = []
    i = len(prices) - 1
    
    if i >= 20:
        # Buy when price is below lower band and VWAP is trending up
        if (prices[i] < lower[i] and 
            vwap[i] > vwap[i-1] and 
            current_position == 0):
            trades.append(("BUY", 1))
            
        # Sell when price is above upper band or reaches middle band
        elif (current_position > 0 and 
              (prices[i] > upper[i] or prices[i] >= middle[i])):
            trades.append(("SELL", current_position))
    
    return trades

def momentum_strategy(data, current_position=0):
    """Enhanced Momentum Strategy using RSI, MACD, and Volume Analysis"""
    prices = data['close']
    volumes = data['volume']
    
    if len(prices) < 50:  # Increased minimum data points for more reliable signals
        return []
    
    # Calculate indicators
    rsi = calculate_rsi(prices)
    macd, signal, hist = calculate_macd(prices)
    volume_ma = pd.Series(volumes).rolling(window=20).mean().tolist()
    
    trades = []
    i = len(prices) - 1
    
    if i >= 49:  # Ensure we have enough data for all indicators
        # Strong momentum buy signal conditions
        buy_conditions = (
            rsi[i] > 40 and rsi[i] < 60 and  # RSI in neutral zone
            hist[i] > 0 and hist[i-1] <= 0 and  # MACD histogram crossover
            macd[i] > signal[i] and  # MACD line above signal line
            volumes[i] > volume_ma[i] * 1.2 and  # Above average volume
            current_position == 0
        )
        
        # Exit signal conditions
        sell_conditions = (
            current_position > 0 and (
                rsi[i] > 70 or  # Overbought
                (hist[i] < 0 and hist[i-1] >= 0) or  # MACD histogram reversal
                (macd[i] < signal[i] and macd[i-1] >= signal[i-1])  # MACD crossover down
            )
        )
        
        if buy_conditions:
            position_size = max(1, int(100000 / prices[i]))  # Dynamic position sizing
            trades.append(("BUY", position_size))
        elif sell_conditions:
            trades.append(("SELL", current_position))
    
    return trades
def vwap_strategy(data, current_position=0):
    """Enhanced VWAP Strategy with Multiple Time Frame Analysis"""
    prices = data['close']
    volumes = data['volume']
    highs = data['high']
    lows = data['low']
    
    if len(prices) < 50:
        return []
    
    # Calculate indicators
    vwap = calculate_vwap(prices, volumes)
    volume_ma = pd.Series(volumes).rolling(window=20).mean().tolist()
    atr = calculate_atr(highs, lows, prices, period=14)
    
    # Calculate multiple timeframe VWAP
    vwap_5 = calculate_vwap(prices[-5:], volumes[-5:])[-1] if len(prices) >= 5 else None
    vwap_15 = calculate_vwap(prices[-15:], volumes[-15:])[-1] if len(prices) >= 15 else None
    
    trades = []
    i = len(prices) - 1
    
    if i >= 49 and vwap_5 and vwap_15 and atr[i]:
        current_price = prices[i]
        
        # Enhanced buy conditions
        buy_conditions = (
            current_price > vwap[i] and  # Price above VWAP
            current_price > vwap_5 and  # Above short-term VWAP
            current_price > vwap_15 and  # Above medium-term VWAP
            volumes[i] > volume_ma[i] * 1.5 and  # Strong volume
            current_position == 0
        )
        
        # Enhanced sell conditions
        sell_conditions = (
            current_position > 0 and (
                current_price < vwap[i] or  # Price below VWAP
                volumes[i] < volume_ma[i] * 0.7 or  # Volume dry up
                (current_price - vwap[i]) > (2 * atr[i])  # Extended too far from VWAP
            )
        )
        
        if buy_conditions:
            # Position size based on ATR
            risk_amount = 0.02 * 100000  # 2% risk
            position_size = max(1, int(risk_amount / (atr[i] * 2)))
            trades.append(("BUY", position_size))
        elif sell_conditions:
            trades.append(("SELL", current_position))
    
    return trades

# Trading Execution
def execute_trade(side, quantity, symbol):
    """Execute trade and log performance"""
    market_order_data = MarketOrderRequest(
        symbol=symbol,
        qty=quantity,
        side=OrderSide.BUY if side == "BUY" else OrderSide.SELL,
        time_in_force=TimeInForce.DAY
    )
    
    try:
        logger.info(f"Placing order: {side} {quantity} shares of {symbol}")
        market_order = trading_client.submit_order(order_data=market_order_data)
        logger.info(f"Order placed: {market_order.id}")
        return market_order
    except Exception as e:
        logger.error(f"Error placing order: {e}")
        return None

def run_bot(bot_type, symbol):
    """Main bot loop with performance tracking"""
    logger.info(f"Starting {bot_type} bot for {symbol}")
    strategy_map = {
        'ema': ema_strategy,
        'macd': momentum_strategy,
        'rsi': mean_reversion_strategy,
        'mean_reversion': mean_reversion_strategy,
        'momentum': momentum_strategy,
        'vwap': vwap_strategy
    }
    
    while active_bots[bot_type].get(symbol, False):
        try:
            positions = trading_client.get_all_positions()
            current_position = next((float(p.qty) for p in positions if p.symbol == symbol), 0)
            
            end = datetime.now()
            start = end - timedelta(hours=6)
            data = get_historical_data(symbol, TimeFrame.Minute, start, end)
            
            if not data:
                logger.info(f"Waiting for data for {symbol}")
                time.sleep(60)
                continue
            
            strategy = strategy_map[bot_type]
            trades = strategy(data, current_position)
            
            if trades:
                last_trade = trades[-1]
                order = execute_trade(last_trade[0], last_trade[1], symbol)
                
                if order:
                    bot_performance[bot_type]['trades'].append({
                        'timestamp': datetime.now().isoformat(),
                        'symbol': symbol,
                        'side': last_trade[0],
                        'quantity': last_trade[1],
                        'price': float(order.filled_avg_price) if order.filled_avg_price else None
                    })
            
            time.sleep(60)
            
        except Exception as e:
            logger.error(f"Error in {bot_type} bot for {symbol}: {e}")
            time.sleep(60)

# Store active bots and their performance
active_bots = {
    'ema': {},
    'macd': {},
    'rsi': {},
    'mean_reversion': {},
    'momentum': {},
    'vwap': {}
}

bot_performance = {
    'ema': {'trades': [], 'pnl': []},
    'macd': {'trades': [], 'pnl': []},
    'rsi': {'trades': [], 'pnl': []},
    'mean_reversion': {'trades': [], 'pnl': []},
    'momentum': {'trades': [], 'pnl': []},
    'vwap': {'trades': [], 'pnl': []}
}

# Data Fetching Function
def get_historical_data(symbol, timeframe, start, end):
    """Fetch historical price data from Alpaca"""
    try:
        request_params = StockBarsRequest(
            symbol_or_symbols=[symbol],
            timeframe=timeframe,
            start=start,
            end=end
        )
        bars = stock_data_client.get_stock_bars(request_params)
        df = bars.df
        
        if not df.empty:
            logger.info(f"Fetched {len(df)} data points for {symbol}")
            return {
                'close': df['close'].tolist(),
                'open': df['open'].tolist(),
                'high': df['high'].tolist(),
                'low': df['low'].tolist(),
                'volume': df['volume'].tolist()
            }
        return None
    except Exception as e:
        logger.error(f"Error fetching data for {symbol}: {e}")
        return None

# Flask Routes
@app.route('/')
def index():
    return render_template('index.html')


@app.route('/account_info')
def account_info():
    account = trading_client.get_account()
    return jsonify({
        'buying_power': account.buying_power,
        'cash': account.cash,
        'portfolio_value': account.portfolio_value
    })


@app.route('/orders', methods=['GET', 'POST'])
def handle_orders():
    if request.method == 'GET':
        orders = trading_client.get_orders()
        return jsonify([order.dict() for order in orders])
    
    elif request.method == 'POST':
        try:
            data = request.json
            market_order_data = MarketOrderRequest(
                symbol=data['symbol'],
                qty=data['qty'],
                side=OrderSide.BUY if data['side'] == 'BUY' else OrderSide.SELL,
                time_in_force=TimeInForce.DAY
            )
            
            logger.info(f"Submitting order: {data}")
            order = trading_client.submit_order(order_data=market_order_data)
            logger.info(f"Order submitted successfully: {order.dict()}")
            
            return jsonify(order.dict())
        except Exception as e:
            logger.error(f"Order submission failed: {str(e)}")
            return jsonify({'message': str(e)}), 400


@app.route('/orders/<order_id>', methods=['DELETE'])
def cancel_order(order_id):
    try:
        trading_client.cancel_order(order_id)
        return jsonify({'message': 'Order cancelled successfully'})
    except Exception as e:
        return jsonify({'message': str(e)}), 400

def execute_trade(side, quantity, symbol):
    """Execute trade and log performance"""
    market_order_data = MarketOrderRequest(
        symbol=symbol,
        qty=quantity,
        side=OrderSide.BUY if side == "BUY" else OrderSide.SELL,
        time_in_force=TimeInForce.DAY
    )
    
    try:
        logger.info(f"Placing order: {side} {quantity} shares of {symbol}")
        market_order = trading_client.submit_order(order_data=market_order_data)
        logger.info(f"Order placed: {market_order.id}")
        return market_order
    except Exception as e:
        logger.error(f"Error placing order: {e}")
        return None



@app.route('/bot_performance')
def get_bot_performance():
    return jsonify(bot_performance)

@app.route('/market_data')
def get_market_data():
    stock_symbols = ["SPY", "AAPL", "TSLA", "MSFT", "AMZN"]
    
    try:
        stock_request = StockLatestQuoteRequest(symbol_or_symbols=stock_symbols)
        stock_quotes = stock_data_client.get_stock_latest_quote(stock_request)
        
        response_data = {
            'stocks': {},
            'crypto': {}  # Empty for now
        }
        
        for symbol, quote in stock_quotes.items():
            response_data['stocks'][symbol] = {
                'ask_price': float(quote.ask_price) if quote.ask_price else 0,
                'bid_price': float(quote.bid_price) if quote.bid_price else 0,
                'price_change': 0
            }
            
        return jsonify(response_data)
        
    except Exception as e:
        logger.error(f"Error fetching market data: {str(e)}")
        return jsonify({
            'stocks': {},
            'crypto': {},
            'error': str(e)
        }), 200

@app.route('/backtest', methods=['POST'])
def run_backtest():
    try:
        data = request.json
        strategy_name = data.get('strategy')
        symbol = data.get('symbol')
        start_date = datetime.strptime(data.get('start_date'), '%Y-%m-%d')
        end_date = datetime.strptime(data.get('end_date'), '%Y-%m-%d')
        
        logger.info(f"Running backtest for {symbol} from {start_date} to {end_date}")
        
        historical_data = get_historical_data(symbol, TimeFrame.Day, start_date, end_date)
        
        if historical_data is None:
            return jsonify({'error': 'Failed to fetch historical data'}), 400
        
        engine = BacktestEngine(initial_capital=100000)
        
        strategy_map = {
            'ema': ema_strategy,
            'macd': momentum_strategy,
            'rsi': mean_reversion_strategy,
            'mean_reversion': mean_reversion_strategy,
            'momentum': momentum_strategy,
            'vwap': vwap_strategy
        }
        
        strategy_func = strategy_map.get(strategy_name)
        if not strategy_func:
            return jsonify({'error': 'Invalid strategy name'}), 400
        
        result = engine.run_backtest(historical_data, strategy_func)
        
        response = {
            'equity_curve': [
                {
                    'date': str(start_date + timedelta(days=i)),
                    'equity': float(value)
                }
                for i, value in enumerate(result['equity_curve'])
            ],
            'metrics': {
                'total_return': result['metrics'].get('total_return', 0.0),
                'sharpe_ratio': result['metrics'].get('sharpe_ratio', 0.0),
                'max_drawdown': result['metrics'].get('max_drawdown', 0.0),
                'win_rate': result['metrics'].get('win_rate', 0.0),
                'total_trades': result['metrics'].get('total_trades', 0),
                'avg_trade_pnl': result['metrics'].get('avg_trade_pnl', 0.0)
            },
            'trades': [
                {
                    'date': str(trade.get('date', '')),
                    'side': trade.get('side', ''),
                    'price': float(trade.get('price', 0.0)),
                    'quantity': int(trade.get('quantity', 0)),
                    'pnl': float(trade.get('pnl', 0.0))
                }
                for trade in result['trades']
            ]
        }
        
        return jsonify(response)
        
    except Exception as e:
        logger.error(f"Backtest error: {str(e)}")
        return jsonify({
            'error': str(e),
            'metrics': {
                'total_return': 0.0,
                'sharpe_ratio': 0.0,
                'max_drawdown': 0.0,
                'win_rate': 0.0,
                'total_trades': 0,
                'avg_trade_pnl': 0.0
            },
            'equity_curve': [],
            'trades': []
        }), 500
        
@app.route('/positions')
def get_positions():
    positions = trading_client.get_all_positions()
    return jsonify([position.dict() for position in positions])

@app.route('/toggle_bot', methods=['POST'])
def toggle_bot():
    data = request.json
    bot_type = data['bot_type']
    symbol = data['symbol']
    enabled = data['enabled']
    
    if enabled:
        if not active_bots[bot_type].get(symbol, False):
            active_bots[bot_type][symbol] = True
            thread = threading.Thread(
                target=run_bot,
                args=(bot_type, symbol),
                daemon=True
            )
            thread.start()
    else:
        active_bots[bot_type][symbol] = False
    
    return jsonify({'status': 'success'})

@app.route('/bot_status')
def get_bot_status():
    return jsonify({
        'active_bots': active_bots,
        'performance': bot_performance
    })

if __name__ == '__main__':
    app.run(debug=True)