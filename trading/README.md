# Automated Revision-Based Trading System

## UK-Available Platforms for Automated Trading (2026)

### 1. **Alpaca Trading** ✅ TOP RATED FOR ALGO TRADING
- **Best for**: Algorithmic trading, commission-free US stocks/ETFs
- **API**: REST API, WebSocket streaming, Python SDK
- **Features**:
  - Zero commission on stocks and ETFs
  - Paper trading sandbox for testing
  - Real-time and historical data included free
  - Advanced order types (VWAP, TWAP)
  - Low latency execution
- **Install**: `pip install alpaca-trade-api`
- **UK Available**: Yes
- **Best for our strategy**: US ETFs (SPY, QQQ) for economic release reactions

### 2. **Interactive Brokers (IBKR)** ✅ RECOMMENDED FOR SERIOUS TRADERS
- **Best for**: Comprehensive market access, professional features
- **API**: TWS API, Client Portal Web API, FIX API
- **Features**:
  - Trade: Stocks, ETFs, Futures, Options, Forex globally
  - Very low commissions
  - Paper trading available (port 7497)
  - Most comprehensive instrument coverage
- **Install**: `pip install ib_insync`
- **UK Available**: Yes (Interactive Brokers UK Limited, FCA regulated)
- **Our bot uses this**: See `broker_ibkr.py`

### 3. **IG Markets** - FCA Regulated
- **Best for**: CFDs, Spread Betting (tax-free gains in UK!)
- **API**: REST API, FIX API for DMA
- **Features**:
  - UK-based, well-established
  - Trade: Indices, Forex, Commodities, Shares CFDs
  - Demo account available
  - FIX API for institutional-grade access
- **Tax advantage**: Spread betting profits are tax-free in UK

### 4. **Saxo Bank** - Professional Grade
- **Best for**: Multi-asset trading
- **API**: OpenAPI for Excel, REST APIs, third-party platform support
- **Features**:
  - No minimum deposit (UK Classic account)
  - Wide range: Forex, stocks, ETFs, bonds, options, futures
  - Good for large accounts

### 5. **OANDA** - Good for Beginners
- **Best for**: Forex, user-friendly API
- **API**: REST API with good documentation
- **Features**:
  - No minimum account size
  - Intuitive platform
  - Good for automated forex strategies

### ⚠️ Platforms WITHOUT API (Avoid for Automation)
- **Trading 212**: No official API
- **DEGIRO**: No official API
- **Plus500**: No API
- **Webull**: No official API

---

## What We Trade

Based on our revision model, we trade the REACTION to economic releases:

| Economic Release | Revision Bias | What to Trade | Direction |
|------------------|---------------|---------------|-----------|
| GDP Miss | 99% revises UP | SPY/SPX, QQQ | BUY the dip |
| GDP Beat | 99% revises UP | Already priced | HOLD |
| NFP Miss | 62% revises UP | SPY, Financials | BUY the dip |
| NFP Beat | May revise down | Careful | SMALL position |
| Retail Sales Beat | 98% revises DOWN | Consumer stocks | FADE |
| Industrial Prod Beat | 92% revises DOWN | Industrial ETFs | FADE |

---

## Strategy Logic

1. **Monitor** economic calendar for releases
2. **Capture** the initial release value
3. **Compare** to consensus/threshold
4. **Check** our revision probability
5. **If edge exists** → Execute trade
6. **Hold** until revision or sentiment shifts

---

## Quick Start Guide

### Option 1: Alpaca (Easiest for UK)

Alpaca is available to UK residents (190+ countries supported). No minimum deposit.

```bash
# 1. Install dependencies
pip install alpaca-py

# 2. Create account and get API keys:
#    a) Sign up at https://alpaca.markets/ (verify identity with passport/licence)
#    b) Go to Paper Trading dashboard: https://app.alpaca.markets/paper/dashboard/overview
#    c) Click "View" under "API Keys" on the right side
#    d) Copy your APCA-API-KEY-ID and APCA-API-SECRET-KEY

# 3. Set environment variables (use YOUR actual keys!)
export ALPACA_API_KEY='PKXXXXXXXXXXXXXXXXXX'    # Your Key ID (starts with PK for paper)
export ALPACA_SECRET_KEY='xxxxxxxxxxxxxxxx'      # Your Secret Key

# 4. Test authentication works (should return your account info, NOT 403)
curl -H "APCA-API-KEY-ID: $ALPACA_API_KEY" \
     -H "APCA-API-SECRET-KEY: $ALPACA_SECRET_KEY" \
     https://paper-api.alpaca.markets/v2/account

# 5. Run in paper mode
python trading/bot.py --paper --broker alpaca

# 6. Simulate a release to test
python trading/bot.py --broker alpaca --simulate PAYEMS,150000,200000,180000
```

**Common error**: 403 Forbidden = API keys not set or incorrect. Check your environment variables.

### Option 2: Interactive Brokers

```bash
# 1. Install dependencies
pip install ib_insync

# 2. Download and install TWS or IB Gateway from IBKR website

# 3. Enable API in TWS:
#    - Edit > Global Configuration > API > Settings
#    - Enable "ActiveX and Socket Clients"
#    - Socket port: 7497 (paper) or 7496 (live)

# 4. Run TWS and log in

# 5. Run the bot
python trading/bot.py --paper --broker ibkr
```

---

## Alert Setup (Optional but Recommended)

### Telegram Alerts
```bash
# 1. Create bot via @BotFather on Telegram
# 2. Get your chat ID via @userinfobot
# 3. Set environment variables:
export TELEGRAM_BOT_TOKEN='your-bot-token'
export TELEGRAM_CHAT_ID='your-chat-id'
```

### Discord Alerts
```bash
# 1. Create webhook in Discord channel settings
# 2. Set environment variable:
export DISCORD_WEBHOOK_URL='your-webhook-url'
```

---

## Running the Bot

```bash
# Demo mode (no real trades, for testing)
python trading/bot.py --demo

# Paper trading (simulated money, real market data)
python trading/bot.py --paper --broker alpaca
python trading/bot.py --paper --broker ibkr

# Live trading (REAL MONEY - be careful!)
python trading/bot.py --live --broker alpaca
python trading/bot.py --live --broker ibkr

# Simulate a specific release
python trading/bot.py --simulate GDPC1,2.1,2.5,2.3
python trading/bot.py --simulate PAYEMS,150000,200000,180000
python trading/bot.py --simulate RSXFS,1.2,0.8,0.9
```

---

## Files in this Directory

| File | Purpose |
|------|---------|
| `bot.py` | Main trading bot entry point |
| `broker_alpaca.py` | Alpaca broker implementation |
| `broker_ibkr.py` | Interactive Brokers implementation |
| `strategy.py` | Trading strategy based on revision biases |
| `economic_calendar.py` | Economic release calendar |
| `release_monitor.py` | Real-time release monitoring |
| `alerts.py` | Alert system (Telegram, Discord, Email) |

---

## Risk Management

The bot has built-in safeguards:
- **Maximum position size**: 7% of portfolio per trade
- **Maximum total risk**: 15% of portfolio
- **Default stop loss**: 2%
- **Default take profit**: 4%
- **Paper trading recommended** for at least 1 month before going live

---

## Sources

- [Best Brokers for Algorithmic Trading in UK 2026](https://brokerchooser.com/best-brokers/best-brokers-for-algo-trading-in-the-united-kingdom)
- [Alpaca Markets](https://alpaca.markets/)
- [Interactive Brokers UK](https://www.interactivebrokers.co.uk/)
- [Best API Brokers 2026](https://investingintheweb.com/brokers/best-api-brokers/)
