#!/bin/bash
# ============================================================
# AUTOTRADER SETUP SCRIPT
# ============================================================
# Run this once to configure everything:
#   chmod +x setup.sh && ./setup.sh
# ============================================================

set -e

echo "============================================================"
echo "ECONOMIC REVISION AUTOTRADER - SETUP"
echo "============================================================"
echo ""

PROJECT_DIR="$(cd "$(dirname "$0")" && pwd)"
ENV_FILE="$PROJECT_DIR/.env"

# 1. Check Python
echo "1. Checking Python..."
if command -v python3 &>/dev/null; then
    PYTHON=$(command -v python3)
    echo "   Found: $PYTHON ($(python3 --version))"
else
    echo "   ERROR: Python 3 not found. Install it first."
    exit 1
fi

# 2. Install dependencies
echo ""
echo "2. Installing dependencies..."
pip3 install -r "$PROJECT_DIR/requirements.txt" --quiet
pip3 install alpaca-py python-dotenv --quiet
echo "   Done."

# 3. Collect API keys
echo ""
echo "3. Setting up API keys..."

if [ -f "$ENV_FILE" ]; then
    echo "   Found existing .env file"
    source "$ENV_FILE" 2>/dev/null || true
fi

# FRED API Key
if [ -z "$FRED_API_KEY" ]; then
    echo ""
    echo "   FRED API Key (get from https://fred.stlouisfed.org/docs/api/api_key.html):"
    read -r FRED_KEY
    echo "FRED_API_KEY=$FRED_KEY" >> "$ENV_FILE"
else
    echo "   FRED API Key: set"
fi

# Alpaca API Key
if [ -z "$ALPACA_API_KEY" ]; then
    echo ""
    echo "   Alpaca API Key (get from https://app.alpaca.markets):"
    read -r ALP_KEY
    echo "ALPACA_API_KEY=$ALP_KEY" >> "$ENV_FILE"
else
    echo "   Alpaca API Key: set"
fi

# Alpaca Secret Key
if [ -z "$ALPACA_SECRET_KEY" ]; then
    echo ""
    echo "   Alpaca Secret Key:"
    read -r ALP_SECRET
    echo "ALPACA_SECRET_KEY=$ALP_SECRET" >> "$ENV_FILE"
else
    echo "   Alpaca Secret Key: set"
fi

# Telegram (optional)
echo ""
echo "   Telegram alerts (optional, press Enter to skip):"
echo "   Bot Token:"
read -r TG_TOKEN
if [ -n "$TG_TOKEN" ]; then
    echo "TELEGRAM_BOT_TOKEN=$TG_TOKEN" >> "$ENV_FILE"
    echo "   Chat ID:"
    read -r TG_CHAT
    echo "TELEGRAM_CHAT_ID=$TG_CHAT" >> "$ENV_FILE"
fi

echo ""
echo "   .env file saved to: $ENV_FILE"

# 4. Test connections
echo ""
echo "4. Testing connections..."

python3 -c "
import os
from dotenv import load_dotenv
load_dotenv('$ENV_FILE')

# Test FRED
import requests
key = os.getenv('FRED_API_KEY', '')
if key:
    r = requests.get(f'https://api.stlouisfed.org/fred/series?series_id=GDPC1&api_key={key}&file_type=json', timeout=10)
    if r.status_code == 200:
        print('   FRED API: OK')
    else:
        print(f'   FRED API: FAILED ({r.status_code})')
else:
    print('   FRED API: NO KEY')

# Test Alpaca
ak = os.getenv('ALPACA_API_KEY', '')
sk = os.getenv('ALPACA_SECRET_KEY', '')
if ak and sk:
    r = requests.get('https://paper-api.alpaca.markets/v2/account',
        headers={'APCA-API-KEY-ID': ak, 'APCA-API-SECRET-KEY': sk}, timeout=10)
    if r.status_code == 200:
        data = r.json()
        print(f'   Alpaca API: OK (Balance: \${float(data[\"portfolio_value\"]):,.2f})')
    else:
        print(f'   Alpaca API: FAILED ({r.status_code})')
else:
    print('   Alpaca API: NO KEYS')
"

# 5. Create run scripts
echo ""
echo "5. Creating run scripts..."

# Paper trading script
cat > "$PROJECT_DIR/run_paper.sh" << 'RUNEOF'
#!/bin/bash
cd "$(dirname "$0")"
source .env 2>/dev/null
export FRED_API_KEY ALPACA_API_KEY ALPACA_SECRET_KEY TELEGRAM_BOT_TOKEN TELEGRAM_CHAT_ID
python3 trading/autotrader.py --paper
RUNEOF
chmod +x "$PROJECT_DIR/run_paper.sh"

# Live trading script
cat > "$PROJECT_DIR/run_live.sh" << 'RUNEOF'
#!/bin/bash
cd "$(dirname "$0")"
source .env 2>/dev/null
export FRED_API_KEY ALPACA_API_KEY ALPACA_SECRET_KEY TELEGRAM_BOT_TOKEN TELEGRAM_CHAT_ID
python3 trading/autotrader.py --live
RUNEOF
chmod +x "$PROJECT_DIR/run_live.sh"

# Test script
cat > "$PROJECT_DIR/run_test.sh" << 'RUNEOF'
#!/bin/bash
cd "$(dirname "$0")"
source .env 2>/dev/null
export FRED_API_KEY ALPACA_API_KEY ALPACA_SECRET_KEY
python3 trading/autotrader.py --test
RUNEOF
chmod +x "$PROJECT_DIR/run_test.sh"

echo "   Created: run_paper.sh, run_live.sh, run_test.sh"

# 6. Create launchd plist for auto-start on Mac
echo ""
echo "6. Setting up auto-start on login (macOS)..."

PLIST_DIR="$HOME/Library/LaunchAgents"
PLIST_FILE="$PLIST_DIR/com.econ-revision.autotrader.plist"
mkdir -p "$PLIST_DIR"

cat > "$PLIST_FILE" << PLISTEOF
<?xml version="1.0" encoding="UTF-8"?>
<!DOCTYPE plist PUBLIC "-//Apple//DTD PLIST 1.0//EN" "http://www.apple.com/DTDs/PropertyList-1.0.dtd">
<plist version="1.0">
<dict>
    <key>Label</key>
    <string>com.econ-revision.autotrader</string>
    <key>ProgramArguments</key>
    <array>
        <string>$PROJECT_DIR/run_paper.sh</string>
    </array>
    <key>RunAtLoad</key>
    <true/>
    <key>KeepAlive</key>
    <true/>
    <key>StandardOutPath</key>
    <string>$PROJECT_DIR/autotrader_stdout.log</string>
    <key>StandardErrorPath</key>
    <string>$PROJECT_DIR/autotrader_stderr.log</string>
    <key>EnvironmentVariables</key>
    <dict>
        <key>PATH</key>
        <string>/usr/local/bin:/usr/bin:/bin:/Library/Frameworks/Python.framework/Versions/3.14/bin</string>
    </dict>
</dict>
</plist>
PLISTEOF

echo "   Created: $PLIST_FILE"
echo ""
echo "   To enable auto-start on login:"
echo "     launchctl load $PLIST_FILE"
echo ""
echo "   To disable auto-start:"
echo "     launchctl unload $PLIST_FILE"
echo ""
echo "   To check status:"
echo "     launchctl list | grep autotrader"

echo ""
echo "============================================================"
echo "SETUP COMPLETE!"
echo "============================================================"
echo ""
echo "QUICK START:"
echo "  1. Test:   ./run_test.sh"
echo "  2. Paper:  ./run_paper.sh"
echo "  3. Live:   ./run_live.sh"
echo ""
echo "AUTO-START ON LOGIN:"
echo "  launchctl load ~/Library/LaunchAgents/com.econ-revision.autotrader.plist"
echo ""
echo "SET CONSENSUS BEFORE RELEASE DAY:"
echo "  python3 -c \"from trading.live_calendar import LiveCalendar; c=LiveCalendar(); c.set_consensus('PAYEMS', '2026-02-07', 200000, 180000)\""
echo ""
echo "============================================================"
