import gradio as gr
import yfinance as yf
import pandas as pd

# --- 核心分析邏輯 ---
def get_stock_data(ticker):
    try:
        stock = yf.Ticker(ticker)
        df = stock.history(period="2y") 
        return df
    except:
        return pd.DataFrame()

def analyze_single_stock(ticker):
    ticker = ticker.strip().upper()
    df = get_stock_data(ticker)
    spy_df = get_stock_data("SPY")

    if df.empty or len(df) < 200:
        return {"代號": ticker, "狀態": "❌ 資料不足或代號錯誤"}

    current_price = df['Close'].iloc[-1]
    
    # 1. 均線計算
    df['MA20'] = df['Close'].rolling(window=20).mean()
    df['MA50'] = df['Close'].rolling(window=50).mean()
    df['MA200'] = df['Close'].rolling(window=200).mean()
    
    ma20 = df['MA20'].iloc[-1]
    ma50 = df['MA50'].iloc[-1]
    ma200 = df['MA200'].iloc[-1]
    ma200_prev = df['MA200'].iloc[-20]
    
    # 2. 趨勢分數 (4分滿分)
    trend_score = 0
    if current_price > ma20: trend_score += 1
    if ma20 > ma50: trend_score += 1
    if ma50 > ma200: trend_score += 1
    if ma200 > ma200_prev: trend_score += 1 
    
    # 3. RS 強度 (過去半年 vs SPY)
    if len(df) > 126 and len(spy_df) > 126:
        stock_perf = (df['Close'].iloc[-1] - df['Close'].iloc[-126]) / df['Close'].iloc[-126]
        spy_perf = (spy_df['Close'].iloc[-1] - spy_df['Close'].iloc[-126]) / spy_df['Close'].iloc[-126]
        rs_text = "🔥 強於大盤" if stock_perf > spy_perf else "🧊 弱於大盤"
    else:
        rs_text = "N/A"

    # 4. R/R 計算
    support = ma50
    resistance = df['High'].tail(252).max()
    
    if current_price >= resistance * 0.98:
        resistance = current_price * 1.2
    
    risk = current_price - support
    reward = resistance - current_price
    rr = reward / risk if risk > 0 else 0
    
    # 5. 綜合評價
    if trend_score == 4 and rr >= 2:
        verdict = "💎 強力買點"
    elif trend_score == 4 and risk < 0:
         verdict = "⚠️ 跌破MA50"
    elif rr > 3 and trend_score < 2:
        verdict = "🗡️ 逆勢接刀"
    else:
        verdict = "👀 觀察中"

    return {
        "代號": ticker,
        "現價": round(current_price, 2),
        "趨勢分數": f"{trend_score}/4",
        "RS強度": rs_text,
        "R/R值": round(rr, 2),
        "停損(MA50)": round(support, 2),
        "停利(前高)": round(resistance, 2),
        "AI短評": verdict
    }

def app_main(tickers_input):
    if not tickers_input:
        return pd.DataFrame()
    tickers = tickers_input.replace(" ", ",").split(",")
    results = []
    for t in tickers:
        if t.strip():
            results.append(analyze_single_stock(t))
    return pd.DataFrame(results)

# --- 啟動介面 ---
iface = gr.Interface(
    fn=app_main,
    inputs=gr.Textbox(label="輸入股票代號 (例如: NVDA, VRT)", placeholder="輸入代號，用逗號分隔..."),
    outputs=gr.Dataframe(label="分析結果"),
    title="📈 AI 順勢交易 R/R 計算機",
    description="輸入美股代號，AI 將自動計算：趨勢分數、RS強度、R/R風險報酬比",
    allow_flagging="never"
)

if __name__ == "__main__":
    iface.launch()