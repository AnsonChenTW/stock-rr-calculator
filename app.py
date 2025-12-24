import gradio as gr
import yfinance as yf
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import io
import base64
import re
from datetime import datetime, timedelta
import pytz
import twstock
import os
import google.generativeai as genai

# --- 設定 Gemini API ---
# 從 Hugging Face Secrets 讀取 Key
GEMINI_KEY = os.getenv("GEMINI_API_KEY")

if GEMINI_KEY:
    genai.configure(api_key=GEMINI_KEY)
else:
    print("⚠️ 警告: 未偵測到 GEMINI_API_KEY，AI 分析功能將失效。")

# --- AI 分析核心邏輯 ---
def ask_gemini_analysis(stock_list_str):
    if not GEMINI_KEY:
        return "<div style='color:#c0392b;'>⚠️ 請先至 Settings 設定 GEMINI_API_KEY 才能啟用 AI 總經分析功能。</div>"
    
    today_str = datetime.now().strftime("%Y-%m-%d")
    
    # 設計 Prompt (提示詞)
    prompt = f"""
    你是一位專業的華爾街交易員與總體經濟分析師。
    今天是 {today_str}。
    使用者正在關注這幾檔股票：{stock_list_str}。

    請針對「未來 7 天」的市場狀況，生成一份簡短的「交易員早報」。
    
    請包含以下兩部分 (請用 HTML 格式輸出，使用 <ul> <li> <b> 等標籤)：
    
    1. **🚨 未來7日市場關鍵事件 (Macro & Tech Events)**：
       - 請列出即將到來的 FED 決策、CPI/PCE 數據、非農就業數據。
       - 如果近期有 CES, Computex, NVIDIA GTC 等重大科技展覽，務必列出。
       - 若無重大事件，請簡述目前市場情緒 (如：AI 泡沫擔憂、降息預期等)。
       
    2. **⚡ 個股焦點與風險提示**：
       - 針對使用者輸入的股票 ({stock_list_str})，是否有即將到來的財報 (Earnings)、拆股、產品發表會？
       - 根據這些股票的屬性 (如 AI, 半導體, 航運)，給出一個一句話的操作心法。

    **要求**：
    - 繁體中文回答。
    - 語氣專業、精簡、直接點出重點 (Bullet points)。
    - 重大日期與事件請用 <b>粗體紅色</b> 標示。
    - 總字數控制在 300 字以內，不要廢話。
    """
    
    try:
        # 使用 Gemini 1.5 Flash (速度快且免費額度高)
        model = genai.GenerativeModel('gemini-1.5-flash')
        response = model.generate_content(prompt)
        return response.text
    except Exception as e:
        return f"<div style='color:red;'>AI 分析連線失敗: {str(e)}</div>"

# --- 原有邏輯：獲取台股中文名稱 ---
def get_tw_stock_name(ticker_code):
    try:
        code_only = ticker_code.split('.')[0]
        if code_only in twstock.codes:
            return twstock.codes[code_only].name
    except:
        pass
    return None

# --- 原有邏輯：獲取數據 ---
def get_stock_data(ticker):
    try:
        stock = yf.Ticker(ticker)
        df_daily = stock.history(period="2y")
        df_intraday = stock.history(period="5d", interval="1m")
        return df_daily, df_intraday, stock
    except:
        return pd.DataFrame(), pd.DataFrame(), None

# --- 繪圖邏輯 ---
def create_chart_image(df, ticker):
    if len(df) < 50: return None
    plot_df = df.tail(252).copy()
    fig, ax = plt.subplots(figsize=(6, 3.5), dpi=90)
    fig.patch.set_facecolor('white') 
    ax.set_facecolor('white')
    ax.plot(plot_df.index, plot_df['Close'], label='Price', color='#333333', linewidth=1.5)
    ax.plot(plot_df.index, plot_df['MA20'], label='MA20', color='#f39c12', linewidth=1, alpha=0.8)
    ax.plot(plot_df.index, plot_df['MA50'], label='MA50', color='#27ae60', linewidth=1, alpha=0.8)
    ax.plot(plot_df.index, plot_df['MA200'], label='MA200', color='#2980b9', linewidth=1.5, alpha=0.8)
    ax.set_title(f"{ticker} Trend Chart", fontsize=11, color='black', fontweight='bold', pad=10)
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%b')) 
    ax.xaxis.set_major_locator(mdates.MonthLocator(interval=2)) 
    plt.xticks(fontsize=9, color='#666666')
    ax.yaxis.set_visible(False)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['left'].set_visible(False)
    ax.spines['bottom'].set_color('#cccccc')
    legend = ax.legend(loc='upper left', fontsize='small', frameon=False, ncol=4)
    plt.setp(legend.get_texts(), color='black')
    plt.tight_layout()
    buf = io.BytesIO()
    plt.savefig(buf, format='png', transparent=False, facecolor='white')
    plt.close()
    buf.seek(0)
    image_base64 = base64.b64encode(buf.read()).decode('utf-8')
    return f'<img src="data:image/png;base64,{image_base64}" style="width:100%; border-radius:8px; margin-top:5px;">'

# --- 核心邏輯：分析單一股票 ---
def analyze_single_stock(ticker):
    ticker = ticker.strip().upper()
    df, df_intraday, stock_obj = get_stock_data(ticker)
    
    if df.empty or len(df) < 200:
        return None 

    is_tw_stock = ".TW" in ticker or ".TWO" in ticker
    display_name = ticker
    if is_tw_stock:
        cn_name = get_tw_stock_name(ticker)
        if cn_name:
            code_only = ticker.split('.')[0]
            display_name = f"{code_only} {cn_name}"
    
    # 這裡只抓財報跟除息作為備用，主要依賴 Gemini 分析
    upcoming_events = [] # 簡化邏輯，交給 Gemini 統一處理總經與重大財報

    tw_tz = pytz.timezone('Asia/Taipei')

    if not df_intraday.empty:
        current_price = df_intraday['Close'].iloc[-1]
        last_timestamp = df_intraday.index[-1]
        if last_timestamp.tzinfo is None:
            last_timestamp = pytz.utc.localize(last_timestamp)
        last_timestamp_tw = last_timestamp.tz_convert(tw_tz)
        last_date_str = last_timestamp_tw.strftime('%Y-%m-%d %H:%M:%S')
    else:
        current_price = df['Close'].iloc[-1]
        last_date_str = df.index[-1].strftime('%Y-%m-%d') + " (日線)"

    if len(df) >= 2:
        prev_close = df['Close'].iloc[-2]
        change_pct = ((current_price - prev_close) / prev_close) * 100
        change_color = "#27ae60" if change_pct > 0 else "#c0392b"
        sign = "+" if change_pct > 0 else ""
        change_text = f"<span style='color:{change_color}; font-weight:bold;'>{sign}{change_pct:.2f}%</span>"
    else:
        change_text = "0.00%"

    df['MA20'] = df['Close'].rolling(window=20).mean()
    df['MA50'] = df['Close'].rolling(window=50).mean()
    df['MA200'] = df['Close'].rolling(window=200).mean()
    
    ma20 = df['MA20'].iloc[-1]
    ma50 = df['MA50'].iloc[-1]
    ma200 = df['MA200'].iloc[-1]
    ma50_prev = df['MA50'].iloc[-20]
    ma200_prev = df['MA200'].iloc[-20]
    
    benchmark_symbol = "^TWII" if is_tw_stock else "SPY"
    rs_score = 0
    try:
        bm = yf.Ticker(benchmark_symbol)
        bm_df = bm.history(period="2y")
        if len(df) > 126 and len(bm_df) > 126:
            stock_perf = (df['Close'].iloc[-1] - df['Close'].iloc[-126]) / df['Close'].iloc[-126]
            bm_perf = (bm_df['Close'].iloc[-1] - bm_df['Close'].iloc[-126]) / bm_df['Close'].iloc[-126]
            if stock_perf > bm_perf: rs_score = 1
    except:
        pass

    score = 0
    if current_price > ma20: score += 1
    if current_price > ma50: score += 1
    if current_price > ma200: score += 2 
    if ma20 > ma50: score += 1
    if ma50 > ma200: score += 2 
    if ma50 > ma50_prev: score += 1 
    if ma200 > ma200_prev: score += 1 
    if rs_score == 1: score += 1
    
    if score >= 9: score_comment = "🔥 強勢多頭 (Strong)"
    elif score >= 7: score_comment = "📈 多頭排列 (Uptrend)"
    elif score >= 5: score_comment = "👀 盤整/中性 (Neutral)"
    elif score >= 3: score_comment = "📉 轉弱/空頭 (Weak)"
    else: score_comment = "❌ 強勢空頭 (Downtrend)"

    support = ma50
    resistance = df['High'].tail(252).max()
    if current_price >= resistance * 0.98: resistance = current_price * 1.2
    
    risk = current_price - support
    reward = resistance - current_price
    
    if risk > 0.1:
        rr_val = reward / risk
        rr_display = f"1 : {rr_val:.1f}"
    else:
        rr_val = 0
        rr_display = "⚠️ 風險高"

    chart_html = create_chart_image(df, ticker)

    return {
        "ticker": display_name,
        "raw_ticker": ticker,
        "price": current_price,
        "date_str": last_date_str,
        "change_text": change_text,
        "trend_score": score,
        "score_comment": score_comment,
        "rr_val": rr_val,
        "rr_display": rr_display,
        "support": support,
        "target": resistance,
        "chart_html": chart_html,
        "market_type": "TW" if is_tw_stock else "US"
    }

# --- 排序與列表 ---
def generate_ranking_report(sorted_data_list):
    if not sorted_data_list: return ""
    
    html = "<div style='background-color:#f8f9fa; color:#333; padding:15px; border-radius:10px; margin-bottom:20px; border:1px solid #e0e0e0;'>"
    html += "<h3 style='margin-top:0; color:#2c3e50; border-bottom:1px solid #ddd; padding-bottom:8px;'>🤖 AI 資金效率排序</h3>"
    
    for i, item in enumerate(sorted_data_list):
        rank_num = f"#{i+1}"
        if i == 0: rank_num = "👑 首選"
        
        if item['trend_score'] >= 8: tag_color = "#27ae60" 
        elif item['trend_score'] <= 3: tag_color = "#c0392b" 
        else: tag_color = "#7f8c8d" 
        
        mkt_tag = "<span style='font-size:10px; background:#eee; padding:2px 4px; border-radius:4px; margin-left:5px;'>TW</span>" if item['market_type'] == "TW" else "<span style='font-size:10px; background:#e3f2fd; color:#1976d2; padding:2px 4px; border-radius:4px; margin-left:5px;'>US</span>"

        html += f"""
        <div style='display:flex; justify-content:space-between; margin-bottom:8px; font-size:15px; border-bottom:1px dashed #eee; padding-bottom:4px;'>
            <span><b style='color:{tag_color}'>{rank_num} {item['ticker']}</b>{mkt_tag}</span>
            <span style='color:#555;'>評分: {item['trend_score']} (R/R: {item['rr_display']})</span>
        </div>
        """
    html += "</div>"
    return html

def format_results_as_cards(sorted_data_list):
    cards_html = ""
    for item in sorted_data_list:
        rr_color = "#2c3e50"
        if item['rr_val'] >= 3: rr_color = "#27ae60"
        elif item['rr_val'] <= 0: rr_color = "#c0392b"
        currency_symbol = "NT$" if item['market_type'] == "TW" else "$"

        cards_html += f"""
        <div style="border:1px solid #e0e0e0; border-radius:12px; padding:16px; margin-bottom:20px; background-color: white; color: #333333; box-shadow: 0 4px 6px rgba(0,0,0,0.05);">
            <div style="display:flex; justify-content:space-between; align-items:flex-start; margin-bottom:12px;">
                <div>
                    <h2 style="margin:0; color:#2c3e50; font-size:22px; letter-spacing:0.5px;">{item['ticker']}</h2>
                    <div style="font-size:12px; color:#999; margin-top:4px;">🕒 {item['date_str']}</div>
                </div>
                <div style="text-align:right;">
                    <div style="font-size:22px; font-weight:800; color:#2c3e50;">{currency_symbol}{item['price']:.2f}</div>
                    <div style="font-size:15px; margin-top:2px;">{item['change_text']}</div>
                </div>
            </div>
            
            <div style="background-color:#f4f6f7; padding:10px; border-radius:8px; margin-bottom:12px; text-align:center;">
                <div style="font-size:12px; color:#7f8c8d; letter-spacing:0.5px;">趨勢評分 (Trend Score)</div>
                <div style="font-size:24px; font-weight:bold; color:#2c3e50; line-height:1.2;">{item['trend_score']} <span style="font-size:16px; color:#95a5a6;">/ 10</span></div>
                <div style="font-size:13px; color:#2980b9; font-weight:600; margin-top:4px;">{item['score_comment']}</div>
            </div>

            <div style="border:1px solid #eee; border-radius:8px; padding:12px; margin-bottom:15px;">
                <div style="display:flex; justify-content:space-between; align-items:center; margin-bottom:8px; border-bottom:1px solid #f0f0f0; padding-bottom:8px;">
                    <span style="font-size:14px; color:#555;">風報比 (R/R)</span>
                    <span style="font-size:18px; font-weight:bold; color:{rr_color};">{item['rr_display']}</span>
                </div>
                <div style="display:flex; justify-content:space-between; margin-bottom:4px;">
                    <span style="font-size:13px; color:#888;">停損參考 (MA50)</span>
                    <span style="font-size:13px; font-weight:600; color:#c0392b;">${item['support']:.2f}</span>
                </div>
                <div style="display:flex; justify-content:space-between;">
                    <span style="font-size:13px; color:#888;">目標價 (前高/延伸)</span>
                    <span style="font-size:13px; font-weight:600; color:#27ae60;">${item['target']:.2f}</span>
                </div>
            </div>
            {item['chart_html']}
        </div>
        """
    return cards_html

# --- 主程式 (雙輸入) ---
def app_main(us_input, tw_input):
    if not us_input and not tw_input: return ""
    
    data_list = []
    all_tickers_str = [] # 用來餵給 AI 的代號列表
    split_pattern = r'[ ,\n]+' 
    
    # 1. 處理美股
    if us_input:
        us_tickers = re.split(split_pattern, us_input)
        for t in us_tickers:
            if t.strip():
                res = analyze_single_stock(t)
                if res: 
                    data_list.append(res)
                    all_tickers_str.append(t)
    
    # 2. 處理台股
    if tw_input:
        tw_tickers = re.split(split_pattern, tw_input)
        for t in tw_tickers:
            t = t.strip()
            if t:
                if t.isdigit(): t = f"{t}.TW"
                res = analyze_single_stock(t)
                if res: 
                    data_list.append(res)
                    all_tickers_str.append(t)
    
    if not data_list: 
        return "<div style='padding:20px; text-align:center; color:#e74c3c;'>找不到有效資料，請檢查代號。</div>"

    # --- 呼叫 Gemini 進行總經分析 ---
    # 將所有代號結合成字串，例如 "NVDA, TSLA, 2330.TW"
    tickers_context = ", ".join(all_tickers_str)
    ai_analysis_html = ask_gemini_analysis(tickers_context)
    
    # 包裝 AI 分析結果區塊
    ai_section = f"""
    <div style='background-color:#fffbeb; color:#2c3e50; padding:20px; border-radius:12px; margin-bottom:25px; border:2px solid #f1c40f; box-shadow: 0 4px 10px rgba(0,0,0,0.05);'>
        <h3 style='margin-top:0; color:#d35400; border-bottom:1px solid #f39c12; padding-bottom:10px;'>🧠 Gemini 交易員早報 (AI 實時分析)</h3>
        <div style='font-size:15px; line-height:1.6;'>
            {ai_analysis_html}
        </div>
        <div style='margin-top:10px; font-size:12px; color:#aaa; text-align:right;'>
            Powered by Google Gemini 1.5 Flash
        </div>
    </div>
    """

    sorted_data = sorted(data_list, key=lambda x: (x['trend_score'], x['rr_val']), reverse=True)
    ranking_html = generate_ranking_report(sorted_data)
    cards_html = format_results_as_cards(sorted_data)
    
    # 組合：AI 分析置頂 + 排名 + 卡片
    return ai_section + ranking_html + cards_html

# --- Gradio 介面 ---
with gr.Blocks(title="AI 順勢交易助手", theme=gr.themes.Soft()) as demo:
    gr.Markdown("## 📱 AI 跨國界順勢交易分析 (Gemini Upgrade)")
    
    output_area = gr.HTML(label="分析結果", value="<div style='padding:20px; text-align:center; color:#999; border:1px dashed #ccc; border-radius:8px;'>請先至 Settings 設定 Gemini API Key，然後輸入代號</div>")
    
    with gr.Column():
        gr.Markdown("### 🇺🇸 美股輸入")
        us_box = gr.Textbox(show_label=False, placeholder="例如: NVDA TSLA (空白分隔)...", lines=3)
        
        gr.Markdown("### 🇹🇼 台股輸入")
        tw_box = gr.Textbox(show_label=False, placeholder="例如: 2330 2603 (自動加.TW)...", lines=3)
        
        submit_btn = gr.Button("🚀 啟動 Gemini 全面分析", variant="primary", scale=1)
    
    submit_btn.click(fn=app_main, inputs=[us_box, tw_box], outputs=output_area)

demo.launch()
