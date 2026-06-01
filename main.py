import polars as pl
import estimators as est
import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


def fetch_data(tickers):
    df_options = pd.DataFrame()
    for ticker_symbol in tickers:
        print(f"{ticker_symbol} のデータを取得中...")
        ticker = yf.Ticker(ticker_symbol)

        current_price = ticker.fast_info["last_price"]

        options_dates = ticker.options
        if not options_dates:
            print(f"{ticker_symbol}のオプションデータは見つかりませんでした。")
            continue

        # Option 何日先までData取得するか選択可能
        for nearest_date in options_dates[:100]:

            option_chain = ticker.option_chain(nearest_date)
            calls = option_chain.calls
            puts = option_chain.puts

            calls_subset = calls[
                ["strike", "lastPrice", "bid", "ask", "volume", "impliedVolatility"]
            ].copy()
            puts_subset = puts[
                ["strike", "lastPrice", "bid", "ask", "volume", "impliedVolatility"]
            ].copy()

            calls_subset["midPrice"] = np.where(
                (calls_subset["bid"] == 0) & (calls_subset["ask"] == 0),
                calls_subset["lastPrice"],
                (calls_subset["bid"] + calls_subset["ask"]) / 2,
            )

            puts_subset["midPrice"] = np.where(
                (puts_subset["bid"] == 0) & (puts_subset["ask"] == 0),
                puts_subset["lastPrice"],
                (puts_subset["bid"] + puts_subset["ask"]) / 2,
            )
            calls_subset["Option_Type"] = "Call"
            puts_subset["Option_Type"] = "Put"

            calls_subset["Ticker"] = ticker_symbol
            calls_subset["Underlying_Price"] = current_price
            calls_subset["Expiry_Date"] = nearest_date

            puts_subset["Ticker"] = ticker_symbol
            puts_subset["Underlying_Price"] = current_price
            puts_subset["Expiry_Date"] = nearest_date

            df_options = pd.concat(
                [df_options, calls_subset, puts_subset], ignore_index=True
            )

    df_options = df_options[
        [
            "Ticker",
            "Underlying_Price",
            "Expiry_Date",
            "Option_Type",
            "strike",
            "lastPrice",
            "midPrice",
            "bid",
            "ask",
            "volume",
            "impliedVolatility",
        ]
    ]
    df_options_pl = pl.from_pandas(df_options)
    print("\n=== 取得完了！オプションデータ ===")

    df_history_pd = pd.DataFrame()

    for ticker_symbol in tickers:
        ticker = yf.Ticker(ticker_symbol)

        hist_6mo = ticker.history(period="6mo")

        hist_6mo["Ticker"] = ticker_symbol

        hist_6mo = hist_6mo.reset_index()

        df_history_pd = pd.concat([df_history_pd, hist_6mo], ignore_index=True)

    df_history_pd = df_history_pd[
        ["Date", "Ticker", "Open", "High", "Low", "Close", "Volume"]
    ]
    df_history_pl = pl.from_pandas(df_history_pd)

    print("\n=== 取得完了！過去6ヶ月の株価データ ===")

    return df_options_pl, df_history_pl


def future_depicter(hf, ticker, r, t, color="blue"):
    fig, ax = plt.subplots(figsize=(10, 6))

    hf_filtered = hf.filter(pl.col("Ticker") == ticker)
    x = hf_filtered.get_column("Date").to_numpy()
    y = hf_filtered.get_column("Close").to_numpy()

    last_date = x[-1]
    last_price = y[-1]

    ax.plot(x, y, label=f"{ticker} (Historical)", color=color, linestyle="-")

    business_days_ahead = int(t * 252)
    future_dates = pd.bdate_range(start=last_date, periods=business_days_ahead + 1)
    t_array = np.arange(business_days_ahead + 1) / 252.0
    y_future = est.calculation_future(last_price, r, t_array)

    ax.plot(
        future_dates,
        y_future,
        label=f"{ticker} (Prediction)",
        color="red",
        linestyle="--",
    )

    ax.set_title(f"Stock Price and Future Prediction: {ticker}", fontsize=14)
    ax.set_xlabel("Date", fontsize=12)
    ax.set_ylabel("Price ($)", fontsize=12)
    ax.grid(True, alpha=0.5, linestyle="--")
    ax.legend(loc="upper left")

    fig.autofmt_xdate()
    plt.tight_layout()

    plt.show()
    plt.close(fig)


def plot_term_structure_with_theory(
    df, hf, ticker, target_strike, r, option_type="Call", color="green"
):
    close_prices = hf.get_column("Close").to_numpy()

    current_price = close_prices[-1]  # 最新の株価を使うこと
    sigma_hv = est.historical_volatility(close_prices, 1 / 252.0)

    fig, ax = plt.subplots(figsize=(10, 6))

    df_filtered = df.filter(
        (pl.col("Option_Type") == option_type) & (pl.col("strike") == target_strike)
    ).sort("Expiry_Date")

    if df_filtered.height == 0:
        print(
            f"⚠️ {ticker} の Strike {target_strike} ({option_type}) のデータは見つかりませんでした。"
        )
        plt.close(fig)
        return

    x_dates_str = df_filtered.get_column("Expiry_Date").to_numpy()
    x_dates = pd.to_datetime(x_dates_str)

    y_prices_market = df_filtered.get_column("midPrice").to_numpy()

    y_prices_theory_hv = []
    y_prices_theory_iv = []

    today_date = pd.Timestamp.today().date()

    for i, expiry_date in enumerate(x_dates):
        exp_date_obj = expiry_date.date()
        business_days = np.busday_count(today_date, exp_date_obj)
        t = max(business_days / 252.0, 0.0001)

        market_price = y_prices_market[i]

        a, b = 0.0001, 5.0
        try:
            if option_type == "Call":
                calc_iv = est.implied_volatility_european_call(
                    current_price, target_strike, r, t, market_price, a, b
                )
            else:
                calc_iv = est.implied_volatility_european_put(
                    current_price, target_strike, r, t, market_price, a, b
                )
        except ValueError:
            print(
                f"  [Info] {exp_date_obj} のIV逆算が収束しなかったため代替値を使用します。"
            )
            calc_iv = np.nan

        if option_type == "Call":
            theory_hv = est.black_scholes_merton_european_call_option(
                current_price, target_strike, r, t, sigma_hv
            )
            theory_iv = est.black_scholes_merton_european_call_option(
                current_price, target_strike, r, t, calc_iv
            )
        else:
            theory_hv = est.black_scholes_merton_european_put_option(
                current_price, target_strike, r, t, sigma_hv
            )
            theory_iv = est.black_scholes_merton_european_put_option(
                current_price, target_strike, r, t, calc_iv
            )

        y_prices_theory_hv.append(theory_hv)
        y_prices_theory_iv.append(theory_iv)

    ax.plot(
        x_dates,
        y_prices_market,
        label=f"Market Price (Mid)",
        color=color,
        marker="o",
        linestyle="-",
        linewidth=2.5,
    )

    ax.plot(
        x_dates,
        y_prices_theory_iv,
        label=f"Theory (Custom IV)",
        color="orange",
        marker="^",
        linestyle="--",
        alpha=0.8,
    )

    ax.plot(
        x_dates,
        y_prices_theory_hv,
        label=f"Theory (HV: {sigma_hv:.1%})",
        color="black",
        marker="x",
        linestyle="--",
        alpha=0.8,
    )

    ax.set_title(
        f"Term Structure & BS Theory: {ticker} {option_type} @ Strike {target_strike}",
        fontsize=14,
    )
    ax.set_xlabel("Expiry Date", fontsize=12)
    ax.set_ylabel("Option Price ($)", fontsize=12)
    ax.grid(True, alpha=0.5, linestyle="--")
    ax.legend(loc="upper left")

    fig.autofmt_xdate()
    plt.tight_layout()

    plt.show()
    plt.close(fig)


def plot_iv_term_structure(
    df, hf, ticker, target_strike, r, option_type="Call", color="blue"
):
    close_prices = hf.get_column("Close").to_numpy()
    current_price = close_prices[-1]
    sigma_hv = est.historical_volatility(close_prices, 1 / 252.0)

    fig, ax = plt.subplots(figsize=(10, 6))

    # データフィルタリング
    df_filtered = df.filter(
        (pl.col("Option_Type") == option_type) & (pl.col("strike") == target_strike)
    ).sort("Expiry_Date")

    if df_filtered.height == 0:
        return

    x_dates_str = df_filtered.get_column("Expiry_Date").to_numpy()
    x_dates = pd.to_datetime(x_dates_str)
    y_prices_market = df_filtered.get_column("midPrice").to_numpy()
    plot_x_dates = []
    y_iv_calculated = []
    today_date = pd.Timestamp.today().date()

    for i, expiry_date in enumerate(x_dates):
        exp_date_obj = expiry_date.date()

        # 💡 先ほどのガタガタを解消！営業日ではなくカレンダー日数で計算
        calendar_days = (exp_date_obj - today_date).days
        if calendar_days < 10:
            continue
        t = max(calendar_days / 365.0, 0.0001)

        market_price = y_prices_market[i]
        a, b = 0.0001, 5.0

        try:
            if option_type == "Call":
                calc_iv = est.implied_volatility_european_call(
                    current_price, target_strike, r, t, market_price, a, b
                )
            else:
                calc_iv = est.implied_volatility_european_put(
                    current_price, target_strike, r, t, market_price, a, b
                )
        except ValueError:
            calc_iv = np.nan  # 収束しない場合は欠損値扱い

        # パーセント表記（%）にするために100をかける
        plot_x_dates.append(expiry_date)
        y_iv_calculated.append(calc_iv * 100 if not np.isnan(calc_iv) else np.nan)

    # 📈 IVのプロット
    ax.plot(
        plot_x_dates,
        y_iv_calculated,
        label=f"Implied Volatility (IV)",
        color=color,
        marker="o",
        linestyle="-",
        linewidth=2.5,
    )

    ax.axhline(
        y=sigma_hv * 100,
        color="black",
        linestyle="--",
        label=f"Historical Volatility (HV: {sigma_hv*100:.1f}%)",
        alpha=0.8,
    )

    ax.set_title(
        f"IV Term Structure: {ticker} {option_type} @ Strike {target_strike}",
        fontsize=14,
    )
    ax.set_xlabel("Expiry Date", fontsize=12)
    ax.set_ylabel("Implied Volatility (%)", fontsize=12)
    ax.grid(True, alpha=0.5, linestyle="--")
    ax.legend(loc="upper left")

    fig.autofmt_xdate()
    plt.tight_layout()

    plt.show()
    plt.close(fig)


def main():
    target_symbol = "TSLA"  # ここで企業変更可能

    print("無リスク金利(Risk-free rate)を取得中...")
    treasury_ticker = "^IRX"
    ticker_tbill = yf.Ticker(treasury_ticker)
    r = ticker_tbill.fast_info["last_price"] / 100
    print(f"Risk-free rate (r): {r:.4f}")

    df, hf = fetch_data([target_symbol])

    target_ticker = yf.Ticker(target_symbol)
    current_price = target_ticker.fast_info["last_price"]
    print(f"\n{target_symbol} の現在の株価 (yfinance直): {current_price:.2f}")

    available_strikes = (
        df.filter(pl.col("Ticker") == target_symbol)
        .get_column("strike")
        .unique()
        .to_numpy()
    )

    my_target_strike = available_strikes[
        np.argmin(np.abs(available_strikes - current_price))
    ]
    print(f"自動選択された ATM ストライク: {my_target_strike}")

    print(f"\n[1/3] {target_symbol} の株価推移と未来予測を描画します...")
    future_depicter(hf, target_symbol, r, t=0.5, color="blue")

    print(
        f"\n[2/3] {target_symbol} の Strike {my_target_strike} (Call) の理論値比較を描画します..."
    )
    plot_term_structure_with_theory(
        df, hf, target_symbol, my_target_strike, r, option_type="Call", color="green"
    )

    print(
        f"\n[3/3] {target_symbol} の Strike {my_target_strike} (Put) の理論値比較を描画します..."
    )
    plot_term_structure_with_theory(
        df, hf, target_symbol, my_target_strike, r, option_type="Put", color="purple"
    )
    print(
        f"\n[Extra] {target_symbol} の Strike {my_target_strike} の IV タームストラクチャーを描画します..."
    )
    plot_iv_term_structure(
        df, hf, target_symbol, my_target_strike, r, option_type="Call", color="red"
    )

    print("\nすべての描画が完了しました！")


if __name__ == "__main__":
    main()
