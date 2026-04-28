import warnings
warnings.filterwarnings("ignore")

import os
import numpy as np
import pandas as pd
from dataclasses import dataclass
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

def load_yahoo(ticker: str, start="2022-07-01", end=None) -> pd.DataFrame:
    try:
        import yfinance as yf
    except ImportError:
        raise ImportError("Zainstaluj: pip install yfinance")

    df = yf.download(ticker, start=start, end=end, auto_adjust=False, progress=False)
    if df is None or df.empty:
        raise ValueError(f"Brak danych dla {ticker} z Yahoo.")

    if isinstance(df.columns, pd.MultiIndex):
        last_level = df.columns.get_level_values(-1)
        if ticker in last_level:
            df = df.xs(ticker, axis=1, level=-1)
        else:
            df = df.droplevel(-1, axis=1)

    df = df.rename(columns=str.title)
    needed = ["Open", "High", "Low", "Close", "Volume"]
    df = df[[c for c in needed if c in df.columns]].dropna()

    if not all(c in df.columns for c in needed):
        raise ValueError(f"{ticker}: Brak wymaganych kolumn OHLCV. Kolumny: {df.columns}")

    df.index = pd.to_datetime(df.index)
    return df

def ema(s: pd.Series, span: int) -> pd.Series:
    return s.ewm(span=span, adjust=False).mean()

def rsi(close: pd.Series, period: int = 14) -> pd.Series:
    delta = close.diff()
    gain = delta.clip(lower=0.0)
    loss = -delta.clip(upper=0.0)
    avg_gain = gain.rolling(period).mean()
    avg_loss = loss.rolling(period).mean()
    rs = avg_gain / (avg_loss + 1e-12)
    return 100 - (100 / (1 + rs))

def atr(high: pd.Series, low: pd.Series, close: pd.Series, period: int = 14) -> pd.Series:
    prev_close = close.shift(1)
    tr = pd.concat([
        (high - low).abs(),
        (high - prev_close).abs(),
        (low - prev_close).abs()
    ], axis=1).max(axis=1)
    return tr.rolling(period).mean()

def macd(close: pd.Series, fast: int = 12, slow: int = 26, signal: int = 9):
    macd_line = ema(close, fast) - ema(close, slow)
    signal_line = ema(macd_line, signal)
    hist = macd_line - signal_line
    return macd_line, signal_line, hist

def bollinger(close: pd.Series, period: int = 20, n_std: float = 2.0):
    ma = close.rolling(period).mean()
    sd = close.rolling(period).std()
    upper = ma + n_std * sd
    lower = ma - n_std * sd
    width = (upper - lower) / (ma + 1e-12)
    return upper, lower, width

def obv(close: pd.Series, volume: pd.Series) -> pd.Series:
    direction = np.sign(close.diff()).fillna(0)
    return (direction * volume).cumsum()

def make_features(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    close = out["Close"].squeeze()
    high, low, volume = out["High"], out["Low"], out["Volume"]

    out["ret_1"] = close.pct_change(1)
    out["ret_3"] = close.pct_change(3)
    out["ret_5"] = close.pct_change(5)
    out["ret_10"] = close.pct_change(10)

    for w in [5, 10, 20, 50, 100, 200]:
        out[f"sma_{w}"] = close.rolling(w).mean()
        out[f"dist_sma_{w}"] = (close / (out[f"sma_{w}"] + 1e-12)) - 1

    out["ema_12"] = ema(close, 12)
    out["ema_26"] = ema(close, 26)
    out["dist_ema12"] = (close / (out["ema_12"] + 1e-12)) - 1

    out["rsi_14"] = rsi(close, 14)

    m_line, s_line, hist = macd(close)
    out["macd_line"] = m_line
    out["macd_signal"] = s_line
    out["macd_hist"] = hist

    out["atr_14"] = atr(high, low, close, 14)
    out["atr_pct"] = out["atr_14"] / (close + 1e-12)
    out["vol_20"] = out["ret_1"].rolling(20).std()

    bb_u, bb_l, bb_w = bollinger(close, 20, 2.0)
    out["bb_width"] = bb_w
    out["bb_pos"] = (close - bb_l) / ((bb_u - bb_l) + 1e-12)

    out["vol_z20"] = (volume - volume.rolling(20).mean()) / (volume.rolling(20).std() + 1e-12)
    out["obv"] = obv(close, volume)
    out["obv_z20"] = (out["obv"] - out["obv"].rolling(20).mean()) / (out["obv"].rolling(20).std() + 1e-12)

    out["hl_range"] = (high - low) / (close + 1e-12)
    out["co_range"] = (out["Close"] - out["Open"]) / (close + 1e-12)

    out = out.replace([np.inf, -np.inf], np.nan)
    return out

def base_direction(df_feat: pd.DataFrame) -> pd.Series:
    close = df_feat["Close"].squeeze()
    sma20 = df_feat["sma_20"]
    sma50 = df_feat["sma_50"]
    sma200 = df_feat["sma_200"]

    long_cond = (sma20 > sma50) & (close > sma200)
    dir_ = pd.Series(0, index=df_feat.index, dtype=float)
    dir_[long_cond] = 1.0
    return dir_

def make_trade_label(df_feat: pd.DataFrame, horizon: int = 5, thr_abs: float = 0.012):
    close = df_feat["Close"].squeeze()
    fwd_ret = close.shift(-horizon) / close - 1
    y = (fwd_ret.abs() > thr_abs).astype(int)
    return y, fwd_ret

@dataclass
class WFConfig:
    train_months: int = 18
    test_months: int = 3
    horizon: int = 5

    trade_thr_abs: float = 0.012
    trade_p: float = 0.55

    fee_bps: int = 5
    random_state: int = 42

    use_sl_tp: bool = True
    sltp_mode: str = "atr"
    fixed_sl_pct: float = 0.02
    fixed_tp_pct: float = 0.04
    atr_sl_mult: float = 1.5
    atr_tp_mult: float = 3.0

    use_intraday_hilo: bool = True


def month_floor(ts: pd.Timestamp) -> pd.Timestamp:
    return pd.Timestamp(year=ts.year, month=ts.month, day=1)

def add_months(ts: pd.Timestamp, n: int) -> pd.Timestamp:
    y = ts.year + (ts.month - 1 + n) // 12
    m = (ts.month - 1 + n) % 12 + 1
    return pd.Timestamp(year=y, month=m, day=1)


def walk_forward_ml_filter(df: pd.DataFrame, cfg: WFConfig):
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.preprocessing import StandardScaler

    feat = make_features(df)
    y_trade, _ = make_trade_label(feat, horizon=cfg.horizon, thr_abs=cfg.trade_thr_abs)

    drop_cols = ["Open", "High", "Low", "Close", "Volume"]
    X = feat.drop(columns=[c for c in drop_cols if c in feat.columns])

    data = pd.concat([df[["Close"]], X, y_trade.rename("y_trade")], axis=1).dropna()
    if len(data) < 250:
        raise ValueError("Za mało danych po featuringu/labelingu dla walk-forward (post-2022).")

    start = month_floor(data.index.min())
    p_trade_all = pd.Series(index=data.index, dtype=float)
    allow_trade_all = pd.Series(index=data.index, dtype=float)
    windows = []

    cur_train_start = start
    while True:
        train_start = cur_train_start
        train_end = add_months(train_start, cfg.train_months) - pd.Timedelta(days=1)
        test_end = add_months(train_start, cfg.train_months + cfg.test_months) - pd.Timedelta(days=1)

        if test_end > data.index.max():
            break

        train_mask = (data.index >= train_start) & (data.index <= train_end)
        test_mask = (data.index > train_end) & (data.index <= test_end)

        train = data.loc[train_mask].copy()
        test = data.loc[test_mask].copy()

        if len(train) < 180 or len(test) < 30:
            cur_train_start = add_months(cur_train_start, cfg.test_months)
            continue

        X_train = train.drop(columns=["Close", "y_trade"])
        y_train = train["y_trade"].astype(int)
        X_test = test.drop(columns=["Close", "y_trade"])

        scaler = StandardScaler()
        X_train_s = scaler.fit_transform(X_train)
        X_test_s = scaler.transform(X_test)

        model = RandomForestClassifier(
            n_estimators=600,
            max_depth=8,
            min_samples_leaf=10,
            class_weight="balanced_subsample",
            random_state=cfg.random_state,
            n_jobs=-1
        )
        model.fit(X_train_s, y_train)

        proba = model.predict_proba(X_test_s)
        classes = model.classes_.tolist()
        idx_trade = classes.index(1) if 1 in classes else None
        p_trade = proba[:, idx_trade] if idx_trade is not None else np.zeros(len(test))

        allow = (p_trade > cfg.trade_p).astype(int)

        p_trade_all.loc[test.index] = p_trade
        allow_trade_all.loc[test.index] = allow

        windows.append({
            "train_start": str(train_start.date()),
            "train_end": str(train_end.date()),
            "test_start": str((train_end + pd.Timedelta(days=1)).date()),
            "test_end": str(test_end.date()),
            "n_train": len(train),
            "n_test": len(test),
            "trade_rate_test": float(allow.mean()),
            "pos_rate_train": float(y_train.mean())
        })

        cur_train_start = add_months(cur_train_start, cfg.test_months)

    wf_table = pd.DataFrame(windows)
    p_trade_all = p_trade_all.reindex(df.index)
    allow_trade_all = allow_trade_all.reindex(df.index).fillna(0)

    return p_trade_all, allow_trade_all, wf_table, feat

def backtest_with_sltp(df: pd.DataFrame, position_signal: pd.Series, cfg: WFConfig):
    idx = df.index
    close = df["Close"].astype(float)
    high = df["High"].astype(float)
    low = df["Low"].astype(float)

    feat = make_features(df)
    atr14 = feat["atr_14"].astype(float)

    desired = position_signal.reindex(idx).fillna(0).clip(0, 1).astype(int)

    in_pos = 0
    entry_price = np.nan
    sl_price = np.nan
    tp_price = np.nan

    pos = pd.Series(0, index=idx, dtype=int)
    strat_ret = pd.Series(0.0, index=idx, dtype=float)

    for i in range(1, len(idx)):
        today = idx[i]
        yest = idx[i - 1]
        pos.iloc[i] = in_pos
        if in_pos == 0 and desired.iloc[i] == 1:
            in_pos = 1
            entry_price = close.iloc[i]

            if cfg.use_sl_tp:
                if cfg.sltp_mode == "fixed":
                    sl_price = entry_price * (1 - cfg.fixed_sl_pct)
                    tp_price = entry_price * (1 + cfg.fixed_tp_pct)
                elif cfg.sltp_mode == "atr":
                    a = float(atr14.iloc[i])
                    sl_price = entry_price - cfg.atr_sl_mult * a
                    tp_price = entry_price + cfg.atr_tp_mult * a
                else:
                    raise ValueError("cfg.sltp_mode must be 'fixed' or 'atr'")

            pos.iloc[i] = 1
        if in_pos == 1:
            exit_now = False
            exit_price = np.nan

            if cfg.use_sl_tp and not (np.isnan(sl_price) or np.isnan(tp_price)):
                if cfg.use_intraday_hilo:
                    if low.iloc[i] <= sl_price:
                        exit_now = True
                        exit_price = sl_price
                    elif high.iloc[i] >= tp_price:
                        exit_now = True
                        exit_price = tp_price
                else:
                    if close.iloc[i] <= sl_price:
                        exit_now = True
                        exit_price = close.iloc[i]
                    elif close.iloc[i] >= tp_price:
                        exit_now = True
                        exit_price = close.iloc[i]
            if not exit_now and desired.iloc[i] == 0:
                exit_now = True
                exit_price = close.iloc[i]
            if in_pos == 1:
                if exit_now:
                    strat_ret.iloc[i] = (exit_price / close.loc[yest]) - 1
                    in_pos = 0
                    entry_price = np.nan
                    sl_price = np.nan
                    tp_price = np.nan
                    pos.iloc[i] = 0
                else:
                    strat_ret.iloc[i] = (close.iloc[i] / close.loc[yest]) - 1
                    pos.iloc[i] = 1
        else:
            strat_ret.iloc[i] = 0.0

    turnover = pos.diff().abs().fillna(0)
    fee = (cfg.fee_bps / 1e4) * turnover
    strat_ret = strat_ret - fee

    eq = (1 + strat_ret.fillna(0)).cumprod()
    return strat_ret, eq, pos


def perf_stats(strat_ret: pd.Series):
    strat_ret = strat_ret.dropna()
    if strat_ret.empty:
        return {"CAGR": np.nan, "Sharpe": np.nan, "MaxDD": np.nan, "HitRate": np.nan}

    eq = (1 + strat_ret).cumprod()
    cagr = eq.iloc[-1] ** (252 / len(eq)) - 1
    sharpe = np.sqrt(252) * strat_ret.mean() / (strat_ret.std() + 1e-12)

    peak = eq.cummax()
    dd = (eq / peak) - 1
    maxdd = dd.min()

    hitrate = (strat_ret > 0).mean()
    return {"CAGR": float(cagr), "Sharpe": float(sharpe), "MaxDD": float(maxdd), "HitRate": float(hitrate)}

def safe_filename(ticker: str) -> str:
    return ticker.replace("^", "").replace("/", "_").replace("\\", "_").replace(".", "_")

def run_for_tickers(tickers, cfg: WFConfig, start="2022-07-01", out_dir="outputs_ml_filter_sl_tp"):
    os.makedirs(out_dir, exist_ok=True)
    os.makedirs(os.path.join(out_dir, "plots"), exist_ok=True)
    os.makedirs(os.path.join(out_dir, "signals"), exist_ok=True)

    summary_rows = []

    for t in tickers:
        print(f"\n=== {t} ===")
        df = load_yahoo(t, start=start)

        p_trade, allow_trade, wf_table, feat = walk_forward_ml_filter(df, cfg)

        dir_ta = base_direction(feat)
        desired_position = (dir_ta * allow_trade).clip(0, 1)

        strat_ret, eq, pos = backtest_with_sltp(df, desired_position, cfg)
        stats = perf_stats(strat_ret)

        ret = df["Close"].pct_change().fillna(0)
        bh_eq = (1 + ret).cumprod()
        bh_stats = perf_stats(ret)

        summary_rows.append({
            "Ticker": t,
            **{f"Strat_{k}": v for k, v in stats.items()},
            **{f"BH_{k}": v for k, v in bh_stats.items()},
            "TradeRate": float((pos != 0).mean()),
            "SLTP_Mode": cfg.sltp_mode
        })

        wf_table.to_csv(os.path.join(out_dir, f"{safe_filename(t)}_walkforward.csv"), index=False)

        sig_df = pd.DataFrame({
            "Close": df["Close"],
            "High": df["High"],
            "Low": df["Low"],
            "P_trade": p_trade,
            "AllowTrade": allow_trade,
            "TA_Direction": dir_ta,
            "DesiredPos": desired_position,
            "Position": pos
        })
        sig_df.to_csv(os.path.join(out_dir, "signals", f"{safe_filename(t)}_signals.csv"), index=True)

        print("Performance (Strategy):", {k: round(v, 4) for k, v in stats.items()})
        print("Performance (Buy&Hold):", {k: round(v, 4) for k, v in bh_stats.items()})
        print("Trade rate:", round((pos != 0).mean(), 4))

        plt.figure()
        title_mode = f"SL/TP: {cfg.sltp_mode.upper()}"
        plt.title(f"Equity Curve (ML+TA + {title_mode}): {t}")
        plt.plot(eq.index, eq.values, label="Strategy")
        plt.plot(bh_eq.index, bh_eq.values, label="Buy&Hold")
        plt.legend()
        plt.tight_layout()
        plt.savefig(os.path.join(out_dir, "plots", f"{safe_filename(t)}_equity.png"), dpi=170)
        plt.close()

        last = df.tail(220)
        last_pos = pos.reindex(last.index).fillna(0)

        plt.figure()
        plt.title(f"Price & Position (last 220d): {t}")
        plt.plot(last.index, last["Close"].values, label="Close")

        entry = (last_pos.shift(1).fillna(0) == 0) & (last_pos == 1)
        exit_ = (last_pos.shift(1).fillna(0) == 1) & (last_pos == 0)

        plt.scatter(last.index[entry], last["Close"][entry], marker="^", label="ENTRY")
        plt.scatter(last.index[exit_], last["Close"][exit_], marker="v", label="EXIT")
        plt.legend()
        plt.tight_layout()
        plt.savefig(os.path.join(out_dir, "plots", f"{safe_filename(t)}_signals.png"), dpi=170)
        plt.close()

    summary = pd.DataFrame(summary_rows)
    summary_path = os.path.join(out_dir, "summary_results.csv")
    summary.to_csv(summary_path, index=False)

    print("\n=== SUMMARY ===")
    pd.set_option("display.max_columns", None)
    print(summary.to_string(index=False))
    print(f"\nZapisano:\n- {summary_path}\n- {out_dir}/plots/*.png\n- {out_dir}/signals/*_signals.csv\n- {out_dir}/*_walkforward.csv")

    return summary


if __name__ == "__main__":
    gpw = ["CDR.WA", "PKN.WA", "PKO.WA", "KGH.WA", "PZU.WA"]
    us = ["AAPL", "MSFT", "NVDA", "AMZN", "TSLA", "SPY"]
    tickers = gpw + us

    cfg = WFConfig(
        train_months=18,
        test_months=3,
        horizon=5,
        trade_thr_abs=0.012,
        trade_p=0.55,
        fee_bps=5,
        use_sl_tp=True,
        sltp_mode="atr", atr_sl_mult=1.5, atr_tp_mult=3.0,
        use_intraday_hilo=True
    )

    run_for_tickers(
        tickers=tickers,
        cfg=cfg,
        start="2022-07-01",
        out_dir="outputs_ml_filter_sl_tp"
    )
