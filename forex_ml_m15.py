import warnings
warnings.filterwarnings("ignore")

import os
import copy
import numpy as np
import pandas as pd
from dataclasses import dataclass
from typing import Dict, List, Optional
from datetime import datetime, timedelta

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import matplotlib.dates as mdates

BARS_PER_YEAR_M15 = 24_192

FOREX_PAIRS = {
    "EURUSD=X": {"name": "EUR/USD", "pip": 0.0001, "spread_pips": 1.0,  "csv": "EURUSD_M15.csv"},
    "GBPUSD=X": {"name": "GBP/USD", "pip": 0.0001, "spread_pips": 1.5,  "csv": "GBPUSD_M15.csv"},
    "USDJPY=X": {"name": "USD/JPY", "pip": 0.01,   "spread_pips": 1.0,  "csv": "USDJPY_M15.csv"},
    "USDCHF=X": {"name": "USD/CHF", "pip": 0.0001, "spread_pips": 1.5,  "csv": "USDCHF_M15.csv"},
    "AUDUSD=X": {"name": "AUD/USD", "pip": 0.0001, "spread_pips": 1.5,  "csv": "AUDUSD_M15.csv"},
    "USDCAD=X": {"name": "USD/CAD", "pip": 0.0001, "spread_pips": 1.5,  "csv": "USDCAD_M15.csv"},
    "EURGBP=X": {"name": "EUR/GBP", "pip": 0.0001, "spread_pips": 2.0,  "csv": "EURGBP_M15.csv"},
    "EURJPY=X": {"name": "EUR/JPY", "pip": 0.01,   "spread_pips": 2.0,  "csv": "EURJPY_M15.csv"},
    "GBPJPY=X": {"name": "GBP/JPY", "pip": 0.01,   "spread_pips": 3.0,  "csv": "GBPJPY_M15.csv"},
}

MODEL_COLORS = {
    "RandomForest":    "#2196F3",
    "GradientBoosting":"#FF9800",
    "NeuralNet":       "#9C27B0",
    "XGBoost":         "#4CAF50",
    "Ensemble":        "#F44336",
    "BuyHold":         "#90A4AE",
}

def load_from_csv(path: str, start: str = "2022-01-01") -> pd.DataFrame:
    """
    Wczytuje M15 dane z pliku CSV.
    Obsługiwane formaty:
      A) MetaTrader 5 export: <DATE>\t<TIME>\t<OPEN>\t<HIGH>\t<LOW>\t<CLOSE>\t<TICKVOL>
         nagłówek: DATE  TIME  OPEN  HIGH  LOW  CLOSE  TICKVOL  VOL  SPREAD
      B) Dukascopy / generic: datetime jako pierwsza kolumna lub index, kolumny OHLCV
      C) Dowolny CSV z datetime index i kolumnami Open,High,Low,Close (Volume opcjonalny)
    """
    df = pd.read_csv(path, sep=None, engine="python")

    date_col = next((c for c in df.columns if c.strip().upper() in ("DATE", "<DATE>")), None)
    time_col = next((c for c in df.columns if c.strip().upper() in ("TIME", "<TIME>")), None)

    if date_col and time_col:
        df["datetime"] = pd.to_datetime(df[date_col].astype(str) + " " + df[time_col].astype(str))
        df = df.drop(columns=[date_col, time_col])
        df = df.set_index("datetime")
    else:
        first = df.columns[0]
        try:
            df[first] = pd.to_datetime(df[first])
            df = df.set_index(first)
        except Exception:
            df.index = pd.to_datetime(df.index)

    col_map = {}
    for c in df.columns:
        cu = c.strip().upper().lstrip("<").rstrip(">")
        if cu in ("OPEN", "O"):        col_map[c] = "Open"
        elif cu in ("HIGH", "H"):      col_map[c] = "High"
        elif cu in ("LOW", "L"):       col_map[c] = "Low"
        elif cu in ("CLOSE", "C"):     col_map[c] = "Close"
        elif cu in ("VOL", "VOLUME", "TICKVOL", "TICK_VOLUME", "V"): col_map[c] = "Volume"
    df = df.rename(columns=col_map)

    needed = ["Open", "High", "Low", "Close"]
    df = df[[c for c in df.columns if c in needed + ["Volume"]]].dropna(subset=needed)
    if "Volume" not in df.columns or df["Volume"].sum() == 0:
        df["Volume"] = 1.0

    df.index = pd.to_datetime(df.index)
    df = df.sort_index()
    df = df[~df.index.duplicated(keep="first")]
    df = df[df.index >= pd.Timestamp(start)]
    return df


def load_from_yahoo_m15(ticker: str, start: str = "2022-01-01") -> pd.DataFrame:
    """
    Pobiera M15 z Yahoo Finance w kawałkach po 55 dni.

    UWAGA: Yahoo Finance przechowuje dane M15 tylko przez ostatnie ~60 dni.
    Niezależnie od parametru `start`, otrzymasz co najwyżej ostatnie 60 dni danych.
    Dla 3 lat historii użyj load_from_csv() z danymi MT5 / Dukascopy.
    """
    try:
        import yfinance as yf
    except ImportError:
        raise ImportError("pip install yfinance")

    end_dt    = datetime.utcnow()
    start_dt  = max(pd.Timestamp(start).to_pydatetime(), end_dt - timedelta(days=59))
    chunk_days = 55
    chunks    = []

    cur = start_dt
    while cur < end_dt:
        nxt = min(cur + timedelta(days=chunk_days), end_dt)
        try:
            chunk = yf.download(
                ticker,
                start=cur.strftime("%Y-%m-%d"),
                end=nxt.strftime("%Y-%m-%d"),
                interval="15m",
                auto_adjust=False,
                progress=False,
            )
            if chunk is not None and not chunk.empty:
                if isinstance(chunk.columns, pd.MultiIndex):
                    lvl = chunk.columns.get_level_values(-1)
                    chunk = chunk.xs(ticker, axis=1, level=-1) if ticker in lvl \
                            else chunk.droplevel(-1, axis=1)
                chunk = chunk.rename(columns=str.title)
                chunks.append(chunk)
        except Exception as e:
            print(f"    [WARN] chunk {cur.date()}–{nxt.date()}: {e}")
        cur = nxt

    if not chunks:
        raise ValueError(f"Brak danych M15 z Yahoo dla {ticker}. Użyj CSV z MT5.")

    df = pd.concat(chunks)
    df = df[~df.index.duplicated(keep="first")].sort_index()
    needed = ["Open", "High", "Low", "Close"]
    df = df[[c for c in df.columns if c in needed + ["Volume"]]].dropna(subset=needed)
    if "Volume" not in df.columns or df["Volume"].sum() == 0:
        df["Volume"] = 1.0
    return df


def load_m15(ticker: str, cfg, csv_dir: str = "data_m15") -> pd.DataFrame:
    """Wczytuje dane M15: CSV (pełna historia) lub Yahoo (ostatnie ~60 dni)."""
    meta     = FOREX_PAIRS.get(ticker, {})
    csv_name = meta.get("csv", f"{ticker.replace('=X','').replace('=','')}_M15.csv")
    csv_path = os.path.join(csv_dir, csv_name)

    if os.path.exists(csv_path):
        print(f"  [CSV] Wczytuję {csv_path}")
        df = load_from_csv(csv_path, start=cfg.start)
    else:
        print(f"  [Yahoo] Brak pliku {csv_path} — pobieram M15 z Yahoo Finance (~60 dni)")
        df = load_from_yahoo_m15(ticker, start=cfg.start)

    df.index = pd.to_datetime(df.index).tz_localize(None)
    return df


def load_from_mt5(ticker_mt5: str, start: str = "2022-01-01", timeframe_str: str = "M15") -> pd.DataFrame:
    """
    Pobiera dane bezpośrednio z MetaTrader 5 (wymaga zainstalowanego MT5 i otwartego terminala).
    pip install MetaTrader5

    ticker_mt5: symbol w MT5, np. "EURUSD", "GBPUSD"
    """
    try:
        import MetaTrader5 as mt5
    except ImportError:
        raise ImportError("pip install MetaTrader5   (wymaga MT5 na Windows)")

    if not mt5.initialize():
        raise RuntimeError(f"MT5 initialize() failed: {mt5.last_error()}")

    tf_map = {"M1": mt5.TIMEFRAME_M1, "M5": mt5.TIMEFRAME_M5,
              "M15": mt5.TIMEFRAME_M15, "M30": mt5.TIMEFRAME_M30,
              "H1": mt5.TIMEFRAME_H1,  "H4": mt5.TIMEFRAME_H4,
              "D1": mt5.TIMEFRAME_D1}
    tf = tf_map.get(timeframe_str, mt5.TIMEFRAME_M15)

    from_dt = pd.Timestamp(start).to_pydatetime()
    rates   = mt5.copy_rates_from(ticker_mt5, tf, from_dt, 500_000)
    mt5.shutdown()

    if rates is None or len(rates) == 0:
        raise ValueError(f"MT5: brak danych dla {ticker_mt5}")

    df = pd.DataFrame(rates)
    df["time"] = pd.to_datetime(df["time"], unit="s")
    df = df.set_index("time").rename(columns={
        "open": "Open", "high": "High", "low": "Low",
        "close": "Close", "tick_volume": "Volume",
    })
    df = df[["Open", "High", "Low", "Close", "Volume"]]
    df = df[df.index >= pd.Timestamp(start)]
    return df


def filter_session(df: pd.DataFrame, start_hour_utc: int = 7, end_hour_utc: int = 20) -> pd.DataFrame:
    """
    Zostawia tylko świece z aktywnych sesji handlowych (UTC).
    Domyślnie: 07:00–20:00 UTC = Londyn + Nowy Jork.
    Eliminuje niską płynność nocy i weekendy.
    """
    h = df.index.hour
    mask = (h >= start_hour_utc) & (h < end_hour_utc)
    mask &= df.index.dayofweek < 5
    return df[mask].copy()



def _ema(s: pd.Series, span: int) -> pd.Series:
    return s.ewm(span=span, adjust=False).mean()

def _rsi(close: pd.Series, period: int = 14) -> pd.Series:
    d = close.diff()
    ag = d.clip(lower=0).ewm(com=period - 1, adjust=False).mean()
    al = (-d.clip(upper=0)).ewm(com=period - 1, adjust=False).mean()
    return 100 - 100 / (1 + ag / (al + 1e-12))

def _atr(h, l, c, period=14) -> pd.Series:
    pc = c.shift(1)
    tr = pd.Series(
        np.maximum(np.maximum((h - l).abs().values, (h - pc).abs().values), (l - pc).abs().values),
        index=c.index,
    )
    return tr.ewm(com=period - 1, adjust=False).mean()

def _macd(c, fast=12, slow=26, signal=9):
    ml = _ema(c, fast) - _ema(c, slow)
    sl = _ema(ml, signal)
    return ml, sl, ml - sl

def _bollinger(c, period=20, n_std=2.0):
    ma = c.rolling(period).mean()
    sd = c.rolling(period).std()
    upper = ma + n_std * sd
    lower = ma - n_std * sd
    return upper, lower, (upper - lower) / (ma + 1e-12)

def _stochastic(h, l, c, k=14, d=3):
    lo = l.rolling(k).min()
    hi = h.rolling(k).max()
    kp = 100 * (c - lo) / (hi - lo + 1e-12)
    return kp, kp.rolling(d).mean()

def _cci(h, l, c, period=20) -> pd.Series:
    tp  = (h + l + c) / 3
    ma  = tp.rolling(period).mean()
    mad = tp.rolling(period).apply(lambda x: np.abs(x - x.mean()).mean(), raw=True)
    return (tp - ma) / (0.015 * mad + 1e-12)

def _adx(h, l, c, period=14):
    ph, pl, pc = h.shift(1), l.shift(1), c.shift(1)
    pdm = (h - ph).clip(lower=0).where((h - ph) > (pl - l), 0)
    mdm = (pl - l).clip(lower=0).where((pl - l) > (h - ph), 0)
    tr  = pd.Series(
        np.maximum(np.maximum((h - l).abs().values, (h - pc).abs().values), (l - pc).abs().values),
        index=c.index,
    )
    ts  = tr.ewm(com=period - 1, adjust=False).mean()
    pdi = 100 * pdm.ewm(com=period - 1, adjust=False).mean() / (ts + 1e-12)
    mdi = 100 * mdm.ewm(com=period - 1, adjust=False).mean() / (ts + 1e-12)
    dx  = 100 * (pdi - mdi).abs() / (pdi + mdi + 1e-12)
    return dx.ewm(com=period - 1, adjust=False).mean(), pdi, mdi


def make_features_m15(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    c   = out["Close"].squeeze()
    h   = out["High"].squeeze()
    l   = out["Low"].squeeze()
    o   = out["Open"].squeeze()

    for p in [1, 4, 8, 20, 48, 96]:
        out[f"ret_{p}"] = c.pct_change(p)

    for w in [20, 50, 96, 200, 384]:
        sma = c.rolling(w).mean()
        out[f"sma_{w}"]      = sma
        out[f"dist_sma_{w}"] = (c - sma) / (sma + 1e-12)

    for e in [8, 21, 55, 144]:
        em = _ema(c, e)
        out[f"ema_{e}"]      = em
        out[f"dist_ema_{e}"] = (c - em) / (em + 1e-12)

    for p in [9, 14, 21]:
        out[f"rsi_{p}"] = _rsi(c, p)

    ml, sl, hist = _macd(c)
    out["macd_line"]     = ml
    out["macd_signal"]   = sl
    out["macd_hist"]     = hist
    out["macd_hist_chg"] = hist.diff()

    out["atr_14"]  = _atr(h, l, c, 14)
    out["atr_pct"] = out["atr_14"] / (c + 1e-12)
    for w in [20, 96]:
        out[f"vol_{w}"] = out["ret_1"].rolling(w).std()

    bb_u, bb_l, bb_w = _bollinger(c, 20)
    out["bb_upper"] = bb_u
    out["bb_lower"] = bb_l
    out["bb_width"] = bb_w
    out["bb_pos"]   = (c - bb_l) / ((bb_u - bb_l) + 1e-12)

    kp, dp = _stochastic(h, l, c)
    out["stoch_k"]  = kp
    out["stoch_d"]  = dp
    out["stoch_kd"] = kp - dp

    out["cci_20"] = _cci(h, l, c, 20)

    adx_v, pdi, mdi = _adx(h, l, c, 14)
    out["adx"]      = adx_v
    out["plus_di"]  = pdi
    out["minus_di"] = mdi
    out["di_diff"]  = pdi - mdi

    out["hl_range"]      = (h - l) / (c + 1e-12)
    out["co_range"]      = (c - o) / (o + 1e-12)
    out["upper_shadow"]  = (h - c.clip(lower=o))  / (h - l + 1e-12)
    out["lower_shadow"]  = (c.clip(upper=o) - l)   / (h - l + 1e-12)

    for p in [20, 96, 384]:
        out[f"mom_{p}"] = c / c.shift(p) - 1

    out["trend_20_50"]   = (out["sma_20"]  > out["sma_50"]).astype(float)
    out["trend_50_200"]  = (out["sma_50"]  > out["sma_200"]).astype(float)
    out["trend_200_384"] = (out["sma_200"] > out["sma_384"]).astype(float)

    for w in [20, 96]:
        hmax = h.rolling(w).max()
        lmin = l.rolling(w).min()
        out[f"pct_from_high_{w}"] = (c - hmax) / (hmax + 1e-12)
        out[f"pct_from_low_{w}"]  = (c - lmin) / (lmin + 1e-12)

    minute_rad = 2 * np.pi * (df.index.hour * 60 + df.index.minute) / (24 * 60)
    dow_rad    = 2 * np.pi * df.index.dayofweek / 5

    out["min_sin"] = np.sin(minute_rad)
    out["min_cos"] = np.cos(minute_rad)
    out["dow_sin"] = np.sin(dow_rad)
    out["dow_cos"] = np.cos(dow_rad)

    h_utc = df.index.hour
    out["session_london"] = ((h_utc >= 7)  & (h_utc < 16)).astype(float)
    out["session_ny"]     = ((h_utc >= 13) & (h_utc < 22)).astype(float)
    out["session_asia"]   = ((h_utc >= 0)  & (h_utc < 8)).astype(float)
    out["session_overlap"]= ((h_utc >= 13) & (h_utc < 16)).astype(float)  # London+NY

    out = out.replace([np.inf, -np.inf], np.nan)
    return out

def make_direction_label_m15(df_feat: pd.DataFrame, horizon: int = 20, thr: float = 0.0003):
    """
    horizon: liczba świec M15 do przodu (domyślnie 20 = 5 godzin)
    thr: próg zwrotu (0.0003 = 0.03% ≈ 3 pipy EURUSD)
    """
    c       = df_feat["Close"].squeeze()
    fwd_ret = c.shift(-horizon) / c - 1
    y       = pd.Series(0, index=df_feat.index, dtype=int)
    y[fwd_ret >  thr] =  1
    y[fwd_ret < -thr] = -1
    return y, fwd_ret

def build_models_m15(random_state: int = 42) -> Dict:
    from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
    from sklearn.neural_network import MLPClassifier

    models = {
        "RandomForest": RandomForestClassifier(
            n_estimators=300, max_depth=10, min_samples_leaf=20,
            class_weight="balanced_subsample", random_state=random_state, n_jobs=-1,
        ),
        "GradientBoosting": GradientBoostingClassifier(
            n_estimators=150, max_depth=5, learning_rate=0.05,
            subsample=0.8, random_state=random_state,
        ),
        "NeuralNet": MLPClassifier(
            hidden_layer_sizes=(128, 64, 32), activation="relu",
            learning_rate_init=0.001, max_iter=300,
            early_stopping=True, validation_fraction=0.1,
            random_state=random_state,
        ),
    }
    try:
        from xgboost import XGBClassifier
        models["XGBoost"] = XGBClassifier(
            n_estimators=200, max_depth=6, learning_rate=0.05,
            subsample=0.8, colsample_bytree=0.8,
            eval_metric="mlogloss", random_state=random_state,
            n_jobs=-1, verbosity=0,
        )
        print("  [INFO] XGBoost OK")
    except ImportError:
        print("  [INFO] XGBoost nie zainstalowany — pomijam")
    return models



@dataclass
class ForexM15Config:
    train_bars:     int   = 16_000
    test_bars:      int   = 2_880   
    horizon:        int   = 20      
    direction_thr:  float = 0.0003  
    conf_threshold: float = 0.45
    spread_pips:    float = 1.0
    pip_size:       float = 0.0001
    use_sl_tp:      bool  = True
    sltp_mode:      str   = "atr"
    atr_sl_mult:    float = 1.5
    atr_tp_mult:    float = 2.5
    fixed_sl_pct:   float = 0.003
    fixed_tp_pct:   float = 0.006
    filter_session: bool  = True
    session_start:  int   = 7
    session_end:    int   = 20
    random_state:   int   = 42
    start:          str   = "2022-01-01"
    bars_per_year:  int   = BARS_PER_YEAR_M15

_DROP_M15 = [
    "Open", "High", "Low", "Close", "Volume",
    "sma_20", "sma_50", "sma_96", "sma_200", "sma_384",
    "ema_8", "ema_21", "ema_55", "ema_144",
    "bb_upper", "bb_lower", "atr_14",
    "macd_line", "macd_signal",
]


def walk_forward_m15(df: pd.DataFrame, cfg: ForexM15Config):
    from sklearn.preprocessing import StandardScaler

    feat       = make_features_m15(df)
    y_dir, _   = make_direction_label_m15(feat, horizon=cfg.horizon, thr=cfg.direction_thr)
    X_cols     = [c for c in feat.columns if c not in _DROP_M15]
    data       = pd.concat(
        [df[["Close", "High", "Low"]], feat[X_cols], y_dir.rename("y")], axis=1
    ).dropna()

    n = len(data)
    min_needed = cfg.train_bars + cfg.test_bars
    if n < min_needed:
        raise ValueError(
            f"Za mało danych: {n} świec M15, potrzeba ≥ {min_needed} "
            f"(train_bars={cfg.train_bars} + test_bars={cfg.test_bars}).\n"
            f"Yahoo Finance zwraca ~4 100 świec (60 dni). "
            f"Dla 3 lat historii użyj CSV z MT5 — patrz nagłówek pliku."
        )

    models_reg = build_models_m15(cfg.random_state)
    model_names = list(models_reg.keys())

    pred_all = {m: pd.Series(np.nan, index=data.index) for m in model_names}
    pup_all  = {m: pd.Series(np.nan, index=data.index) for m in model_names}
    pdn_all  = {m: pd.Series(np.nan, index=data.index) for m in model_names}
    windows  = []

    start_idx = 0
    win_num   = 0

    while start_idx + cfg.train_bars + cfg.test_bars <= n:
        tr_end = start_idx + cfg.train_bars
        te_end = tr_end + cfg.test_bars

        safe_end = max(tr_end - cfg.horizon, start_idx + 50)
        train_fit = data.iloc[start_idx:safe_end]
        test      = data.iloc[tr_end:te_end]

        if len(train_fit) < 200 or len(test) < 50:
            start_idx += cfg.test_bars
            continue

        X_tr = train_fit.drop(columns=["Close", "High", "Low", "y"])
        y_tr = train_fit["y"].astype(int)
        X_te = test.drop(columns=["Close", "High", "Low", "y"])

        scaler = StandardScaler()
        X_tr_s = scaler.fit_transform(X_tr)
        X_te_s = scaler.transform(X_te)

        win = {
            "window":      win_num,
            "train_start": str(data.index[start_idx].date()),
            "train_end":   str(data.index[safe_end - 1].date()),
            "test_start":  str(data.index[tr_end].date()),
            "test_end":    str(data.index[te_end - 1].date()),
            "n_train":     len(train_fit),
            "n_test":      len(test),
        }

        for mname, proto in models_reg.items():
            model = copy.deepcopy(proto)
            try:
                y_fit = (y_tr + 1) if mname == "XGBoost" else y_tr
                model.fit(X_tr_s, y_fit)
                proba   = model.predict_proba(X_te_s)
                classes = ([c - 1 for c in model.classes_] if mname == "XGBoost"
                           else list(model.classes_))
                p_up = proba[:, classes.index( 1)] if  1 in classes else np.zeros(len(test))
                p_dn = proba[:, classes.index(-1)] if -1 in classes else np.zeros(len(test))
                pred = np.where(p_up > cfg.conf_threshold,  1,
                       np.where(p_dn > cfg.conf_threshold, -1, 0))
                pred_all[mname].loc[test.index] = pred
                pup_all[mname].loc[test.index]  = p_up
                pdn_all[mname].loc[test.index]  = p_dn
                win[f"{mname}_acc"] = float((pred == test["y"].values).mean())
            except Exception as e:
                print(f"    [WARN] {mname}: {e}")

        windows.append(win)
        start_idx += cfg.test_bars
        win_num   += 1

    if not windows:
        raise ValueError("Żadne okno walk-forward nie zostało przetworzone.")

    avg_up = pd.concat([pup_all[m] for m in model_names], axis=1).mean(axis=1)
    avg_dn = pd.concat([pdn_all[m] for m in model_names], axis=1).mean(axis=1)
    ens = pd.Series(0, index=data.index, dtype=int)
    ens[avg_up > cfg.conf_threshold] =  1
    ens[avg_dn > cfg.conf_threshold] = -1

    all_names = model_names + ["Ensemble"]
    pred_all["Ensemble"] = ens.reindex(df.index).fillna(0)
    pup_all["Ensemble"]  = avg_up.reindex(df.index).fillna(0)
    pdn_all["Ensemble"]  = avg_dn.reindex(df.index).fillna(0)
    for m in model_names:
        pred_all[m] = pred_all[m].reindex(df.index).fillna(0)
        pup_all[m]  = pup_all[m].reindex(df.index).fillna(0)
        pdn_all[m]  = pdn_all[m].reindex(df.index).fillna(0)

    print(f"  Walk-forward: {len(windows)} okien | "
          f"train≈{cfg.train_bars} świec | test≈{cfg.test_bars} świec")
    return pred_all, pup_all, pdn_all, pd.DataFrame(windows), feat.reindex(df.index), all_names


def backtest_m15(df: pd.DataFrame, signal: pd.Series, cfg: ForexM15Config, feat: pd.DataFrame):
    idx   = df.index
    c     = df["Close"].astype(float).reindex(idx)
    h     = df["High"].astype(float).reindex(idx)
    l     = df["Low"].astype(float).reindex(idx)
    atr14 = feat["atr_14"].astype(float).reindex(idx)

    desired = signal.reindex(idx).fillna(0).shift(1).fillna(0).astype(int)
    spread_cost = (cfg.spread_pips * cfg.pip_size) / max(c.mean(), 1e-6)

    in_pos = 0
    entry_price = sl_price = tp_price = np.nan
    pos       = pd.Series(0, index=idx, dtype=int)
    strat_ret = pd.Series(0.0, index=idx, dtype=float)

    for i in range(1, len(idx)):
        yest    = idx[i - 1]
        new_sig = int(desired.iloc[i])
        pos.iloc[i] = in_pos

        if in_pos == 0 and new_sig != 0:
            in_pos = new_sig
            entry_price = c.iloc[i]
            a = float(atr14.iloc[i]) if not np.isnan(atr14.iloc[i]) else c.iloc[i] * 0.0005
            if cfg.use_sl_tp:
                if cfg.sltp_mode == "atr":
                    sl_price = entry_price - in_pos * cfg.atr_sl_mult * a
                    tp_price = entry_price + in_pos * cfg.atr_tp_mult * a
                else:
                    sl_price = entry_price * (1 - in_pos * cfg.fixed_sl_pct)
                    tp_price = entry_price * (1 + in_pos * cfg.fixed_tp_pct)
            pos.iloc[i] = in_pos

        if in_pos != 0:
            exit_now   = False
            exit_price = np.nan
            if cfg.use_sl_tp and not np.isnan(sl_price):
                if in_pos == 1:
                    if l.iloc[i] <= sl_price:
                        exit_now, exit_price = True, sl_price
                    elif h.iloc[i] >= tp_price:
                        exit_now, exit_price = True, tp_price
                else:
                    if h.iloc[i] >= sl_price:
                        exit_now, exit_price = True, sl_price
                    elif l.iloc[i] <= tp_price:
                        exit_now, exit_price = True, tp_price
            if not exit_now and new_sig != in_pos:
                exit_now, exit_price = True, c.iloc[i]

            if exit_now:
                strat_ret.iloc[i] = in_pos * (exit_price / c.loc[yest] - 1) - spread_cost
                in_pos = 0
                entry_price = sl_price = tp_price = np.nan
                pos.iloc[i] = 0
            else:
                strat_ret.iloc[i] = in_pos * (c.iloc[i] / c.loc[yest] - 1)
                pos.iloc[i] = in_pos

    eq = (1 + strat_ret.fillna(0)).cumprod()
    return strat_ret, eq, pos


def perf_stats_m15(ret: pd.Series, bars_per_year: int = BARS_PER_YEAR_M15) -> Dict:
    ret = ret.dropna()
    if len(ret) < 5:
        return {"CAGR": np.nan, "Sharpe": np.nan, "Sortino": np.nan,
                "MaxDD": np.nan, "MaxDD_Bars": 0, "HitRate": np.nan,
                "ProfitFactor": np.nan, "Trades": 0}
    eq      = (1 + ret).cumprod()
    cagr    = eq.iloc[-1] ** (bars_per_year / len(ret)) - 1
    sharpe  = np.sqrt(bars_per_year) * ret.mean() / (ret.std() + 1e-12)
    dn_std  = ret[ret < 0].std()
    sortino = np.sqrt(bars_per_year) * ret.mean() / (dn_std + 1e-12)
    peak    = eq.cummax()
    dd      = (eq / peak) - 1
    maxdd   = dd.min()
    in_dd   = dd < 0
    grp     = (in_dd != in_dd.shift()).cumsum()
    dd_lens = in_dd[in_dd].groupby(grp[in_dd]).count()
    maxdd_bars = int(dd_lens.max()) if len(dd_lens) > 0 else 0
    wins = ret[ret > 0].sum()
    loss = ret[ret < 0].abs().sum()
    return {
        "CAGR":         float(cagr),
        "Sharpe":       float(sharpe),
        "Sortino":      float(sortino),
        "MaxDD":        float(maxdd),
        "MaxDD_Bars":   maxdd_bars,
        "HitRate":      float((ret > 0).mean()),
        "ProfitFactor": float(wins / (loss + 1e-12)),
        "Trades":       int((ret != 0).sum()),
    }



def get_feature_importance_m15(df: pd.DataFrame, cfg: ForexM15Config) -> Optional[pd.Series]:
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.preprocessing import StandardScaler

    feat  = make_features_m15(df)
    y, _  = make_direction_label_m15(feat, horizon=cfg.horizon, thr=cfg.direction_thr)
    X_cols = [c for c in feat.columns if c not in _DROP_M15]
    data  = pd.concat([feat[X_cols], y.rename("y")], axis=1).dropna()
    if len(data) < 500:
        return None
    X_d = data.drop(columns=["y"])
    y_d = data["y"].astype(int)
    scaler = StandardScaler()
    rf = RandomForestClassifier(n_estimators=150, max_depth=8,
                                 random_state=cfg.random_state, n_jobs=-1)
    rf.fit(scaler.fit_transform(X_d), y_d)
    return pd.Series(rf.feature_importances_, index=X_d.columns).sort_values(ascending=False)


def _safe_fn(ticker: str) -> str:
    return ticker.replace("=", "").replace("/", "_").replace("\\", "_").replace(".", "_")

def _fmt_dates(ax, n=6):
    ax.xaxis.set_major_locator(mdates.AutoDateLocator(minticks=4, maxticks=n))
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%d/%m'%y"))
    plt.setp(ax.get_xticklabels(), rotation=30, ha="right", fontsize=7)


def plot_m15_dashboard(
    ticker: str, df: pd.DataFrame, feat: pd.DataFrame,
    signals: Dict, probs_up: Dict, probs_dn: Dict,
    equities: Dict, stats_all: Dict,
    cfg: ForexM15Config, out_dir: str,
    last_n_bars: int = 2000,
):
    name = FOREX_PAIRS.get(ticker, {}).get("name", ticker)
    c    = df["Close"].squeeze()

    fig  = plt.figure(figsize=(22, 18))
    fig.suptitle(f"Forex M15 Dashboard — {name}", fontsize=14, fontweight="bold", y=0.99)
    gs   = gridspec.GridSpec(5, 2, figure=fig, hspace=0.50, wspace=0.32,
                              height_ratios=[2.5, 1.0, 1.0, 2.0, 1.5])

    ax_price  = fig.add_subplot(gs[0, :])
    ax_rsi    = fig.add_subplot(gs[1, 0])
    ax_macd   = fig.add_subplot(gs[1, 1])
    ax_stoch  = fig.add_subplot(gs[2, 0])
    ax_adx    = fig.add_subplot(gs[2, 1])
    ax_eq     = fig.add_subplot(gs[3, :])
    ax_prob   = fig.add_subplot(gs[4, 0])
    ax_dd     = fig.add_subplot(gs[4, 1])

    df_plot  = df.tail(last_n_bars)
    feat_plt = feat.reindex(df_plot.index)
    c_plt    = df_plot["Close"].squeeze()

    ax_price.plot(c_plt.index, c_plt.values, color="#212121", lw=0.7, label="Close")
    for w, col, ls in [(20, "#1565C0", "--"), (96, "#E65100", "--"), (200, "#1B5E20", "-.")]:
        fc = f"sma_{w}"
        if fc in feat_plt.columns:
            ax_price.plot(feat_plt.index, feat_plt[fc].values,
                          color=col, lw=0.7, linestyle=ls, alpha=0.8, label=f"SMA{w}")
    if "bb_upper" in feat_plt.columns:
        ax_price.fill_between(feat_plt.index, feat_plt["bb_lower"], feat_plt["bb_upper"],
                               alpha=0.07, color="#757575", label="BB(20,2)")

    ens_sig  = signals.get("Ensemble", pd.Series(0, index=c.index)).reindex(c_plt.index).fillna(0).astype(int)
    prev     = ens_sig.shift(1).fillna(0).astype(int)
    ax_price.scatter(c_plt.index[(prev == 0) & (ens_sig ==  1)],
                     c_plt[(prev == 0) & (ens_sig ==  1)],
                     marker="^", s=40, color="#00C853", zorder=5, label="Long")
    ax_price.scatter(c_plt.index[(prev == 0) & (ens_sig == -1)],
                     c_plt[(prev == 0) & (ens_sig == -1)],
                     marker="v", s=40, color="#D50000", zorder=5, label="Short")
    ax_price.scatter(c_plt.index[(prev != 0) & (ens_sig == 0)],
                     c_plt[(prev != 0) & (ens_sig == 0)],
                     marker="o", s=20, color="#FF6F00", zorder=5, alpha=0.6, label="Exit")
    ax_price.set_title(f"Cena + MAs + Sygnały Ensemble (ostatnie {last_n_bars} świec M15)", fontsize=9, loc="left")
    ax_price.legend(fontsize=7, ncol=5, loc="upper left")
    ax_price.set_ylabel("Price", fontsize=8)
    _fmt_dates(ax_price)

    if "rsi_14" in feat_plt.columns:
        r = feat_plt["rsi_14"]
        ax_rsi.plot(r.index, r.values, color="#6A1B9A", lw=0.8)
        ax_rsi.axhline(70, color="#C62828", lw=0.6, linestyle="--")
        ax_rsi.axhline(30, color="#2E7D32", lw=0.6, linestyle="--")
        ax_rsi.fill_between(r.index, 70, r.values.clip(70),       alpha=0.2, color="#EF5350")
        ax_rsi.fill_between(r.index, 30, r.values.clip(None, 30), alpha=0.2, color="#66BB6A")
        ax_rsi.set_ylim(0, 100); ax_rsi.set_title("RSI(14)", fontsize=9, loc="left")
        _fmt_dates(ax_rsi)

    if "macd_hist" in feat_plt.columns:
        mh   = feat_plt["macd_hist"]
        ml_s = feat_plt["macd_line"]
        ms   = feat_plt["macd_signal"]
        bar_c = ["#43A047" if v >= 0 else "#E53935" for v in mh.values]
        ax_macd.bar(mh.index, mh.values, color=bar_c, alpha=0.55, width=0.0005)
        ax_macd.plot(ml_s.index, ml_s.values, color="#1565C0", lw=0.7, label="MACD")
        ax_macd.plot(ms.index,   ms.values,   color="#E64A19", lw=0.7, label="Signal")
        ax_macd.axhline(0, color="black", lw=0.4)
        ax_macd.set_title("MACD", fontsize=9, loc="left"); ax_macd.legend(fontsize=7)
        _fmt_dates(ax_macd)

    if "stoch_k" in feat_plt.columns:
        ax_stoch.plot(feat_plt.index, feat_plt["stoch_k"].values, color="#1E88E5", lw=0.8, label="%K")
        ax_stoch.plot(feat_plt.index, feat_plt["stoch_d"].values, color="#E53935", lw=0.8, label="%D")
        ax_stoch.axhline(80, color="red",   lw=0.5, linestyle="--")
        ax_stoch.axhline(20, color="green", lw=0.5, linestyle="--")
        ax_stoch.set_ylim(0, 100); ax_stoch.set_title("Stochastic", fontsize=9, loc="left")
        ax_stoch.legend(fontsize=7); _fmt_dates(ax_stoch)

    if "adx" in feat_plt.columns:
        ax_adx.plot(feat_plt.index, feat_plt["adx"].values,      color="#5E35B1", lw=0.9, label="ADX")
        ax_adx.plot(feat_plt.index, feat_plt["plus_di"].values,  color="#43A047", lw=0.7, label="+DI")
        ax_adx.plot(feat_plt.index, feat_plt["minus_di"].values, color="#E53935", lw=0.7, label="-DI")
        ax_adx.axhline(25, color="black", lw=0.5, linestyle="--", alpha=0.5)
        ax_adx.set_title("ADX", fontsize=9, loc="left"); ax_adx.legend(fontsize=7)
        _fmt_dates(ax_adx)

    bh_eq = (1 + c.pct_change().fillna(0)).cumprod()
    ax_eq.plot(bh_eq.index, bh_eq.values, color=MODEL_COLORS["BuyHold"],
               lw=1.0, linestyle="--", alpha=0.7, label="Buy&Hold")
    for mname, eq_s in equities.items():
        st    = stats_all.get(mname, {})
        label = f"{mname} (Sh={st.get('Sharpe', np.nan):.2f}, CAGR={st.get('CAGR', np.nan)*100:.1f}%)"
        ax_eq.plot(eq_s.index, eq_s.values, color=MODEL_COLORS.get(mname, "#555"),
                   lw=1.1, label=label)
    ax_eq.axhline(1.0, color="black", lw=0.4)
    ax_eq.set_title("Equity Curves — wszystkie modele (pełna historia)", fontsize=9, loc="left")
    ax_eq.set_ylabel("Equity (start=1)", fontsize=8)
    ax_eq.legend(fontsize=7, loc="upper left"); _fmt_dates(ax_eq)

    pu = probs_up.get("Ensemble", pd.Series(0.0, index=c.index)).reindex(c.index).fillna(0)
    pd_ = probs_dn.get("Ensemble", pd.Series(0.0, index=c.index)).reindex(c.index).fillna(0)
    ax_prob.stackplot(pu.index, pu.values, pd_.values,
                      labels=["P(UP)", "P(DOWN)"],
                      colors=["#4CAF50", "#F44336"], alpha=0.65)
    ax_prob.axhline(cfg.conf_threshold, color="black", lw=0.7, linestyle="--",
                    label=f"Próg={cfg.conf_threshold}")
    ax_prob.set_ylim(0, 1); ax_prob.set_title("Ensemble ML Confidence", fontsize=9, loc="left")
    ax_prob.legend(fontsize=7); _fmt_dates(ax_prob)

    ens_eq = equities.get("Ensemble")
    if ens_eq is not None:
        peak = ens_eq.cummax(); dd_s = (ens_eq / peak) - 1
        ax_dd.fill_between(dd_s.index, dd_s.values, 0, color="#EF5350", alpha=0.65)
        ax_dd.plot(dd_s.index, dd_s.values, color="#B71C1C", lw=0.7)
        ax_dd.set_title("Ensemble Drawdown", fontsize=9, loc="left"); _fmt_dates(ax_dd)

    path = os.path.join(out_dir, "plots", f"{_safe_fn(ticker)}_m15_dashboard.png")
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"    [chart] {path}")


def plot_session_analysis(ticker: str, ret: pd.Series, out_dir: str):
    r = ret[ret != 0].copy()
    if len(r) < 100:
        return

    fig, axes = plt.subplots(1, 2, figsize=(16, 5))
    name = FOREX_PAIRS.get(ticker, {}).get("name", ticker)
    fig.suptitle(f"Analiza Sesji M15 — {name} (Ensemble)", fontsize=12, fontweight="bold")

    ax1 = axes[0]
    by_hour = r.groupby(r.index.hour).mean() * 1e4  # w pipsach (×10000)
    colors  = ["#43A047" if v > 0 else "#E53935" for v in by_hour.values]
    ax1.bar(by_hour.index, by_hour.values, color=colors, edgecolor="white")
    ax1.axhline(0, color="black", lw=0.6)
    ax1.set_xlabel("Godzina UTC")
    ax1.set_ylabel("Śr. zwrot (×10⁴)")
    ax1.set_title("Zwroty wg godziny UTC (aktywność sesji)")
    ax1.set_xticks(range(0, 24, 2))

    ax2 = axes[1]
    days    = ["Pon", "Wt", "Śr", "Czw", "Pt"]
    by_dow  = r.groupby(r.index.dayofweek).mean() * 1e4
    by_dow  = by_dow.reindex(range(5)).fillna(0)
    colors2 = ["#43A047" if v > 0 else "#E53935" for v in by_dow.values]
    ax2.bar(range(5), by_dow.values, color=colors2, edgecolor="white")
    ax2.set_xticks(range(5)); ax2.set_xticklabels(days)
    ax2.axhline(0, color="black", lw=0.6)
    ax2.set_xlabel("Dzień tygodnia")
    ax2.set_ylabel("Śr. zwrot (×10⁴)")
    ax2.set_title("Zwroty wg dnia tygodnia")

    fig.tight_layout()
    path = os.path.join(out_dir, "plots", f"{_safe_fn(ticker)}_m15_session_analysis.png")
    fig.savefig(path, dpi=150); plt.close(fig)
    print(f"    [chart] {path}")


def plot_monthly_heatmap_m15(ticker: str, ret: pd.Series, out_dir: str):
    monthly = ret.resample("ME").apply(lambda x: (1 + x).prod() - 1)
    df_m    = monthly.to_frame("ret")
    df_m["year"]  = df_m.index.year
    df_m["month"] = df_m.index.month
    pivot = df_m.pivot(index="year", columns="month", values="ret")
    pivot.columns = ["Jan","Feb","Mar","Apr","May","Jun","Jul","Aug","Sep","Oct","Nov","Dec"]

    vals = pivot.values.astype(float)
    vmax = max(abs(vals[~np.isnan(vals)]).max(), 0.005) if np.any(~np.isnan(vals)) else 0.02

    fig, ax = plt.subplots(figsize=(15, max(3, len(pivot) * 0.75 + 1.5)))
    im = ax.imshow(vals, aspect="auto", cmap="RdYlGn", vmin=-vmax, vmax=vmax)
    plt.colorbar(im, ax=ax, fraction=0.015, label="Miesięczny zwrot")
    ax.set_xticks(range(12)); ax.set_xticklabels(pivot.columns, fontsize=9)
    ax.set_yticks(range(len(pivot))); ax.set_yticklabels(pivot.index.tolist(), fontsize=9)
    for i in range(len(pivot)):
        for j in range(12):
            v = vals[i, j]
            if not np.isnan(v):
                ax.text(j, i, f"{v*100:.1f}%", ha="center", va="center", fontsize=7)
    name = FOREX_PAIRS.get(ticker, {}).get("name", ticker)
    ax.set_title(f"Miesięczne zwroty — {name} M15 (Ensemble)", fontsize=11)
    fig.tight_layout()
    path = os.path.join(out_dir, "plots", f"{_safe_fn(ticker)}_m15_monthly_heatmap.png")
    fig.savefig(path, dpi=150); plt.close(fig)
    print(f"    [chart] {path}")


def plot_feature_importance_m15(ticker: str, fi: pd.Series, out_dir: str, top_n: int = 30):
    top    = fi.head(top_n)
    colors = plt.cm.RdYlGn(np.linspace(0.25, 0.85, len(top)))[::-1]
    fig, ax = plt.subplots(figsize=(11, 9))
    ax.barh(range(len(top)), top.values[::-1], color=colors[::-1], edgecolor="white")
    ax.set_yticks(range(len(top)))
    ax.set_yticklabels(top.index[::-1], fontsize=8)
    ax.set_xlabel("Feature Importance (RF)", fontsize=9)
    name = FOREX_PAIRS.get(ticker, {}).get("name", ticker)
    ax.set_title(f"Top {top_n} Features — {name} M15", fontsize=11)
    fig.tight_layout()
    path = os.path.join(out_dir, "plots", f"{_safe_fn(ticker)}_m15_feature_importance.png")
    fig.savefig(path, dpi=150); plt.close(fig)
    print(f"    [chart] {path}")


def plot_summary_m15(summary_df: pd.DataFrame, out_dir: str):
    ens = summary_df[summary_df["Model"] == "Ensemble"].copy()
    if ens.empty:
        return
    pairs  = ens["Pair"].tolist()
    labels = [FOREX_PAIRS.get(p, {}).get("name", p) for p in pairs]
    x      = np.arange(len(pairs))

    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    fig.suptitle("Forex M15 — Podsumowanie (Ensemble)", fontsize=12, fontweight="bold")

    for ax, metric, title in zip(axes,
            ["Sharpe", "CAGR", "MaxDD"],
            ["Sharpe Ratio", "CAGR", "Max Drawdown"]):
        vals   = ens[metric].fillna(0).values
        if metric == "CAGR":   vals = vals * 100
        if metric == "MaxDD":  vals = vals * 100
        colors = ["#43A047" if v > 0 else "#E53935" for v in vals]
        ax.bar(x, vals, color=colors, edgecolor="white")
        ax.set_xticks(x); ax.set_xticklabels(labels, rotation=35, ha="right", fontsize=8)
        ax.axhline(0, color="black", lw=0.5)
        ax.set_title(title, fontsize=10)
        for bar, v in zip(ax.patches, vals):
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height(),
                    f"{v:.2f}", ha="center", va="bottom", fontsize=8)

    fig.tight_layout()
    path = os.path.join(out_dir, "plots", "m15_summary.png")
    fig.savefig(path, dpi=150); plt.close(fig)
    print(f"    [chart] {path}")


def run_forex_m15(
    tickers:  Optional[List[str]] = None,
    cfg:      Optional[ForexM15Config] = None,
    out_dir:  str = "outputs_forex_m15",
    csv_dir:  str = "data_m15",
):
    if cfg     is None: cfg     = ForexM15Config()
    if tickers is None: tickers = list(FOREX_PAIRS.keys())

    for d in [out_dir, f"{out_dir}/plots", f"{out_dir}/signals"]:
        os.makedirs(d, exist_ok=True)

    summary_rows = []

    for ticker in tickers:
        meta  = FOREX_PAIRS.get(ticker, {})
        name  = meta.get("name", ticker)
        print(f"\n{'='*60}")
        print(f"  {name}  ({ticker})  [M15]")
        print(f"{'='*60}")

        pair_cfg             = copy.copy(cfg)
        pair_cfg.spread_pips = meta.get("spread_pips", cfg.spread_pips)
        pair_cfg.pip_size    = meta.get("pip",         cfg.pip_size)

        try:
            df = load_m15(ticker, pair_cfg, csv_dir=csv_dir)
            if pair_cfg.filter_session:
                df = filter_session(df, pair_cfg.session_start, pair_cfg.session_end)
            print(f"  Dane: {len(df):,} świec M15 | "
                  f"{df.index[0].strftime('%Y-%m-%d %H:%M')} → "
                  f"{df.index[-1].strftime('%Y-%m-%d %H:%M')}")
        except Exception as e:
            print(f"  [SKIP] {e}"); continue

        try:
            pred_all, pup_all, pdn_all, wf_table, feat, model_names = \
                walk_forward_m15(df, pair_cfg)
        except Exception as e:
            print(f"  [ERROR] Walk-forward: {e}"); continue

        try:
            fi = get_feature_importance_m15(df, pair_cfg)
            if fi is not None:
                plot_feature_importance_m15(ticker, fi, out_dir)
        except Exception as e:
            print(f"  [WARN] Feature importance: {e}")

        equities  = {}
        stats_all = {}

        for mname in model_names:
            sig = pred_all[mname].reindex(df.index).fillna(0).astype(int)
            try:
                s_ret, eq, pos = backtest_m15(df, sig, pair_cfg, feat)
                st = perf_stats_m15(s_ret, pair_cfg.bars_per_year)
                equities[mname]  = eq
                stats_all[mname] = st
                summary_rows.append({
                    "Pair": ticker, "Name": name, "Model": mname, **st,
                    "Spread_Pips": pair_cfg.spread_pips,
                    "TradeRate":   float((pos != 0).mean()),
                })
                print(f"  {mname:20s} | CAGR={st['CAGR']:+.3f} | Sh={st['Sharpe']:.2f} "
                      f"| So={st['Sortino']:.2f} | MaxDD={st['MaxDD']:.3f} "
                      f"| Trades={st['Trades']:,}")
            except Exception as e:
                print(f"  [WARN] {mname}: {e}")

        bh_ret   = df["Close"].pct_change().fillna(0)
        bh_stats = perf_stats_m15(bh_ret, pair_cfg.bars_per_year)
        print(f"  {'Buy&Hold':20s} | CAGR={bh_stats['CAGR']:+.3f} | Sh={bh_stats['Sharpe']:.2f} "
              f"| MaxDD={bh_stats['MaxDD']:.3f}")

        sig_df = df[["Close", "High", "Low"]].copy()
        for mname in model_names:
            sig_df[f"signal_{mname}"] = pred_all[mname].reindex(df.index).fillna(0)
            sig_df[f"prob_up_{mname}"] = pup_all[mname].reindex(df.index).fillna(0)
        sig_df.to_csv(f"{out_dir}/signals/{_safe_fn(ticker)}_m15_signals.csv")
        wf_table.to_csv(f"{out_dir}/{_safe_fn(ticker)}_m15_walkforward.csv", index=False)

        try:
            plot_m15_dashboard(ticker, df, feat, pred_all, pup_all, pdn_all,
                               equities, stats_all, pair_cfg, out_dir)
        except Exception as e:
            print(f"  [WARN] Dashboard: {e}")

        try:
            ens_eq = equities.get("Ensemble")
            if ens_eq is not None:
                ens_ret = ens_eq.pct_change().fillna(0)
                plot_session_analysis(ticker, ens_ret, out_dir)
                plot_monthly_heatmap_m15(ticker, ens_ret, out_dir)
        except Exception as e:
            print(f"  [WARN] Heatmap/Session: {e}")

    if not summary_rows:
        print("\n[ERROR] Brak wyników.")
        return []

    summary = pd.DataFrame(summary_rows)
    summary.to_csv(f"{out_dir}/m15_summary_results.csv", index=False)

    print(f"\n{'='*70}")
    print("ENSEMBLE M15 — WYNIKI")
    print(f"{'='*70}")
    ens_df = summary[summary["Model"] == "Ensemble"][
        ["Name", "CAGR", "Sharpe", "Sortino", "MaxDD", "HitRate", "ProfitFactor", "Trades", "TradeRate"]
    ].set_index("Name")
    pd.set_option("display.float_format", "{:.4f}".format)
    pd.set_option("display.max_columns", None)
    print(ens_df.to_string())

    try: plot_summary_m15(summary, out_dir)
    except Exception as e: print(f"[WARN] Summary: {e}")

    print(f"\nOutputy w: {out_dir}/")
    print(f"  plots/   — dashboardy, session analysis, monthly heatmap, feature importance")
    print(f"  signals/ — sygnały CSV per para")
    print(f"  m15_summary_results.csv")
    return summary_rows


if __name__ == "__main__":

    cfg = ForexM15Config(
        train_bars     = 1_500,
        test_bars      = 500,
        horizon        = 20,
        direction_thr  = 0.0003,
        conf_threshold = 0.45,
        use_sl_tp      = True,
        sltp_mode      = "atr",
        atr_sl_mult    = 1.5,
        atr_tp_mult    = 2.5,
        filter_session = False,
        random_state   = 42,
        start          = "2022-01-01",
    )

    tickers = [
        "EURUSD=X", "GBPUSD=X", "USDJPY=X",
        "AUDUSD=X", "USDCAD=X", "EURGBP=X",
    ]

    run_forex_m15(
        tickers = tickers,
        cfg     = cfg,
        out_dir = "outputs_forex_m15",
        csv_dir = "data_m15",
    )
