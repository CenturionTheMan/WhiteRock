import warnings
warnings.filterwarnings("ignore")

import os
import copy
import numpy as np
import pandas as pd
from dataclasses import dataclass
from typing import Dict, List, Optional

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import matplotlib.dates as mdates


COMMODITIES = {
    "GC=F": {"name": "Gold",          "sector": "Metals",  "spread_pct": 0.00008, "direction_thr": 0.004, "unit": "$/oz"},
    "SI=F": {"name": "Silver",        "sector": "Metals",  "spread_pct": 0.00025, "direction_thr": 0.007, "unit": "$/oz"},
    "HG=F": {"name": "Copper",        "sector": "Metals",  "spread_pct": 0.00020, "direction_thr": 0.006, "unit": "$/lb"},
    "PL=F": {"name": "Platinum",      "sector": "Metals",  "spread_pct": 0.00025, "direction_thr": 0.006, "unit": "$/oz"},
    "CL=F": {"name": "WTI Crude Oil", "sector": "Energy",  "spread_pct": 0.00015, "direction_thr": 0.008, "unit": "$/bbl"},
    "BZ=F": {"name": "Brent Crude",   "sector": "Energy",  "spread_pct": 0.00015, "direction_thr": 0.008, "unit": "$/bbl"},
    "NG=F": {"name": "Natural Gas",   "sector": "Energy",  "spread_pct": 0.00100, "direction_thr": 0.020, "unit": "$/MMBtu"},
    "ZW=F": {"name": "Wheat",         "sector": "Grains",  "spread_pct": 0.00030, "direction_thr": 0.008, "unit": "¢/bu"},
    "ZC=F": {"name": "Corn",          "sector": "Grains",  "spread_pct": 0.00025, "direction_thr": 0.007, "unit": "¢/bu"},
    "ZS=F": {"name": "Soybeans",      "sector": "Grains",  "spread_pct": 0.00020, "direction_thr": 0.006, "unit": "¢/bu"},
}

SECTOR_COLORS = {"Metals": "#FFD700", "Energy": "#FF6B35", "Grains": "#7CB342"}

MODEL_COLORS = {
    "RandomForest":     "#2196F3",
    "GradientBoosting": "#FF9800",
    "NeuralNet":        "#9C27B0",
    "XGBoost":          "#4CAF50",
    "Ensemble":         "#F44336",
    "BuyHold":          "#90A4AE",
}


def load_commodity(ticker: str, start: str = "2018-01-01", end=None) -> pd.DataFrame:
    try:
        import yfinance as yf
    except ImportError:
        raise ImportError("pip install yfinance")

    df = yf.download(ticker, start=start, end=end, auto_adjust=False, progress=False)
    if df is None or df.empty:
        raise ValueError(f"Brak danych dla {ticker}")

    if isinstance(df.columns, pd.MultiIndex):
        last = df.columns.get_level_values(-1)
        df = df.xs(ticker, axis=1, level=-1) if ticker in last else df.droplevel(-1, axis=1)

    df = df.rename(columns=str.title)
    df.index = pd.to_datetime(df.index)
    needed = ["Open", "High", "Low", "Close"]
    df = df[[c for c in df.columns if c in needed + ["Volume"]]].dropna(subset=needed)

    if "Volume" not in df.columns or df["Volume"].sum() == 0:
        df["Volume"] = 1.0

    return df



def _ema(s: pd.Series, span: int) -> pd.Series:
    return s.ewm(span=span, adjust=False).mean()

def _rsi(close: pd.Series, period: int = 14) -> pd.Series:
    delta    = close.diff()
    avg_gain = delta.clip(lower=0).ewm(com=period - 1, adjust=False).mean()
    avg_loss = (-delta.clip(upper=0)).ewm(com=period - 1, adjust=False).mean()
    return 100 - (100 / (1 + avg_gain / (avg_loss + 1e-12)))

def _atr(high: pd.Series, low: pd.Series, close: pd.Series, period: int = 14) -> pd.Series:
    pc = close.shift(1)
    tr = np.maximum(np.maximum((high - low).abs(), (high - pc).abs()), (low - pc).abs())
    return pd.Series(tr, index=close.index).ewm(com=period - 1, adjust=False).mean()

def _macd(close: pd.Series, fast=12, slow=26, signal=9):
    ml = _ema(close, fast) - _ema(close, slow)
    sl = _ema(ml, signal)
    return ml, sl, ml - sl

def _bollinger(close: pd.Series, period=20, n_std=2.0):
    ma    = close.rolling(period).mean()
    sd    = close.rolling(period).std()
    upper = ma + n_std * sd
    lower = ma - n_std * sd
    return upper, lower, (upper - lower) / (ma + 1e-12)

def _stochastic(high: pd.Series, low: pd.Series, close: pd.Series, k=14, d=3):
    lo    = low.rolling(k).min()
    hi    = high.rolling(k).max()
    k_pct = 100 * (close - lo) / (hi - lo + 1e-12)
    return k_pct, k_pct.rolling(d).mean()

def _williams_r(high: pd.Series, low: pd.Series, close: pd.Series, period=14) -> pd.Series:
    hh = high.rolling(period).max()
    ll = low.rolling(period).min()
    return -100 * (hh - close) / (hh - ll + 1e-12)

def _cci(high: pd.Series, low: pd.Series, close: pd.Series, period=20) -> pd.Series:
    tp  = (high + low + close) / 3
    ma  = tp.rolling(period).mean()
    mad = tp.rolling(period).apply(lambda x: np.abs(x - x.mean()).mean(), raw=True)
    return (tp - ma) / (0.015 * mad + 1e-12)

def _adx(high: pd.Series, low: pd.Series, close: pd.Series, period=14):
    ph, pl = high.shift(1), low.shift(1)
    plus_dm  = (high - ph).clip(lower=0).where((high - ph) > (pl - low), 0)
    minus_dm = (pl - low).clip(lower=0).where((pl - low) > (high - ph), 0)
    tr_s = _atr(high, low, close, period)
    pdi  = 100 * plus_dm.ewm(com=period - 1, adjust=False).mean() / (tr_s + 1e-12)
    mdi  = 100 * minus_dm.ewm(com=period - 1, adjust=False).mean() / (tr_s + 1e-12)
    dx   = 100 * (pdi - mdi).abs() / (pdi + mdi + 1e-12)
    return dx.ewm(com=period - 1, adjust=False).mean(), pdi, mdi

def _obv(close: pd.Series, volume: pd.Series) -> pd.Series:
    sign = np.sign(close.diff().fillna(0))
    return (sign * volume).cumsum()



_DROP_FROM_X = [
    "Open", "High", "Low", "Close", "Volume",
    "sma_5", "sma_10", "sma_20", "sma_50", "sma_100", "sma_200",
    "ema_8", "ema_13", "ema_21", "ema_55",
    "bb_upper", "bb_lower", "atr_14",
    "macd_line", "macd_signal",
    "obv",
]


def make_features(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    c   = out["Close"].squeeze()
    h   = out["High"].squeeze()
    l   = out["Low"].squeeze()
    o   = out["Open"].squeeze()
    vol = out["Volume"].squeeze()

    for p in [1, 2, 3, 5, 10, 21]:
        out[f"ret_{p}"] = c.pct_change(p)

    for w in [5, 10, 20, 50, 100, 200]:
        sma = c.rolling(w).mean()
        out[f"sma_{w}"]      = sma
        out[f"dist_sma_{w}"] = (c - sma) / (sma + 1e-12)

    for e in [8, 13, 21, 55]:
        em = _ema(c, e)
        out[f"ema_{e}"]      = em
        out[f"dist_ema_{e}"] = (c - em) / (em + 1e-12)

    for w in [20, 60, 120]:
        roll_mean = c.rolling(w).mean()
        roll_std  = c.rolling(w).std()
        out[f"zscore_{w}"] = (c - roll_mean) / (roll_std + 1e-12)

    
    for p in [9, 14, 21]:
        out[f"rsi_{p}"] = _rsi(c, p)

   
    ml, sl, hist = _macd(c)
    out["macd_line"]     = ml
    out["macd_signal"]   = sl
    out["macd_hist"]     = hist
    out["macd_hist_chg"] = hist.diff()

    
    out["atr_14"]  = _atr(h, l, c, 14)
    out["atr_pct"] = out["atr_14"] / (c + 1e-12)
    for w in [5, 10, 20]:
        out[f"vol_{w}"] = out["ret_1"].rolling(w).std()
    out["vol_regime"] = out["vol_5"] / (out["vol_20"] + 1e-12)

    
    bb_u, bb_l, bb_w = _bollinger(c)
    out["bb_upper"] = bb_u
    out["bb_lower"] = bb_l
    out["bb_width"] = bb_w
    out["bb_pos"]   = (c - bb_l) / ((bb_u - bb_l) + 1e-12)

    
    k_pct, d_pct = _stochastic(h, l, c)
    out["stoch_k"]  = k_pct
    out["stoch_d"]  = d_pct
    out["stoch_kd"] = k_pct - d_pct

    
    out["williams_r"] = _williams_r(h, l, c)

    
    out["cci_20"] = _cci(h, l, c, 20)

    
    adx_v, pdi, mdi = _adx(h, l, c, 14)
    out["adx"]     = adx_v
    out["plus_di"] = pdi
    out["minus_di"]= mdi
    out["di_diff"] = pdi - mdi

    
    out["hl_range"]     = (h - l) / (c + 1e-12)
    out["co_range"]     = (c - o) / (o + 1e-12)
    out["upper_shadow"] = (h - c.clip(lower=o)) / (h - l + 1e-12)
    out["lower_shadow"] = (c.clip(upper=o) - l)  / (h - l + 1e-12)

    for p in [5, 10, 21]:
        out[f"mom_{p}"] = c / c.shift(p) - 1

    out["trend_20_50"]  = (out["sma_20"] > out["sma_50"]).astype(float)
    out["trend_50_200"] = (out["sma_50"] > out["sma_200"]).astype(float)

    for w in [5, 10, 20]:
        hmax = h.rolling(w).max()
        lmin = l.rolling(w).min()
        out[f"pct_from_high_{w}"] = (c - hmax) / (hmax + 1e-12)
        out[f"pct_from_low_{w}"]  = (c - lmin) / (lmin  + 1e-12)

    obv_raw = _obv(c, vol)
    out["obv"] = obv_raw
    out["obv_chg"] = obv_raw.pct_change(5).fillna(0)
    for w in [10, 20]:
        vol_sma = vol.rolling(w).mean()
        out[f"vol_ratio_{w}"] = vol / (vol_sma + 1e-12)
    out["vol_zscore_20"] = (vol - vol.rolling(20).mean()) / (vol.rolling(20).std() + 1e-12)

    idx = out.index
    out["month_sin"]     = np.sin(2 * np.pi * idx.month / 12)
    out["month_cos"]     = np.cos(2 * np.pi * idx.month / 12)
    out["dayofyear_sin"] = np.sin(2 * np.pi * idx.dayofyear / 365)
    out["dayofyear_cos"] = np.cos(2 * np.pi * idx.dayofyear / 365)
    out["quarter_sin"]   = np.sin(2 * np.pi * idx.quarter / 4)
    out["quarter_cos"]   = np.cos(2 * np.pi * idx.quarter / 4)

    out = out.replace([np.inf, -np.inf], np.nan)
    return out


def make_direction_label(df_feat: pd.DataFrame, horizon: int = 5, thr: float = 0.005):
    c       = df_feat["Close"].squeeze()
    fwd_ret = c.shift(-horizon) / c - 1
    y       = pd.Series(0, index=df_feat.index, dtype=int)
    y[fwd_ret >  thr] =  1
    y[fwd_ret < -thr] = -1
    return y, fwd_ret



def build_models(random_state: int = 42) -> Dict:
    from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
    from sklearn.neural_network import MLPClassifier

    models = {
        "RandomForest": RandomForestClassifier(
            n_estimators=400, max_depth=8, min_samples_leaf=10,
            class_weight="balanced_subsample", random_state=random_state, n_jobs=-1,
        ),
        "GradientBoosting": GradientBoostingClassifier(
            n_estimators=200, max_depth=4, learning_rate=0.05,
            subsample=0.8, random_state=random_state,
        ),
        "NeuralNet": MLPClassifier(
            hidden_layer_sizes=(128, 64, 32), activation="relu",
            learning_rate_init=0.001, max_iter=400,
            early_stopping=True, validation_fraction=0.1,
            random_state=random_state,
        ),
    }

    try:
        from xgboost import XGBClassifier
        models["XGBoost"] = XGBClassifier(
            n_estimators=300, max_depth=5, learning_rate=0.05,
            subsample=0.8, colsample_bytree=0.8,
            eval_metric="mlogloss", random_state=random_state,
            n_jobs=-1, verbosity=0,
        )
        print("  [INFO] XGBoost available — included.")
    except ImportError:
        print("  [INFO] XGBoost not installed (pip install xgboost). Skipping.")

    return models


@dataclass
class CommodityConfig:
    train_months:   int   = 24
    test_months:    int   = 3
    horizon:        int   = 5        
    direction_thr:  float = 0.005    
    conf_threshold: float = 0.45
    spread_pct:     float = 0.0002   
    use_sl_tp:      bool  = True
    sltp_mode:      str   = "atr"    
    atr_sl_mult:    float = 1.5
    atr_tp_mult:    float = 3.0
    fixed_sl_pct:   float = 0.015
    fixed_tp_pct:   float = 0.030
    random_state:   int   = 42
    start:          str   = "2018-01-01"


def _month_floor(ts: pd.Timestamp) -> pd.Timestamp:
    return pd.Timestamp(year=ts.year, month=ts.month, day=1)

def _add_months(ts: pd.Timestamp, n: int) -> pd.Timestamp:
    y = ts.year + (ts.month - 1 + n) // 12
    m = (ts.month - 1 + n) % 12 + 1
    return pd.Timestamp(year=y, month=m, day=1)


def walk_forward_multi_model(df: pd.DataFrame, cfg: CommodityConfig):
    from sklearn.preprocessing import StandardScaler

    feat       = make_features(df)
    y_dir, _   = make_direction_label(feat, horizon=cfg.horizon, thr=cfg.direction_thr)
    X_cols     = [c for c in feat.columns if c not in _DROP_FROM_X]
    data       = pd.concat(
        [df[["Close", "High", "Low"]], feat[X_cols], y_dir.rename("y")], axis=1
    ).dropna()

    if len(data) < 300:
        raise ValueError(f"Za mało danych: {len(data)} wierszy po feature engineering.")

    models_registry = build_models(cfg.random_state)
    model_names     = list(models_registry.keys())

    pred_all = {m: pd.Series(np.nan, index=data.index, dtype=float) for m in model_names}
    pup_all  = {m: pd.Series(np.nan, index=data.index, dtype=float) for m in model_names}
    pdn_all  = {m: pd.Series(np.nan, index=data.index, dtype=float) for m in model_names}

    windows   = []
    cur_start = _month_floor(data.index.min())

    while True:
        train_end = _add_months(cur_start, cfg.train_months) - pd.Timedelta(days=1)
        test_end  = _add_months(cur_start, cfg.train_months + cfg.test_months) - pd.Timedelta(days=1)

        if test_end > data.index.max():
            break

        train = data[(data.index >= cur_start) & (data.index <= train_end)].copy()
        test  = data[(data.index >  train_end) & (data.index <= test_end)].copy()

        if len(train) < 200 or len(test) < 20:
            cur_start = _add_months(cur_start, cfg.test_months)
            continue

        train_fit = train.iloc[:-cfg.horizon] if len(train) > cfg.horizon + 50 else train

        X_tr = train_fit.drop(columns=["Close", "High", "Low", "y"])
        y_tr = train_fit["y"].astype(int)
        X_te = test.drop(columns=["Close", "High", "Low", "y"])

        scaler  = StandardScaler()
        X_tr_s  = scaler.fit_transform(X_tr)
        X_te_s  = scaler.transform(X_te)

        win = {
            "train_start": str(cur_start.date()),
            "train_end":   str(train_end.date()),
            "test_start":  str((train_end + pd.Timedelta(days=1)).date()),
            "test_end":    str(test_end.date()),
            "n_train": len(train), "n_test": len(test),
        }

        for mname, proto in models_registry.items():
            model = copy.deepcopy(proto)
            try:
                y_tr_fit = (y_tr + 1) if mname == "XGBoost" else y_tr
                model.fit(X_tr_s, y_tr_fit)
                proba   = model.predict_proba(X_te_s)
                classes = [c - 1 for c in model.classes_] if mname == "XGBoost" else list(model.classes_)
                p_up    = proba[:, classes.index( 1)] if  1 in classes else np.zeros(len(test))
                p_dn    = proba[:, classes.index(-1)] if -1 in classes else np.zeros(len(test))
                pred    = np.where(p_up > cfg.conf_threshold,  1,
                          np.where(p_dn > cfg.conf_threshold, -1, 0))

                pred_all[mname].loc[test.index] = pred
                pup_all[mname].loc[test.index]  = p_up
                pdn_all[mname].loc[test.index]  = p_dn
                win[f"{mname}_acc"] = float((pred == test["y"].values).mean())
            except Exception as e:
                print(f"    [WARN] {mname}: {e}")

        windows.append(win)
        cur_start = _add_months(cur_start, cfg.test_months)

    avg_up   = pd.concat([pup_all[m] for m in model_names], axis=1).mean(axis=1)
    avg_dn   = pd.concat([pdn_all[m] for m in model_names], axis=1).mean(axis=1)
    ens_pred = pd.Series(0, index=data.index, dtype=int)
    ens_pred[avg_up > cfg.conf_threshold] =  1
    ens_pred[avg_dn > cfg.conf_threshold] = -1

    all_names = model_names + ["Ensemble"]
    pred_all["Ensemble"] = ens_pred.reindex(df.index).fillna(0)
    pup_all["Ensemble"]  = avg_up.reindex(df.index).fillna(0)
    pdn_all["Ensemble"]  = avg_dn.reindex(df.index).fillna(0)
    for m in model_names:
        pred_all[m] = pred_all[m].reindex(df.index).fillna(0)
        pup_all[m]  = pup_all[m].reindex(df.index).fillna(0)
        pdn_all[m]  = pdn_all[m].reindex(df.index).fillna(0)

    return pred_all, pup_all, pdn_all, pd.DataFrame(windows), feat.reindex(df.index), all_names


def backtest_commodity(df: pd.DataFrame, signal: pd.Series, cfg: CommodityConfig, feat: pd.DataFrame):
    idx    = df.index
    c      = df["Close"].astype(float).reindex(idx)
    h      = df["High"].astype(float).reindex(idx)
    l      = df["Low"].astype(float).reindex(idx)
    atr14  = feat["atr_14"].astype(float).reindex(idx)
    
    desired = signal.reindex(idx).fillna(0).shift(1).fillna(0).astype(int)

    in_pos      = 0
    entry_price = sl_price = tp_price = np.nan
    pos         = pd.Series(0, index=idx, dtype=int)
    strat_ret   = pd.Series(0.0, index=idx, dtype=float)

    for i in range(1, len(idx)):
        yest    = idx[i - 1]
        new_sig = int(desired.iloc[i])
        pos.iloc[i] = in_pos

        if in_pos == 0 and new_sig != 0:
            in_pos      = new_sig
            entry_price = c.iloc[i]
            a           = float(atr14.iloc[i]) if not np.isnan(atr14.iloc[i]) else c.iloc[i] * 0.01
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
                strat_ret.iloc[i] = in_pos * (exit_price / c.loc[yest] - 1) - cfg.spread_pct
                in_pos = 0
                entry_price = sl_price = tp_price = np.nan
                pos.iloc[i] = 0
            else:
                strat_ret.iloc[i] = in_pos * (c.iloc[i] / c.loc[yest] - 1)
                pos.iloc[i] = in_pos

    eq = (1 + strat_ret.fillna(0)).cumprod()
    return strat_ret, eq, pos


def perf_stats(ret: pd.Series, freq: int = 252) -> Dict:
    ret = ret.dropna()
    if len(ret) < 5:
        return {"CAGR": np.nan, "Sharpe": np.nan, "Sortino": np.nan,
                "MaxDD": np.nan, "MaxDD_Days": 0, "HitRate": np.nan,
                "ProfitFactor": np.nan, "Trades": 0}

    eq      = (1 + ret).cumprod()
    cagr    = eq.iloc[-1] ** (freq / len(ret)) - 1
    sharpe  = np.sqrt(freq) * ret.mean() / (ret.std() + 1e-12)
    dn_std  = ret[ret < 0].std()
    sortino = np.sqrt(freq) * ret.mean() / (dn_std + 1e-12)
    peak    = eq.cummax()
    dd      = eq / peak - 1
    maxdd   = dd.min()

    in_dd     = dd < 0
    grp       = (in_dd != in_dd.shift()).cumsum()
    dd_lens   = in_dd[in_dd].groupby(grp[in_dd]).count()
    maxdd_days = int(dd_lens.max()) if len(dd_lens) > 0 else 0

    wins = ret[ret > 0].sum()
    loss = ret[ret < 0].abs().sum()
    return {
        "CAGR":         float(cagr),
        "Sharpe":       float(sharpe),
        "Sortino":      float(sortino),
        "MaxDD":        float(maxdd),
        "MaxDD_Days":   maxdd_days,
        "HitRate":      float((ret > 0).mean()),
        "ProfitFactor": float(wins / (loss + 1e-12)),
        "Trades":       int((ret != 0).sum()),
    }


def get_feature_importance(df: pd.DataFrame, cfg: CommodityConfig) -> Optional[pd.Series]:
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.preprocessing import StandardScaler

    feat   = make_features(df)
    y, _   = make_direction_label(feat, horizon=cfg.horizon, thr=cfg.direction_thr)
    X_cols = [c for c in feat.columns if c not in _DROP_FROM_X]
    data   = pd.concat([feat[X_cols], y.rename("y")], axis=1).dropna()

    if len(data) < 200:
        return None

    X_d = data.drop(columns=["y"])
    y_d = data["y"].astype(int)
    rf  = RandomForestClassifier(n_estimators=200, max_depth=6,
                                  random_state=cfg.random_state, n_jobs=-1)
    scaler = StandardScaler()
    rf.fit(scaler.fit_transform(X_d), y_d)
    return pd.Series(rf.feature_importances_, index=X_d.columns).sort_values(ascending=False)


def _fmt_dates(ax, n_ticks=6):
    ax.xaxis.set_major_locator(mdates.AutoDateLocator(minticks=4, maxticks=n_ticks))
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%b'%y"))
    plt.setp(ax.get_xticklabels(), rotation=30, ha="right", fontsize=7)


def safe_filename(ticker: str) -> str:
    return ticker.replace("=", "").replace("/", "_").replace("\\", "_").replace(".", "_")


def plot_pair_dashboard(
    ticker: str, df: pd.DataFrame, feat: pd.DataFrame,
    signals: Dict, probs_up: Dict, probs_dn: Dict,
    equities: Dict, stats_all: Dict,
    cfg: CommodityConfig, out_dir: str,
):
    meta = COMMODITIES.get(ticker, {})
    name = meta.get("name", ticker)
    unit = meta.get("unit", "")
    c    = df["Close"].squeeze()

    fig  = plt.figure(figsize=(20, 17))
    fig.suptitle(f"Commodities ML Dashboard — {name}  [{unit}]",
                 fontsize=14, fontweight="bold", y=0.99)
    gs   = gridspec.GridSpec(4, 2, figure=fig, hspace=0.50, wspace=0.32,
                              height_ratios=[2.5, 1.2, 2.0, 1.5])

    ax_price = fig.add_subplot(gs[0, :])
    ax_rsi   = fig.add_subplot(gs[1, 0])
    ax_macd  = fig.add_subplot(gs[1, 1])
    ax_eq    = fig.add_subplot(gs[2, :])
    ax_prob  = fig.add_subplot(gs[3, 0])
    ax_dd    = fig.add_subplot(gs[3, 1])

    ax_price.plot(c.index, c.values, color="#212121", lw=0.9, label="Close")
    for period, color, ls in [(20, "#1565C0", "--"), (50, "#E65100", "--"), (200, "#1B5E20", "-.")]:
        col = f"sma_{period}"
        if col in feat.columns:
            ax_price.plot(feat.index, feat[col].values, color=color, lw=0.8,
                          linestyle=ls, alpha=0.85, label=f"SMA{period}")
    if "bb_upper" in feat.columns:
        ax_price.fill_between(feat.index, feat["bb_lower"], feat["bb_upper"],
                               alpha=0.07, color="#757575", label="BB(20,2)")

    ens_sig = signals.get("Ensemble", pd.Series(0, index=c.index)).reindex(c.index).fillna(0).astype(int)
    prev    = ens_sig.shift(1).fillna(0).astype(int)
    ax_price.scatter(c.index[(prev == 0) & (ens_sig ==  1)],
                     c[(prev == 0) & (ens_sig ==  1)],
                     marker="^", s=55, color="#00C853", zorder=5, label="Long entry")
    ax_price.scatter(c.index[(prev == 0) & (ens_sig == -1)],
                     c[(prev == 0) & (ens_sig == -1)],
                     marker="v", s=55, color="#D50000", zorder=5, label="Short entry")
    ax_price.scatter(c.index[(prev != 0) & (ens_sig ==  0)],
                     c[(prev != 0) & (ens_sig ==  0)],
                     marker="o", s=30, color="#FF6F00", zorder=5, alpha=0.7, label="Exit")

    ax_price.set_title("Price + Moving Averages + Ensemble Signals", fontsize=9, loc="left")
    ax_price.legend(fontsize=7, ncol=5, loc="upper left")
    ax_price.set_ylabel(f"Price ({unit})", fontsize=8)
    _fmt_dates(ax_price)

    
    if "rsi_14" in feat.columns:
        r14 = feat["rsi_14"].reindex(c.index)
        ax_rsi.plot(r14.index, r14.values, color="#6A1B9A", lw=0.9)
        ax_rsi.axhline(70, color="#C62828", lw=0.7, linestyle="--")
        ax_rsi.axhline(30, color="#2E7D32", lw=0.7, linestyle="--")
        ax_rsi.fill_between(r14.index, 70, r14.values.clip(70),      alpha=0.2, color="#EF5350")
        ax_rsi.fill_between(r14.index, 30, r14.values.clip(None, 30), alpha=0.2, color="#66BB6A")
        ax_rsi.set_ylim(0, 100)
        ax_rsi.set_title("RSI(14)", fontsize=9, loc="left")
        _fmt_dates(ax_rsi)

    
    if "macd_hist" in feat.columns:
        mh   = feat["macd_hist"].reindex(c.index)
        ml_s = feat["macd_line"].reindex(c.index)
        ms   = feat["macd_signal"].reindex(c.index)
        ax_macd.bar(mh.index, mh.values,
                    color=["#43A047" if v >= 0 else "#E53935" for v in mh.values],
                    alpha=0.55, width=1.2)
        ax_macd.plot(ml_s.index, ml_s.values, color="#1565C0", lw=0.8, label="MACD")
        ax_macd.plot(ms.index,   ms.values,   color="#E64A19", lw=0.8, label="Signal")
        ax_macd.axhline(0, color="black", lw=0.5)
        ax_macd.set_title("MACD", fontsize=9, loc="left")
        ax_macd.legend(fontsize=7)
        _fmt_dates(ax_macd)

    
    bh_eq = (1 + c.pct_change().fillna(0)).cumprod()
    ax_eq.plot(bh_eq.index, bh_eq.values, color=MODEL_COLORS["BuyHold"],
               lw=1.0, linestyle="--", alpha=0.7, label="Buy&Hold")
    for mname, eq_s in equities.items():
        s     = stats_all.get(mname, {})
        label = f"{mname} (Sh={s.get('Sharpe', np.nan):.2f}, CAGR={s.get('CAGR', np.nan)*100:.1f}%)"
        ax_eq.plot(eq_s.index, eq_s.values,
                   color=MODEL_COLORS.get(mname, "#555"), lw=1.2, label=label)
    ax_eq.axhline(1.0, color="black", lw=0.4)
    ax_eq.set_title("Equity Curves — All Models", fontsize=9, loc="left")
    ax_eq.set_ylabel("Equity (start=1)", fontsize=8)
    ax_eq.legend(fontsize=7, loc="upper left")
    _fmt_dates(ax_eq)

    
    pu  = probs_up.get("Ensemble", pd.Series(0.0, index=c.index)).reindex(c.index).fillna(0)
    pd_ = probs_dn.get("Ensemble", pd.Series(0.0, index=c.index)).reindex(c.index).fillna(0)
    ax_prob.stackplot(pu.index, pu.values, pd_.values,
                      labels=["P(UP)", "P(DOWN)"],
                      colors=["#4CAF50", "#F44336"], alpha=0.65)
    ax_prob.axhline(cfg.conf_threshold, color="black", lw=0.8, linestyle="--",
                    label=f"Threshold={cfg.conf_threshold}")
    ax_prob.set_ylim(0, 1)
    ax_prob.set_title("Ensemble ML Confidence", fontsize=9, loc="left")
    ax_prob.set_ylabel("Probability", fontsize=8)
    ax_prob.legend(fontsize=7)
    _fmt_dates(ax_prob)

    # Drawdown
    ens_eq = equities.get("Ensemble")
    if ens_eq is not None:
        peak = ens_eq.cummax()
        dd_s = ens_eq / peak - 1
        ax_dd.fill_between(dd_s.index, dd_s.values, 0, color="#EF5350", alpha=0.65)
        ax_dd.plot(dd_s.index, dd_s.values, color="#B71C1C", lw=0.7)
        ax_dd.set_title("Ensemble Drawdown", fontsize=9, loc="left")
        ax_dd.set_ylabel("Drawdown", fontsize=8)
        _fmt_dates(ax_dd)

    path = os.path.join(out_dir, "plots", f"{safe_filename(ticker)}_dashboard.png")
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"    [chart] {path}")


def plot_feature_importance(ticker: str, fi: pd.Series, out_dir: str, top_n: int = 25):
    top    = fi.head(top_n)
    colors = plt.cm.RdYlGn(np.linspace(0.25, 0.85, len(top)))[::-1]

    fig, ax = plt.subplots(figsize=(11, 8))
    ax.barh(range(len(top)), top.values[::-1], color=colors[::-1], edgecolor="white")
    ax.set_yticks(range(len(top)))
    ax.set_yticklabels(top.index[::-1], fontsize=8)
    ax.set_xlabel("Feature Importance (RandomForest)", fontsize=9)
    ax.set_title(f"Top {top_n} Features — {COMMODITIES.get(ticker, {}).get('name', ticker)}", fontsize=11)
    fig.tight_layout()

    path = os.path.join(out_dir, "plots", f"{safe_filename(ticker)}_feature_importance.png")
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"    [chart] {path}")


def plot_monthly_heatmap(ticker: str, ret: pd.Series, out_dir: str):
    monthly   = ret.resample("ME").apply(lambda x: (1 + x).prod() - 1)
    df_m      = monthly.to_frame("ret")
    df_m["year"]  = df_m.index.year
    df_m["month"] = df_m.index.month
    pivot = df_m.pivot(index="year", columns="month", values="ret")
    pivot.columns = ["Jan","Feb","Mar","Apr","May","Jun","Jul","Aug","Sep","Oct","Nov","Dec"]

    vals = pivot.values[~np.isnan(pivot.values.astype(float))]
    vmax = max(abs(vals).max(), 0.01) if len(vals) else 0.05

    fig, ax = plt.subplots(figsize=(15, max(3, len(pivot) * 0.75 + 1.5)))
    im = ax.imshow(pivot.values.astype(float), aspect="auto",
                   cmap="RdYlGn", vmin=-vmax, vmax=vmax)
    plt.colorbar(im, ax=ax, fraction=0.015, label="Monthly Return")
    ax.set_xticks(range(12))
    ax.set_xticklabels(pivot.columns, fontsize=9)
    ax.set_yticks(range(len(pivot)))
    ax.set_yticklabels(pivot.index.tolist(), fontsize=9)
    for i in range(len(pivot)):
        for j in range(12):
            v = pivot.values[i, j]
            if not np.isnan(float(v)):
                ax.text(j, i, f"{float(v)*100:.1f}%", ha="center", va="center",
                        fontsize=7, color="black")

    name = COMMODITIES.get(ticker, {}).get("name", ticker)
    ax.set_title(f"Monthly Returns Heatmap — {name} (Ensemble)", fontsize=11)
    fig.tight_layout()
    path = os.path.join(out_dir, "plots", f"{safe_filename(ticker)}_monthly_heatmap.png")
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"    [chart] {path}")


def plot_seasonality(ticker: str, df: pd.DataFrame, eq: pd.Series, out_dir: str):
    """Average strategy return by calendar month — highlights seasonal patterns."""
    ret = eq.pct_change().fillna(0)
    bh  = df["Close"].pct_change().fillna(0)

    month_strat = ret.groupby(ret.index.month).mean() * 100
    month_bh    = bh.groupby(bh.index.month).mean() * 100

    months = ["Jan","Feb","Mar","Apr","May","Jun","Jul","Aug","Sep","Oct","Nov","Dec"]
    x      = np.arange(12)
    width  = 0.38

    fig, ax = plt.subplots(figsize=(13, 5))
    ax.bar(x - width/2, [month_strat.get(i+1, 0) for i in range(12)],
           width, color="#2196F3", alpha=0.85, label="Ensemble")
    ax.bar(x + width/2, [month_bh.get(i+1, 0) for i in range(12)],
           width, color="#90A4AE", alpha=0.65, label="Buy&Hold")
    ax.axhline(0, color="black", lw=0.6)
    ax.set_xticks(x)
    ax.set_xticklabels(months, fontsize=9)
    ax.set_ylabel("Avg daily return (%)", fontsize=9)
    name = COMMODITIES.get(ticker, {}).get("name", ticker)
    ax.set_title(f"Seasonality — Average Return by Month: {name}", fontsize=11)
    ax.legend(fontsize=9)
    fig.tight_layout()

    path = os.path.join(out_dir, "plots", f"{safe_filename(ticker)}_seasonality.png")
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"    [chart] {path}")


def plot_sector_comparison(summary_df: pd.DataFrame, out_dir: str):
    """Sharpe and CAGR grouped by commodity sector (Metals / Energy / Grains)."""
    ens = summary_df[summary_df["Model"] == "Ensemble"].copy()
    if ens.empty:
        return

    ens["Sector"] = ens["Ticker"].map(lambda t: COMMODITIES.get(t, {}).get("sector", "Other"))

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    fig.suptitle("Sector Comparison — Ensemble (avg per sector)", fontsize=12, fontweight="bold")

    for ax, metric in zip(axes, ["Sharpe", "CAGR"]):
        sector_avg = ens.groupby("Sector")[metric].mean().sort_values(ascending=False)
        colors     = [SECTOR_COLORS.get(s, "#888") for s in sector_avg.index]
        bars = ax.bar(sector_avg.index, sector_avg.values, color=colors, edgecolor="white", width=0.5)
        ax.axhline(0, color="black", lw=0.5)
        ax.set_title(f"Avg {metric} by Sector", fontsize=10)
        ax.set_ylabel(metric, fontsize=9)
        for bar, val in zip(bars, sector_avg.values):
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height(),
                    f"{val:.3f}", ha="center", va="bottom", fontsize=9)

    fig.tight_layout()
    path = os.path.join(out_dir, "plots", "sector_comparison.png")
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"    [chart] {path}")


def plot_model_comparison_bar(summary_df: pd.DataFrame, out_dir: str):
    metrics = ["Sharpe", "CAGR", "MaxDD"]
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    fig.suptitle("Model Performance Comparison (average across commodities)", fontsize=12, fontweight="bold")

    for ax, metric in zip(axes, metrics):
        grp    = summary_df.groupby("Model")[metric].mean().sort_values(ascending=(metric == "MaxDD"))
        colors = [MODEL_COLORS.get(m, "#888") for m in grp.index]
        grp.plot(kind="bar", ax=ax, color=colors, edgecolor="white")
        ax.set_title(f"Avg {metric}", fontsize=10)
        ax.tick_params(axis="x", rotation=30)
        ax.axhline(0, color="black", lw=0.5)
        for bar, val in zip(ax.patches, grp.values):
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height(),
                    f"{val:.3f}", ha="center", va="bottom", fontsize=8)

    fig.tight_layout()
    path = os.path.join(out_dir, "plots", "model_comparison_bar.png")
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"    [chart] {path}")


def plot_all_equity_curves(results: Dict, out_dir: str):
    pairs = list(results.keys())
    ncols = 2
    nrows = (len(pairs) + 1) // 2

    fig, axes = plt.subplots(nrows, ncols, figsize=(17, nrows * 4), squeeze=False)
    fig.suptitle("All Commodities — Equity Curves by Model", fontsize=13, fontweight="bold")

    for idx, ticker in enumerate(pairs):
        row, col = divmod(idx, ncols)
        ax   = axes[row][col]
        res  = results[ticker]
        bh   = (1 + res["df"]["Close"].pct_change().fillna(0)).cumprod()
        ax.plot(bh.index, bh.values, color=MODEL_COLORS["BuyHold"],
                lw=0.8, linestyle="--", alpha=0.6, label="B&H")
        for mname, eq_s in res["equities"].items():
            ax.plot(eq_s.index, eq_s.values,
                    color=MODEL_COLORS.get(mname, "#555"), lw=0.9, alpha=0.85, label=mname)
        ax.axhline(1.0, color="black", lw=0.4)
        name   = COMMODITIES.get(ticker, {}).get("name", ticker)
        sector = COMMODITIES.get(ticker, {}).get("sector", "")
        ax.set_title(f"{name}  [{sector}]", fontsize=10)
        ax.legend(fontsize=6, loc="upper left")
        _fmt_dates(ax)

    for idx in range(len(pairs), nrows * ncols):
        axes[divmod(idx, ncols)[0]][divmod(idx, ncols)[1]].set_visible(False)

    fig.tight_layout()
    path = os.path.join(out_dir, "plots", "all_commodities_equity.png")
    fig.savefig(path, dpi=130, bbox_inches="tight")
    plt.close(fig)
    print(f"    [chart] {path}")


def plot_summary_scatter(summary_df: pd.DataFrame, out_dir: str):
    ens = summary_df[summary_df["Model"] == "Ensemble"].copy()
    if ens.empty:
        return

    ens["Sector"] = ens["Ticker"].map(lambda t: COMMODITIES.get(t, {}).get("sector", "Other"))

    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    fig.suptitle("Commodities Ensemble — Cross-Market Summary", fontsize=12, fontweight="bold")

    ax1, ax2 = axes
    names     = ens["Name"].tolist()
    sh_vals   = ens["Sharpe"].fillna(0).values
    cagr_vals = ens["CAGR"].fillna(0).values * 100
    dd_vals   = ens["MaxDD"].fillna(0).values * 100

    colors = [SECTOR_COLORS.get(COMMODITIES.get(t, {}).get("sector", ""), "#888")
              for t in ens["Ticker"]]
    x = np.arange(len(names))
    ax1.bar(x, sh_vals, color=colors, edgecolor="white")
    ax1.set_xticks(x)
    ax1.set_xticklabels(names, rotation=35, ha="right", fontsize=8)
    ax1.axhline(0, color="black", lw=0.6)
    ax1.set_ylabel("Sharpe Ratio")
    ax1.set_title("Sharpe per Commodity (Ensemble)", fontsize=10)

    sc = ax2.scatter(np.abs(dd_vals), sh_vals, s=90,
                     c=cagr_vals, cmap="RdYlGn", edgecolors="black", linewidths=0.6, zorder=3)
    plt.colorbar(sc, ax=ax2, label="CAGR (%)")
    for i, lbl in enumerate(names):
        ax2.annotate(lbl, (abs(dd_vals[i]), sh_vals[i]),
                     fontsize=7, xytext=(4, 4), textcoords="offset points")
    ax2.axhline(0, color="gray", lw=0.6, linestyle="--")
    ax2.set_xlabel("|Max Drawdown| (%)")
    ax2.set_ylabel("Sharpe Ratio")
    ax2.set_title("Risk-Return: Sharpe vs |MaxDD|", fontsize=10)

    fig.tight_layout()
    path = os.path.join(out_dir, "plots", "summary_scatter.png")
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"    [chart] {path}")


def run_commodities(
    tickers: Optional[List[str]] = None,
    cfg: Optional[CommodityConfig] = None,
    out_dir: str = "outputs_commodities_ml",
):
    if cfg     is None: cfg     = CommodityConfig()
    if tickers is None: tickers = list(COMMODITIES.keys())

    for d in [out_dir, f"{out_dir}/plots", f"{out_dir}/signals"]:
        os.makedirs(d, exist_ok=True)

    all_results  = {}
    summary_rows = []

    for ticker in tickers:
        meta   = COMMODITIES.get(ticker, {})
        name   = meta.get("name", ticker)
        sector = meta.get("sector", "?")
        print(f"\n{'='*60}")
        print(f"  {name}  [{sector}]  ({ticker})")
        print(f"{'='*60}")

        pair_cfg              = copy.copy(cfg)
        pair_cfg.spread_pct   = meta.get("spread_pct",    cfg.spread_pct)
        pair_cfg.direction_thr= meta.get("direction_thr", cfg.direction_thr)

        try:
            df = load_commodity(ticker, start=cfg.start)
            print(f"  Data: {len(df)} bars | {df.index[0].date()} → {df.index[-1].date()}")
        except Exception as e:
            print(f"  [SKIP] {e}")
            continue

        try:
            pred_all, pup_all, pdn_all, wf_table, feat, model_names = \
                walk_forward_multi_model(df, pair_cfg)
        except Exception as e:
            print(f"  [ERROR] Walk-forward: {e}")
            continue

        try:
            fi = get_feature_importance(df, pair_cfg)
            if fi is not None:
                plot_feature_importance(ticker, fi, out_dir)
        except Exception as e:
            print(f"  [WARN] Feature importance: {e}")

        equities  = {}
        stats_all = {}

        for mname in model_names:
            sig = pred_all[mname].reindex(df.index).fillna(0).astype(int)
            try:
                s_ret, eq, pos = backtest_commodity(df, sig, pair_cfg, feat)
                st = perf_stats(s_ret)
                equities[mname]  = eq
                stats_all[mname] = st
                summary_rows.append({
                    "Ticker": ticker, "Name": name, "Sector": sector, "Model": mname,
                    **st,
                    "Spread_Pct":  pair_cfg.spread_pct,
                    "DirThr":      pair_cfg.direction_thr,
                    "TradeRate":   float((pos != 0).mean()),
                })
                print(f"  {mname:20s} | CAGR={st['CAGR']:+.3f} | Sh={st['Sharpe']:.2f} "
                      f"| So={st['Sortino']:.2f} | MaxDD={st['MaxDD']:.3f} "
                      f"| Trades={st['Trades']}")
            except Exception as e:
                print(f"  [WARN] {mname}: {e}")

        bh_stats = perf_stats(df["Close"].pct_change().fillna(0))
        print(f"  {'Buy&Hold':20s} | CAGR={bh_stats['CAGR']:+.3f} | Sh={bh_stats['Sharpe']:.2f} "
              f"| MaxDD={bh_stats['MaxDD']:.3f}")

        sig_df = df[["Close", "High", "Low"]].copy()
        for mname in model_names:
            sig_df[f"signal_{mname}"] = pred_all[mname].reindex(df.index).fillna(0)
            sig_df[f"prob_up_{mname}"] = pup_all[mname].reindex(df.index).fillna(0)
            sig_df[f"prob_dn_{mname}"] = pdn_all[mname].reindex(df.index).fillna(0)
        sfn = safe_filename(ticker)
        sig_df.to_csv(f"{out_dir}/signals/{sfn}_signals.csv")
        wf_table.to_csv(f"{out_dir}/{sfn}_walkforward.csv", index=False)

        try:
            plot_pair_dashboard(ticker, df, feat, pred_all, pup_all, pdn_all,
                                equities, stats_all, pair_cfg, out_dir)
        except Exception as e:
            print(f"  [WARN] Dashboard: {e}")

        try:
            ens_eq = equities.get("Ensemble")
            if ens_eq is not None:
                plot_monthly_heatmap(ticker, ens_eq.pct_change().fillna(0), out_dir)
                plot_seasonality(ticker, df, ens_eq, out_dir)
        except Exception as e:
            print(f"  [WARN] Monthly/Seasonality: {e}")

        all_results[ticker] = {"df": df, "equities": equities, "feat": feat}

    if not summary_rows:
        print("\n[ERROR] Brak wyników — sprawdź połączenie lub tickery.")
        return []

    summary = pd.DataFrame(summary_rows)
    summary.to_csv(f"{out_dir}/summary_results.csv", index=False)

    print(f"\n{'='*70}")
    print("ENSEMBLE RESULTS")
    print(f"{'='*70}")
    ens_df = summary[summary["Model"] == "Ensemble"][
        ["Name", "Sector", "CAGR", "Sharpe", "Sortino", "MaxDD",
         "HitRate", "ProfitFactor", "Trades", "TradeRate"]
    ].set_index("Name")
    pd.set_option("display.float_format", "{:.4f}".format)
    pd.set_option("display.max_columns", None)
    print(ens_df.to_string())

    try: plot_model_comparison_bar(summary, out_dir)
    except Exception as e: print(f"[WARN] model_comparison: {e}")

    try: plot_summary_scatter(summary, out_dir)
    except Exception as e: print(f"[WARN] summary_scatter: {e}")

    try: plot_sector_comparison(summary, out_dir)
    except Exception as e: print(f"[WARN] sector_comparison: {e}")

    try:
        if all_results:
            plot_all_equity_curves(all_results, out_dir)
    except Exception as e:
        print(f"[WARN] all_equity: {e}")

    print(f"\nOutputy w: {out_dir}/")
    print(f"  plots/          — wykresy PNG (dashboard, heatmap, sezonowość, sektory)")
    print(f"  signals/        — sygnały CSV per surowiec")
    print(f"  *_walkforward.csv")
    print(f"  summary_results.csv")
    return summary_rows


if __name__ == "__main__":
    cfg = CommodityConfig(
        train_months   = 24,
        test_months    = 3,
        horizon        = 5,
        conf_threshold = 0.45,
        use_sl_tp      = True,
        sltp_mode      = "atr",
        atr_sl_mult    = 1.5,
        atr_tp_mult    = 3.0,
        random_state   = 42,
        start          = "2018-01-01",
    )

    tickers = [
        "GC=F",   # Gold
        "SI=F",   # Silver
        "CL=F",   # WTI Crude Oil
        "NG=F",   # Natural Gas
        "ZW=F",   # Wheat
        "ZC=F",   # Corn
        "ZS=F",   # Soybeans
        "HG=F",   # Copper
    ]

    run_commodities(tickers=tickers, cfg=cfg, out_dir="outputs_commodities_ml")
