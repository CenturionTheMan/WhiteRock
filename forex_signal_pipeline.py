import warnings
warnings.filterwarnings("ignore")

import os
import sys
import json
import argparse
import numpy as np
import pandas as pd
from dataclasses import dataclass, asdict, field
from datetime import datetime
from typing import Dict, List, Optional

try:
    import joblib
except ImportError:
    raise ImportError("pip install joblib")

from forex_ml import (
    load_forex,
    make_features,
    make_direction_label,
    FOREX_PAIRS,
    _DROP_FROM_X,
    safe_filename,
)

PAIR_MODEL_MAP: Dict[str, str] = {
    "EURUSD=X": "RandomForest",
    "GBPUSD=X": "RandomForest",
    "USDJPY=X": "RandomForest",
    "AUDUSD=X": "RandomForest",
    "EURGBP=X": "RandomForest",
    "EURJPY=X": "RandomForest",
    "GBPJPY=X": "NeuralNet",
    "USDCAD=X": "NeuralNet",
}


@dataclass
class PipelineConfig:
    train_months:   int   = 24
    horizon:        int   = 5
    direction_thr:  float = 0.003
    conf_threshold: float = 0.45
    atr_sl_mult:    float = 1.5
    atr_tp_mult:    float = 3.0
    models_dir:     str   = "saved_models"
    signals_dir:    str   = "live_signals"
    random_state:   int   = 42
    start:          str   = "2020-01-01"

def _build_single_model(model_type: str, random_state: int = 42):
    """Zwraca nową (nie wytrenowaną) instancję wybranego modelu."""
    if model_type == "RandomForest":
        from sklearn.ensemble import RandomForestClassifier
        return RandomForestClassifier(
            n_estimators=400, max_depth=8, min_samples_leaf=10,
            class_weight="balanced_subsample",
            random_state=random_state, n_jobs=-1,
        )
    if model_type == "NeuralNet":
        from sklearn.neural_network import MLPClassifier
        return MLPClassifier(
            hidden_layer_sizes=(128, 64, 32), activation="relu",
            learning_rate_init=0.001, max_iter=400,
            early_stopping=True, validation_fraction=0.1,
            random_state=random_state,
        )
    raise ValueError(f"Nieznany typ modelu: '{model_type}'. Dostępne: RandomForest, NeuralNet")


def _model_dir(ticker: str, model_type: str, base: str) -> str:
    return os.path.join(base, f"{safe_filename(ticker)}_{model_type}")


def train_and_save(cfg: PipelineConfig) -> List[Dict]:
    """
    Trenuje model produkcyjny dla każdej pary na ostatnich `train_months` miesiącach.
    Zapisuje: model.joblib, scaler.joblib, metadata.json
    """
    from sklearn.preprocessing import StandardScaler

    os.makedirs(cfg.models_dir, exist_ok=True)
    saved = []

    for ticker, model_type in PAIR_MODEL_MAP.items():
        meta_pair = FOREX_PAIRS.get(ticker, {})
        name      = meta_pair.get("name", ticker)

        print(f"\n{'='*58}")
        print(f"  {name}  ({ticker})  →  {model_type}")
        print(f"{'='*58}")

        try:
            df = load_forex(ticker, start=cfg.start)
        except Exception as e:
            print(f"  [SKIP] {e}")
            continue

        cutoff   = df.index[-1] - pd.DateOffset(months=cfg.train_months)
        df_train = df[df.index >= cutoff].copy()
        print(f"  Dane: {df_train.index[0].date()} → {df_train.index[-1].date()}"
              f"  ({len(df_train)} wierszy)")

        feat      = make_features(df_train)
        y, _      = make_direction_label(feat, horizon=cfg.horizon, thr=cfg.direction_thr)
        X_cols    = [c for c in feat.columns if c not in _DROP_FROM_X]
        data      = pd.concat([feat[X_cols], y.rename("y")], axis=1).dropna()

        data = data.iloc[:-cfg.horizon]

        if len(data) < 100:
            print(f"  [SKIP] Za mało próbek: {len(data)}")
            continue

        X_df   = data.drop(columns=["y"])
        y_data = data["y"].astype(int)

        scaler  = StandardScaler()
        X_scaled = scaler.fit_transform(X_df)

        model = _build_single_model(model_type, cfg.random_state)
        model.fit(X_scaled, y_data)

        class_counts = y_data.value_counts().to_dict()
        print(f"  Klasy: {class_counts}  |  n={len(data)}")

        out_dir = _model_dir(ticker, model_type, cfg.models_dir)
        os.makedirs(out_dir, exist_ok=True)
        joblib.dump(model,  os.path.join(out_dir, "model.joblib"))
        joblib.dump(scaler, os.path.join(out_dir, "scaler.joblib"))

        metadata = {
            "ticker":         ticker,
            "name":           name,
            "model_type":     model_type,
            "feature_cols":   list(X_df.columns),
            "train_start":    str(df_train.index[0].date()),
            "train_end":      str(df_train.index[-1].date()),
            "n_train":        int(len(data)),
            "horizon":        cfg.horizon,
            "direction_thr":  cfg.direction_thr,
            "conf_threshold": cfg.conf_threshold,
            "atr_sl_mult":    cfg.atr_sl_mult,
            "atr_tp_mult":    cfg.atr_tp_mult,
            "pip_size":       meta_pair.get("pip",         0.0001),
            "spread_pips":    meta_pair.get("spread_pips", 1.5),
            "classes":        sorted(map(int, model.classes_)),
            "saved_at":       datetime.utcnow().isoformat(timespec="seconds"),
        }
        with open(os.path.join(out_dir, "metadata.json"), "w") as f:
            json.dump(metadata, f, indent=2, ensure_ascii=False)

        print(f"  [OK] Zapisano → {out_dir}")
        saved.append(metadata)

    print(f"\n{'='*58}")
    print(f"Zapisano {len(saved)}/{len(PAIR_MODEL_MAP)} modeli  →  {cfg.models_dir}/")
    return saved




@dataclass
class Signal:
    ticker:          str
    instrument:      str
    model_type:      str
    signal_date:     str

    direction:       str
    is_new:          bool  
    confidence:      float 
    prob_long:       float
    prob_short:      float
    prob_flat:       float

    execution:       str
    entry_price:     float
    sl_price:        float
    tp_price:        float
    sl_pips:         float
    tp_pips:         float
    risk_reward:     str
    spread_pips:     float

    current_close:   float
    atr:             float
    atr_pips:        float
    rsi_14:          float
    adx:             float
    bb_pos:          float
    macd_hist:       float
    stoch_k:         float
    trend_20_50:     str
    trend_50_200:    str

    horizon_days:    int
    train_period:    str


def generate_signals(cfg: PipelineConfig) -> List[Signal]:
    os.makedirs(cfg.signals_dir, exist_ok=True)
    signals = []

    for ticker, model_type in PAIR_MODEL_MAP.items():
        meta_pair = FOREX_PAIRS.get(ticker, {})
        name      = meta_pair.get("name", ticker)
        out_dir   = _model_dir(ticker, model_type, cfg.models_dir)
        meta_path = os.path.join(out_dir, "metadata.json")

        if not os.path.exists(meta_path):
            print(f"  [SKIP] {name}: brak modelu w {out_dir}. Uruchom --train.")
            continue

        with open(meta_path) as f:
            meta = json.load(f)
        model  = joblib.load(os.path.join(out_dir, "model.joblib"))
        scaler = joblib.load(os.path.join(out_dir, "scaler.joblib"))

        feature_cols = meta["feature_cols"]
        pip_size     = meta.get("pip_size",     0.0001)
        spread_pips  = meta.get("spread_pips",  1.5)
        thr          = meta.get("conf_threshold", cfg.conf_threshold)
        sl_mult      = meta.get("atr_sl_mult",    cfg.atr_sl_mult)
        tp_mult      = meta.get("atr_tp_mult",    cfg.atr_tp_mult)

        try:
            df = load_forex(ticker, start=cfg.start)
        except Exception as e:
            print(f"  [SKIP] {name}: {e}")
            continue

        feat = make_features(df)

        last_feat  = feat.iloc[[-1]]
        last_date  = last_feat.index[0]
        last_close = float(df["Close"].iloc[-1])

        missing = [c for c in feature_cols if c not in last_feat.columns]
        if missing:
            print(f"  [WARN] {name}: brakuje kolumn {missing[:3]}… Przetrenuj model.")
            continue

        X_row = last_feat[feature_cols]
        if X_row.isnull().any(axis=1).iloc[0]:
            print(f"  [WARN] {name}: NaN w cechach ostatniego baru — za mało historii.")
            continue

        X_scaled = scaler.transform(X_row)
        proba    = model.predict_proba(X_scaled)[0]
        classes  = list(model.classes_)

        p_up = float(proba[classes.index( 1)]) if  1 in classes else 0.0
        p_dn = float(proba[classes.index(-1)]) if -1 in classes else 0.0
        p_fl = max(0.0, 1.0 - p_up - p_dn)

        if p_up >= thr and p_up >= p_dn:
            direction  = "LONG"
            confidence = p_up
        elif p_dn >= thr and p_dn > p_up:
            direction  = "SHORT"
            confidence = p_dn
        else:
            direction  = "FLAT"
            confidence = max(p_up, p_dn)

        atr_val = float(feat["atr_14"].iloc[-1]) if "atr_14" in feat.columns else last_close * 0.005

        if direction == "LONG":
            sl = last_close - sl_mult * atr_val
            tp = last_close + tp_mult * atr_val
        elif direction == "SHORT":
            sl = last_close + sl_mult * atr_val
            tp = last_close - tp_mult * atr_val
        else:
            sl = tp = float("nan")

        sl_pips = abs(last_close - sl) / pip_size if not np.isnan(sl) else 0.0
        tp_pips = abs(tp - last_close) / pip_size if not np.isnan(tp) else 0.0
        rr_str  = f"1 : {tp_pips / sl_pips:.2f}" if sl_pips > 0 else "—"

        def _f(col: str, default: float = 0.0) -> float:
            if col not in feat.columns:
                return default
            v = feat[col].iloc[-1]
            return float(v) if not np.isnan(v) else default

        prev_path = os.path.join(cfg.signals_dir, f"{safe_filename(ticker)}_prev.json")
        prev_dir  = "FLAT"
        if os.path.exists(prev_path):
            with open(prev_path) as f:
                prev_dir = json.load(f).get("direction", "FLAT")
        with open(prev_path, "w") as f:
            json.dump({"direction": direction, "date": str(last_date.date())}, f)

        sig = Signal(
            ticker        = ticker,
            instrument    = name,
            model_type    = model_type,
            signal_date   = str(last_date.date()),
            direction     = direction,
            is_new        = (direction != prev_dir),
            confidence    = round(confidence, 4),
            prob_long     = round(p_up, 4),
            prob_short    = round(p_dn, 4),
            prob_flat     = round(p_fl, 4),
            execution     = "Wejście na otwarciu następnej sesji",
            entry_price   = round(last_close, 6),
            sl_price      = round(sl, 6) if not np.isnan(sl) else 0.0,
            tp_price      = round(tp, 6) if not np.isnan(tp) else 0.0,
            sl_pips       = round(sl_pips, 1),
            tp_pips       = round(tp_pips, 1),
            risk_reward   = rr_str,
            spread_pips   = spread_pips,
            current_close = round(last_close, 6),
            atr           = round(atr_val, 6),
            atr_pips      = round(atr_val / pip_size, 1),
            rsi_14        = round(_f("rsi_14"), 2),
            adx           = round(_f("adx"), 2),
            bb_pos        = round(_f("bb_pos"), 4),
            macd_hist     = round(_f("macd_hist"), 8),
            stoch_k       = round(_f("stoch_k"), 2),
            trend_20_50   = "Bullish" if _f("trend_20_50") > 0.5 else "Bearish",
            trend_50_200  = "Bullish" if _f("trend_50_200") > 0.5 else "Bearish",
            horizon_days  = meta.get("horizon", cfg.horizon),
            train_period  = f"{meta.get('train_start','')} → {meta.get('train_end','')}",
        )

        _print_signal(sig)
        signals.append(sig)

    if signals:
        _save_signals(signals, cfg)

    return signals




_DIR_ICON = {"LONG": "▲", "SHORT": "▼", "FLAT": "━"}
_DIR_COLOR = {
    "LONG":  "\033[92m",
    "SHORT": "\033[91m",
    "FLAT":  "\033[90m",
}
_RESET = "\033[0m"


def _print_signal(sig: Signal):
    icon    = _DIR_ICON.get(sig.direction, "?")
    color   = _DIR_COLOR.get(sig.direction, "")
    new_tag = "  [NOWY SYGNAŁ]" if sig.is_new else "  [bez zmian]"

    print(f"\n{'─'*62}")
    print(f"  {color}{icon} {sig.direction:5}{_RESET}  │  {sig.instrument:10}  │  {sig.signal_date}{new_tag}")
    print(f"  Model: {sig.model_type:15}  Confidence: {sig.confidence:.1%}   "
          f"(L:{sig.prob_long:.1%} / S:{sig.prob_short:.1%} / F:{sig.prob_flat:.1%})")
    print(f"{'─'*62}")
    print(f"  Kurs (close):  {sig.current_close:.5f}   ATR: {sig.atr:.5f} ({sig.atr_pips:.0f} pip)")

    if sig.direction != "FLAT":
        print(f"  ─── Parametry pozycji ───────────────────────────────")
        print(f"  Wykonanie:     {sig.execution}")
        print(f"  Entry:         {sig.entry_price:.5f}")
        print(f"  Stop Loss:     {sig.sl_price:.5f}  ({sig.sl_pips:.0f} pip)")
        print(f"  Take Profit:   {sig.tp_price:.5f}  ({sig.tp_pips:.0f} pip)")
        print(f"  Risk/Reward:   {sig.risk_reward}   Spread: ~{sig.spread_pips} pip")

    print(f"  ─── Kontekst rynkowy ────────────────────────────────")
    print(f"  RSI(14): {sig.rsi_14:6.2f}   ADX: {sig.adx:5.1f}   "
          f"Stoch %K: {sig.stoch_k:.1f}   BB pos: {sig.bb_pos:.2f}")
    print(f"  Trend SMA20/50:  {sig.trend_20_50:8}   "
          f"Trend SMA50/200: {sig.trend_50_200}")
    print(f"  MACD hist:     {sig.macd_hist:.6f}")
    print(f"  ─── Model ───────────────────────────────────────────")
    print(f"  Trenowany na:  {sig.train_period}")
    print(f"  Horyzont:      {sig.horizon_days} dni")



def _save_signals(signals: List[Signal], cfg: PipelineConfig):
    out_dir = cfg.signals_dir
    os.makedirs(out_dir, exist_ok=True)
    ts      = datetime.utcnow().strftime("%Y%m%d_%H%M")
    rows    = [asdict(s) for s in signals]
    df_all  = pd.DataFrame(rows)

    full_csv = os.path.join(out_dir, f"signals_{ts}.csv")
    df_all.to_csv(full_csv, index=False)

    df_active = df_all[df_all["direction"] != "FLAT"]
    if not df_active.empty:
        active_csv = os.path.join(out_dir, f"active_signals_{ts}.csv")
        df_active.to_csv(active_csv, index=False)

    json_path = os.path.join(out_dir, f"signals_{ts}.json")
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(rows, f, indent=2, ensure_ascii=False)

    df_all.to_csv(os.path.join(out_dir, "signals_latest.csv"), index=False)
    with open(os.path.join(out_dir, "signals_latest.json"), "w", encoding="utf-8") as f:
        json.dump(rows, f, indent=2, ensure_ascii=False)

    txt_path = os.path.join(out_dir, f"report_{ts}.txt")
    _write_text_report(signals, txt_path, ts)

    n_active = sum(1 for s in signals if s.direction != "FLAT")
    n_new    = sum(1 for s in signals if s.is_new)
    n_long   = sum(1 for s in signals if s.direction == "LONG")
    n_short  = sum(1 for s in signals if s.direction == "SHORT")

    print(f"\n{'='*62}")
    print(f"  SYGNAŁY ZAPISANE  ({ts} UTC)")
    print(f"{'='*62}")
    print(f"  Wszystkich par:  {len(signals)}")
    print(f"  Aktywnych:       {n_active}  (Long: {n_long}, Short: {n_short})")
    print(f"  Nowych sygnałów: {n_new}")
    print(f"\n  Pliki:")
    print(f"  • {full_csv}")
    print(f"  • {json_path}")
    print(f"  • {txt_path}")
    print(f"  • {out_dir}/signals_latest.csv  (zawsze nadpisywany)")
    print(f"{'='*62}")


def _write_text_report(signals: List[Signal], path: str, ts: str):
    """Czytelny raport .txt gotowy do wysłania e-mailem / wklejenia."""
    lines = [
        "=" * 62,
        f"  FOREX ML SIGNAL REPORT  |  {ts} UTC",
        "=" * 62,
        "",
    ]

    active  = [s for s in signals if s.direction != "FLAT"]
    passive = [s for s in signals if s.direction == "FLAT"]

    if active:
        lines.append("  ▶  AKTYWNE POZYCJE")
        lines.append("  " + "─" * 58)
        for s in active:
            icon = _DIR_ICON.get(s.direction, "?")
            new  = "[NOWY]" if s.is_new else "[HOLD]"
            lines += [
                f"",
                f"  {icon} {s.direction}  {s.instrument}  {new}  —  {s.signal_date}",
                f"  Model: {s.model_type}  |  Confidence: {s.confidence:.1%}",
                f"  Entry:       {s.entry_price:.5f}",
                f"  Stop Loss:   {s.sl_price:.5f}  ({s.sl_pips:.0f} pip)",
                f"  Take Profit: {s.tp_price:.5f}  ({s.tp_pips:.0f} pip)",
                f"  R:R:         {s.risk_reward}   Spread: ~{s.spread_pips} pip",
                f"  Wykonanie:   {s.execution}",
                f"  RSI:{s.rsi_14:.1f}  ADX:{s.adx:.1f}  BB:{s.bb_pos:.2f}"
                f"  Trend:{s.trend_20_50}/{s.trend_50_200}",
                f"  ATR: {s.atr:.5f} ({s.atr_pips:.0f} pip)",
                f"  P(L):{s.prob_long:.1%}  P(S):{s.prob_short:.1%}  P(F):{s.prob_flat:.1%}",
            ]

    if passive:
        lines += ["", "  ━  FLAT (brak pozycji)", "  " + "─" * 58]
        for s in passive:
            lines.append(
                f"  {s.instrument:10}  Close:{s.current_close:.5f}"
                f"  RSI:{s.rsi_14:.1f}  Conf:{s.confidence:.1%}"
            )

    lines += [
        "",
        "=" * 62,
        "  LEGENDA:",
        "  Entry    — kurs do wykonania zlecenia (ostatni close)",
        "  SL       — stop loss (entry ± ATR × mnożnik)",
        "  TP       — take profit (entry ± ATR × mnożnik)",
        "  Confidence — prawdopodobieństwo wybranego kierunku",
        "  [NOWY]   — sygnał zmienił się od ostatniego generowania",
        "  Wykonanie — wejście na otwarciu następnej sesji",
        "=" * 62,
    ]

    with open(path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Forex ML Signal Pipeline",
        formatter_class=argparse.RawTextHelpFormatter,
    )
    parser.add_argument("--train",       action="store_true",
                        help="Trenuj i zapisz modele")
    parser.add_argument("--signals",     action="store_true",
                        help="Generuj sygnały (wymaga wcześniejszego --train)")
    parser.add_argument("--models-dir",  default="saved_models",
                        help="Folder z modelami (domyślnie: saved_models)")
    parser.add_argument("--signals-dir", default="live_signals",
                        help="Folder z sygnałami (domyślnie: live_signals)")
    parser.add_argument("--train-months",type=int, default=24,
                        help="Miesiące danych do trenowania (domyślnie: 24)")
    args = parser.parse_args()

    do_train   = args.train   or (not args.train and not args.signals)
    do_signals = args.signals or (not args.train and not args.signals)

    cfg = PipelineConfig(
        train_months    = args.train_months,
        horizon         = 5,
        direction_thr   = 0.003,
        conf_threshold  = 0.45,
        atr_sl_mult     = 1.5,
        atr_tp_mult     = 3.0,
        models_dir      = args.models_dir,
        signals_dir     = args.signals_dir,
        random_state    = 42,
        start           = "2020-01-01",
    )

    if do_train:
        print("\n" + "=" * 62)
        print("  FAZA 1: TRENING I ZAPIS MODELI")
        print("=" * 62)
        train_and_save(cfg)

    if do_signals:
        print("\n" + "=" * 62)
        print("  FAZA 2: GENEROWANIE SYGNAŁÓW")
        print("=" * 62)
        sigs = generate_signals(cfg)
        if not sigs:
            print("\n  Brak sygnałów. Sprawdź czy modele są wytrenowane (--train).")
            sys.exit(1)
