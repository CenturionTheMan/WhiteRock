from model import FinancialLSTMModel, cleanup
import pandas as pd
import os, random, numpy as np, tensorflow as tf



# SEED = 2222
# os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
# tf.config.experimental_run_functions_eagerly(False)
# random.seed(SEED)
# np.random.seed(SEED)
# tf.random.set_seed(SEED)

# -------------------------- PARAMS --------------------------

CSV_PATHS = [
    './../data/AAPL_1h.csv', 
    './../data/GOOGL_1h.csv', 
    './../data/^NDX_1h.csv', 
    './../data/^GSPC_1h.csv'
    ]

DATE_COL = 'Datetime'
TARGET = 'direction'

SEQ_LENGTH = 60
BATCH_SIZE = 32
LEARNING_RATE = 0.001
EPOCHS = 100
TEST_RATIO = 0.1
VAL_SPLIT = 0.1

REPS = 10
OUTPUT_DIR='./../res/'


# FEATURES = [
#     [("Close", "minmax")],

#     [("Close", "minmax"),
#      ("rsi_14", "minmax"),
#      ("rsi_28", "minmax"),
#      ("rsi_50", "minmax"),
#      ("rsi_7", "minmax"),],

#     [("Close", "minmax"),
#      ("macd", "standard"),
#      ],

#     [("Close", "minmax"),
#      ("ema_10", "standard"),
#      ("ema_20", "standard"),
#      ("ema_50", "standard"),
#      ("ema_100", "standard"),
#      ("ema_200", "standard"),
#      ],

#     [("Close", "minmax"),
#      ("stoch_k", "minmax"),
#      ("stoch_d", "minmax"),
#      ],

#     [("Close", "minmax"),
#      ("roc", "standard"),
#      ],

#     [("Close", "minmax"),
#      ("adx", "minmax"),
#      ("di_plus", "minmax"),
#      ("di_minus", "minmax"),
#      ],

#     [("Close", "minmax"),
#      ("atr_14", "standard"),
#      ("atr_20", "standard"),
#      ],

#     [("Close", "minmax"),
#      ("close_pos", "none"),
#      ],


#     [("Close", "minmax"),
#      ("body_range_ratio", "none"),
#      ],


#     [("Close", "minmax"),   
#      ("Volume", "minmax"),
#      ],

#         # ---------- Wariant A ----------
#      [
#         ("Close", "minmax"),
#         ("ema_20", "standard"),
#         ("ema_50", "standard"),
#         ("macd", "standard"),
#         ("rsi_14", "minmax"),
#         ("atr_20", "standard"),
#         ("volume_zscore_50", "standard"),
#     ],

#         # ---------- Wariant B ----------
#     [
#         ("Close", "minmax"),
#         ("rsi_14", "minmax"),
#         ("rsi_28", "minmax"),
#         ("stoch_k", "minmax"),
#         ("stoch_d", "minmax"),
#         ("bb_upper_20", "standard"),
#         ("bb_middle_20", "standard"),
#         ("bb_lower_20", "standard"),
#         ("atr_14", "standard"),
#     ],

#         # ---------- Wariant C ----------
#     [
#         ("Close", "minmax"),
#         ("adx", "minmax"),
#         ("atr_20", "standard"),
#         ("bb_upper_20", "standard"),
#         ("bb_middle_20", "standard"),
#         ("bb_lower_20", "standard"),
#         ("ema_20", "standard"),
#         ("ema_100", "standard"),
#         ("volume_zscore_50", "standard"),
#     ],

#         # ---------- Wariant D ----------
#     [
#         ("Close", "minmax"),
#         ("rsi_14", "minmax"),
#         ("rsi_50", "minmax"),
#         ("ema_10", "standard"),
#         ("ema_50", "standard"),
#         ("ema_200", "standard"),
#         ("atr_20", "standard"),
#         ("obv", "standard"),
#         ("bb_width_20", "standard"),
#     ],
# ]

FEATURES = []
data_tmp = pd.read_csv(CSV_PATHS[0], parse_dates=[DATE_COL])
all_cols = data_tmp.columns.tolist()
FEATURES = [[(feat, "minmax") for feat in all_cols if feat not in ['Datetime', 'returns', 'direction']]]
print("Using features:", FEATURES)

def build_hidden_layers():
    return [
        tf.keras.layers.LSTM(96, return_sequences=True, recurrent_dropout=0.1),
        tf.keras.layers.LayerNormalization(),
        tf.keras.layers.LSTM(96, return_sequences=False, recurrent_dropout=0.1),
        tf.keras.layers.LayerNormalization(),
        tf.keras.layers.Dense(128, activation='relu'),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.Dropout(0.1),
        tf.keras.layers.Dense(64, activation='relu'),
        tf.keras.layers.BatchNormalization(),
    ]


# -------------------------- VALIDATION CHECK --------------------------


for file in CSV_PATHS:
    if not os.path.exists(file):
        raise FileNotFoundError(f"File {file} does not exist.")

    df = pd.read_csv(file)
    for feature_set in FEATURES:
        for feature, _ in feature_set:
            if feature not in df.columns:
                raise ValueError(f"Feature {feature} not found in {file}.")


# -------------------------- TEST --------------------------

for r in range(0, REPS):
    res = []
    for file in CSV_PATHS:
        for idx, feat in enumerate(FEATURES):
            model = FinancialLSTMModel(
                csv_path=file,
                features_scales=feat,
                target_col="direction",
                datetime_col="Datetime",

                seq_length=SEQ_LENGTH,
                batch_size=BATCH_SIZE,
                learning_rate=LEARNING_RATE,
                epochs=EPOCHS,
                test_ratio=TEST_RATIO,
                val_split=VAL_SPLIT,
            )

            print(f" ---- Running file {file}, feature {idx+1}/{len(FEATURES)}, repetition {r+1}/{REPS} ---- ")

            model.prepare_data()
            model.build_model(build_hidden_layers())
            model.train()
            ev = model.evaluate()
            print(f" > Evaluation results: {ev}")

            res.append({
                "file_path": file,
                "feature_set_index": idx,
                "features": [f[0] for f in feat],
                "repetition": r,
                **ev
            })

            df_res = pd.DataFrame(res)
            df_res.to_csv(os.path.join(OUTPUT_DIR, f'model1_testing_rep{r}.csv'), index=False)
            print("")
            
            model.close()
            cleanup(model, df_res)
            tf.keras.backend.clear_session()
            tf.compat.v1.reset_default_graph()
    cleanup(res)
            

