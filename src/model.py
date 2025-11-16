import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix
from tensorflow.keras import layers, optimizers
import tensorflow as tf


class FinancialLSTMModel:

    def __init__(
        self,
        csv_path,
        date_col,
        features, 
        target,
        seq_length=60,
        batch_size=32,
        learning_rate=0.001,
        epochs=100,
        test_ratio=0.1,
        val_split=0.1,
        shuffle=False,
        training_ranges=None,
        testing_ranges=None
    ):
        self.test_ratio = test_ratio
        self.shuffle = shuffle
        self.csv_path = csv_path
        self.date_col = date_col
        self.features = features  # holds tuples now
        self.feature_names = [f[0] for f in features]
        self.target = target
        self.seq_length = seq_length
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.epochs = epochs
        self.val_split = val_split
        
        self.training_ranges = training_ranges or []
        self.testing_ranges = testing_ranges or []

        self.model = None
        self.df = None

        self.X_train = self.X_val = self.X_test = None
        self.y_train = self.y_val = self.y_test = None

        # Dictionary: {feature_name: scaler_object or None}
        self.feature_scalers = {}
        self.dates = None

    # ----------------------- DATA PREPARATION ----------------------- #

    def _find_closest_date_index(self, dates_array, target_date):
        target = np.datetime64(target_date)
        return np.argmin(np.abs(dates_array - target))

    def _get_samples_in_range(self, start_date, end_date):
        beg = self._find_closest_date_index(self.dates, start_date)
        end = self._find_closest_date_index(self.dates, end_date)
        return self.X[beg:end+1], self.y[beg:end+1]

    def prepare_data(self):
        self.df = pd.read_csv(self.csv_path, parse_dates=[self.date_col])
        self.df = self.df.sort_values(self.date_col).reset_index(drop=True)
        self.df = self.df[[self.date_col] + self.feature_names + [self.target]]
        self.df.dropna(inplace=True)
        
        if self.training_ranges.count() == 0 or self.testing_ranges.count() == 0:
            train_test_split = int(len(self.df) * (1 - self.test_ratio))
            date_train_beg = self.df[self.date_col].iloc[0]
            date_train_end = self.df[self.date_col].iloc[train_test_split - 1]
            date_test_beg = self.df[self.date_col].iloc[train_test_split]
            date_test_end = self.df[self.date_col].iloc[-1]
            self.training_ranges = [(date_train_beg, date_train_end)]
            self.testing_ranges = [(date_test_beg, date_test_end)]
            print(f"Training range: {self.training_ranges}")
            print(f"Testing range: {self.testing_ranges}")

        feature_data = self.df[self.feature_names]
        target_data = self.df[[self.target]].values

        X, y, dates = [], [], []
        for i in range(self.seq_length, len(feature_data)):
            X.append(feature_data.iloc[i - self.seq_length:i].values)
            y.append(target_data[i])
            dates.append(self.df[self.date_col].values[i])

        self.X = np.array(X)
        self.y = np.array(y)
        self.dates = np.array(dates)

        # ---------------- TRAIN / VALIDATION SPLIT ---------------- #

        X_train_val, y_train_val = [], []
        for beg, end in self.training_ranges:
            _x, _y = self._get_samples_in_range(beg, end)
            X_train_val.extend(_x)
            y_train_val.extend(_y)

        X_train_val, y_train_val = np.array(X_train_val), np.array(y_train_val)

        split_idx = int(len(X_train_val) * (1 - self.val_split))
        self.X_train = X_train_val[:split_idx]
        self.y_train = y_train_val[:split_idx]
        self.X_val = X_train_val[split_idx:]
        self.y_val = y_train_val[split_idx:]

        # ---------------- TEST SET ---------------- #

        X_test, y_test = [], []
        for beg, end in self.testing_ranges:
            _x, _y = self._get_samples_in_range(beg, end)
            X_test.extend(_x)
            y_test.extend(_y)

        self.X_test = np.array(X_test)
        self.y_test = np.array(y_test)


        if self.val_split > 0.0:
            print(f"Training samples: {len(self.X_train)}, Validation samples: {len(self.X_val)}, Testing samples: {len(self.X_test)}")
        else:
            print(f"Training samples: {len(self.X_train)}, Testing samples: {len(self.X_test)}")

        # ---------------- PER-FEATURE SCALING ---------------- #

        for idx, (feat_name, scaler_type) in enumerate(self.features):

            if scaler_type == "standard":
                scaler = StandardScaler()
            elif scaler_type == "minmax":
                scaler = MinMaxScaler()
            else:
                scaler = None

            if scaler is not None:
                # Fit only using training data
                scaler.fit(self.X_train[:, :, idx].reshape(-1, 1))

                # Apply scaling across all datasets
                self.X_train[:, :, idx] = scaler.transform(self.X_train[:, :, idx].reshape(-1, 1)).reshape(
                    self.X_train.shape[0], self.X_train.shape[1]
                )
                if self.val_split > 0.0:
                    self.X_val[:, :, idx] = scaler.transform(self.X_val[:, :, idx].reshape(-1, 1)).reshape(
                        self.X_val.shape[0], self.X_val.shape[1]
                    )
                self.X_test[:, :, idx] = scaler.transform(self.X_test[:, :, idx].reshape(-1, 1)).reshape(
                    self.X_test.shape[0], self.X_test.shape[1]
                )

            self.feature_scalers[feat_name] = scaler

    # -------------------------- MODEL ----------------------------- #

    def build_model(self):
        inputs = layers.Input(shape=(self.seq_length, len(self.feature_names)))

        x = layers.LSTM(96, return_sequences=True, recurrent_dropout=0.1)(inputs)
        x = layers.LayerNormalization()(x)
        x = layers.LSTM(96, return_sequences=False, recurrent_dropout=0.1)(x)
        x = layers.LayerNormalization()(x)

        x = layers.Dense(128, activation='relu')(x)
        x = layers.BatchNormalization()(x)
        x = layers.Dropout(0.1)(x)

        x = layers.Dense(64, activation='relu')(x)
        x = layers.BatchNormalization()(x)

        outputs = layers.Dense(1, activation='sigmoid')(x)

        self.model = tf.keras.Model(inputs, outputs)
        self.model.compile(
            optimizer=optimizers.Adam(learning_rate=self.learning_rate),
            loss='binary_crossentropy',
            metrics=['accuracy']
        )

    # ------------------------ TRAINING ---------------------------- #

    def train(self):
        callbacks = [
            tf.keras.callbacks.ReduceLROnPlateau(monitor='loss', factor=0.5, patience=2),
            tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=15, restore_best_weights=True, verbose=1)
        ]

        val = (self.X_val, self.y_val) if self.val_split > 0 else None

        self.model.fit(
            self.X_train, self.y_train,
            epochs=self.epochs,
            validation_data=val,
            batch_size=self.batch_size,
            callbacks=callbacks,
            shuffle=self.shuffle,
            verbose=0
        )

    # ----------------------- EVALUATION --------------------------- #

    def evaluate(self):
        preds_prob = self.model.predict(self.X_test, verbose=0)
        preds = (preds_prob > 0.5).astype(int).flatten()
        y_true = self.y_test.flatten()

        first_correct = preds[0] == y_true[0]

        return {
            "first_prediction_correct": first_correct,
            "accuracy": float(accuracy_score(y_true, preds)),
            "f1_score": float(f1_score(y_true, preds)),
            "confusion_matrix": confusion_matrix(y_true, preds).tolist()
        }
