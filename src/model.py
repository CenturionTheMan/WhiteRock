from typing import List, Tuple
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix
import tensorflow as tf
from tensorflow.keras.layers import (
    InputLayer,
    LSTM,
    LayerNormalization,
    Dense,
    BatchNormalization,
    Dropout
)
from tensorflow.keras import optimizers


class FinancialLSTMModel:

    def __init__(
        self,
        csv_path : str,
        features_scales : List[Tuple['str', 'str']], 
        target_col : str,
        datetime_col :str,
        
        seq_length : int,
        batch_size : int,
        learning_rate : float,
        epochs : int,
        test_ratio : float,
        val_split : float,
    ):
        self.csv_path = csv_path
        self.features_scales = features_scales
        self.feature_names = [f[0] for f in features_scales]
        self.target_col = target_col
        self.datetime_col = datetime_col
        self.seq_length = seq_length
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.epochs = epochs
        self.test_ratio = test_ratio
        self.val_split = val_split
        self.model = None
        
    def prepare_data(self):
        df = pd.read_csv(self.csv_path, parse_dates=[self.datetime_col])
        df = df.sort_values(self.datetime_col).reset_index(drop=True)
        df = df[[self.datetime_col] + self.feature_names + [self.target_col]]
        df.dropna(inplace=True)
        
        feature_data = df[self.feature_names]
        target_data = df[[self.target_col]].values
        
        X, y, dates = [], [], []
        for i in range(self.seq_length, len(feature_data)):
            X.append(feature_data.iloc[i - self.seq_length:i].values)
            y.append(target_data[i])
            dates.append(df[self.datetime_col].values[i])
        
        X, y, dates = np.array(X), np.array(y), np.array(dates)
        
        test_size = int(len(X) * self.test_ratio)
        val_size = int((len(X) - test_size) * self.val_split)
        train_size = len(X) - test_size - val_size

        self.X_train, self.y_train = X[:train_size], y[:train_size]
        self.X_val, self.y_val = X[train_size:train_size + val_size], y[train_size:train_size + val_size]
        self.X_test, self.y_test = X[train_size + val_size:], y[train_size + val_size:]      
        
        for i, (feature, scale) in enumerate(self.features_scales):
            if scale == 'minmax':
                scaler = MinMaxScaler()
            elif scale == 'standard':
                scaler = StandardScaler()
            else:
                scaler = None
            
            if scaler:
                flat_X_train = self.X_train[:, :, i].reshape(-1, 1)
                scaler.fit(flat_X_train)
                
                self.X_train[:, :, i] = scaler.transform(flat_X_train).reshape(self.X_train.shape[0], self.seq_length)
                flat_X_val = self.X_val[:, :, i].reshape(-1, 1)
                self.X_val[:, :, i] = scaler.transform(flat_X_val).reshape(self.X_val.shape[0], self.seq_length)
                flat_X_test = self.X_test[:, :, i].reshape(-1, 1)
                self.X_test[:, :, i] = scaler.transform(flat_X_test).reshape(self.X_test.shape[0], self.seq_length)

                
    def build_model(self, hidden_layers: List[tf.keras.layers.Layer]):
        model = tf.keras.Sequential()
        
        model.add(InputLayer(shape=(self.seq_length, len(self.feature_names))))
        for layer in hidden_layers:
            model.add(layer)
        model.add(Dense(1, activation='sigmoid'))
        
        self.model = model
        self.model.compile(
            optimizer=optimizers.Adam(learning_rate=self.learning_rate),
            loss='binary_crossentropy',
            metrics=['accuracy']
        )
        
        
    def train(self):
        callbacks = [
            tf.keras.callbacks.ReduceLROnPlateau(monitor='loss', factor=0.5, patience=2, verbose=0),
            tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=15, restore_best_weights=True, verbose=1)
        ]

        val = (self.X_val, self.y_val) if self.val_split > 0 else None

        self.history = self.model.fit(
            self.X_train, self.y_train,
            epochs=self.epochs,
            validation_data=val,
            batch_size=self.batch_size,
            callbacks=callbacks,
            verbose=0
        )
        
    def evaluate(self):
        preds_prob = self.model.predict(self.X_test, verbose=0)
        preds = (preds_prob > 0.5).astype(int).flatten()
        y_true = self.y_test.flatten()

        first_correct = preds[0] == y_true[0]
        
        auc_roc = tf.keras.metrics.AUC(curve='ROC')(y_true, preds_prob).numpy()

        last_epoch_num = len(self.history.history['loss']) - 1

        return {
            "first_prediction_correct": first_correct,
            "accuracy": float(accuracy_score(y_true, preds)),
            "f1_score": float(f1_score(y_true, preds)),
            "precision": float(tf.keras.metrics.Precision()(y_true, preds).numpy()),
            "recall": float(tf.keras.metrics.Recall()(y_true, preds).numpy()),
            "auc_roc": float(auc_roc),
            "confusion_matrix": confusion_matrix(y_true, preds).tolist(),
            "last epoch num": last_epoch_num
        }