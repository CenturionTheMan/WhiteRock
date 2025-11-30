from typing import List, Tuple

import pandas as pd
import numpy as np

from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix
from sklearn.metrics import balanced_accuracy_score
import tensorflow as tf
from tensorflow.keras import layers, optimizers
from tensorflow.keras import backend as K
import gc
import time
from tensorflow.keras.utils import to_categorical
from sklearn.utils.class_weight import compute_class_weight
from imblearn.under_sampling import RandomUnderSampler

VERBOSE = 1
PRINT_MODEL = False

def cleanup(*args):
    for obj in args:
        try:
            del obj
        except:
            pass
    K.clear_session()
    gc.collect()
    time.sleep(1)
    gc.collect()
    time.sleep(5)
    

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
        weight_adj_factors : List['float'] = [1,1,1],
        under_sample_imbalanced = False
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
        self.weight_adj_factors = weight_adj_factors
        self.under_sample_imbalanced = under_sample_imbalanced
        
    def close(self):
        attrs_to_kill = [
            getattr(self, "model", None),
            getattr(self, "X_train", None),
            getattr(self, "y_train", None),
            getattr(self, "X_val", None),
            getattr(self, "y_val", None),
            getattr(self, "X_test", None),
            getattr(self, "y_test", None),
            getattr(self, "history", None),
        ]

        cleanup(*attrs_to_kill)

        for name in [
            "model", "X_train", "y_train", "X_val", "y_val",
            "X_test", "y_test", "history"
        ]:
            if hasattr(self, name):
                delattr(self, name)

        
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
        
        if self.under_sample_imbalanced:
            n_samples, n_timesteps, n_features = self.X_train.shape
            X_train_flat = self.X_train.reshape(n_samples, n_timesteps * n_features)

            rus = RandomUnderSampler(sampling_strategy='majority', random_state=42)
            
            X_resampled_flat, y_resampled = rus.fit_resample(X_train_flat, self.y_train)

            self.X_train = X_resampled_flat.reshape(-1, n_timesteps, n_features)
            self.y_train = y_resampled # y_train is now balanced
        
        self.y_train = to_categorical(self.y_train, num_classes=3)
        self.y_val = to_categorical(self.y_val, num_classes=3)
        self.y_test = to_categorical(self.y_test, num_classes=3)
        
        if PRINT_MODEL:
            print(f"Data prepared: {self.X_train.shape[0]} train samples, {self.X_val.shape[0]} val samples, {self.X_test.shape[0]} test samples.")
            train_df = pd.DataFrame(self.X_train.reshape(-1, len(self.feature_names)), columns=self.feature_names)
            print("Train data feature stats:")
            print(train_df.describe())
            if self.under_sample_imbalanced:
                print(f"y_train class distribution after undersampling: {np.sum(self.y_train, axis=0)}")


    def calculate_weights(self):
        y_integers = np.argmax(self.y_train, axis=1) 
        classes = np.unique(y_integers)
        weights = compute_class_weight(
            class_weight='balanced', 
            classes=classes, 
            y=y_integers
        )
        class_weight_dict = dict(zip(classes, weights))
        
        if 0 in class_weight_dict:
            class_weight_dict[0] *= self.weight_adj_factors[0]
        if 1 in class_weight_dict:
            class_weight_dict[1] *= self.weight_adj_factors[1]
        if 2 in class_weight_dict:
            class_weight_dict[2] *= self.weight_adj_factors[2]
        
        if PRINT_MODEL:
            print(f"Calculated Class Weights: {class_weight_dict}")
        
        return class_weight_dict
                
    def build_model(self, hidden_layers: List[tf.keras.layers.Layer]):
        model = tf.keras.Sequential()
        
        model.add(layers.InputLayer(shape=(self.seq_length, len(self.feature_names))))
        
        for layer in hidden_layers:
            new_layer = layer.__class__.from_config(layer.get_config())
            model.add(new_layer)
            
        model.add(layers.Dense(3, activation='softmax'))
        
        self.model = model
        self.model.compile(
            optimizer=optimizers.Adam(learning_rate=self.learning_rate),
            loss='categorical_crossentropy',
            metrics=[balanced_accuracy, tf.keras.metrics.AUC(name='auc_roc')]
        )
        
        if PRINT_MODEL:
            self.model.summary()
        
    def train(self):
        class_weights = self.calculate_weights()
        
        callbacks = [
            tf.keras.callbacks.ReduceLROnPlateau(monitor='loss', factor=0.5, patience=3, verbose=VERBOSE),
            tf.keras.callbacks.EarlyStopping(monitor='val_balanced_accuracy', mode='max', patience=30, restore_best_weights=True, verbose=1)
        ]

        val = (self.X_val, self.y_val) if self.val_split > 0 else None

        self.history = self.model.fit(
            self.X_train, self.y_train,
            epochs=self.epochs,
            validation_data=val,
            batch_size=self.batch_size,
            callbacks=callbacks,
            class_weight=class_weights,
            verbose=VERBOSE
        )
        
    def evaluate(self):
        preds_prob = self.model.predict(self.X_test, verbose=0)
        preds = np.argmax(preds_prob, axis=1)
        y_true = np.argmax(self.y_test, axis=1)
        acc = accuracy_score(y_true, preds)
        f1 = f1_score(y_true, preds, average='weighted')
        bal_acc = balanced_accuracy_score(y_true, preds)
        precision = tf.keras.metrics.Precision()(y_true, preds).numpy()
        recall = tf.keras.metrics.Recall()(y_true, preds).numpy()
        cm = confusion_matrix(y_true, preds)
        return {
            'accuracy': acc,
            'f1_score': f1,
            'balanced_accuracy': bal_acc,
            'precision': precision,
            'recall': recall,
            'confusion_matrix': cm
        }        
                
        
@tf.keras.utils.register_keras_serializable()
def balanced_accuracy(y_true, y_pred):
    y_pred_labels = tf.argmax(y_pred, axis=1)
    y_true_labels = tf.argmax(y_true, axis=1)
    cm = tf.math.confusion_matrix(y_true_labels, y_pred_labels, num_classes=3)
    per_class_acc = tf.linalg.diag_part(cm) / tf.reduce_sum(cm, axis=1)
    bal_acc = tf.reduce_mean(per_class_acc)
    return bal_acc
