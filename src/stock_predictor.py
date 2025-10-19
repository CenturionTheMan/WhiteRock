import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout, Input
import tabulate as tb
import joblib 
from tensorflow.keras.models import load_model

class StockPredictor:
    def __init__(self, df_train, scaler_features, scaler_target, model):
        self.df_base = df_train.copy()
        self.epochs = None
        self.batch_size = None
        self.seq_length = None
        self.features = None
        self.target = None
        self.X = None
        self.y = None
        self.scaler_features = scaler_features
        self.scaler_target = scaler_target
        self.model = model
        self.scaler_features_path = None
        self.scaler_target_path = None
        self.model_path = None
        
    
    @classmethod
    def create_new(cls, df_train, scaler_features_path, scaler_target_path, model_path, 
                          epochs, batch_size, seq_length, features, target):
        obj = cls(
            df_train=df_train,
            scaler_features=None,
            scaler_target=None,
            model=None
        )
        obj.scaler_features_path = scaler_features_path
        obj.scaler_target_path = scaler_target_path
        obj.model_path = model_path
        obj.epochs = epochs
        obj.batch_size = batch_size
        obj.seq_length = seq_length
        obj.features = features
        obj.target = target
        obj.__preprocess_data()
        obj.__train_model_full()
        return obj
    
    
    @classmethod
    def load_for_batching(cls, df_train, scaler_features_path, scaler_target_path, model_path, 
                          epochs, batch_size, seq_length, features, target):
        scaler_features = joblib.load(scaler_features_path)
        scaler_target = joblib.load(scaler_target_path)
        model = load_model(model_path)
        
        obj = cls(
            df_train=df_train,
            scaler_features=scaler_features,
            scaler_target=scaler_target,
            model=model
        )
        
        obj.epochs = epochs
        obj.batch_size = batch_size
        obj.seq_length = seq_length
        obj.features = features
        obj.target = target
        obj.preprocess_data()
        return obj
        
    def __preprocess_data(self):
        input_scaler = MinMaxScaler()
        scaled_features = input_scaler.fit_transform(self.df_base[self.features])
        
        target_values = self.df_base[self.target].values.reshape(-1, 1)
        target_scaler = MinMaxScaler()
        scaled_target = target_scaler.fit_transform(target_values)
        
        X, y = [], []

        for i in range(self.seq_length, len(scaled_features)):
            X.append(scaled_features[i - self.seq_length:i])
            y.append(scaled_target[i, 0])

        self.X, self.y = np.array(X), np.array(y)
        self.scaler_features = input_scaler
        self.scaler_target = target_scaler

    
    def __create_model(self):
        """
        Create LSTM model similar to the paper's architecture
        """
        model = Sequential()
        
        model.add(Input(shape=(self.seq_length, len(self.features))))
        
        model.add(LSTM(96, return_sequences=True))
        model.add(Dropout(0.2))
        
        model.add(LSTM(96, return_sequences=True))
        model.add(Dropout(0.2))
        
        model.add(LSTM(96, return_sequences=True))
        model.add(Dropout(0.2))
        
        model.add(LSTM(96))
        model.add(Dropout(0.2))
        
        model.add(Dense(1))
        
        model.compile(optimizer='adam', loss='mean_squared_error', metrics=['mae'])

        self.model = model
        
    def __train_model_full(self):
        if self.model is None:
            self.__create_model()
        
        self.model.fit(
            self.X, self.y,
            epochs=self.epochs,
            batch_size=self.batch_size,
            verbose=1
        )
        
    def predict(self, data_window):
        """
        Predict the next target value given a data window of shape (seq_length, n_features).
        """
        if len(data_window) != self.seq_length:
            raise ValueError(f"Expected {self.seq_length} rows, got {len(data_window)}")
            
        scaled_window = self.scaler_features.transform(data_window)
        X_input = np.expand_dims(scaled_window, axis=0)  # shape (1, seq_length, n_features)
        
        scaled_prediction = self.model.predict(X_input, verbose=0)
        prediction = self.scaler_target.inverse_transform(scaled_prediction)
        return prediction[0, 0]
    
    def update_model_online(self, data_window, target_value):
        """
        Update the model with a single training step using the given window and target value.
        
        Parameters:
            data_window (pd.DataFrame): The last seq_length rows of feature data.
            target_value (float): The actual target value for the next day.
        """
        scaled_window = self.scaler_features.transform(data_window)
        X_input = np.expand_dims(scaled_window, axis=0)  # shape (1, seq_length, n_features)

        y_scaled = self.scaler_target.transform(np.array([[target_value]]))  # shape (1, 1)

        self.model.train_on_batch(X_input, y_scaled)
        
    def save(self):
        self.model.save('lstm_stock_model.h5')
        joblib.dump(self.scaler_features, 'scaler_features.save')
        joblib.dump(self.scaler_target, 'scaler_target.save')
        self.df_base.to_csv('training_data.csv')
        
def increment_date(df, date, max_date, num_of_days):
    cur_date = pd.to_datetime(date) + pd.Timedelta(days=num_of_days)
    while cur_date <= max_date:
        if cur_date in df.index:
            return cur_date
        cur_date += pd.Timedelta(days=1)
    return None
        
