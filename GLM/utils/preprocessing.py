import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler

class DataProcessor:
    def __init__(self, data_path="data/train.csv"):
        self.data_path = data_path
        self.scaler = StandardScaler()
        # Wichtige Spalten definieren
        self.numeric_cols = ["temp", "atemp", "humidity", "windspeed"]
        self.categorical_cols = ["season", "holiday", "workingday", "weather"]
        self.drop_cols = ["casual", "registered", "datetime"]

    def load_and_engineer(self):
        # Daten laden
        df = pd.read_csv(self.data_path)
        df['datetime'] = pd.to_datetime(df['datetime'])
        
        # Features generieren (wie vorher)
        dt = df['datetime'].dt
        df['hour'] = dt.hour.astype(np.int8)
        df['month'] = dt.month.astype(np.int8)
        
        df['hour_sin'] = np.sin(2 * np.pi * df['hour'] / 24).astype(np.float32)
        df['hour_cos'] = np.cos(2 * np.pi * df['hour'] / 24).astype(np.float32)
        
        # Kategorische Variablen umwandeln & aufräumen
        for col in self.categorical_cols:
            df[col] = df[col].astype('category')
            
        df = df.drop(columns=[col for col in self.drop_cols if col in df.columns] + ['hour', 'month'])
        return pd.get_dummies(df, drop_first=True, dtype=np.float32)

    def get_train_test_split(self, test_size=0.2):
        df = self.load_and_engineer()
        
        # Simpler chronologischer Split
        split_idx = int(len(df) * (1 - test_size))
        train, test = df.iloc[:split_idx], df.iloc[split_idx:]
        
        y_train = train['count'].values
        y_test = test['count'].values
        
        X_train = train.drop(columns=['count'])
        X_test = test.drop(columns=['count'])
        
        # Skalieren der numerischen Features (Wichtig für Fehlerfreiheit!)
        num_cols = [col for col in X_train.columns if col in self.numeric_cols]
        if num_cols:
            X_train[num_cols] = self.scaler.fit_transform(X_train[num_cols])
            X_test[num_cols] = self.scaler.transform(X_test[num_cols])
            
        return X_train, X_test, y_train, y_test