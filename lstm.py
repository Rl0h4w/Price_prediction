import pandas as pd
import numpy as np
import seaborn as sns
import tensorflow as tf
import os
from sklearn.preprocessing import LabelEncoder, MinMaxScaler, StandardScaler
import joblib
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
import re


def get_dataframe():
    """
    Загружает данные за 2018, 2019 и 2020 годы и объединяет их в один DataFrame.

    Returns:
        pd.DataFrame: Объединенный DataFrame.
    """
    df1 = pd.read_csv("data/Данные 2018 год.xlsx - Лист1.csv")
    df2 = pd.read_csv("data/Данные 2019 год.xlsx - Лист1.csv")
    df3 = pd.read_csv("data/Данные 2020 год.xlsx - Лист1.csv")
    df = pd.concat([df1, df2, df3], ignore_index=True)
    return df

def preprocess(df):
    """
    Выполняет предварительную обработку данных.

    Args:
        df (pd.DataFrame): Исходный DataFrame.

    Returns:
        pd.DataFrame: Обработанный DataFrame.
    """
    df['Дата'] = pd.to_datetime(df['Дата'])
    df['Год'] = df['Дата'].dt.year
    df['Месяц'] = df['Дата'].dt.month
    df['День'] = df['Дата'].dt.day
    df['Час'] = df['Дата'].dt.hour
    df['ДН'] = df['Дата'].dt.weekday
    
    df['Месяц_sin'] = np.sin(2 * np.pi * df['Месяц'] / 12)
    df['Месяц_cos'] = np.cos(2 * np.pi * df['Месяц'] / 12)
    df['День_sin'] = np.sin(2 * np.pi * df['День'] / 31)
    df['День_cos'] = np.cos(2 * np.pi * df['День'] / 31)
    df['Час_sin'] = np.sin(2 * np.pi * df['Час'] / 24)
    df['Час_cos'] = np.cos(2 * np.pi * df['Час'] / 24)
    df['ДН_sin'] = np.sin(2 * np.pi * df['ДН'] / 7)
    df['ДН_cos'] = np.cos(2 * np.pi * df['ДН'] / 7)
    
    df.drop(['Год', 'Месяц', 'День', 'Час', 'ДН'], axis=1, inplace=True)
    
    del df["Дата"]
    del df["ЦЗ"]
    return df

def split(df, size_of_train=30):
    """
    Разбивает данные на обучающую и тестовую выборки.

    Args:
        df (pd.DataFrame): Исходный DataFrame.
        size_of_train (int): Размер обучающей выборки в днях.

    Returns:
        tuple: Кортеж из обучающих и тестовых выборок (x, y).
    """
    rows_per_day = 360
    days_per_month = size_of_train
    rows_per_month = rows_per_day * days_per_month
    n_full_months = len(df) // rows_per_month
    df = df.iloc[:n_full_months * rows_per_month]
    monthly_splits = np.array_split(df, n_full_months)
    
    x = np.array([month.to_numpy() for month in monthly_splits])[:-1]
    y = np.array([[day_chunk["Цена продажи, руб./МВт*ч"].mean() for day_chunk in np.array_split(month, days_per_month)] for month in monthly_splits])[1:]
    
    x = x.reshape((n_full_months - 1, rows_per_month, df.shape[1]))
    y = y.reshape((n_full_months - 1, days_per_month))
    
    return x, y


def show_history(history):
    """
    Отображает график потерь во время обучения и валидации модели.

    Args:
        history (History): История обучения модели.
    """
    train_loss = history.history['loss']
    val_loss = history.history['val_loss']
    epochs = range(1, len(train_loss) + 1)
    plt.plot(epochs, train_loss, 'b', label='Train Loss')
    plt.plot(epochs, val_loss, 'r', label='Validation Loss')
    plt.title('Training and Validation Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.show()

def show_dists(df):
    """
    Отображает распределения всех признаков в DataFrame.

    Args:
        df (pd.DataFrame): Исходный DataFrame.
    """
    for column in df.columns:
        sns.histplot(data=df, x=column)
        plt.show()

def get_model(shape):
    """
    Создает и компилирует модель LSTM.

    Args:
        shape (tuple): Форма входных данных.

    Returns:
        Model: Скомпилированная модель LSTM.
    """
    inputs = tf.keras.layers.Input(shape=shape[1:])

#     x = tf.keras.layers.LSTM(512, return_sequences=True)(inputs)
#     x = tf.keras.layers.Dropout(0.5)(x)
    
#     x = tf.keras.layers.LSTM(256, return_sequences=True)(inputs)
#     x = tf.keras.layers.Dropout(0.5)(x)
    
    x = tf.keras.layers.LSTM(128)(inputs)
    x = tf.keras.layers.Dropout(0.5)(x)
    
    x = tf.keras.layers.Dense(64, activation='relu')(x)
    x = tf.keras.layers.Dropout(0.5)(x)
    
    outputs = tf.keras.layers.Dense(30)(x)
    
    model = tf.keras.models.Model(inputs=inputs, outputs=outputs)
    
    model.compile(optimizer=tf.keras.optimizers.Adam(0.001), loss="mse", metrics=["mae", "mse"])
    
    return model

def train_model():
    """
    Обучает модель на данных.
    """
    strategy = tf.distribute.MirroredStrategy()
    with strategy.scope():
        df = get_dataframe()
        df = preprocess(df)
        
        label_encoder = LabelEncoder()
        df["ЗСП"] = label_encoder.fit_transform(df["ЗСП"])
        
        x, y = split(df, 30)
        x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)
        
        scaler_x = StandardScaler()
        scaler_y = MinMaxScaler()

        x_train_shape = x_train.shape
        x_test_shape = x_test.shape
        y_train_shape = y_train.shape
        y_test_shape = y_test.shape

        x_train = x_train.reshape(-1, x_train_shape[-1])
        x_test = x_test.reshape(-1, x_test_shape[-1])
        
        y_train = y_train.reshape(-1, 1)
        y_test = y_test.reshape(-1, 1)

        x_train = scaler_x.fit_transform(x_train).reshape(x_train_shape)
        x_test = scaler_x.transform(x_test).reshape(x_test_shape)

        y_train = scaler_y.fit_transform(y_train).reshape(y_train_shape[0], y_train_shape[1])
        y_test = scaler_y.transform(y_test).reshape(y_test_shape[0], y_test_shape[1])

        joblib.dump(scaler_x, "scaler_x.pkl")
        joblib.dump(scaler_y, "scaler_y.pkl")
        joblib.dump(label_encoder, "label_encoder.pkl")
        model = get_model(x_train.shape)
        callbacks = [tf.keras.callbacks.ModelCheckpoint("lstm_model_loss:{val_loss:.3f}_mae:{val_mae:.3f}.keras", save_best_only=True, monitor="val_loss"),
            tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=40, restore_best_weights=True),
                    ]
        history = model.fit(x=x_train, y=y_train, validation_data=(x_test, y_test), callbacks=callbacks, epochs=200, shuffle=True)
        show_history(history)


def get_best_model():
    """
    Загружает лучшую сохраненную модель на основе наименьшей валидационной потери и MAE.

    Returns:
        Model: Лучшая модель.
    """
    model_dir = "/"
    model_files = [f for f in os.listdir(model_dir) if f.endswith(".keras")]
    
    pattern = r"lstm_model_loss:(\d+\.\d+)_mae:(\d+\.\d+).keras"
    models_info = []
    
    for file in model_files:
        
        match = re.match(pattern, file)
        
        if match:
            val_loss = float(match.group(1))
            val_mae = float(match.group(2))
            models_info.append((val_loss, val_mae, file))
            
    if not models_info:
        raise ValueError("Не найдены файлы моделей, соответствующие шаблону.")
    
    models_info.sort(key=lambda x: (x[0], x[1]))

    best_model_file = models_info[0][2]
    best_model_path = os.path.join(model_dir, best_model_file)

    best_model = tf.keras.models.load_model(best_model_path)
    print(f"""Best model: val_loss: {models_info[0][0]}, val_mae: {models_info[0][1]}""")
    return best_model

        
if __name__=="__main__":
    try:
        model = get_best_model()
    except ValueError:
        train_model()
        model = get_best_model()
    scaler_x = joblib.load("scaler_x.pkl")
    scaler_y = joblib.load("scaler_y.pkl")
    label_encoder = joblib.load("label_encoder.pkl")
    
    data_to_prediction = get_dataframe()
    data_to_prediction["ЗСП"] = label_encoder.transform(data_to_prediction["ЗСП"])
    data_to_prediction = preprocess(data_to_prediction)[-360*30:]
    
    data_to_prediction = scaler_x.transform(data_to_prediction.to_numpy())
    data_to_prediction = data_to_prediction.reshape(1, 10800, 23)
    
    predicted_price = model.predict(data_to_prediction)
    predicted_price = scaler_y.inverse_transform(predicted_price)
    
    sns.lineplot(predicted_price[0])
