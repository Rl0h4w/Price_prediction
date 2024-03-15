import pandas as pd
import numpy as np
import seaborn as sns
import tensorflow as tf
from sklearn.preprocessing import LabelEncoder, MinMaxScaler, StandardScaler
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split

pd.set_option("display.max_columns", None)

def get_dataframe():
    df1 = pd.read_csv("data/Данные 2018 год.xlsx - Лист1.csv")
    df2 = pd.read_csv("data/Данные 2019 год.xlsx - Лист1.csv")
    df3 = pd.read_csv("data/Данные 2020 год.xlsx - Лист1.csv")
    df = pd.concat([df1, df2, df3], ignore_index=True)
    return df

def preprocess(df):
    df['Дата'] = pd.to_datetime(df['Дата'])
    df["День недели"] = df['Дата'].dt.weekday
    df["Номер месяца"] = df["Дата"].dt.month
    df["Дата"] = df["Дата"].astype(int)//10**9
    del df["ЦЗ"]
    mean_of_price = df["Цена продажи, руб./МВт*ч"].mean()
    df["Цена продажи, руб./МВт*ч"] = df["Цена продажи, руб./МВт*ч"].apply(lambda x: x if x >=300 and x <= 2500 else mean_of_price)
    df["Наличие АЭС"] = df["План АЭС, МВт*ч"].apply(lambda x: 1 if x!= 0 else 0)
    df["Наличие ГЭС"] = df["План ГЭС, МВт*ч"].apply(lambda x: 1 if x!= 0 else 0)
    df["Наличие ТЭС"] = df["План ТЭС, МВт*ч"].apply(lambda x: 1 if x!= 0 else 0)
    df["Наличие потребления"] = df["Потребление, МВт*ч"].apply(lambda x: 1 if x!= 0 else 0)
    df["Наличие экспорта"] = df["Экспорт, МВт*ч"].apply(lambda x: 1 if x!= 0 else 0)
    df["Наличие импорта"] = df["Импорт, МВт*ч"].apply(lambda x: 1 if x!= 0 else 0)
    
    scaler_dict = dict()
    for column_name in df.keys():
        if column_name in ["План ВИЭ, МВт*ч",  "Номер месяца", "День недели", "Час", "Дата","Ценопринимающее предложение, МВт*ч", "Ценопринимание сверх минимальной мощности, МВт*ч", "Цена продажи, руб./МВт*ч"]:
            scaler = StandardScaler()
            df[column_name] = scaler.fit_transform(df[column_name].values.reshape(-1, 1))
            scaler_dict[column_name] = scaler
        elif column_name =="ЗСП":
            scaler = LabelEncoder()
            df[column_name] = scaler.fit_transform(df[column_name])
            scaler_dict[column_name] = scaler
        elif column_name in ["Наличие АЭС", "Наличие ГЭС", "Наличие ТЭС", "Наличие потребления", "Наличие экспорта", "Наличие импорта"]:
            continue
        else:
            scaler = MinMaxScaler()
            df[column_name] = scaler.fit_transform(df[column_name].values.reshape(-1, 1))
            scaler_dict[column_name] = scaler
    return df, scaler_dict["Цена продажи, руб./МВт*ч"]
        
def split(df, size_of_train):
    days = np.split(df, len(df)//360)
    x = []
    y = []
    for i in range(len(days)-(size_of_train+30)):
        x_i = np.array(list(map(lambda x: x.to_numpy(), days[i:i+size_of_train])))
        y_i = np.array(list(map(lambda x: x.mean(), map(lambda x: np.array(x["Цена продажи, руб./МВт*ч"]), days[i+size_of_train:i+size_of_train+30]))))
        x.append(x_i)
        y.append(y_i)
    x = np.array(x)
    x = x.reshape((x.shape[0], x.shape[1]*x.shape[2], x.shape[3]))
    y = np.array(y)
    return x, y

def get_model(shape):
    inputs = tf.keras.layers.Input(shape=shape[1:])
    x = tf.keras.layers.LSTM(256, return_sequences=True)(inputs)
    x = tf.keras.layers.LSTM(128, return_sequences=True)(x)
    x = tf.keras.layers.LSTM(64)(x)
    outputs = tf.keras.layers.Dense(30)(x)
    model = tf.keras.models.Model(inputs=inputs, outputs=outputs)
    model.compile(optimizer=tf.keras.optimizers.Adam(0.001), loss = "mse", metrics=["mae", "mse"])
    return model


def show_history(history):
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
    for column in df.columns:
        sns.histplot(data=df, x=column)
        plt.show()
    
if __name__ == "__main__":
    df = get_dataframe()
    df, scaler = preprocess(df)
    x, y = split(df, size_of_train=30)
    # x_train, x_val, y_train, y_val = train_test_split(x, y, test_size=0.2, random_state=42)
    # model = get_model(x_train.shape)
    # callbacks = [tf.keras.callbacks.ModelCheckpoint("lstm_model_val_mae:{val_mae:.4f}.keras", save_best_only=True, monitor="val_mae"), tf.keras.callbacks.EarlyStopping(monitor='val_mae', patience=10)]
    # history = model.fit(x=x_train, y=y_train, validation_data=[x_val, y_val], callbacks=callbacks, epochs = 100, shuffle=True)
    # show_history(history)
    model = tf.keras.models.load_model("lstm_model_val_mae_0.1597.keras") #input_shape = (None, 10800, 25) 
    predict_data = df.iloc[-360*30:].to_numpy()  # shape = (10800, 25)
    predict_data = np.expand_dims(predict_data, axis=0)
    raw_result = model.predict(predict_data)
    result = scaler.inverse_transform(raw_result)
    sns.lineplot(x=range(1, 31), y=result.flatten())
    plt.show()
    
    

