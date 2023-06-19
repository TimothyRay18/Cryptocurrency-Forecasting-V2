from pandas import read_csv
from pandas import DataFrame
from pandas import concat
from sklearn.preprocessing import MinMaxScaler
import tensorflow as tf
import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.layers import Bidirectional, LSTM
from tensorflow.keras import initializers
tf.random.set_seed(42)

class Preprocessing:
    def sort_df(df):
        df_sorted = df.sort_values(by=['Date']).copy()
        df_sorted_index = df_sorted.reset_index(drop=True)
        return df_sorted_index
    def minmax_scale(df):
        FEATURES = ['Open', 'High', 'Low', 'Volume USDT', 'Close']
        scaler = MinMaxScaler()
        df_scaled = scaler.fit_transform(np.array(df[FEATURES]))
        
        scaler_pred = MinMaxScaler()
        pred_scaled = scaler_pred.fit_transform(np.array(df['Close']).reshape(-1,1))
        return df_scaled, scaler_pred
    def inverse_scaler(pred, scaler):
        pred_inversed = scaler.inverse_transform(pred)
        return pred_inversed
    def splitting_data(df):
        training_size = int(len(df)*0.8)
        test_size = len(df)-training_size
        train_data,test_data = df[0:training_size,:],df[training_size:len(df),:]
        return train_data, test_data
    def create_dataset(dataset, time_step=1, index=4):
        dataX = []
        dataY = []
        
        for i in range(len(dataset)-time_step):
            dataX.append(dataset[i:(i+time_step)])
            dataY.append(float(dataset[i+time_step][index]))
        return np.array(dataX), np.array(dataY)

class Evaluation:
    def rmse (y, yhat):
        differences = [y[i] - yhat[i] for i in range(len(y))]
        squared_differences = [d**2 for d in differences]
        sum_squared_differences = sum(squared_differences)
        mean_squared_error = sum_squared_differences / len(y)
        return (mean_squared_error**0.5)[0]
    def mae (y, yhat):
        differences = [y[i] - yhat[i] for i in range(len(y))]
        absolute_differences = [abs(x) for x in differences]
        sum_absolute_difference = sum(absolute_differences)
        mean_absolute_error = sum_absolute_difference / len(y)
        return mean_absolute_error[0]
    def mape (y, yhat):
        divided_differences = [abs((y[i] - yhat[i])/y[i]) for i in range(len(y))]
        sum_absolute_difference = sum(divided_differences)
        mean_absolute_percentage_error = sum_absolute_difference / len(y)
        return (mean_absolute_percentage_error*100)[0]
    
class NeuralNetwork:
    def train_lstm(train_X, train_y, test_X, test_y, unit, epoch, batch):
        model = Sequential()
        model.add(LSTM(unit, kernel_initializer=initializers.GlorotUniform(seed=42), 
                       input_shape=(train_X.shape[1], train_X.shape[2])))
        model.add(Dense(units=1, kernel_initializer=initializers.GlorotUniform(seed=42)))
        model.compile(loss='mse',optimizer='adam')
        history = model.fit(train_X, train_y, epochs=epoch, batch_size=batch, 
                            validation_data=(test_X, test_y), verbose=2, shuffle=False)
        return model, history
    def train_bilstm(train_X, train_y, test_X, test_y, unit, epoch, batch):
        model = Sequential()
        model.add(Bidirectional(LSTM(unit, kernel_initializer=initializers.GlorotUniform(seed=42), 
                                     input_shape=(train_X.shape[1], train_X.shape[2]))))
        model.add(Dense(units=1, kernel_initializer=initializers.GlorotUniform(seed=42)))
        model.compile(loss='mse',optimizer='adam')
        history = model.fit(train_X, train_y, epochs=epoch, batch_size=batch, 
                            validation_data=(test_X, test_y), verbose=2, shuffle=False)
        return model, history
    def save_model(model, category, crypto_name, hyperparam):
        if category == 0:
            model.save('LSTM_'+ crypto_name +str(hyperparam)+'.h5')
        else:
            model.save('BiLSTM_'+ crypto_name +str(hyperparam)+'.h5')