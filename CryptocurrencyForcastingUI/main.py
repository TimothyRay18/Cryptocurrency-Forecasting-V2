from flask import Flask, render_template, request
from flask_bootstrap import Bootstrap
from tensorflow import keras
from Skripsi import Preprocessing
import pandas as pd

app = Flask(__name__, static_folder='static', static_url_path='/static/')
Bootstrap(app)
app.jinja_env.globals.update(len=len)


@app.route("/", methods=['GET', 'POST'])
def index():
    if request.method == "POST":
        crypto = request.form.get("crypto")
        method = request.form.get("method")
        batch = request.form.get("batch")
        epoch = request.form.get("epoch")
        unit = request.form.get("unit")
        time = request.form.get("time")
        yhat, date_test, ytrue = forecast(crypto, method, batch, epoch, unit)
        return render_template('index.html', crypto=crypto, method=method, batch=batch, epoch=epoch, unit=unit,
                               time=time, yhat=yhat.reshape(-1).tolist()[-int(time):],
                               date_test=date_test.values.tolist()[-int(time):],
                               ytrue=ytrue.reshape(-1).tolist()[-int(time):])
    return render_template('index.html')


def prepare_dataset(name):
    df = pd.DataFrame()
    if name == 'BTC':
        df = pd.read_csv('dataset/Binance_BTCUSDT_1h.csv')
    elif name == 'ETH':
        df = pd.read_csv('dataset/Binance_ETHUSDT_1h.csv')
    elif name == 'LTC':
        df = pd.read_csv('dataset/Binance_LTCUSDT_1h.csv')
    df_5_input = Preprocessing.feature_selection(df)
    df_no_dup = Preprocessing.handle_duplicate(df_5_input)
    miss = Preprocessing.handle_missing_value(df_no_dup)
    x, scaler = Preprocessing.minmax_scale(miss)
    train, test = Preprocessing.splitting_data(x)
    train_X, train_y = Preprocessing.create_dataset(train, 25)
    test_X, test_y = Preprocessing.create_dataset(test, 25)
    sorted_df = df.sort_values(by=['Date']).copy()
    date = sorted_df['Date']
    date_test = date.loc[5204:]
    return train_X, train_y, test_X, test_y, date_test, scaler


def forecast(crypto, method, batch, epoch, unit):
    train_X, train_y, test_X, test_y, date_test, scaler = prepare_dataset(crypto)
    model_name = method + '_' + crypto + '(' + batch + ', ' + epoch + ', ' + unit + ').h5'
    model = keras.models.load_model('model/Model ' + method + ' ' + crypto + '/' + model_name)
    yhat = model.predict(test_X)
    ypred = Preprocessing.inverse_scaler(yhat, scaler)
    ytrue = Preprocessing.inverse_scaler(test_y.reshape(-1, 1), scaler)
    return ypred, date_test, ytrue


if __name__ == '__main__':
    app.run(port=5000, debug=True)
