import pandas as pd
import yfinance as yf
import datetime
import numpy as np
from statsmodels.tsa.seasonal import seasonal_decompose
from statsmodels.tsa.arima.model import ARIMA
from pmdarima.arima import auto_arima
import matplotlib.pyplot as plt
import plotly.express as px
from prophet import Prophet
from prophet.plot import add_changepoints_to_plot, plot_plotly, plot_components_plotly


# bibliotecas testes
from sklearn.preprocessing import MinMaxScaler
from keras.models import Sequential
from keras.layers import Dense, Dropout, LSTM
import matplotlib.pyplot as plt

'''
Action that being used:
MGLU3.SA
BBAS3.SA
WEGE3.SA
PETR3.SA
'''


def data(name):
    ticker = yf.Ticker(name)
    df = pd.DataFrame(ticker.history(period='5y')['Close'])
    df.reset_index(inplace=True)
    df['Date'] = df['Date'].dt.strftime('%Y-%m-%d')
    df.set_index('Date', inplace=True)
    # df_data = pd.DataFrame({'Date': pd.date_range(df['Date'].iloc[-1] + timedelta(days=1), periods=30)})
    # df = df.append(df_data)
    df.fillna(0, inplace=True)
    return df

def arima(data):
    modelo = auto_arima(data, suppress_warnings=True, error_action='ignore')
    previsoes = modelo.predict(15)
    treinamento = data[:len(data) - 15]
    teste = data[len(data) - 15:]

    modelo2 = auto_arima(treinamento, suppress_warnings=True, error_action='ignore')
    previsoes = pd.DataFrame(modelo2.predict(n_periods=365, index='Date'))
    previsoes.columns = ['previsoes']

    plt.figure(figsize=(20, 10))
    plt.plot(treinamento, label='Treinamento')
    plt.plot(teste, label='Teste')
    plt.plot(previsoes, label='Previsões')
    plt.legend()
    plt.show()

def prophet(data):
    data.reset_index(inplace=True)
    data.rename(columns={"Date": "ds", "Close": "y"}, inplace=True)
    modelo = Prophet()
    modelo.fit(data)
    futuro = modelo.make_future_dataframe(periods=8)
    previsoes = modelo.predict(futuro)
    modelo.plot(previsoes, xlabel='Data', ylabel='Preço');
    plot_plotly(modelo, previsoes)
    plt.show()

def lstm(data):
    # separar treino e teste
    train = data[:-90].copy()
    valid = data[-90:].copy()

    # Suavizar dados
    scaler = MinMaxScaler(feature_range=(0, 1))
    scaled_data = scaler.fit_transform(data)


    # 90 dias historico para cada dado
    x_train, y_train = [], []
    for i in range(90, len(train)):
        x_train.append(scaled_data[i - 90:i, 0])
        y_train.append(scaled_data[i, 0])
    x_train, y_train = np.array(x_train), np.array(y_train)

    x_train = np.reshape(x_train, (x_train.shape[0], x_train.shape[1], 1))

    # Criando o modelo LSTM
    model = Sequential()
    model.add(LSTM(units=50, return_sequences=True, input_shape=(x_train.shape[1], 1)))
    model.add(LSTM(units=50))

    # funcao para minimizar o erro quadratico medio
    model.compile(loss='mean_squared_error', optimizer='adam')
    model.fit(x_train, y_train, epochs=4, batch_size=1, verbose=2)

    #
    inputs = data[len(data) - len(valid) - 90:].values
    inputs = inputs.reshape(-1, 1)
    inputs = scaler.transform(inputs)

    x_test = []
    for i in range(90, inputs.shape[0]):
        x_test.append(inputs[i - 90:i, 0])
    x_test = np.array(x_test)
    x_test = np.reshape(x_test, (x_test.shape[0], x_test.shape[1], 1))
    closing_price = model.predict(x_test)
    closing_price = scaler.inverse_transform(closing_price)
    closing_price = closing_price.tolist()
    closing_price_m = []
    for x in closing_price:
        closing_price_m.append(x[24])

    # Visualizando a Previsão
    plt.rcParams.update({'font.size': 22})

    plt.figure(figsize=(20, 10))
    train = data[:-90]
    valid = data[-90:]
    valid['Predictions'] = closing_price_m
    plt.ylabel('Preço da Ação')
    plt.xlabel('Data')
    plt.plot(train['Close'], label="Treino")
    plt.plot(valid['Close'], label='Observado')
    plt.plot(valid['Predictions'], label='Previsão')
    plt.legend(bbox_to_anchor=(0., 1.02, 1., .102), loc='lower left',
               ncol=2, mode="expand", borderaxespad=0.)
    plt.show()

data= data('MGLU3.SA')
prophet = lstm(data)