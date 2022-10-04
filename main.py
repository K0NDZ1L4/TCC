import pandas as pd
import yfinance as yf
from datetime import timedelta
import numpy as np
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


def Data(name):
    ticker = yf.Ticker(name)
    df = pd.DataFrame(ticker.history(period='5y'))
    df.reset_index(inplace=True)
    df_data = pd.DataFrame({'Date': pd.date_range(df['Date'].iloc[-1] + timedelta(days=1), periods=30)})
    df = df.append(df_data)
    df.fillna(0, inplace=True)
    return df




Magalu = Data("BBAS3.SA")
Magalu = Magalu[['Date', 'Close']]
Magalu.set_index('Date', inplace=True)


train = Magalu.iloc[0:1100]
valid = Magalu.iloc[1100:]
x_train, y_train = [], []
for i in range(90, len(train)):
    x_train.append(Magalu.iloc[i-90:i])
    y_train.append(Magalu.iloc[i])
x_train, y_train = np.array(x_train), np.array(y_train)
x_train = np.reshape(x_train, (x_train.shape[0],x_train.shape[1],1))


model = Sequential()
model.add(LSTM(units=50, return_sequences=True, input_shape=(x_train.shape[1],1)))
model.add(LSTM(units=50))

model.compile(loss='mean_squared_error', optimizer='adam')
model.fit(x_train, y_train, epochs=4, batch_size=1, verbose=2)

inputs = Magalu[len(Magalu) - len(valid) - 90:].values
inputs = inputs.reshape(-1,1)


X_test = []
for i in range(90,inputs.shape[0]):
    X_test.append(inputs[i-90:i,0])
X_test = np.array(X_test)

X_test = np.reshape(X_test, (X_test.shape[0],X_test.shape[1],1))
closing_price = model.predict(X_test)

valid['Predictions'] = closing_price
print(valid[['Close', 'Predictions']].tail(20))
#Visualizando a Previsão
plt.rcParams.update({'font.size': 22})


plt.figure(figsize=(20,10))
train = Magalu[:1100]
t_2020 = train['2020']
valid = Magalu[1100:]

plt.ylabel('Preço da Ação')
plt.xlabel('Data')
plt.plot(train['Close'], label = "Treino")
plt.plot(valid['Close'], label = 'Observado')
plt.plot(valid['Predictions'], label = 'Previsão')
plt.legend(bbox_to_anchor=(0., 1.02, 1., .102), loc='lower left',
           ncol=2, mode="expand", borderaxespad=0.)


