import numpy as numpy
import pandas as pd
import matplotlib.pyplot as plt
from keras.models import load_model
import streamlit as st
import yfinance as yf
from datetime import date


def predictionBasedOnData():
    start = '2010-01-01'
    end = date.today().strftime("%Y-%m-%d")

    st.title('Stock & Forex Prediction Based On Data')

    stocks = ('GOOG', 'AAPL', 'MSFT', 'GME','HBL.KA', 'USDEUR=X','USDAED=X','USDCAD=X','USDGBP=X')
    selected_stock = st.selectbox('Select dataset for prediction', stocks)

    df = yf.download(selected_stock, start , end)
    df.head()

    #Describing Data
    st.subheader('Data from 2010 - 2024')
    st.write(df.describe())

    #Visualizations
    st.subheader('Closing Price vs Time chart')
    fig = plt.figure(figsize = (12,6))
    plt.plot(df.Close)
    st.pyplot(fig)


    st.subheader('Closing Price vs Time chart with 100MA')
    ma100 = df.Close.rolling(100).mean()
    fig = plt.figure(figsize = (12,6))
    plt.plot(ma100)
    plt.plot(df.Close)
    st.pyplot(fig)

    st.subheader('Closing Price vs Time chart with 100MA & 200MA')
    ma100 = df.Close.rolling(100).mean()
    ma200 = df.Close.rolling(200).mean()
    fig = plt.figure(figsize = (12,6))
    plt.plot(ma100, 'r')
    plt.plot(ma200, 'g')
    plt.plot(df.Close, 'b')
    st.pyplot(fig)

    #Splitting Data into Training and Testing
    data_training = pd.DataFrame(df['Close'][0:int(len(df)*0.70)])
    data_testing = pd.DataFrame(df['Close'][int(len(df)*0.70): int(len(df))])

    # print(data_trainign.shape)
    # print(data_testing.shape)

    from sklearn.preprocessing import MinMaxScaler
    scaler = MinMaxScaler(feature_range=(0,1))

    data_training_array = scaler.fit_transform(data_training)

    #splitting Data into x_train and y_train
    # x_train = []
    # y_train = []

    # for i in range(100, data_training_array.shape[0]):
    #     x_train.append(data_training_array[i-100: i])
    #     y_train.append(data_training_array[i,0]) 

    # x_train, y_train = np.array(x_train), np.array(y_train)

    # #ML Model
    # from keras.layers import Dense, Dropout, LSTM
    # from keras.models import Sequential

    # model = Sequential()
    # model.add(LSTM(units = 50, activation = 'relu',return_sequences = True, input_shape =(x_train.shape[1], 1)))
    # model.add(Dropout(0.2))

    # model.add(LSTM(units = 60, activation = 'relu',return_sequences = True))
    # model.add(Dropout(0.3))

    # model.add(LSTM(units = 80, activation = 'relu',return_sequences = True))
    # model.add(Dropout(0.4))

    # model.add(LSTM(units = 120, activation = 'relu'))
    # model.add(Dropout(0.5))

    # model.add(Dense(units = 1))
    # model.compile(optimizer='adam',loss = 'mean_squared_error')
    # mmodel.fit(x_train, y_train, epochs = 50)
    # model.save('keras_model.h5')
    #end ML MOdel


    #load my model
    model = load_model('keras_model.h5')

    #Testing Part
    past_100_days = data_training.tail(100)
    final_df = past_100_days._append (data_testing, ignore_index=True)
    input_data = scaler.fit_transform(final_df)

    x_test = []
    y_test = []

    for i in range(100, input_data.shape[0]):
        x_test.append(input_data[i-100: i])
        y_test.append(input_data[i, 0])

    x_test, y_test = numpy.array(x_test), numpy.array(y_test)
    y_predicted = model.predict(x_test)
    scaler = scaler.scale_
    scale_factor = 1/scaler[0]
    y_predicted = y_predicted * scale_factor
    y_test = y_test * scale_factor

    #final Graph
    st.subheader('Predictions Vs original')
    fig2 = plt.figure(figsize=(12,6))
    plt.plot(y_test, 'b', label = 'Original Price')
    plt.plot(y_predicted,'r',label = 'predicted Price')
    plt.xlabel('Time')
    plt.ylabel('Price')
    plt.legend()
    st.pyplot(fig2)