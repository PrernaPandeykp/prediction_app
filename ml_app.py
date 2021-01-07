import streamlit as st
import pandas_datareader as web
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from statsmodels.tsa.stattools import adfuller
from statsmodels.tsa.arima_model import ARIMA
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense,LSTM
from sklearn.metrics import mean_squared_error 
from sklearn.metrics import mean_absolute_error
from math import sqrt
from statsmodels.tools.eval_measures import rmse
from sklearn.metrics import mean_absolute_percentage_error
from pmdarima import auto_arima 
import warnings 

st.title("ML project")
st.write("""
# Prediction of Stock Prices
""")
st.subheader("Visualize the Time Series Data")
model=st.sidebar.selectbox("select MODEL",("ARIMA","LSTM") )
dataset=st.sidebar.selectbox("select dataset",("AAPL","GOOGL","INTC"))
with st.echo():
	#fetch dataset
	df=web.DataReader(dataset,data_source="yahoo",start="2009-01-01",end="2019-12-17")
	st.write("shape of data",df.shape)
	st.write(df.head())
st.set_option('deprecation.showPyplotGlobalUse', False)
select_column=st.multiselect("Select Columns to plot",df.columns)
plot_type=st.selectbox("Selct Type of plot",("line","bar","box","hist"))
if st.button("Plot"):
    cust_type=df[select_column].plot(kind=plot_type)
    st.write(cust_type)
    st.pyplot()


prd_col=st.selectbox("On which attribute you are going to predcit",df.columns)
df1=df[prd_col]
if model=="ARIMA":
    t1=0
    t2=0
    st.header("ARIMA Model Implementation")
    st.subheader("Check for stationarity")
    st.subheader("Augmented Dickey-Fuller Test")
    #adf_test
    def check_stationarity(ts_data):
    
    # Rolling statistics
        roll_mean = ts_data.rolling(20).mean()
        roll_std = ts_data.rolling(20).std()
    
    # Plot rolling statistics
        fig, ax = plt.subplots()
        if st.button("Plot",ts_data):
	 
            ax.plot(ts_data)
            ax.plot(roll_mean)
            ax.plot(roll_std)
            ax.set_title("plot among original data, roll mean & roll_standard_deviation",fontsize = 12)
            st.pyplot(fig)
    
        dftest=adfuller(ts_data,autolag="AIC")
        st.write("ADF:",dftest[0])
        st.write("P-Value:",dftest[1])
        global t1
        t1=dftest[0]
        global t2
        t2=dftest[1]

    def fit(stat_data):
        st.header("Fitting the Model")
        with st.echo():
            #To find the order of ARIMA model
            warnings.filterwarnings("ignore") 
            st.write("Best Model ",auto_arima(df1,trace=True,suppress_warnings = True))
    

        train = stat_data.iloc[:int(len(df)*0.90)] 
        test =stat_data.iloc[int(len(df)*0.90):]
        #a=st.selectbox("orders",("order=(2,0,3)","order=(1,0,1)"))
        user_input = st.text_input("input order", "2 1 0")
        model = ARIMA(stat_data,order = tuple(map(int, user_input.split())))
        fit_model=model.fit()
        #fit_model.summary()

        start = len(train) 
        end = len(train) + len(test) - 1
        predictions = fit_model.predict(start=start, end=end,typ = 'levels').rename("ARIMA Predictions") 
#plot predictions and actual values 
        st.subheader("Plot predictions and actual values")
        fig, ax = plt.subplots()
        ax.set_title("predictions and actual values",fontsize = 12)
        
        ax.plot(predictions)
        ax.plot(test)
        st.pyplot(fig)
    
        model = ARIMA(stat_data,order =tuple(map(int, user_input.split())))
        result = model.fit() 

# Forecast for the next 1 year 

        forecast = result.predict(start =start,end =end,typ = 'levels').rename('Forecast') 

# Plot the forecast values 
        st.subheader("Plot the forecast values")
	fig, ax = plt.subplots()
        ax.set_title("oringial data with forecast values",fontsize = 12)
        ax.plot(stat_data)
        ax.plot(forecast)
        st.pyplot(fig)
        
        st.subheader("Errors")    
        st.write('mean_absolute_percentage_error', round(mean_absolute_percentage_error(test,predictions),5 ))   
        st.write('MSE ', round(mean_squared_error(test,predictions),5))
        st.write("mean_absolute_error",round(mean_absolute_error(predictions, test),5))
           
    st.markdown("""
    A Stationary series has no trend, its variations around its mean have a constant amplitude.  
    For stationary: P-value< 0.05 and ADF < 2.91""")
    check_stationarity(df1)
    if (t1<2.91 and t2<0.05)==True :
        st.success("Dataset become stationary")
        fit(df1)
    else:
        st.warning("not stationary")
        st.write("Taking logrithmic transformatiom")
        df1_log = np.log(df1)
        #df1_log.dropna(inplace=True)
        check_stationarity(df1_log)
        if (t1<2.91 and t2<0.05)==True :
            st.write(t1,t2)
            st.success("Dataset become stationary")
            fit(df1_log) 
        else:
            st.warning("not stationary")
            st.write("Taking log differencing transformatiom")
            df1_log_diff =  df1_log - df1_log.shift(1)
            df1_log_diff.dropna(inplace=True)
            check_stationarity(df1_log_diff)
            if (t1<2.91 and t2<0.05)==True :
                st.success("Dataset become stationary")
                fit(df1_log_diff) 
            else:
                df1_diff =  df1 - df1.shift()
                df1_diff.dropna(inplace=True)
                check_stationarity(df1_log_diff)
                fit(df1_diff) 

    #fit model
else:
    #preprocessing
    from sklearn.preprocessing import MinMaxScaler
    scaler = MinMaxScaler(feature_range=(0, 1)) 
    df1 = scaler.fit_transform(np.array(df1).reshape(-1,1))
    st.write(df1)
    #split train&test
    train_size=int(len(df1)*0.80)
    test_size=len(df1)-train_size
    train_data,test_data=df1[0:train_size,:],df1[train_size:len(df1), :1]
    st.write("Train and Test size",train_size,test_size)
    
    def create_dataset(dataset,time_step=1):
        dataX, dataY =[] ,[]
        for i in range(len(dataset)-time_step-1):
            a=dataset[i:(i+time_step),0]
            dataX.append(a)
            dataY.append(dataset[i+ time_step,0])
        return np.array(dataX),np.array(dataY)

    time_step=100
    x_train,y_train=create_dataset(train_data,time_step)
    x_test,y_test=create_dataset(test_data,time_step)

    st.subheader("After converting an array of values into a dataset matrix")
    st.write(x_train.shape,y_train.shape)
    st.write(x_test.shape,y_test.shape)
    
    # reshape input to be [samples, time steps, features]
    x_train=x_train.reshape(x_train.shape[0],x_train.shape[1],1)
    x_test=x_test.reshape(x_test.shape[0],x_test.shape[1],1)
    st.subheader("After reshaping")
    st.write(x_train.shape,x_test.shape)

    model=Sequential()
    model.add(LSTM(50,return_sequences=True,input_shape=(100,1)))
    model.add(LSTM(50,return_sequences=True))
    model.add(LSTM(50))
    model.add(Dense(1))
    model.compile(loss="mean_squared_error",optimizer="adam")

    model.fit(x_train, y_train,validation_data=(x_test,y_test),epochs=5, batch_size=1, verbose=2)

    train_pred=model.predict(x_train)
    test_pred=model.predict(x_test)
    train_predict=scaler.inverse_transform(train_pred)
    test_predict=scaler.inverse_transform(test_pred)

    ax.set_title("predictions and test values",fontsize = 12)
    ax.plot(test_pred)
    ax.plot(y_test)
    st.lineplot(fig)

    st.subheader("Errors")
    st.write("mean_square_error",round(np.mean(((test_pred- y_test)**2)),5))
    st.write("mean_absolute_error",round(mean_absolute_error(test_pred,y_test),5))
    st.write('mean_absolute_percentage_error', round(mean_absolute_percentage_error(test_pred,y_test),5 ))
	

