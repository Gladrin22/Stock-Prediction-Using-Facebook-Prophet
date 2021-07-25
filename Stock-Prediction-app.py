import streamlit as st                 #importting the packages 
from datetime import date 
import yfinance as yf

from fbprophet import Prophet 
from fbprophet.plot import plot_plotly
from plotly import graph_objs as go


START = "2015-01-01"    # fixing the starting date      data frequencies !(From START to TODAY)

TODAY = date.today().strftime("%Y-%m-%d")  #Until today 


st.title('Stock Prediction:')   #Title of our web-app
st.write('(Tesla , Google , Microsoft, Facebook , Nvidia , Paypal , Adobe , Netflix)')

stocks = ('TSLA','GOOG','MSFT','FB','AAPL','NVDA','PYPL','ADBE','NFLX')  #Stocks we are using to predict

selected_stocks = st.selectbox('Select Dataset for prediction',stocks)   

n_years = st.slider("Years of prediction :",1,5)   #Number of years to do the prediction 
period = n_years * 365

@st.cache

def load_data(ticker):                               # A function to download the data of the selected stocks
    data = yf.download(ticker,START , TODAY)
    data.reset_index(inplace = True)
    return data

data_load_state = st.text('Loading Data...')         #Text to be displayed before and after downloading of the data 
data = load_data(selected_stocks)
data_load_state.text('Done !!')

st.subheader('The Data:')
st.write(data.tail())

def plot_data():
    fig= go.Figure()
    fig.add_trace(go.scatter(x = data['Date'], y =data['Open'], name = 'Stock_Open'))         # Getting and tabulating the data 
    fig.add_trace(go.scatter(x = data['Date'], y =data['Close'], name = 'Stock_Close'))       # of stock open and close 
    fig.layout.update(title_text ="Time Series data with ranger slider",xaxis_rangeslider_visible = True)
    st.plotly_chart(fig)

    plot_data

# P R E D I C T I O N -----> with Facebook Prophet

data_frame_train = data [['Date','Close']]
data_frame_train = data_frame_train.rename(columns={"Date":"ds","Close":"y"})

p = Prophet()
p.fit(data_frame_train)

future = p.make_future_dataframe(periods=period)
forecast = p.predict(future)

# Visualizing the output ....

st.subheader('Predicted output:')
st.write(forecast.tail())

st.subheader(f'The Forecast for {n_years} year :')
fig1 = plot_plotly(p, forecast)
st.plotly_chart(fig1)



st.write("Forecast components")
fig2 = p.plot_components(forecast)
st.write(fig2)



