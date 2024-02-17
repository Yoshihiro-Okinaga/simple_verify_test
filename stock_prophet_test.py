import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import pandas.tseries.offsets as offsets
import matplotlib.dates as mdates
import datetime
import os
import sys
from sklearn import preprocessing
import japanize_matplotlib

sys.path.append("../utility")
import stock

if os.name == 'posix':
    from fbprophet import Prophet
elif os.name == 'nt':
    from prophet import Prophet

TEST_PERIOD_DAYS = 2
TEST_DAYS = 1
GRAPH_PERIOD_DAYS = 100

def load_data(url: str) -> pd.DataFrame:
    df = pd.read_csv(url, parse_dates=[0])
    df = df.sort_values(by='日付')
    df = df.query('日付' + ' >= "2018/01/01"')
    df = df[['日付', '始値']]
    df.columns = ['ds', 'y']
    return df

def split_data(df: pd.DataFrame) -> tuple:
    mday = df['ds'].iloc[-1] - offsets.Day(TEST_PERIOD_DAYS)
    graph_start_mday = df['ds'].iloc[-1] - offsets.Day(TEST_PERIOD_DAYS + GRAPH_PERIOD_DAYS)
    train_index = df['ds'] <= mday
    test_index = df['ds'] > mday
    graph_train_index = (df['ds'] > graph_start_mday) & train_index
    x_train = df[train_index]
    x_test = df[test_index]
    x_graph_test = df[graph_train_index]
    return x_train, x_test, x_graph_test

def create_and_train_model(x_train: pd.DataFrame) -> Prophet:
    model = Prophet(yearly_seasonality=False, weekly_seasonality=False,
        daily_seasonality=False,
        seasonality_mode='multiplicative')
    model.fit(x_train)
    return model

def predict(model: Prophet, periods: int) -> pd.DataFrame:
    future = model.make_future_dataframe(periods=periods, freq='d')
    forecast = model.predict(future)
    return forecast

def plot_results(ypred: np.ndarray, ytest: np.ndarray) -> None:
    fig, ax = plt.subplots(figsize=(12, 6))
    ax.plot(ypred, label='予測結果', c='r')
    ax.plot(ytest, label='正解データ', c='b')
    ax.grid()
    ax.legend()
    ax.set_title('テスト予測')
    plt.show()

def main() -> None:
    url = '../PythonData/FXCFDData/USD_JPY.txt'
    df = load_data(url)
    x_train, x_test, x_graph_test = split_data(df)
    model = create_and_train_model(x_train)
    forecast = predict(model, TEST_PERIOD_DAYS+TEST_DAYS)
    ypred = forecast[-(TEST_PERIOD_DAYS+TEST_DAYS):]['yhat'].to_numpy()
    ytest = x_test['y'].values
    last_ytest_data = x_graph_test['y'].values[-1]
    if ytest.size > 0:
        last_ytest_data = ytest[-1]
    for i in range(TEST_DAYS):
        ytest = np.append(ytest, last_ytest_data)
    ypred_total = np.hstack([x_graph_test['y'].values, ypred])
    ytest_total = np.hstack([x_graph_test['y'].values, ytest])
    y_total = np.stack([ypred_total, ytest_total], 1)
    total_df = pd.DataFrame(data=y_total,
                             columns=['pred', 'test'])
    total_df = total_df.dropna(subset='test')
    plot_results(total_df['pred'], total_df['test'])

if __name__ == '__main__':
    main()
