import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import pandas.tseries.offsets as offsets
import matplotlib.dates as mdates
import datetime
import os
import sys
from sklearn import preprocessing
# matplotlib日本語化対応
import japanize_matplotlib

sys.path.append("../utility")
import stock

if os.name == 'posix':
    from fbprophet import Prophet
elif os.name == 'nt':
    from prophet import Prophet

# 例えば 2024/02/16までのデータがあるとします。
# NO_TRAINING_DAYS=15だと、2024/02/01までのデータを使ってトレーニングをします
# FORECAST_DAYS=5なら、2/24/02/21まで予測をします。

NO_TRAINING_DAYS = 20
FORECAST_DAYS = 1
GRAPH_DAYS = 100
DATA_URL = '../PythonData/FXCFDData/USD_JPY.txt'
START_DATE = '2005/01/01'
END_DATE = '2024/12/31'

# データ読み込み
def load_data(url: str) -> pd.DataFrame:
    df: pd.DataFrame = pd.read_csv(url, parse_dates=[0])
    df = df.sort_values(by='日付')
    df = df.query('日付 >= @START_DATE')
    df = df.query('日付 < @END_DATE')
    df = df[['日付', '始値']]
    df.columns = ['ds', 'y']
    return df

# データをトレーニング、テスト、グラフ用に分割
def split_data(df: pd.DataFrame, no_training_days:int, graph_days:int) -> tuple:
    mday: pd.Timestamp = df['ds'].iloc[-1] - offsets.Day(no_training_days)
    graph_start_mday: pd.Timestamp = df['ds'].iloc[-1] - offsets.Day(no_training_days + graph_days)
    train_index = df['ds'] <= mday
    test_index = df['ds'] > mday
    graph_train_index = (df['ds'] > graph_start_mday) & train_index
    x_train: pd.DataFrame = df[train_index]
    x_test: pd.DataFrame = df[test_index]
    x_graph_test: pd.DataFrame = df[graph_train_index]
    return x_train, x_test, x_graph_test

# 学習モデル作成
def create_and_train_model(x_train: pd.DataFrame) -> Prophet:
    # 欠損値がある日を見つける
    missing_dates = x_train[x_train['y'].isnull()]['ds']

    # ホリデーとして扱う
    holidays = pd.DataFrame({
        'holiday': 'missing_data',
        'ds': missing_dates,
        'lower_window': 0,
        'upper_window': 1,
    })

    model: Prophet = Prophet(yearly_seasonality=False, weekly_seasonality=False,
        daily_seasonality=False, holidays=holidays, changepoint_prior_scale=0.5,
        seasonality_mode='multiplicative')
    model.fit(x_train)
    return model

# 予測
def predict(model: Prophet, periods: int) -> pd.DataFrame:
    # 予測用データ作成
    future: pd.DataFrame = model.make_future_dataframe(periods=periods, freq='d')
    # 予測
    forecast: pd.DataFrame = model.predict(future)
    return forecast

# 結果をグラフに
def plot_results(ypred: np.ndarray, ytest: np.ndarray) -> None:
    fig, ax = plt.subplots(figsize=(12, 6))
    ax.plot(ypred, label='予測結果', c='r')
    ax.plot(ytest, label='正解データ', c='b')
    ax.grid()
    ax.legend()
    ax.set_title('テスト予測')
    plt.show()

# メイン
def prophet_test(url:str, no_training_days:int, forecast_days:int, graph_days: int) -> None:
    df = load_data(url)

    x_train, x_test, x_graph_test = split_data(df, no_training_days, graph_days)
    model = create_and_train_model(x_train)

    forecast = predict(model, no_training_days+forecast_days)
    ypred = forecast[-(no_training_days+forecast_days):]['yhat'].to_numpy()
    ytest = x_test['y'].values

    last_ytest_data = x_graph_test['y'].values[-1]
    if ytest.size > 0:
        last_ytest_data = ytest[-1]
    for i in range(forecast_days):
        ytest = np.append(ytest, last_ytest_data)

    ypred_total = np.hstack([x_graph_test['y'].values, ypred])
    ytest_total = np.hstack([x_graph_test['y'].values, ytest])
    y_total = np.stack([ypred_total, ytest_total], 1)
    total_df = pd.DataFrame(data=y_total,
                             columns=['pred', 'test'])
    total_df = total_df.dropna(subset='test')
    plot_results(total_df['pred'], total_df['test'])

def main() -> None:
    prophet_test(DATA_URL, NO_TRAINING_DAYS, FORECAST_DAYS, GRAPH_DAYS)

if __name__ == '__main__':
    main()
