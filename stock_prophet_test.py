import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import pandas.tseries.offsets as offsets
import matplotlib.dates as mdates
import datetime
import os
import sys
from sklearn import preprocessing
from sklearn.metrics import mean_squared_error

sys.path.append("../utility")
import stock
import stock_database

if os.name == 'posix':
    from fbprophet import Prophet
elif os.name == 'nt':
    from prophet import Prophet

# 例えば 2024/02/16までのデータがあるとします。
# NO_TRAINING_DAYS=15だと、2024/02/01までのデータを使ってトレーニングをします
# FORECAST_DAYS=5なら、2/24/02/21まで予測をします。

NO_TRAINING_DAYS = 21
FORECAST_DAYS = 1
GRAPH_DAYS = 100
DATA_URL = '../PythonData/FXCFDData/USD_JPY.txt'
START_DATE = '2005/01/01'
END_DATE = '2024/12/31'


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

    model: Prophet = Prophet(
        yearly_seasonality=False,
        weekly_seasonality=False,
        daily_seasonality=False,
        holidays=holidays,
        changepoint_prior_scale=0.5,
        changepoint_range=0.9,
        seasonality_mode='multiplicative'
    )
    assert not np.isnan(x_train.iloc[-1]['y']), '検証値の最後はnullにしないで'
    model.fit(x_train)
    return model


# 予測
def predict(model: Prophet, periods: int) -> pd.DataFrame:
    # 予測用データ作成
    future: pd.DataFrame = model.make_future_dataframe(periods=periods, freq='d')
    # 予測
    forecast: pd.DataFrame = model.predict(future)
    return forecast


# 結果をプロット
def plot_results(ypred: np.ndarray, ytest: np.ndarray) -> None:
    fig, ax = plt.subplots(figsize=(12, 6))
    ax.plot(ypred, label='pred', c='r')
    ax.plot(ytest, label='test', c='b')
    ax.grid()
    ax.legend()
    ax.set_title('test')
    plt.show()


# メイン
def prophet_test(url:str, no_training_days:int, forecast_days:int, graph_days: int) -> None:
    database = stock_database.StockDatabase('始値')
    database.load(DATA_URL, start_date=START_DATE, end_date=END_DATE)

    df = database.data_frame[['日付', '始値']]
    df.columns = ['ds', 'y']

    training_days = len(df) - no_training_days
    retX, retY, ret_date, start_idx, end_idx = \
        database.create_training_basic_data(
            df[['y']],
            df[['ds']],
            1,
            1,
            0,
            training_days + 1,
            True
        )

    model = create_and_train_model(df[start_idx:end_idx+1])

    forecast = predict(model, no_training_days+forecast_days)
    y_date = forecast[-(no_training_days+forecast_days):]['ds'].dt.date.to_numpy()
    y_pred = forecast[-(no_training_days+forecast_days):]['yhat'].to_numpy()
    x_graph_test: pd.DataFrame = df.iloc[-no_training_days-graph_days:]

    #y_test = x_test['y'].values
    y_test = df['y'].iloc[end_idx+1:].values

    print(len(y_test))
    print(len(y_date))
    print(type(y_test))

    for i in range(len(y_date) - len(y_test)):
        y_test = np.append(y_test, np.nan)
    print(len(y_test))
    test_file = open('../../TemporaryFolder/prophet_result.txt', 'w', encoding=stock.BASE_ENCODING)

    test_file.write(str(ret_date[start_idx]) + '\n')
    test_file.write(str(ret_date[end_idx]) + '\n')
    test_file.write(str(ret_date[end_idx+1]) + '\n')
    test_file.write(str(ret_date[-1]) + '\n')

    for date, actual, predicted in zip(y_date, y_test, y_pred):
        test_file.write(str(date) + ',' + str(actual) + ',' + str(predicted) + '\n')

    test_file.close()

    mask = ~np.isnan(y_pred) & ~np.isnan(y_test)
    y_pred_true = y_pred[mask]
    y_test_true = y_test[mask]
    mse = mean_squared_error(y_pred_true, y_test_true)
    print(mse)
    y_pred_total = np.hstack([x_graph_test['y'].values, y_pred])
    y_test_total = np.hstack([x_graph_test['y'].values, y_test])
    y_total = np.stack([y_pred_total, y_test_total], 1)
    total_df = pd.DataFrame(data=y_total,
                             columns=['pred', 'test'])
    total_df = total_df.dropna(subset='test')
    plot_results(total_df['pred'], total_df['test'])
    


def main() -> None:
    stock.reset_random()

    prophet_test(DATA_URL, NO_TRAINING_DAYS, FORECAST_DAYS, GRAPH_DAYS)


if __name__ == '__main__':
    main()
