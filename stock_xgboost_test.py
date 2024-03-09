import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn import preprocessing
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import mean_squared_error

sys.path.append("../utility")
import stock
import stock_database

class Prediction:

    def __init__(self, finish_days, length_of_sequences):
        self.__length_of_sequences = length_of_sequences
        self.__finish_days = finish_days


    def create_model(self) -> xgb.XGBRegressor:
        model = xgb.XGBRegressor(
                    objective ='reg:squarederror',
                    learning_rate = 0.2,
                    max_depth = 5,
                    n_estimators = 50
                )
        return model


    def train(self, X_train, y_train) -> xgb.XGBRegressor:
        model = self.create_model()
        '''
        param_grid = {
            'learning_rate': [0.01, 0.1, 0.2, 0.3],
            'max_depth': [3, 4, 5, 6, 7],
            'n_estimators': [50, 100, 150, 200]
        }

        # GridSearchCVを設定します
        grid_search = GridSearchCV(estimator=model, param_grid=param_grid, cv=3)

        # モデルの学習を行います
        grid_search.fit(X_train, y_train)

        # 最適なパラメータを表示します
        print("Best parameters found: ", grid_search.best_params_)
        '''

        model.fit(X_train, y_train)
        return model


def verify(
        database,
        base_str,
        finish_days,
        training_days_rate,
        length_of_sequences
    ) -> None:

    dataframe = database.data_frame
    prediction = Prediction(finish_days, length_of_sequences)

    training_days_pos = int(len(dataframe) * training_days_rate)
    x_train, y_train, date_train, train_start_index, train_end_index = \
        database.create_training_basic_data(
            dataframe[[base_str]].iloc[0:training_days_pos],
            dataframe[[stock.DAY]].iloc[0:training_days_pos],
            length_of_sequences,
            finish_days
        )
    x_test, y_test, date_test, test_start_index, test_end_index = \
    database.create_training_basic_data(
            dataframe[[base_str]].iloc[training_days_pos:],
            dataframe[[stock.DAY]].iloc[training_days_pos:],
            length_of_sequences,
            finish_days
        )
    model = prediction.train(
        x_train[:train_end_index-train_start_index+1].reshape((x_train[:train_end_index-train_start_index+1].shape[0], -1)),
        y_train[:train_end_index-train_start_index+1]
    )

    test_file = open('../../TemporaryFolder/xgboost_result.txt', 'w', encoding=stock.BASE_ENCODING)

    test_file.write(str(date_train[0]) + '\n')
    test_file.write(str(date_train[train_end_index-train_start_index]) + '\n')
    test_file.write(str(date_test[0]) + '\n')
    test_file.write(str(date_test[test_end_index-test_start_index]) + '\n')

    predicted = model.predict(x_test.reshape(x_test.shape[0], -1))
    result = pd.DataFrame(predicted)
    for date, x_te, res, y_te in zip(date_test, x_test, predicted, y_test):
        test_file.write(str(date) + ',' + str(x_te[-1]) + ',' + str(res) + ',' + str(y_te) + '\n')

    y_test_new =  np.concatenate(y_test)
    mask = ~np.isnan(predicted) & ~np.isnan(y_test_new)
    predicted_true = predicted[mask]
    y_test_true = y_test_new[mask]

    mse = mean_squared_error(predicted_true, y_test_true)
    test_file.write(str(mse) + '\n')

    test_file.close()
    # DataFrameに変換
    predicted_true_df = pd.DataFrame(predicted_true, columns=['predicted'])
    predicted_true_df['actual'] = y_test_true

    # グラフをプロット
    predicted_true_df.plot()
    plt.show()


DATA_URL = '../PythonData/FXCFDData/USD_JPY.txt'
LENGTH_OF_SEQUENCE = 20
AVERAGE_DAYS = 5
FINISH_DAYS = 1
TRAINING_DAYS_RATE = 0.99

def xgboost_test() -> None:
    base_str = '始値'
    database = stock_database.StockDatabase(base_str)
    database.load(DATA_URL, start_date='2005/01/01')
    database.create_basic_data()
    database.set_length_of_sequences(LENGTH_OF_SEQUENCE)
    database.set_average_days(AVERAGE_DAYS)

    # x_list, y_list, \
    # training_start_index, training_end_index, \
    # test_start_index, test_end_index = database.create_training(base_str, FINISH_DAYS, TRAINING_DAYS_RATE, 100.0)

    verify(
        database,
        base_str,
        FINISH_DAYS,
        TRAINING_DAYS_RATE,
        LENGTH_OF_SEQUENCE
    )


def main() -> None:
    stock.reset_random()

    xgboost_test()


if __name__ == '__main__':
    main()
