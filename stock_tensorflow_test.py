import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn import preprocessing
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Activation, LSTM, SimpleRNN
from sklearn.metrics import mean_squared_error

sys.path.append("../utility")
import stock
import stock_database

class Prediction:

    def __init__(self, finish_days, length_of_sequences):
        self.__length_of_sequences = length_of_sequences
        self.__finish_days = finish_days


    def create_model(self) -> Sequential:
        model = Sequential()
        model.add(SimpleRNN(300, return_sequences=False))
        model.add(Dense(64, activation="linear"))
        model.add(Dense(128, activation="linear"))
        model.add(Dense(1, activation="linear"))
        model.compile(loss="mape", optimizer="adam")
        return model


    def train(self, X_train, y_train, fit_epochs, fit_batch_size) -> Sequential:
        model = self.create_model()
        model.fit(
            X_train,
            y_train,
            epochs=fit_epochs,
            batch_size=fit_batch_size,
            validation_split=0.05,
            verbose=0
        )
        return model


def verify(
        database,
        base_str,
        finish_days,
        training_days_rate,
        length_of_sequences,
        fit_epochs,
        fit_batch_size
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
        x_train[train_start_index:train_end_index+1],
        y_train[train_start_index:train_end_index+1],
        fit_epochs,
        fit_batch_size
    )

    test_file = open('../../TemporaryFolder/tensorflow_result.txt', 'w', encoding=stock.BASE_ENCODING)

    test_file.write(str(date_train[train_start_index]) + '\n')
    test_file.write(str(date_train[train_end_index]) + '\n')
    test_file.write(str(date_test[test_start_index]) + '\n')
    test_file.write(str(date_test[test_end_index]) + '\n')

    predicted = model.predict(x_test[test_start_index:], 32, 0)
    result = pd.DataFrame(predicted)
    for date, x_te, res, y_te in zip(date_test[test_start_index:], x_test[test_start_index:], predicted, y_test[test_start_index:]):
        test_file.write(str(date) + ',' + str(x_te[-1]) + ',' + str(res) + ',' + str(y_te) + '\n')

    mask = ~np.isnan(predicted) & ~np.isnan(y_test[test_start_index:])
    predicted_true = predicted[mask]
    y_test_true = y_test[test_start_index:][mask]

    mse = mean_squared_error(predicted_true, y_test_true)
    test_file.write(str(mse) + '\n')

    test_file.close()
    # result.columns = ['predict']
    # result['actual'] = y_test
    # result.plot()
    # plt.show()


DATA_URL = '../PythonData/FXCFDData/USD_JPY.txt'
LENGTH_OF_SEQUENCE = 20
AVERAGE_DAYS = 5
FINISH_DAYS = 1
TRAINING_DAYS_RATE = 0.95
FIT_EPOCHS = 100
FIT_BATCH_SIZE = 30


def tensorflow_test() -> None:
    base_str = '始値'
    database = stock_database.StockDatabase(base_str)
    database.load(DATA_URL, start_date='2005/01/01')
    database.create_basic_data()
    database.set_length_of_sequences(LENGTH_OF_SEQUENCE)
    database.set_average_days(AVERAGE_DAYS)

    verify(
        database,
        base_str,
        FINISH_DAYS,
        TRAINING_DAYS_RATE,
        LENGTH_OF_SEQUENCE,
        FIT_EPOCHS,
        FIT_BATCH_SIZE
    )


def main() -> None:
    stock.reset_random()

    tensorflow_test()


if __name__ == '__main__':
    main()