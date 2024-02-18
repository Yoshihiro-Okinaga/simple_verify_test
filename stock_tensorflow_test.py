import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn import preprocessing
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Activation, LSTM, SimpleRNN

sys.path.append("../utility")
import stock
import stock_database

class Prediction:

    def __init__(self, finish_days, length_of_sequences):
        self.__length_of_sequences = length_of_sequences
        self.__finish_days = finish_days

    def load_data(self, data, date, n_prev) -> tuple:
        X, Y, Date = [], [], []
        start_idx = n_prev-1
        end_idx = -1
        data_len = len(data)
        for i in range(data_len-(n_prev-1)):
            X.append(data.iloc[i:(i+n_prev)].to_numpy())
            Date.append(date.iloc[i])

            finish_idx = i+(n_prev-1)+self.__finish_days
            if finish_idx < data_len:
                Y.append(data.iloc[finish_idx].to_numpy())
                end_idx += 1
            else:
                Y.append(data.iloc[0].to_numpy())
        retX = np.array(X)
        retY = np.array(Y)
        retDate = np.array(Date)
        return retX, retY, retDate, start_idx, start_idx + end_idx

    def create_model(self) -> Sequential:
        model = Sequential()
        model.add(SimpleRNN(300, return_sequences=False))
        model.add(Dense(1, activation="linear"))
        model.compile(loss="mape", optimizer="adam")
        return model

    def train(self, X_train, y_train, fit_epochs, fit_batch_size) -> Sequential:
        model = self.create_model()
        model.fit(X_train, y_train, epochs=fit_epochs, batch_size=fit_batch_size, verbose=0)
        return model


def verify(dataframe, base_str, finish_days, training_days_rate, length_of_sequences, fit_epochs, fit_batch_size) -> None:

    prediction = Prediction(finish_days, length_of_sequences)

    training_days_pos = int(len(dataframe) * training_days_rate)
    x_train, y_train, date_train, train_start_index, train_end_index = \
        prediction.load_data(
            dataframe[[base_str]].iloc[0:training_days_pos],
            dataframe[[stock.DAY]].iloc[0:training_days_pos],
            length_of_sequences
        )
    x_test, y_test, date_test, test_start_index, test_end_index = \
        prediction.load_data(
            dataframe[[base_str]].iloc[training_days_pos:],
            dataframe[[stock.DAY]].iloc[training_days_pos:],
            length_of_sequences
        )
    model = prediction.train(
        x_train[:train_end_index-train_start_index+1],
        y_train[:train_end_index-train_start_index+1],
        fit_epochs,
        fit_batch_size
    )

    print(date_train[0])
    print(date_train[train_end_index-train_start_index])
    print(date_test[0])
    print(date_test[test_end_index-test_start_index])

    predicted = model.predict(x_test, 32, 0)
    result = pd.DataFrame(predicted)
    result.columns = ['predict']
    result['actual'] = y_test
    result.plot()
    plt.show()


DATA_URL = '../PythonData/FXCFDData/USD_JPY.txt'
LENGTH_OF_SEQUENCE = 20
AVERAGE_DAYS = 5
FINISH_DAYS = 1
TRAINING_DAYS_RATE = 0.95
FIT_EPOCHS = 10
FIT_BATCH_SIZE = 30

def tensorflow_test() -> None:
    base_str = '始値'
    database = stock_database.StockDatabase(base_str, LENGTH_OF_SEQUENCE, AVERAGE_DAYS)
    database.load(DATA_URL, start_date='2005/01/01')

    verify(database.data_frame, base_str, FINISH_DAYS, TRAINING_DAYS_RATE, LENGTH_OF_SEQUENCE, FIT_EPOCHS, FIT_BATCH_SIZE)


def main() -> None:
    tensorflow_test()


if __name__ == '__main__':
    main()