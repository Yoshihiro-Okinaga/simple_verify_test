import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

ORG = True

from sklearn import preprocessing
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Activation, LSTM, SimpleRNN
from tensorflow.keras.optimizers import Adam, SGD, RMSprop
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import GridSearchCV

if not ORG:
    from kerastuner.tuners import RandomSearch

sys.path.append("../utility")
import stock
import stock_database

def build_model(hp):
    model = Sequential()
    model.add(SimpleRNN(units=hp.Int('units', min_value=64, max_value=512, step=32),
                        return_sequences=False))
    model.add(Dense(64, activation='relu'))
    model.add(Dense(128, activation='relu'))
    model.add(Dense(1, activation='linear'))

    hp_learning_rate = hp.Choice('learning_rate', values=[1e-2, 1e-3, 1e-4])
    hp_optimizer = hp.Choice('optimizer', values=['adam', 'sgd', 'rmsprop'])

    if hp_optimizer == 'adam':
        optimizer = Adam(learning_rate=hp_learning_rate)
    elif hp_optimizer == 'sgd':
        optimizer = SGD(learning_rate=hp_learning_rate)
    else:
        optimizer = RMSprop(learning_rate=hp_learning_rate)

    model.compile(loss='mean_squared_error', optimizer=optimizer, metrics=['accuracy'])
    return model


def build_model_second(hp):
    model = Sequential()

    hp_activation = hp.Choice('activation', values=['softmax', 'linear', 'relu'])

    model.add(SimpleRNN(300, return_sequences=False))
    model.add(Dense(64, activation=hp_activation))
    model.add(Dense(128, activation=hp_activation))
    model.add(Dense(1, activation=hp_activation))

    hp_loss = hp.Choice('loss', values=['mean_squared_error', 'mean_absolute_error', 'hinge'])
    hp_optimizer = hp.Choice('optimizer', values=['adam', 'sgd', 'rmsprop'])

    model.compile(loss=hp_loss, optimizer=hp_optimizer, metrics=['accuracy'])
    return model


def create_model(activation, loss, optimizer) -> Sequential:
    model = Sequential()
    model.add(SimpleRNN(300, return_sequences=False))
    model.add(Dense(64, activation=activation))
    model.add(Dense(128, activation=activation))
    model.add(Dense(1, activation=activation))
    model.compile(loss=loss, optimizer=optimizer, metrics=['accuracy'])
    return model


def train(X_train, y_train, fit_epochs, fit_batch_size) -> Sequential:
    model = create_model('linear', 'mape', 'Adam')
    model.fit(
        X_train,
        y_train,
        epochs=fit_epochs,
        batch_size=fit_batch_size,
        validation_split=0.05,
        verbose=1
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
    
    if ORG:
        model = train(
            x_train[train_start_index:train_end_index+1],
            y_train[train_start_index:train_end_index+1],
            fit_epochs,
            fit_batch_size
        )

        predicted = model.predict(x_test[test_start_index:], 32, 0)

    else:
        # モデルのチューニング
        tuner = RandomSearch(
            build_model_second,
            objective='val_accuracy', #val_accuracy, val_precision, val_loss
            max_trials=10,
            executions_per_trial=5,
            directory='keras_tuner_logs',
            project_name='stock_prediction')

        tuner.search(x_train[train_start_index:train_end_index+1], y_train[train_start_index:train_end_index+1], validation_split=0.01, epochs=10)

        # ベストなモデルを取得
        best_model = tuner.get_best_models(num_models=1)[0]
        best_hyperparameters = tuner.get_best_hyperparameters()[0]
        print(best_hyperparameters.values)

        # テストデータで評価
        predicted = best_model.predict(x_test[test_start_index:])

        mask = ~np.isnan(predicted) & ~np.isnan(y_test[test_start_index:])
        predicted_true = predicted[mask]
        y_test_true = y_test[test_start_index:][mask]


    test_file = open('../../TemporaryFolder/tensorflow_result.txt', 'w', encoding=stock.BASE_ENCODING)

    test_file.write(str(date_train[train_start_index]) + '\n')
    test_file.write(str(date_train[train_end_index]) + '\n')
    test_file.write(str(date_test[test_start_index]) + '\n')
    test_file.write(str(date_test[test_end_index]) + '\n')

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