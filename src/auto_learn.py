import numpy as np
import h5py
from sklearn.datasets import make_regression
from sklearn.metrics import r2_score
from sklearn.model_selection import train_test_split
from pprint import pprint

from autosklearn.regression import AutoSklearnRegressor
from autosklearn.classification import AutoSklearnClassifier

def main():
    hf = h5py.File("/local/home/hanlonm/mt-matthew/data/00195_HL_SPA_NN/test-2000.h5",
                "r")
    num_points = hf.attrs["num_points"]
    num_angles = hf.attrs["num_angles"]

    histogram_data = hf["histogram_data"][:]
    errors = hf["errors"][:]
    histogram_data = histogram_data.reshape(
        (num_points * num_angles, histogram_data.shape[2]))
    errors = errors.reshape((num_points * num_angles, errors.shape[2]))

    X_train, X_test, y_train, y_test = train_test_split(histogram_data,
                                                        errors,
                                                        random_state=1,test_size=0.1)
    logging_config = {
    "version": 1,
    "disable_existing_loggers": True,
    "formatters": {
        "custom": {
            # More format options are available in the official
            # `documentation <https://docs.python.org/3/howto/logging-cookbook.html>`_
            "format": "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
        }
    },
    # Any INFO level msg will be printed to the console
    "handlers": {
        "console": {
            "level": "INFO",
            "formatter": "custom",
            "class": "logging.StreamHandler",
            "stream": "ext://sys.stdout",
        },
    },
    "loggers": {
        "": {  # root logger
            "level": "DEBUG",
        },
        "Client-EnsembleBuilder": {
            "level": "DEBUG",
            "handlers": ["console"],
        },
    },
}

    automl = AutoSklearnRegressor(n_jobs=6,per_run_time_limit=300, time_left_for_this_task=1200, logging_config=logging_config)
    automl.fit(X_train, y_train, dataset_name="test")
    print(automl.leaderboard())
    pprint(automl.show_models(), indent=4)
    predictions = automl.predict(X_test)
    print("R2 score:", r2_score(y_test, predictions))
    # The configuration space is reduced, i.e. no SVM.
    # print(automl.get_configuration_space(X_train, y_train))

    # print()

if __name__ == '__main__':
    main()