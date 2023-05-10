import numpy as np
import h5py
from sklearn.datasets import make_regression
from sklearn.metrics import r2_score, accuracy_score
from sklearn.model_selection import train_test_split
from pprint import pprint

from autosklearn.regression import AutoSklearnRegressor
from autosklearn.classification import AutoSklearnClassifier
import pickle

def main():

    hf = h5py.File("/local/home/hanlonm/mt-matthew/data/00195_HL_SPA_NN/test-2000.h5",
                    "r")
    num_points = hf.attrs["num_points"]
    num_angles = hf.attrs["num_angles"]

    histogram_data = hf["encodings"][:]
    errors = hf["errors"][:]
    histogram_data = histogram_data.reshape(
        (num_points * num_angles, histogram_data.shape[2]))
    errors: np.ndarray = errors.reshape((num_points * num_angles, errors.shape[2]))

    e_trans = np.array([np.linalg.norm(errors[:,:3], axis=1)]).T
    e_rot = np.array([errors[:,3]]).T

    errors = np.hstack((e_trans, e_rot))

    labels = np.logical_and(errors[:,0]<0.05, errors[:,1]<0.5)
    labels = labels.astype(int)
    print(labels)

    X_train, X_test, y_train, y_test = train_test_split(histogram_data,
                                                        labels,
                                                        random_state=1,test_size=0.1)
    automl = AutoSklearnClassifier(
    time_left_for_this_task=600,
    per_run_time_limit=60, n_jobs=6)
    automl.fit(X_train, y_train, dataset_name="test")

    print(automl.leaderboard())
    pprint(automl.show_models(), indent=4)
    predictions = automl.predict(X_test)
    probas = automl.predict_proba(X_test)
    print("Accuracy score:", accuracy_score(y_test, predictions))
    predictions = automl.predict(X_train)
    print("Accuracy score:", accuracy_score(y_train, predictions))
    print(np.sum(y_test))
    # save model
    # with open('viewpoint-classifier.pkl', 'wb') as f:
    #     pickle.dump(automl, f)
    # print()

if __name__ == '__main__':
    main()
