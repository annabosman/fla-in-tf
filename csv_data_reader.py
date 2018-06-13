import tensorflow as tf
import numpy as np
import csv
from sklearn import preprocessing

class Data:
    """
    Utility class for loading training and test CSV files.
    """
    def __init__(self):
        self.training_features = None
        self.training_labels = None
        self.training_labels_1hot = None

    def load_csv_with_header(filename,
                             target_dtype,
                             features_dtype,
                             target_column=-1):
        """Load dataset from CSV file with a header row."""
        with gfile.Open(filename) as csv_file:
            data_file = csv.reader(csv_file)
            header = next(data_file)
            n_samples = int(header[0])
            n_features = int(header[1])
            data = np.zeros((n_samples, n_features), dtype=features_dtype)
            target = np.zeros((n_samples,), dtype=target_dtype)
            for i, row in enumerate(data_file):
                target[i] = np.asarray(row.pop(target_column), dtype=target_dtype)
                data[i] = np.asarray(row, dtype=features_dtype)
            return data, target

    def load(self, training_filename):
        """
        Load CSV files into class member variables.
        """
        # Load training data using load_csv() function from Tensorflow 0.10.
        training_set = tf.contrib.learn.datasets.base.load_csv_with_header(
            filename=training_filename, features_dtype=np.float32, target_dtype=np.int)

        self.training_features    = training_set.data.astype(np.float32)
        self.training_labels_1hot = self.convert_to_one_hot(training_set.target)
        self.training_labels      = np.reshape(training_set.target,(training_set.target.shape[0], 1))

    def convert_to_one_hot(self, vector, num_classes=None):
        """
        Converts an input 1-D vector of integers into an output
        2-D array of one-hot vectors, where an i'th input value
        of j will set a '1' in the i'th row, j'th column of the
        output array.

        Example:
            v = np.array((1, 0, 4))
            one_hot_v = convert_to_one_hot(v)
            print one_hot_v

            [[0 1 0 0 0]
             [1 0 0 0 0]
             [0 0 0 0 1]]
        """
        assert isinstance(vector, np.ndarray)
        assert len(vector) > 0
        if num_classes is None:
            num_classes = np.max(vector)+1
        else:
            assert num_classes > 0
            assert num_classes >= np.max(vector)
        result = np.zeros(shape=(len(vector), num_classes))
        result[np.arange(len(vector)), vector] = 1
        return result.astype(int)

    def scale_features_to_range(self, low=-1, high=1):
        """
        Scale numerical features to a predefined range.
        Default range: [-1,1]
        """
        self.training_features = preprocessing.minmax_scale(self.training_features, feature_range=(low, high))

    def standardise_features(self):
        """
        Standardise numerical features (Z-score).
        """
        self.training_features = preprocessing.scale(self.training_features)

