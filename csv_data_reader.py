import tensorflow as tf
import numpy as np
import csv
from sklearn import preprocessing
from tensorflow.python.platform import gfile

class Data:
    """
    Utility class for loading training and test CSV files.
    """
    def __init__(self):
        self.training_features = None
        self.training_labels = None
        self.training_labels_1hot = None

    def load_csv_with_header(self, filename,
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

    def load_csv_without_header(self, filename,
                                target_dtype,
                                features_dtype,
                                target_column=-1):
        """Load dataset from CSV file without a header row.
        """
        with gfile.Open(filename) as csv_file:
            data_file = csv.reader(csv_file)
            data, target = [], []
            for row in data_file:
                target.append(row.pop(target_column))
                data.append(np.asarray(row, dtype=features_dtype))

        target = np.array(target, dtype=target_dtype)
        data = np.array(data)
        return data, target

    def load(self, training_filename, header=False):
        """
        Load CSV files into class member variables.
        """
        # Load training data using load_csv() function from Tensorflow 0.10.
        if header:
            self.training_features, targets = self.load_csv_with_header(
                filename=training_filename, features_dtype=np.float32, target_dtype=np.int)
        else:
            self.training_features, targets = self.load_csv_without_header(
                filename=training_filename, features_dtype=np.float32, target_dtype=np.int)

        self.training_labels_1hot = preprocessing.label_binarize(targets, classes=np.unique(targets))
        self.training_labels = np.reshape(targets, (targets.shape[0], 1))

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

