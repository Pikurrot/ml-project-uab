import pandas as pd
from pandas import DataFrame, Series
import matplotlib.pyplot as plt
import numpy as np
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split

import src.utils as utils
from src.logreg import LogReg

def main():
	df = pd.read_csv("diamonds.csv")
	X, y, scaler = utils.preprocessing(df)
	model = LogReg(scaler = scaler, epsilon = .0001, random_state = 42, penalty = "l2", max_iter = 10000)
	X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 42)
	model.cross_validation(X_train, y_train, n_splits = 5, val_size = 0.2)


if __name__ == "__main__":
	main()

"""
data structuring: categorical ->numerical
remove outilers
data normalization
split data into training and validation
train models
msouibgui@cvc.uab.cat
"""
