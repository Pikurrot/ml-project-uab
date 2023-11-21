import pandas as pd
from pandas import DataFrame, Series
import matplotlib.pyplot as plt
import numpy as np
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split
from logreg import LogReg

def preprocessing(df: DataFrame):
	# Encode categorical to numerical
	le_cut = LabelEncoder()
	le_color = LabelEncoder()
	le_clarity = LabelEncoder()
	df["cut"] = le_cut.fit_transform(df["cut"])
	df["color"] = le_color.fit_transform(df["color"])
	df["clarity"] = le_clarity.fit_transform(df["clarity"])

	# Drop unnecesary columns
	df = df.drop(columns=["Unnamed: 0", "x", "y", "z", "price"])

	# Separate X and y
	y_scaled = df["cut"]
	X_scaled = df.drop(columns=["cut"])

	# Standarize
	features = ["carat", "color", "clarity", "depth", "table"]
	scaler = StandardScaler()
	X_scaled = pd.DataFrame(scaler.fit_transform(X_scaled.to_numpy()), columns=features)

	# Remove outliers
	for feature in features:
		Q1 = X_scaled[feature].quantile(0.25)
		Q3 = X_scaled[feature].quantile(0.75)
		IQR = Q3 - Q1

		lower_bound = Q1 - 1.5 * IQR
		upper_bound = Q3 + 1.5 * IQR

		condition = (X_scaled[feature] >= lower_bound) & (X_scaled[feature] <= upper_bound)
		X_scaled = X_scaled[condition]
		y_scaled = y_scaled[condition]
	
	return X_scaled, y_scaled, scaler

def main():
	df = pd.read_csv("diamonds.csv")
	X, y, scaler = preprocessing(df)
	model = LogReg(scaler = scaler, epsilon = .0001, random_state = 42, penalty = "l2", max_iter = 10000)
	X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 42)
	model.cross_validation(X_train, y_train, n_splits = 5, test_size = 0.2)  # test_size actually means validation set size here


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
