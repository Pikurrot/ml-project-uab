import pandas as pd
from pandas import DataFrame, Series
import matplotlib.pyplot as plt
import numpy as np
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split
from logreg import LogReg

def preprocessing(df: DataFrame):
	le_cut = LabelEncoder()
	le_color = LabelEncoder()
	le_clarity = LabelEncoder()
	df["cut"] = le_cut.fit_transform(df["cut"])
	df["color"] = le_color.fit_transform(df["color"])
	df["clarity"] = le_clarity.fit_transform(df["clarity"])
	y = df["cut"]
	X = df.drop(columns=["cut", "Unnamed: 0"])
	scaler = StandardScaler()
	X_scaled = scaler.fit_transform(X.to_numpy())
	return X_scaled, y, scaler

def main():
	df = pd.read_csv("diamonds.csv")
	X, y, scaler = preprocessing(df)
	model = LogReg(scaler = scaler, epsilon = .0001, random_state = 42, penalty = "l2", max_iter = 10000)
	model.cross_validation(X[:500], y[:500], n_splits = 5, test_size = .2)


if __name__ == "__main__":
	main()
