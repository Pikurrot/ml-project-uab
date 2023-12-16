import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, PolynomialFeatures
from sklearn.model_selection import train_test_split
from sklearn.model_selection import train_test_split

def preprocessing_L(df: pd.DataFrame) -> pd.DataFrame:
	"""
	L : Label encode cut, color and clarity.
	"""
	df_copy = df.copy()

	# Encode categorical to numerical, in order according to GIA
	cut_mapping = {"Ideal": 4, "Premium": 3, "Very Good": 2, "Good": 1, "Fair": 0}
	color_mapping = {"D": 6, "E": 5, "F": 4, "G": 3, "H": 2, "I": 1, "J": 0}
	clarity_mapping = {"IF": 7, "VVS1": 6, "VVS2": 5, "VS1": 4, "VS2": 3, "SI1": 2, "SI2": 1, "I1": 0}

	df_copy["cut"] = df_copy["cut"].map(cut_mapping)
	df_copy["color"] = df_copy["color"].map(color_mapping)
	df_copy["clarity"] = df_copy["clarity"].map(clarity_mapping)

	return df_copy

def preprocessing_H(df: pd.DataFrame) -> pd.DataFrame:
	"""
	H : one-Hot encode color and clarity.
	"""
	df_copy = df.copy()

	# One-hot encode categorical features
	cut_mapping = {"Ideal": 4, "Premium": 3, "Very Good": 2, "Good": 1, "Fair": 0}
	df_copy["cut"] = df_copy["cut"].map(cut_mapping)
	df_copy = pd.get_dummies(df_copy, columns=["color", "clarity"])

	return df_copy

def preprocessing_C(df: pd.DataFrame) -> pd.DataFrame:
	"""
	C : Combine x, y, z into volume.
	"""
	df_copy = df.copy()

	# Combine x, y, z into volume
	df_copy["volume"] = df_copy["x"] * df_copy["y"] * df_copy["z"]
	df_copy = df_copy.drop(columns=["x", "y", "z"])

	return df_copy

def preprocessing_O(df: pd.DataFrame,
					IQR_mult: float = 1.5) -> pd.DataFrame:
	"""
	O : remove Outliers from features initially numerical.
	"""
	df_copy = df.copy()

	# Remove outliers	
	numerical = ["carat", "depth", "table", "price", "x", "y", "z", "volume"]
	features = list(set(df_copy.columns) & set(numerical)) # intersection
	
	for feature in features:
		Q1 = df_copy[feature].quantile(0.25)
		Q3 = df_copy[feature].quantile(0.75)
		IQR = Q3 - Q1

		lower_bound = Q1 - IQR_mult * IQR
		upper_bound = Q3 + IQR_mult * IQR

		condition = (df_copy[feature] >= lower_bound) & (df_copy[feature] <= upper_bound)
		df_copy = df_copy[condition]

	return df_copy

def preprocessing_S(df: pd.DataFrame) -> pd.DataFrame:
	"""
	S : Standarize all features except cut and one-hot encoded.
	"""
	df_copy = df.copy()

	# Separate cut and one-hot encoded
	sep = list(df_copy.columns[df_copy.columns.str.contains("_")]) + ["cut"]
	df_sep = df_copy[sep]
	df_copy = df_copy.drop(columns=sep)

	# Standarize
	scaler = StandardScaler()
	scaler.fit(df_copy)
	df_copy = pd.DataFrame(scaler.transform(df_copy), columns=scaler.feature_names_in_)

	df_copy2 = df_copy.reset_index(drop=True)
	df_sep = df_sep.reset_index(drop=True)

	return pd.concat([df_copy2, df_sep], axis=1)

def preprocessing_P(df: pd.DataFrame,
					degree: int = 2) -> pd.DataFrame:
	"""
	P : create new Polynomial features as combinations of carat, depth, table, price, x, y and z.
	"""
	df_copy = df.copy()
	numerical = ["carat", "depth", "table", "price", "x", "y", "z"]
	poly = PolynomialFeatures(degree)
	new_features = poly.fit_transform(df[numerical])[:,1:] # drop first column (1)
	feature_names = poly.get_feature_names_out(numerical)[1:]
	df_copy = pd.concat([df_copy.drop(columns=numerical), pd.DataFrame(new_features, columns=feature_names)], axis=1)
	return df_copy

def preprocessing_LS(df: pd.DataFrame,
					 random_state: int = 42,
					 test_size: float = 0.2):
	"""
	L : Label encode cut, color and clarity.
	S : Standarize all features except cut.

	## Returns
	X_train, X_test, y_train, y_test
	"""
	df_copy = df.copy().drop(columns=["Unnamed: 0"])
	# Encode categorical to numerical
	df_encoded = preprocessing_L(df_copy)
	# Separate train and test
	train, test = train_test_split(df_encoded, test_size=test_size, random_state=random_state)
	# Standarize train different from test
	train = preprocessing_S(train)
	test = preprocessing_S(test)
	# Separate X and y
	return train.drop(columns=["cut"]), test.drop(columns=["cut"]), train["cut"], test["cut"]

def preprocessing_LOS(df: pd.DataFrame,
					 random_state: int = 42,
					 test_size: float = 0.2):
	"""
	L : Label encode cut, color and clarity.
	O : remove Outliers from numerical features.
	S : Standarize all features except cut.

	## Returns
	X_train, X_test, y_train, y_test
	"""
	df_copy = df.copy().drop(columns=["Unnamed: 0"])
	# Encode categorical to numerical
	df_encoded = preprocessing_L(df_copy)
	# Remove outliers
	df_outliers = preprocessing_O(df_encoded)
	# Separate train and test
	train, test = train_test_split(df_outliers, test_size=test_size, random_state=random_state)
	# Standarize train different from test
	train = preprocessing_S(train)
	test = preprocessing_S(test)
	# Separate X and y
	return train.drop(columns=["cut"]), test.drop(columns=["cut"]), train["cut"], test["cut"]

def preprocessing_LS_simple(df: pd.DataFrame):
	"""
	L : Label encode cut, color and clarity.
	S : Standarize all features except cut.

	## Returns
	X, y
	"""
	df_copy = df.copy().drop(columns=["Unnamed: 0"])
	# Encode categorical to numerical
	df_encoded = preprocessing_L(df_copy)
	# Standarize
	df_scaled = preprocessing_S(df_encoded)
	# Separate X and y
	return df_scaled.drop(columns=["cut"]), df_scaled["cut"]

def preprocessing_LOS_simple(df: pd.DataFrame):
	"""
	L : Label encode cut, color and clarity.
	O : remove Outliers from numerical features.
	S : Standarize all features except cut.
	"""
	df_copy = df.copy().drop(columns=["Unnamed: 0"])
	# Encode categorical to numerical
	df_encoded = preprocessing_L(df_copy)
	# Remove outliers
	df_outliers = preprocessing_O(df_encoded)
	# Standarize train different from test
	train = preprocessing_S(df_outliers)
	# Separate X and y
	return train.drop(columns=["cut"]), train["cut"]

def preprocessing_HS(df: pd.DataFrame,
					 random_state: int = 42,
					 test_size: float = 0.2):
	"""
	H : one-Hot encode color and clarity.
	S : Standarize all features except cut and one-hot encoded.

	## Returns
	X_train, X_test, y_train, y_test
	"""
	df_copy = df.copy().drop(columns=["Unnamed: 0"])
	# One-hot encode categorical features
	df_encoded = preprocessing_H(df_copy)
	# Separate train and test
	train, test = train_test_split(df_encoded, test_size=test_size, random_state=random_state)
	# Standarize train different from test
	train = preprocessing_S(train)
	test = preprocessing_S(test)
	# Separate X and y
	return train.drop(columns=["cut"]), test.drop(columns=["cut"]), train["cut"], test["cut"]

def preprocessing_HOS(df: pd.DataFrame,
					  random_state: int = 42,
					  test_size: float = 0.2):
	"""
	H : one-Hot encode color and clarity.
	O : remove Outliers from numerical features.
	S : Standarize all features except cut and one-hot encoded.

	## Returns
	X_train, X_test, y_train, y_test
	"""
	df_copy = df.copy().drop(columns=["Unnamed: 0"])
	# One-hot encode categorical features
	df_encoded = preprocessing_H(df_copy)
	# Remove outliers
	df_outliers = preprocessing_O(df_encoded)
	# Separate train and test
	train, test = train_test_split(df_outliers, test_size=test_size, random_state=random_state)
	# Standarize train different from test
	train = preprocessing_S(train)
	test = preprocessing_S(test)
	# Separate X and y
	return train.drop(columns=["cut"]), test.drop(columns=["cut"]), train["cut"], test["cut"]

def preprocessing_PLS(df: pd.DataFrame,
					   random_state: int = 42,
					   test_size: float = 0.2,
					   degree: int = 2):
	"""
	P : create new Polynomial features as combinations of carat, depth, table, price, x, y and z.
	L : Label encode cut, color and clarity.
	S : Standarize all features except cut and one-hot encoded.
	"""
	return preprocessing_LS(preprocessing_P(df, degree), random_state, test_size)

def preprocessing_PLOS(df: pd.DataFrame,
					   random_state: int = 42,
					   test_size: float = 0.2,
					   degree: int = 2):
	"""
	P : create new Polynomial features as combinations of carat, depth, table, price, x, y and z.
	L : Label encode cut, color and clarity.
	O : remove Outliers from numerical features.
	S : Standarize all features except cut and one-hot encoded.
	"""
	return preprocessing_LOS(preprocessing_P(df, degree), random_state, test_size)
