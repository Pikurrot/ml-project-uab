import pandas as pd
from sklearn.preprocessing import LabelEncoder, StandardScaler

def preprocessing(df: pd.DataFrame):
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


def preprocessing_no_drop(df: pd.DataFrame):
	# Encode categorical to numerical
	le_cut = LabelEncoder()
	le_color = LabelEncoder()
	le_clarity = LabelEncoder()
	df["cut"] = le_cut.fit_transform(df["cut"])
	df["color"] = le_color.fit_transform(df["color"])
	df["clarity"] = le_clarity.fit_transform(df["clarity"])

	# Drop unnecesary columns
	df = df.drop(columns=["Unnamed: 0"])

	# Separate X and y
	y_scaled = df["cut"]
	X_scaled = df.drop(columns=["cut"])

	# Standarize
	features = ["carat", "color", "clarity", "depth", "table", "price", "x", "y", "z"]
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