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
