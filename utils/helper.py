# Main
import os, sys
import base64
import numpy as np
import pandas as pd
import streamlit as st
from itertools import chain

# Sklearn
import sklearn
import sklearn.metrics as metrics
from sklearn.decomposition import PCA
from sklearn.impute import SimpleImputer, KNNImputer
from sklearn import svm, tree, linear_model, neighbors, ensemble
from sklearn.metrics import roc_curve, precision_recall_curve, auc
from sklearn.feature_selection import chi2, mutual_info_classif, f_classif, SelectKBest
from sklearn.model_selection import RepeatedStratifiedKFold, StratifiedKFold, StratifiedShuffleSplit
from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler, LabelEncoder, QuantileTransformer, PowerTransformer

# Plotly Graphs
import plotly
import plotly.express as px
import plotly.graph_objects as go
import plotly.figure_factory as ff
from scipy.spatial.distance import pdist, squareform

# Define common colors
blue_color = '#035672'
red_color = '#f84f57'
gray_color ='#ccc'
