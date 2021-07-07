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

# Define base metrics to be used
scores = ['accuracy', 'roc_auc', 'precision', 'recall', 'f1', 'balanced_accuracy']
scorer_dict = {}
scorer_dict = {metric:metric+'_score' for metric in scores}
scorer_dict = {key: getattr(metrics, metric) for key, metric in scorer_dict.items()}

def make_recording_widget(f, widget_values):
    """
    Return a function that wraps a streamlit widget and records the
    widget's values to a global dictionary.
    """
    def wrapper(label, *args, **kwargs):
        widget_value = f(label, *args, **kwargs)
        widget_values[label] = widget_value
        return widget_value

    return wrapper

class objdict(dict):
    """
    Objdict class to conveniently store a state
    """

    def __getattr__(self, name):
        if name in self:
            return self[name]
        else:
            raise AttributeError("No such attribute: " + name)

    def __setattr__(self, name, value):
        self[name] = value

    def __delattr__(self, name):
        if name in self:
            del self[name]
        else:
            raise AttributeError("No such attribute: " + name)


def main_components():
    """
    Expose external CSS and create & return widgets
    """
    # External CSS
    main_external_css = """
        <style>
            hr {margin: 15px 0px !important; background: #ff3a50}
            .footer {position: absolute; height: 50px; bottom: -150px; width:100%; padding:10px; text-align:center; }
            #MainMenu, .reportview-container .main footer {display: none;}
            .btn-outline-secondary {background: #FFF !important}
            .download_link {color: #f63366 !important; text-decoration: none !important; z-index: 99999 !important;
                            cursor:pointer !important; margin: 15px 0px; border: 1px solid #f63366;
                            text-align:center; padding: 8px !important; width: 200px;}
            .download_link:hover {background: #f63366 !important; color: #FFF !important;}
            h1, h2, h3, h4, h5, h6, a, a:visited {color: #f84f57 !important}
            label, stText, p, .caption {color: #035672}
            .css-17eq0hr {background: #035672 !important;}
            .streamlit-expanderHeader {font-size: 16px !important;}
            .css-17eq0hr label, stText, .caption, .css-j075dz, .css-1t42vg8 {color: #FFF !important}
            .css-17eq0hr a {text-decoration:underline;}
            .tickBarMin, .tickBarMax {color: #f84f57 !important}
            .markdown-text-container p {color: #035672 !important}
            .css-xq1lnh-EmotionIconBase {fill: #ff3a50 !important}
            /*.css-hi6a2p {max-width: 885px !important}*/

            /* Tabs */
            .tabs { position: relative; min-height: 200px; clear: both; margin: 40px auto 0px auto; background: #efefef; box-shadow: 0 48px 80px -32px rgba(0,0,0,0.3); }
            .tab {float: left;}
            .tab label { background: #f84f57; cursor: pointer; font-weight: bold; font-size: 18px; padding: 10px; color: #fff; transition: background 0.1s, color 0.1s; margin-left: -1px; position: relative; left: 1px; top: -29px; z-index: 2; }
            .tab label:hover {background: #035672;}
            .tab [type=radio] { display: none; }
            .content { position: absolute; top: -1px; left: 0; background: #fff; right: 0; bottom: 0; padding: 30px 20px; transition: opacity .1s linear; opacity: 0; }
            [type=radio]:checked ~ label { background: #035672; color: #fff;}
            [type=radio]:checked ~ label ~ .content { z-index: 1; opacity: 1; }

            /* Feature Importance Plotly Link Color */
            .js-plotly-plot .plotly svg a {color: #f84f57 !important}
        </style>
    """
    st.markdown(main_external_css, unsafe_allow_html=True)

    # Fundemental elements
    widget_values = objdict()
    record_widgets = objdict()

    # Sidebar widgets
    sidebar_elements = {
        "button_": st.sidebar.button,
        "slider_": st.sidebar.slider,
        "number_input_": st.sidebar.number_input,
        "selectbox_": st.sidebar.selectbox,
        "multiselect": st.multiselect
    }
    for sidebar_key, sidebar_value in sidebar_elements.items():
        record_widgets[sidebar_key] = make_recording_widget(sidebar_value, widget_values)

    return widget_values, record_widgets

@st.cache(persist=True, show_spinner=True)
def load_data(file_buffer, delimiter):
    """
    Load data to pandas dataframe
    """

    warnings = []
    df = pd.DataFrame()
    if file_buffer is not None:
        if delimiter == "Excel File":
            df = pd.read_excel(file_buffer)

            #check if all columns are strings valid_columns = []
            error = False
            valid_columns = []
            for idx, _ in enumerate(df.columns):
                if isinstance(_, str):
                    valid_columns.append(_)
                else:
                    warnings.append(f'Removing column {idx} with value {_} as type is {type(_)} and not string.')
                    error = True
            if error:
                warnings.append("Errors detected when importing Excel file. Please check that Excel did not convert protein names to dates.")
                df = df[valid_columns]

        elif delimiter == "Comma (,)":
            df = pd.read_csv(file_buffer, sep=',')
        elif delimiter == "Semicolon (;)":
            df = pd.read_csv(file_buffer, sep=';')
    return df, warnings

def get_system_report():
    """
    Returns the package versions
    """
    report = {}
    report['omic_learn_version'] = "v1.1.0"
    report['python_version'] = sys.version[:5]
    report['pandas_version'] = pd.__version__
    report['numpy_version'] = np.version.version
    report['sklearn_version'] = sklearn.__version__
    report['plotly_version'] = plotly.__version__

    return report

def get_download_link(exported_object, name):
    """
    Generate download link for charts in SVG and PDF formats and for dataframes in CSV format
    """
    os.makedirs("downloads/", exist_ok=True)
    extension = name.split(".")[-1]

    if extension == 'svg':
        exported_object.write_image("downloads/"+ name, height=700, width=700, scale=1)
        with open("downloads/" + name) as f:
            svg = f.read()
        b64 = base64.b64encode(svg.encode()).decode()
        href = f'<a class="download_link" href="data:image/svg+xml;base64,%s" download="%s" >Download as *.svg</a>' % (b64, name)
        st.markdown('')
        st.markdown(href, unsafe_allow_html=True)

    elif extension == 'pdf':
        exported_object.write_image("downloads/"+ name, height=700, width=700, scale=1)
        with open("downloads/" + name, "rb") as f:
            pdf = f.read()
        b64 = base64.encodebytes(pdf).decode()
        href = f'<a class="download_link" href="data:application/pdf;base64,%s" download="%s" >Download as *.pdf</a>' % (b64, name)
        st.markdown('')
        st.markdown(href, unsafe_allow_html=True)

    elif extension == 'csv':
        exported_object.to_csv("downloads/"+ name, index=False)
        with open("downloads/" + name, "rb") as f:
            csv = f.read()
        b64 = base64.b64encode(csv).decode()
        href = f'<a class="download_link" href="data:file/csv;base64,%s" download="%s" >Download as *.csv</a>' % (b64, name)
        st.markdown('')
        st.markdown(href, unsafe_allow_html=True)

    else:
        raise NotImplementedError('This output format function is not implemented')
