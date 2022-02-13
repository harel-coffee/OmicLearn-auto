## **Table of Contents**

- [3. Exploratory data analysis (EDA)](#3-exploratory-data-analysis--eda-)
  * [3. 1. Hierarchical clustering](#3-1-hierarchical-clustering)
  * [3. 2. Principal component analysis](#3-2-principal-component-analysis)


---

## 3. Exploratory data analysis (EDA)

Exploratory Data Analysis (EDA) is an important approach to get more insight into the dataset before building a model. It aids in pattern discovery and anomaly detection and allows verifying assumptions via graphical representations.

### 3. 1. Hierarchical clustering

Hierarchical clustering (also known as hierarchical cluster analysis) enables researchers to group similar features hierarchically and it displays the hierarchical relationships between the features with dendrograms. This allows users to visualize different sub-clusters, sets of features that strongly correlate and provide an overview of the dataset. We provide an interactive heatmap so that feature names can quickly be retrieved by hovering over the data point. This allows verifying whether a correlation might be expected or is random.


### [3. 2. Principal component analysis](https://scikit-learn.org/stable/modules/generated/sklearn.decomposition.PCA.html)

Principal component analysis (PCA) is widely used for reducing the dimensionality of the data to make large datasets more interpretable. When creating a graphical representation of data, one is typically limited to 2D or 3D representations, which poses a practical problem when wanting to visualize datasets with much more features. With the help of PCA, it is possible to reduce the number of features while trying to conserve the information content. We employ PCA to reduce dimensionality to display the data in a 2D graph.

One use case of PCA is to identify major sources of variations in a dataset as PCA will cluster similar data points together. Hence, by inspecting datapoints that cluster together, you can assess whether variations can be attributed to your experimental setup, special biological conditions, experimental or sample bias. A common example for this is difference in sample preparation, e.g., by a different laboratory assistant or institute.
