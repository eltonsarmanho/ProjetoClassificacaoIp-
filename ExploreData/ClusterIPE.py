
import itertools
from collections import Counter
from sklearn.decomposition import PCA

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib import cm
from matplotlib.colors import ListedColormap
from sklearn import pipeline
from sklearn import preprocessing
from sklearn.cluster import KMeans
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import RFE, SelectKBest, chi2
from sklearn.manifold import TSNE
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.svm import SVR
from xgboost import XGBClassifier
from sklearn.preprocessing import StandardScaler


def readData(name):
    print('Method readData..')
    try:
        dataset = pd.read_excel(name)
        filetype = 'EXCEL'
    except Exception:
        dataset = pd.read_csv(name)
        filetype = 'CSV'
    print(dataset.columns)
    print(dataset.shape)

    columns_numeric = pd.DataFrame(dataset._get_numeric_data()).columns
    columns_categorical = pd.DataFrame(dataset.select_dtypes(['object'])).columns
    print("Categorical Columns")
    print(columns_categorical)
    drop_columns = (columns_categorical.drop('Especie'))
    dataset_reduce = dataset.drop(columns=drop_columns,axis=1)


    print("Dimensionality reduced from {} to {}.".format(dataset.shape[1], dataset_reduce.shape[1]))
    print("Detect missing values.")
    print(dataset_reduce.isna().sum() / len(dataset_reduce))

    #Remove by value
    values = ['Handroanthus cf. serratifolius','Handroanthus serratifolius']
    dataset_reduce = dataset_reduce[dataset_reduce['Especie'].isin(values)]
    print(Counter(dataset_reduce['Especie']))
    return  dataset_reduce;

def runCrossTab(dataframe):
    fig, ax = plt.subplots(3, figsize=(14, 10))

    # Create a KMeans model with 2 clusters: model
    model = KMeans(n_clusters=2)
    samples = dataframe.drop(columns='Especie', axis=1)
    varieties = dataframe['Especie']

    # Use fit_predict to fit model and obtain cluster labels: labels
    labels = model.fit_predict(samples)
    # Create a DataFrame with labels and varieties as columns: df
    df = pd.DataFrame({'labels': labels, 'varieties': varieties})
    # Create crosstab: ct
    ct = pd.crosstab(df['labels'], df['varieties'])
    print("Without feature standardization:")
    print(ct)

    xs = samples['Espessura_parede_fibras']
    ys = samples['Comprimento_elementos_vaso']
    scatter = ax[0].scatter(x=xs,y=ys,c=df['labels'])
    #ax[0].legend(handles=scatter.legend_elements()[0], labels=labels_)
    ax[0].set_title('Without feature standardization')
    # Annotate the points
    for x, y, labels, especies in zip(xs, ys, df['labels'], df['varieties']):
        ax[0].annotate(especies + "(" + str(labels) + ")", (x, y), fontsize=10, alpha=0.75)

    scaler = preprocessing.StandardScaler()
    # transform data
    samples_scaled = scaler.fit_transform(samples)
    print("With feature standardization:")
    # Create a KMeans model with 2 clusters: model
    model = KMeans(n_clusters=2)
    labels = model.fit_predict(samples_scaled)
    df = pd.DataFrame({'labels': labels, 'varieties': varieties})
    # Create crosstab: ct
    ct = pd.crosstab(df['labels'], df['varieties'])
    print(ct)

    scatter = ax[1].scatter(x=xs,y=ys,c=df['labels'])
    ax[1].set_title('With feature standardization')
    # Annotate the points
    for x, y, labels, especies in zip(xs, ys, df['labels'], df['varieties']):
        ax[1].annotate(especies + "(" + str(labels) + ")", (x, y), fontsize=10, alpha=0.75)

    print("With feature PCA:")
    # Create PCA instance: model
    pca = PCA(n_components=0.95)
    # Apply the fit_transform method of model to sample
    pca_features = pca.fit_transform(samples_scaled)

    # Assign 0th column of pca_features: xs
    xs = pca_features[:, 0]
    # Assign 1st column of pca_features: ys
    ys = pca_features[:, 1]

    model = KMeans(n_clusters=2)
    labels = model.fit_predict(pca_features)
    df = pd.DataFrame({'labels': labels, 'varieties': varieties})
    # Create crosstab: ct
    ct = pd.crosstab(df['labels'], df['varieties'])
    print(ct)

    scatter = ax[2].scatter(x=xs,y=ys,c=df['labels'])
    ax[2].set_title('With feature PCA')
    # Annotate the points
    for x, y, labels,especies in zip(xs, ys, df['labels'],df['varieties']):
        ax[2].annotate(especies+"("+str(labels)+")", (x, y), fontsize=10, alpha=0.75)
    plt.show()


def runAnalisePCA(dataframe):
    samples = dataframe.drop(columns='Especie', axis=1)
    varieties = dataframe['Especie']

    scaler = preprocessing.StandardScaler()
    # transform data
    samples_scaled = scaler.fit_transform(samples)



    pca = PCA(n_components=0.89)
    # Apply the fit_transform method of model to sample
    pca_features = pca.fit_transform(samples_scaled)

    # Assign 0th column of pca_features: xs
    xs = pca_features[:, 0]
    # Assign 1st column of pca_features: ys
    ys = pca_features[:, 1]


    features = range(pca.n_components_)
    plt.bar(features, pca.explained_variance_ratio_)
    plt.xlabel('PCA feature')
    plt.ylabel('variance')
    plt.xticks(features)
    plt.show()
def correlation(dataframe):
    print("correlation...")
    fig, (ax1, ax2) = plt.subplots(1,2, figsize=(24, 16))

    top = cm.get_cmap('Oranges_r', 128)
    bottom = cm.get_cmap('Blues', 128)

    newcolors = np.vstack((top(np.linspace(0, 1, 128)),
                           bottom(np.linspace(0, 1, 128))))
    newcmp = ListedColormap(newcolors, name='OrangeBlue')
    corr = dataframe.corr()
    # Create positive correlation matrix
    corr = dataframe.corr().abs()
    # Create and apply mask
    mask = np.triu(np.ones_like(corr, dtype=bool))
    tri_df = corr.mask(mask)

    sns.heatmap(corr, mask=mask,
                center=0, cmap=newcmp, linewidths=1,
                annot=True, fmt=".2f", ax=ax1)

    # Find columns that meet treshold
    to_drop = [c for c in tri_df.columns if any(tri_df[c] > 0.95)]
    print(to_drop)
    reduced_df = dataframe.drop(to_drop,axis=1)
    print("Dimensionality reduced from {} to {}.".format(dataframe.shape[1], reduced_df.shape[1]))    #Insert Column without erro

    # Create and apply mask
    mask = np.triu(np.ones_like(reduced_df.corr(), dtype=bool))
    sns.heatmap(reduced_df.corr(), mask=mask,
                center=0, cmap=newcmp, linewidths=1,
                annot=True, fmt=".2f", ax=ax2)
    plt.show()
    return  reduced_df;

def visualization(dataframe):
    # Create a TSNE instance: model
    samples = dataframe.drop(columns='Especie', axis=1)
    varieties = dataframe['Especie']
    varieties_number = dataframe.apply(preprocessing.LabelEncoder().fit_transform)['Especie']
    model = TSNE(learning_rate=80)

    # Apply fit_transform to samples: tsne_features
    tsne_features = model.fit_transform(samples)

    # Select the 0th feature: xs
    xs = tsne_features[:, 0]

    # Select the 1st feature: ys
    ys = tsne_features[:, 1]

    # Scatter plot, coloring by variety_numbers
    plt.scatter(xs, ys, alpha=0.5,c=varieties_number)
    # Annotate the points
    for x, y, company in zip(xs, ys,varieties):
        plt.annotate(company, (x, y), fontsize=5, alpha=0.75)
    plt.show()
    pass

def runCluster(dataframe):
    ks = range(1, 6)
    inertias = []
    samples = dataframe.drop(columns='Especie',axis=1)
    scaler = StandardScaler()
    # transform data
    samples_scaled = scaler.fit_transform(samples)
    varieties = dataframe['Especie']
    for k in ks:
        # Create a KMeans instance with k clusters: model
        model = KMeans(n_clusters=k)

        # Fit model to samples
        model.fit(samples_scaled)

        # Append the inertia to the list of inertias
        inertias.append(model.inertia_)

    # Plot ks vs inertias
    plt.plot(ks, inertias, '-o')
    plt.xlabel('number of clusters, k')
    plt.ylabel('inertia')
    plt.xticks(ks)
    plt.show()

if __name__ == '__main__':
    url_csv = "../dataset/dataset.csv"
    dataframe  = readData(url_csv)
    #dataframe = correlation(dataframe)
    #visualization(dataframe)
    runCrossTab(dataframe)