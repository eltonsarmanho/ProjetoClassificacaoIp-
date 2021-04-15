import itertools
from collections import Counter

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib import cm
from matplotlib.colors import ListedColormap
from sklearn import pipeline
from sklearn import preprocessing
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import RFE, SelectKBest, chi2
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.svm import SVR
from xgboost import XGBClassifier


def correlation(dataframe):
    print("correlation...")
    top = cm.get_cmap('Oranges_r', 128)
    bottom = cm.get_cmap('Blues', 128)

    newcolors = np.vstack((top(np.linspace(0, 1, 128)),
                           bottom(np.linspace(0, 1, 128))))
    newcmp = ListedColormap(newcolors, name='OrangeBlue')
    print("Correlation")
    corr = dataframe.corr()
    print(corr)
    # Create positive correlation matrix
    corr = dataframe.corr().abs()
    # Create and apply mask
    mask = np.triu(np.ones_like(corr, dtype=bool))
    tri_df = corr.mask(mask)
    # Find columns that meet treshold
    to_drop = [c for c in tri_df.columns if any(tri_df[c] > 0.95)]
    print(to_drop)
    reduced_df = dataframe.drop(to_drop, axis=1)
    print("Dimensionality reduced from {} to {}.".format(dataframe.shape[1], reduced_df.shape[1]))    #Insert Column without erro

    # Create and apply mask
    mask = np.triu(np.ones_like(reduced_df.corr(), dtype=bool))
    sns.heatmap(reduced_df.corr(), mask=mask,
                center=0, cmap=newcmp, linewidths=1,
                annot=False, fmt=".2f")
    plt.show()
    return  reduced_df;

def DimensionalityReduced(name):
    print('Method showDataSet')
    try:
        dataset = pd.read_excel(name)
        filetype = 'EXCEL'
    except Exception:
        dataset = pd.read_csv(name)
        filetype = 'CSV'
    print(dataset.columns)
    #print(dataset.describe(include=[np.number]))
    #print(dataset['Especie'].describe())
    print(dataset.shape)

    columns_numeric = pd.DataFrame(dataset._get_numeric_data()).columns
    columns_categorical = pd.DataFrame(dataset.select_dtypes(['object'])).columns
    print("Categorical Columns")
    print(columns_categorical)
    drop_columns = columns_numeric
    dataset_reduce = dataset.drop(columns=drop_columns,axis=1)

    print("Dimensionality reduced from {} to {}.".format(dataset.shape[1], dataset_reduce.shape[1]))
    print("Detect missing values.")
    print(dataset_reduce.isna().sum() / len(dataset_reduce))

    #Remove by value
    values = ['Handroanthus cf. serratifolius','Handroanthus serratifolius']
    dataset_reduce = dataset_reduce[dataset_reduce['Especie'].isin(values)]

    return  dataset_reduce;

# Press the green button in the gutter to run the script.
def transformedData(dataframe):
    print("transformedData..")
    column ='Especie'
    print(Counter(dataframe[column]))
    dataframe = dataframe.apply(preprocessing.LabelEncoder().fit_transform)
    #dataframe = correlation(dataframe)
    #print(Counter(dataframe[column]))
    # To convert for array
    y = np.asarray(dataframe['Especie'])
    print(y)
    df = dataframe.drop('Especie', 1)  # Remove the predict variable
    X = np.asarray(df)
    return  X,y;


def classification( X_train, X_test, y_train, y_test,dataframe):
    print("classification...")
    # Number of trees in random forest
    n_estimators = [int(x) for x in np.linspace(start=200, stop=2000, num=10)]
    # Number of features to consider at every split
    max_features = ['auto', 'sqrt']
    # Maximum number of levels in tree
    max_depth = [int(x) for x in np.linspace(10, 110, num=11)]
    max_depth.append(None)
    # Minimum number of samples required to split a node
    min_samples_split = [2, 5, 10]
    # Minimum number of samples required at each leaf node
    min_samples_leaf = [1, 2, 4]
    # Method of selecting samples for training each tree
    bootstrap = [True, False]
    # Create the random grid
    random_grid = {'n_estimators': n_estimators,
                   'max_features': max_features,
                   'max_depth': max_depth,
                   'min_samples_split': min_samples_split,
                   'min_samples_leaf': min_samples_leaf,
                   'bootstrap': bootstrap
                   }
    # Create the random grid
    pipe = Pipeline([
        # the reduce_dim stage is populated by the param_grid
        ('reduce_dim', 'passthrough'),
        ('classify', 'passthrough')
    ])
    N_FEATURES_OPTIONS = [12,13,14,15,16]

    param_grid = [
        {
            'reduce_dim': [RFE(estimator=SVR(kernel='linear'),step=2, verbose=0)],
             'reduce_dim__n_features_to_select': N_FEATURES_OPTIONS,
            'classify': [RandomForestClassifier()],
            'classify__n_estimators':n_estimators,
            'classify__max_features':max_features,
            'classify__max_depth':max_depth,
            'classify__min_samples_split':min_samples_split
        },
        {
            'reduce_dim': [RFE(estimator=RandomForestClassifier(), step=2, verbose=0)],
            'reduce_dim__n_features_to_select': N_FEATURES_OPTIONS,
            'classify': [RandomForestClassifier()],
             'classify__n_estimators':n_estimators,
             'classify__max_features':max_features,
             'classify__max_depth':max_depth,
             'classify__min_samples_split':min_samples_split
        },
        {
            'reduce_dim': [SelectKBest(chi2)],
            'reduce_dim__k': N_FEATURES_OPTIONS,
            'classify': [RandomForestClassifier()],
            'classify__n_estimators':n_estimators,
            'classify__max_features':max_features,
            'classify__max_depth':max_depth,
            'classify__min_samples_split':min_samples_split
        },
    ]


    reducer_labels = ['RFE(SVM)','RFE(RFC)', 'KBest(chi2)']

    grid = GridSearchCV(pipe, n_jobs=1, param_grid=param_grid,scoring= 'accuracy')
    grid.fit(X_train, y_train)

    print("Best parameters set found on development set:")
    print()
    print(grid.best_params_)
    print('Accuracy: %.2f' % (grid.score(X_test, y_test)))
    showConfusionMatrix(X_test, y_test,grid,dataframe)

    mean_scores = np.array(grid.cv_results_['mean_test_score'])
    # scores are in the order of param_grid iteration, which is alphabetical
    mean_scores = mean_scores.reshape(1, -1, len(N_FEATURES_OPTIONS))
    # select score for best C
    mean_scores = mean_scores.max(axis=0)
    bar_offsets = (np.arange(len(N_FEATURES_OPTIONS)) *
                   (len(reducer_labels) + 1) + .5)

    plt.figure()
    COLORS = 'bgrcmyk'
    for i, (label, reducer_scores) in enumerate(zip(reducer_labels, mean_scores)):
        rects = plt.bar(bar_offsets + i, reducer_scores, label=label, color=COLORS[i])
        autolabel(rects,plt)
    plt.title("Comparing feature reduction techniques")
    plt.xlabel('Reduced number of features')
    plt.xticks(bar_offsets + len(reducer_labels) / 4, N_FEATURES_OPTIONS)
    plt.ylabel('classification accuracy')
    plt.ylim((0, 1))
    plt.legend(loc='upper left')

    plt.show()

def autolabel(rects,ax):
    """
    Attach a text label above each bar displaying its height
    """
    for rect in rects:
        height = rect.get_height()
        ax.text(rect.get_x() + rect.get_width()/2., 1.05*height,
                '%.2f' % height, ha='center', va='bottom')

def showConfusionMatrix(X_test,y_test,model,dataframe):
    cnf_matrix = confusion_matrix(y_test, model.predict(X_test),
                                  labels=range(0, len(set(dataframe['Especie']))))
    np.set_printoptions(precision=2)
    plot_confusion_matrix(cnf_matrix, classes=sorted(Counter(dataframe['Especie']).keys()),
                          normalize=True, title='Confusion matrix')

def runReduceRFE( X_train, X_test, y_train, y_test,dataframe):
    rf = RFE(estimator=SVR(kernel='linear', n_features_to_select=16, step=2));
    rf.fit(X_train, y_train)

    # temp = pd.Series(rf.support_, index=dataframe.drop(columns='Especie',axis=1).columns)
    # selected_features_rfe = temp[temp == True].index
    # print(selected_features_rfe)
    # print('Accuracy: %.2f' % (rf.score(X_test,y_test)))
    # evaluate(rf, X, y)
    # showConfusionMatrix(X_test, y_test,rf,dataframe)


def selectionBestFeature( X_train, X_test, y_train, y_test,dataframe):
    print("selectionBestFeature..")

    classify_best = {'max_depth': 70, 'max_features':
        'sqrt','min_samples_split': 2, 'n_estimators': 200, }
    rf =  RFE(estimator=SVR(kernel='linear'),n_features_to_select=16, step=2);

    rf_random = RandomForestClassifier(**classify_best);

    rf.fit(X_train, y_train)
    # Create the random grid
    pipe = Pipeline([
        # the reduce_dim stage is populated by the param_grid
        ('reduce_dim', rf),
        ('classify', rf_random)
    ])

    pipe.fit(X_train, y_train)
    print(pipe.score(X_test, y_test))
    evaluate(pipe, X_test, y_test)
    showConfusionMatrix(X_test, y_test,pipe,dataframe)

def evaluate(clf, X, y):
    print("evaluate...")
    scores = cross_val_score(clf, X, y, scoring='accuracy' , cv=3,n_jobs=-1,verbose=1)
    print("The mean score and the 95%% confidence interval of the score estimate are hence given by Accuracy: %0.2f (+/- %0.2f)" % (scores.mean(), scores.std() * 1.96 ))

def plot_confusion_matrix(cm, classes, normalize=False, title='Confusion matrix', cmap='Blues'):
        """
        This function prints and plots the confusion matrix.
        Normalization can be applied by setting `normalize=True`.
        """
        if normalize:
            cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
            cm = cm * 100
            print("Normalized confusion matrix")
        else:
            print('Confusion matrix, without normalization')

        print(cm)
        # Plot non-normalized confusion matrix
        plt.figure()
        plt.imshow(cm, interpolation='nearest', cmap=cmap)
        plt.title(title)
        plt.colorbar()
        tick_marks = np.arange(len(classes))
        plt.xticks(tick_marks, classes, rotation=90, fontsize=9)
        plt.yticks(tick_marks, classes, fontsize=9)

        fmt = '.2f' if normalize else 'd'
        thresh = cm.max() / 2.
        for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
            plt.text(j, i, format(cm[i, j], fmt) + "%", fontsize=10,
                     horizontalalignment="center",
                     color="white" if cm[i, j] > thresh else "black")

        plt.tight_layout()
        plt.ylabel('True label', fontsize=12)
        plt.xlabel('Predicted label', fontsize=12)
        plt.show()

if __name__ == '__main__':
    url_excel= "https://www.dropbox.com/scl/fi/mob0wl3b7a1d0o6d0av50/Dataset_ipe_09_03_2021.xlsx?dl=1&rlkey=4dyj83gwxi3gz0oiwrjm8j4ti"

    url_csv = "https://www.dropbox.com/s/n7jrz7gd0js9xy6/dataset_ipe.csv?dl=1"

    dataframe  = DimensionalityReduced(url_csv)
    X,y = transformedData(dataframe)
    X_train, X_test, y_train, y_test = train_test_split(X, y,
                                                        test_size=0.3,
                                                        random_state=4,
                                                        stratify=y)
    # Split dataset
    print('Train set:', X_train.shape, y_train.shape)
    print('Test set:', X_test.shape, y_test.shape)

    selectionBestFeature( X_train, X_test, y_train, y_test,dataframe)
    #classification( X_train, X_test, y_train, y_test,dataframe)
# See PyCharm help at https://www.jetbrains.com/help/pycharm/