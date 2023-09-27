import matplotlib.pyplot as plt
import matplotlib as mpl
from mpl_toolkits.axes_grid1.inset_locator import mark_inset, inset_axes, InsetPosition
import os
import numpy as np
import pickle
import sklearn
import joblib
from sklearn.decomposition import PCA
#Classifiers
from xgboost import XGBClassifier
#Eval Metrics
from sklearn.metrics import roc_curve, precision_recall_curve, average_precision_score, precision_recall_fscore_support
import pandas as pd
from sklearn.model_selection import StratifiedKFold, cross_val_score
from hyperopt import hp, fmin, tpe, STATUS_OK, Trials, space_eval
import time


sklearn.set_config(assume_finite=True)


LABEL_INDEX = -2
CAPTURE_INDEX = -1

#DECISION_THRESHOLD = 0.9
DECISION_THRESHOLD = 0.002577161882072687 # For reproducibility

models_folder = 'models/'


plt.rc('font', size=16)          # controls default text sizess
plt.rc('axes', titlesize=30)     # fontsize of the axes title
plt.rc('axes', labelsize=20)    # fontsize of the x and y labels
plt.rc('xtick', labelsize=20)    # fontsize of the tick labels
plt.rc('ytick', labelsize=20)    # fontsize of the tick labels
plt.rc('legend', fontsize=14)    # legend fontsize
plt.rc('figure', titlesize=56)  # fontsize of the figure title

plt.rcParams['axes.xmargin'] = 0
plt.rcParams['axes.ymargin'] = 0
mpl.rcParams['axes.spines.right'] = False
mpl.rcParams['axes.spines.top'] = False


def store_model(model, X_train, save_file_name):
    # Save features names
    model.feature_names = list(X_train.columns.values)

    if not os.path.exists(models_folder):
        os.makedirs(models_folder)
    joblib.dump(model, models_folder+save_file_name)
    

# TODO: Test hyperparameter tuning
class HyperparameterTuning:

    def __init__(self, plFileTrain, statsFileTrain, plFileValidate, statsFileValidate, plFileTest, statsFileTest):
        print("\n=== Gathering training dataset ...")
        self.X_train, self.y_train, _ = gather_dataset(plFileTrain, statsFileTrain)
        print("self.X_train", self.X_train)
        print("\n=== Gathering validation dataset ...")
        self.X_validate, self.y_validate, _ = gather_dataset(plFileValidate, statsFileValidate)
        print("\n=== Gathering testing dataset ...")
        self.X_test, self.y_test, _ = gather_dataset(plFileTest, statsFileTest)

    def search_method(self):
        pass
    
    def store_model(self, opt_model):
        pass

    def search_parameters(self):
        #https://towardsdatascience.com/hyperparameters-optimization-526348bb8e2d
        # https://grabngoinfo.com/hyperparameter-tuning-for-xgboost-grid-search-vs-random-search-vs-bayesian-optimization/
        start_time = time.time()

        best_hyperparameters = self.search_method()

        print("\n=== Creating model with best parameters ...")
        # Train model using the best parameters
        opt_model = XGBClassifier(seed=0, 
                                learning_rate=best_hyperparameters['learning_rate'], 
                                max_depth=best_hyperparameters['max_depth'], 
                                gamma=best_hyperparameters['gamma'], 
                                colsample_bytree=best_hyperparameters['colsample_bytree'], 
                                colsample_bylevel=best_hyperparameters['colsample_bylevel'],
                                reg_alpha=best_hyperparameters['reg_alpha'],
                                reg_lambda=best_hyperparameters['reg_lambda'],
                                n_estimators=best_hyperparameters['n_estimators']
                                ).fit(self.X_train, self.y_train)
        # Make prediction using the best model
        bayesian_opt_predict = opt_model.predict(self.X_test)
        # Get predicted probabilities
        bayesian_opt_predict_prob = opt_model.predict_proba(self.X_test)[:,1]
        # Get performance metrics
        precision, recall, fscore, support = precision_recall_fscore_support(self.y_test, bayesian_opt_predict)
        # Print result
        print(f'The precision value for the xgboost Bayesian optimization is {precision[1]:.4f}')
        print(f'The recall value for the xgboost Bayesian optimization is {recall[1]:.4f}')

        end_time = time.time()
        print("\n=== Hyperparameter tuning time: {}".format(end_time - start_time))

        with open(models_folder+'log_parameters.txt', 'a') as f:
            f.write(f"Hyperparameter tuning time: {end_time - start_time}\n")

        model_save_file = self.store_model(opt_model)
        print("\n=== Stored hyperparameter tuning model")

        probas_ = opt_model.predict_proba(self.X_test)
        plot_precision_recall_curve(self.y_test, probas_, isValidation=True)



class BayesianOptimization(HyperparameterTuning):

    def objective(self, params):
        xgboost = XGBClassifier(seed=0, **params)
        score = cross_val_score(estimator=xgboost, 
                            X=self.X_validate, 
                            y=self.y_validate, 
                            cv=self.kfold, 
                            scoring='precision', 
                            n_jobs=-1).mean()
        # Loss is negative score
        loss = -score

        if not os.path.isfile(models_folder+'log_parameters.txt'):
            write_mode = 'w'
        else:
            write_mode = 'a'
        with open(models_folder+'log_parameters.txt', write_mode) as f:
            f.write(f"Hyperparameters: {params}\n")
            f.write(f"Loss: {loss}\n\n")

        # Dictionary with information for evaluation
        return {'loss': loss, 'params': params, 'status': STATUS_OK}


    def search_method(self):
        np.random.seed(123)
        
        print("\n=== Parameter search with BayesianOptimization  ...")

        save_file_name = 'best_hyperparameters_source_separation.joblib'
        if os.path.isfile(models_folder+save_file_name):
            print("\n=== Loading existing parameters ...")
            best_hyperparameters = joblib.load(models_folder+save_file_name)
            print("\n--- best_hyperparameters", best_hyperparameters)
            return best_hyperparameters

        space = {
            'learning_rate': hp.choice('learning_rate', [0.0001,0.001, 0.01, 0.1, 1]),
            'max_depth' : hp.choice('max_depth', range(3,21,3)),
            'gamma' : hp.choice('gamma', [i/10.0 for i in range(0,5)]),
            'colsample_bytree' : hp.choice('colsample_bytree', [i/10.0 for i in range(3,10)]), 
            'colsample_bylevel': hp.choice('colsample_bylevel', [i/10.0 for i in range(3,10)]),    
            'reg_alpha' : hp.choice('reg_alpha', [1e-5, 1e-2, 0.1, 1, 10, 100]), 
            'reg_lambda' : hp.choice('reg_lambda', [1e-5, 1e-2, 0.1, 1, 10, 100]),
            'n_estimators' : hp.choice('n_estimators', [10, 50, 100, 300, 500, 800])
        }

        # Set up the k-fold cross-validation
        self.kfold = StratifiedKFold(n_splits=3, shuffle=True, random_state=0)

        print("\n=== Bayesian Optimization...")
        trials = Trials()

        #best = fmin(fn = self.objective, space=space, algo=tpe.suggest, max_evals=50, trials=Trials())
        best = fmin(fn=self.objective, space=space, algo=tpe.suggest, max_evals=100, trials=trials)

        best_hyperparameters = space_eval(space, best)

        # Print the index of the best parameters
        print("\n--- best:", best)
        # Print the values of the best parameters
        print("\n--- best_hyperparameters", best_hyperparameters)
        
        if not os.path.exists(models_folder):
            os.makedirs(models_folder)
        joblib.dump(best_hyperparameters, models_folder+save_file_name)

        return best_hyperparameters


    def store_model(self, opt_model):
        save_file = 'source_separation_model_bayesian_optimization.joblib'
        store_model(opt_model, self.X_train, save_file)

        return save_file

def optimal_threshold(tpr, fpr, thresholds):
    optimal_idx = np.argmax(tpr - fpr)
    optimal_threshold = thresholds[optimal_idx]
    
    return optimal_threshold

def plot_precision_recall_curve(y_test, probas_, isValidation=False):
    fig, ax = plt.subplots(figsize=(5, 4))
    
    precision, recall, thresholds = precision_recall_curve(y_test, probas_[:, 1])

    plt.plot(np.insert(recall, 0, recall[0]), np.insert(precision, 0, 0), linewidth=4, color="tab:blue", label="PR curve (AP={})".format(round(average_precision_score(y_test, probas_[:, 1]), 2)))
    plt.plot([0, 1], [0.5, 0.5], linestyle='--', color='orange', label='Random Classifier', linewidth=2)
    plt.plot([0, 1, 1], [1, 1, 0], linestyle=':', color='black', label='Perfect Classifier', linewidth=3)
    plt.ylabel("Precision")
    plt.xlabel("Recall")
    plt.ylim(0.49, 1.01)
    plt.xlim(0, 1.01)
    x_axis = np.arange(0, 1.01, 0.25)
    y_axis = np.arange(0.5, 1.01, 0.25)
    plt.xticks(x_axis, x_axis)
    plt.yticks(y_axis, y_axis)
    plt.legend()
    plt.tight_layout()

    results_folder = 'results/'
    if not os.path.exists(results_folder):
        os.makedirs(results_folder)
    if isValidation == False:
        plt.savefig(results_folder + 'precision_recall_curve_source_separation.pdf')
        plt.savefig(results_folder + 'precision_recall_curve_source_separation.png')
    else:
        plt.savefig(results_folder + 'precision_recall_curve_source_separation_validation.pdf')
        plt.savefig(results_folder + 'precision_recall_curve_source_separation_validation.png')


def plot_precision_recall_curve_zoomin(statsFileTest, model_save_file, results_file):
    
    if os.path.isfile(model_save_file):
        print("Gathering trained model ...")
        model = joblib.load(model_save_file)
    else:
        print("You have to train source separation's model first!")
        print("Exiting ...")
        exit()

    print("Gathering testing dataset ...")
    X_test , y_test, test_captures = gather_dataset(statsFileTest)

    # Predicts the probability of each element to belong to a determined class
    probas_ = model.predict_proba(X_test)

    fig, ax1 = plt.subplots(1, 1, figsize=(5, 4))
    
    precision, recall, thresholds = precision_recall_curve(y_test, probas_[:, 1])

    # for i, p in enumerate(precision):
    #     if round(p, 2) == 0.99:
    #         print("Precision: {}; recall: {}".format(p, recall[i]))

    # for i, r in enumerate(recall):
    #     if round(r, 2) == 0.99:
    #         print("xxx Precision: {}; recall: {}".format(precision[i], r))

    plt.plot(np.insert(recall, 0, recall[0]), np.insert(precision, 0, 0), linewidth=4, color="tab:blue", zorder=0)
    plt.ylabel("Precision")
    plt.xlabel("Recall")
    plt.ylim(0.49, 1.01)
    plt.xlim(0, 1.01)
    plt.xticks([0, 0.2, 0.4, 0.6, 0.8, 1], ["0", "0.2", "0.4", "0.6", "0.8", "1"])
    plt.yticks([0.5, 0.6, 0.7, 0.8, 0.9, 1], ["0.5", "0.6", "0.7", "0.8", "0.9", "1"])

    axins = ax1.inset_axes([0.1, 0.1, 0.80, 0.80])
    axins.spines[['right', 'top']].set_visible(True)
    axins.plot(np.insert(recall, 0, recall[0]), np.insert(precision, 0, 0), linewidth=2)
    axins.set_xlim(0.8, 1)
    axins.set_ylim(0.95, 1)
    axins.set_xticks([0.8, 0.82, 0.84, 0.86, 0.88, 0.9, 0.92, 0.94, 0.96, 0.98, 1], ["0.8", "", "", "", "", "0.9", "", "", "", "",  "1"])
    axins.set_yticks([0.95, 0.96, 0.97, 0.98, 0.99, 1], ["0.95", "", "", "", "", "1"])
    axins.tick_params(axis='both', which='major', labelsize=16)
    axins.set_axes_locator(InsetPosition(ax1, [0.3, 0.3, 0.4, 0.4]))  # Move the zoomed-in plot, posx, posy, width, height
    mark_inset(ax1, axins, loc1=2, loc2=4, fc="none", ec="gray", linestyle='--') # loc1 and loc2 change the connecting corners of the zoomin
    axins.text(x=0.87, y=0.985 ,s="AP={}".format(round(average_precision_score(y_test, probas_[:, 1]), 2)), ha="center", va="center", fontsize=16)
    plt.draw()
    plt.tight_layout()

    results_folder = 'results/'
    if not os.path.exists(results_folder):
        os.makedirs(results_folder)
    plt.savefig(results_folder + '{}.pdf'.format(results_file))
    plt.savefig(results_folder + '{}.png'.format(results_file))

def gather_dataset(statsFile):
    stats = pd.read_csv(statsFile) 

    print("stats:", stats)
    # Transform dtype object columns to numeric
    cols = stats[stats.columns[:LABEL_INDEX]].select_dtypes(exclude=['float']).columns
    stats[cols] = stats[cols].apply(pd.to_numeric, downcast='float', errors='coerce')

    train = stats
    train['Class'] = train['Class'].astype(int)

    # Remove columns that only have zeros
    train.loc[:, (train != 0).any(axis=0)]

    # Shuffle dataset
    train = train.sample(frac = 1)

    y_train = train['Class']
    captures = train['Capture']

    x_train = train[train.columns[:LABEL_INDEX]]

    return x_train, y_train, captures


def train(statsFileTrain, model_save_file):
    print("\n=== Gathering training dataset ...")
    X_train , y_train, _ = gather_dataset(statsFileTrain)

    print("\n=== Creating model ...")
    model = XGBClassifier(seed=0) # Seed is 0 to get reproducible results
    print("\n=== Training model ...")
    model.fit(X_train, y_train)

    store_model(model, X_train, model_save_file)

def hyperparameter_tuning(statsFileTrain, statsFileValidate, statsFileTest, algorithm='BayesianOptimization'):
    if algorithm == 'BayesianOptimization':
        hyperparameter_tuning = BayesianOptimization(statsFileTrain, statsFileValidate, statsFileTest)
    hyperparameter_tuning.search_parameters()

def test(statsFileTest, model_save_file):
    if os.path.isfile(model_save_file):
        print("Gathering trained model ...")
        model = joblib.load(models_folder+model_save_file)
    else:
        print("You have to train source separation's model first!")
        print("Exiting ...")
        exit()

    print("Gathering testing dataset ...")
    X_test , y_test, test_captures = gather_dataset(statsFileTest)

    # Predicts the probability of each element to belong to a determined class
    probas_ = model.predict_proba(X_test)
    plot_precision_recall_curve(y_test, probas_)
    
    return probas_

def test_full_pipeline(dataset_name, statsFileTest, model_save_file, optimal_thr=True):

    if os.path.isfile(model_save_file):
        print("Gathering trained model ...")
        model = joblib.load(model_save_file)
    else:
        print("You have to train source separation's model first!")
        print("Exiting ...")
        exit()

    print("Gathering testing dataset ...")
    X_test , y_test, test_captures = gather_dataset(statsFileTest)

    # Predicts the probability of each element to belong to a determined class
    probas_ = model.predict_proba(X_test)
    plot_precision_recall_curve(y_test, probas_)

    fpr, tpr, thresholds = roc_curve(y_test, probas_[:, 1], pos_label=1)

    if optimal_thr == True:
        decision_threshold = optimal_threshold(tpr, fpr, thresholds)
        print("decision_threshold", decision_threshold)
    else:
        decision_threshold = DECISION_THRESHOLD

    dump_pipeline_features(dataset_name, X_test, probas_[:, 1], test_captures, decision_threshold)
    
    return probas_

def dump_pipeline_features(dataset_name, features, predictions, captures, decision_threshold):
    print("Dumping features for next stage of the pipeline ...")
    
    outputClientFeatures = {}
    outputOSFeatures = {}

    for i in range(len(predictions)):
        if predictions[i] < decision_threshold:
            # use iloc to access a whole row in the dataframe
            outputClientFeatures[captures.iloc[i]] = features.iloc[i]
        else:
            outputOSFeatures[captures.iloc[i]] = features.iloc[i]

    features_folder = 'full_pipeline_features/'
    if not os.path.exists(features_folder):
        os.makedirs(features_folder)
    pickle.dump(outputClientFeatures, open(features_folder+'client_features_source_separation_thr_{}_{}.pickle'.format(decision_threshold, dataset_name), 'wb'))
    pickle.dump(outputOSFeatures, open(features_folder+'os_features_source_separation_thr_{}_{}.pickle'.format(decision_threshold, dataset_name), 'wb'))