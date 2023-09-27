import matplotlib.pyplot as plt
import matplotlib as mpl
from mpl_toolkits.axes_grid1.inset_locator import mark_inset, inset_axes, InsetPosition
import os
import numpy as np
import pickle
import sklearn
import joblib
from xgboost import XGBClassifier
from sklearn.metrics import roc_curve, precision_recall_curve, average_precision_score, precision_recall_fscore_support
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import GridSearchCV, StratifiedKFold, cross_val_score
from hyperopt import hp, fmin, tpe, STATUS_OK, Trials, space_eval
from hyperopt.early_stop import no_progress_loss
import time


sklearn.set_config(assume_finite=True)
pd.set_option('display.max_colwidth', None)


LABEL_INDEX = -2
CAPTURE_INDEX = -1

THRESHOLD = 0.9

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


class HyperparameterTuning:

    def __init__(self, plFileTrain, statsFileTrain, plFileValidate, statsFileValidate, plFileTest, stats_file_test):
        print("\n=== Gathering training dataset ...")
        self.X_train, self.y_train = gather_dataset(statsFileTrain)
        print("\n=== Gathering validation dataset ...")
        self.X_validate, self.y_validate = gather_dataset(statsFileValidate)
        print("\n=== Gathering testing dataset ...")
        self.X_test, self.y_test = gather_dataset(stats_file_test)

    def search_method(self):
        pass
    
    def store_model(self, opt_model):
        pass

    def search_parameters(self):
        # https://towardsdatascience.com/hyperparameters-optimization-526348bb8e2d
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
        plot_precision_recall_curve(self.y_test, probas_, is_validation=True)


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
        loss = - score

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
        
        save_file_name = 'best_hyperparameters_target_separation.joblib'
        if os.path.isfile(models_folder+save_file_name):
            print("\n=== Loading existing parameters ...")
            best_hyperparameters = joblib.load(models_folder+save_file_name)
            print("\n--- best_hyperparameters", best_hyperparameters)
            return best_hyperparameters

        space = {
            'learning_rate': hp.choice('learning_rate', [0.0001, 0.001, 0.01, 0.1, 1]),
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
        save_file = 'target_separation_model_bayesian_optimization.joblib'
        store_model(opt_model, self.X_train, save_file)

        return save_file


def true_false_positive(threshold_vector, y_test):
    true_positive = np.equal(threshold_vector, 1) & np.equal(y_test, 1)
    true_negative = np.equal(threshold_vector, 0) & np.equal(y_test, 0)
    false_positive = np.equal(threshold_vector, 1) & np.equal(y_test, 0)
    false_negative = np.equal(threshold_vector, 0) & np.equal(y_test, 1)

    tpr = true_positive.sum() / (true_positive.sum() + false_negative.sum())
    fpr = false_positive.sum() / (false_positive.sum() + true_negative.sum())

    precision = true_positive.sum() / (true_positive.sum() + false_positive.sum())
    recall = true_positive.sum() / (true_positive.sum() + false_negative.sum())

    return tpr, fpr, precision, recall


def optimal_threshold(tpr, fpr, thresholds):
    optimal_idx = np.argmax(tpr - fpr)
    optimal_threshold = thresholds[optimal_idx]
    
    return optimal_threshold


def get_tpr_fpr_threshold_preds(probabilities, y_test, threshold=THRESHOLD):
    #threshold = 0.9
    threshold_vector = np.greater_equal(probabilities, threshold).astype(int)
    #print("---- threshold {}".format(threshold))
    tpr, fpr, precision, recall = true_false_positive(threshold_vector, y_test)
    #print("tpr {}; fpr {}; precision {}; recall {}".format(tpr, fpr, precision, recall))

    #print("COUNT OCCURRENCES", np.count_nonzero(threshold_vector == 1))
    return threshold_vector


def plot_precision_recall_curve_zoomin(stats_file_test, model_save_file, results_file):
    
    if os.path.isfile(model_save_file):
        print("Gathering trained model ...")
        model = joblib.load(model_save_file)
    else:
        print("You have to train target separation's model first!")
        print("Exiting ...")
        exit()

    print("Gathering testing dataset ...")
    X_test , y_test = gather_dataset(stats_file_test)

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

    plt.plot(np.insert(recall, 0, recall[0]), np.insert(precision, 0, 0), linewidth=4, color="tab:orange", zorder=0)
    plt.ylabel("Precision")
    plt.xlabel("Recall")
    plt.ylim(0.49, 1.01)
    plt.xlim(0, 1.01)
    plt.xticks([0, 0.2, 0.4, 0.6, 0.8, 1], ["0", "0.2", "0.4", "0.6", "0.8", "1"])
    plt.yticks([0.5, 0.6, 0.7, 0.8, 0.9, 1], ["0.5", "0.6", "0.7", "0.8", "0.9", "1"])

    axins = ax1.inset_axes([0.1, 0.1, 0.80, 0.80])
    axins.spines[['right', 'top']].set_visible(True)
    axins.plot(np.insert(recall, 0, recall[0]), np.insert(precision, 0, 0), color="tab:orange", linewidth=2)
    axins.set_xlim(0.9, 1)
    axins.set_ylim(0.6, 1)
    axins.set_xticks([0.9, 0.91, 0.92, 0.93, 0.94, 0.95, 0.96, 0.97, 0.98, 0.99, 1], ["0.90", "", "", "", "", "0.95", "", "", "", "", "1"])
    axins.set_yticks([0.6, 0.7, 0.8, 0.9, 1], ["0.6", "", "0.8", "", "1"])
    axins.tick_params(axis='both', which='major', labelsize=16)
    axins.set_axes_locator(InsetPosition(ax1, [0.3, 0.3, 0.4, 0.4]))  # Move the zoomed-in plot, posx, posy, width, height
    mark_inset(ax1, axins, loc1=2, loc2=4, fc="none", ec="gray", linestyle='--') # loc1 and loc2 change the connecting corners of the zoomin
    axins.text(x=0.935, y=0.80 ,s="AP={}".format(round(average_precision_score(y_test, probas_[:, 1]), 2)), ha="center", va="center", fontsize=16)
    plt.draw()

    plt.tight_layout()

    results_folder = 'results/'
    if not os.path.exists(results_folder):
        os.makedirs(results_folder)
    plt.savefig(results_folder + '{}.pdf'.format(results_file))
    plt.savefig(results_folder + '{}.png'.format(results_file))


def plot_precision_recall_curve(y_test, probas_, is_validation=False, is_full_pipeline=False):
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
    if is_full_pipeline == True:
        plt.savefig(results_folder + 'precision_recall_curve_target_separation_full_pipeline.pdf')
        plt.savefig(results_folder + 'precision_recall_curve_target_separation_full_pipeline.png')
    elif is_validation == True:
        plt.savefig(results_folder + 'precision_recall_curve_target_separation_validation.pdf')
        plt.savefig(results_folder + 'precision_recall_curve_target_separation_validation.png')
    else:
        plt.savefig(results_folder + 'precision_recall_curve_target_separation.pdf')
        plt.savefig(results_folder + 'precision_recall_curve_target_separation.png')


def gather_dataset(statsFile):
    stats = pd.read_csv(statsFile) 

    print("stats before:", stats)

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

    # Fix error ValueError: Invalid classes inferred from unique values of `y`.  Expected: [0], got [1]
    # The class column should start from 0
    le = LabelEncoder()
    y_train = le.fit_transform(y_train)

    x_train = train[train.columns[:LABEL_INDEX]]

    cols = x_train.columns.to_list() + ['Class', 'Capture']
    
    return x_train, y_train


def gather_full_pipeline_dataset():
    clientsFullPipeline = pickle.load(open('../source_separation/full_pipeline_features/client_features_source_separation_thr_0.0010103702079504728_OSTest.pickle', 'rb'))
    captures = list(clientsFullPipeline.keys())

    x_train = pd.DataFrame(clientsFullPipeline.values())
    x_train.reset_index(inplace=True) # add a first column with the line indexes to match non full pipeline features

    y_train = []
    for capture in captures:
        if 'alexa' in capture:
            label = 0
        elif '_hs' in capture:
            label = 0
        else:
            label = 1
        y_train.append(label)

    x_train['Class'] = y_train
    x_train['Capture'] = captures
    print("\n--- x_train full pipeline", x_train)
    
    y_train = x_train['Class']
    x_train = x_train[x_train.columns[:LABEL_INDEX]]
             
    return x_train, y_train, pd.DataFrame(captures, columns =['Capture'])


def train(plFileTrain, statsFileTrain, model_save_file):

    print("\n=== Gathering training dataset ...")
    X_train , y_train = gather_dataset(statsFileTrain)

    print("\n=== Creating model ...")
    model = XGBClassifier()
    print("\n=== Training model ...")
    model.fit(np.asarray(X_train), np.asarray(y_train))

    model.feature_names = list(X_train.columns.values)
    
    if not os.path.exists(models_folder):
        os.makedirs(models_folder)
    joblib.dump(model, models_folder+model_save_file)


def hyperparameter_tuning(plFileTrain, statsFileTrain, plFileValidate, statsFileValidate, plFileTest, stats_file_test, algorithm='GridSearch'):
    if algorithm == 'GridSearch':
        hyperparameter_tuning = GridSearch(plFileValidate, statsFileValidate, plFileTest, stats_file_test)
    elif algorithm == 'BayesianOptimization':
        hyperparameter_tuning = BayesianOptimization(plFileTrain, statsFileTrain, plFileValidate, statsFileValidate, plFileTest, stats_file_test)

    hyperparameter_tuning.search_parameters()


def test(stats_file_test, model_save_file):

    if os.path.isfile(model_save_file):
        print("Gathering trained model ...")
        model = joblib.load(model_save_file)
    else:
        print("You have to train target separation's model first!")
        print("Exiting ...")
        exit()

    print("Gathering testing dataset ...")
    X_test , y_test = gather_dataset(stats_file_test)

    # Predicts the probability of each element to belong to a determined class
    probas_ = model.predict_proba(np.asarray(X_test))
    plot_precision_recall_curve(y_test, probas_)

    return probas_


def test_full_pipeline(dataset_name, model_save_file):

    if os.path.isfile(model_save_file):
        print("Gathering trained model ...")
        model = joblib.load(model_save_file)
    else:
        print("You have to train and test target separation's model and train target separation's model first!")
        print("Exiting ...")
        exit()

    print("Gathering full pipeline testing dataset ...")
    X_test , y_test, test_captures = gather_full_pipeline_dataset()

    # Predicts the probability of each element to belong to a determined class
    probas_ = model.predict_proba(np.asarray(X_test))
    plot_precision_recall_curve(y_test, probas_, is_full_pipeline=True)

    outputClientFeatures = {}
    predictions_final = get_tpr_fpr_threshold_preds(probas_[:, 1], y_test)

    fpr, tpr, thresholds = roc_curve(y_test, probas_[:, 1], pos_label=1)
    decision_threshold = optimal_threshold(tpr, fpr, thresholds)

    predictions_final = get_tpr_fpr_threshold_preds(probas_[:, 1], y_test, threshold=decision_threshold)

    for i in range(len(predictions_final)):
        if predictions_final[i] == 1:     
            outputClientFeatures[test_captures['Capture'].iloc[i]] = X_test.iloc[i]

    features_folder = 'full_pipeline_features/'
    if not os.path.exists(features_folder):
        os.makedirs(features_folder)
    pickle.dump(outputClientFeatures, open(features_folder+'client_features_target_separation_thr_{}_{}.pickle'.format(THRESHOLD, dataset_name), 'wb'))

    return probas_
