import os
import argparse
import tempfile
import joblib

import pandas as pd
import numpy as np
import cupy as cp

from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer, SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder, OrdinalEncoder, MinMaxScaler
from sklearn.feature_selection import VarianceThreshold
from sklearn.decomposition import PCA
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.linear_model import LinearRegression, LogisticRegression, RidgeClassifier
from sklearn.ensemble	import RandomForestClassifier, StackingClassifier, AdaBoostClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC, LinearSVC
from sklearn.naive_bayes import GaussianNB, ComplementNB
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis, LinearDiscriminantAnalysis
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble	import StackingClassifier
from sklearn.neural_network import MLPClassifier
from tabpfn import TabPFNClassifier

from collections import Counter

from sklearn.metrics import (
	accuracy_score, precision_score, recall_score, f1_score,
	matthews_corrcoef, roc_auc_score, confusion_matrix, fbeta_score
)

import xgboost as xgb
import lightgbm as lgb
import catboost as cat

import optuna
from optuna.samplers import TPESampler
from optuna.storages import RDBStorage
import optuna.visualization as vis
from sklearn.model_selection import cross_val_score
from imblearn.over_sampling import SMOTE
from imblearn.under_sampling import RandomUnderSampler
from imblearn.combine import SMOTEENN
from imblearn import pipeline

RANDOM_STATE=384
DATA_PATH="data/heart_raw.csv"

class ToGpuTransformer(BaseEstimator, TransformerMixin):
	def __init__(self):
		self._y = None

	def fit(self, X, y=None):
		self._y = y
		return self

	def transform(self, X):
		return cp.array(X) 

def optimise_pipeline(
	model_cls,
	model_param_grid,
	X,
	y,
	study_name,
	sqlite_file,
	n_trials,
	random_state=RANDOM_STATE,
	n_jobs=2,
	og=False,
):
	storage_url = f'sqlite:///{sqlite_file}'
	storage = RDBStorage(url=storage_url)
	try:
		optuna.delete_study(study_name=study_name, storage=storage)
	except KeyError:
		pass

	optuna_sampler = TPESampler()
	study = optuna.create_study(
		# directions=['maximize', 'maximize', 'maximize'],
		direction='maximize',
		sampler=optuna_sampler,
		storage=storage_url,
		study_name=study_name,
		load_if_exists=False,
	)
	cache_dir = tempfile.mkdtemp()

	def objective(trial):
		# -- scaler --
		scaler = 'passthrough'
		scaler_name = trial.suggest_categorical('scaler',
			['standard', 'minmax']
		)
		if scaler_name == 'standard':
			scaler = StandardScaler()
		elif scaler_name == 'minmax':
			scaler = MinMaxScaler()

		if not og:
			# -- variance threshold --
			vt = trial.suggest_float('var_thresh', 0.0, 0.1)
			var_sel = VarianceThreshold(threshold=vt)

			# -- PCA --
			pca = 'passthrough'
			if trial.suggest_categorical('use_pca', [False, True]):
				ratio = trial.suggest_float('pca_ratio', 0.5, 1.0)
				pca = PCA(n_components=ratio, random_state=random_state)

			# -- sampler --
			samp = trial.suggest_categorical('sampler',
				['none', 'smote', 'undersample', 'smoteenn']
			)

			if samp == 'smote':
				sampler = SMOTE(
					random_state=random_state,
					k_neighbors=trial.suggest_int('smote_k', 1, 10),
				)
			elif samp == 'undersample':
				sampler = RandomUnderSampler(
					random_state=random_state,
				)
			elif samp == 'smoteenn':
				sm = SMOTE(
					random_state=random_state,
					k_neighbors=trial.suggest_int('smoteenn_k', 1, 10),
				)
				sampler = SMOTEENN(random_state=random_state, smote=sm)
			else:
				sampler = 'passthrough'
		else:
			var_sel = 'passthrough'
			pca = 'passthrough'
			sampler = 'passthrough'

		# -- model params --
		model_params = {}
		for name, spec in model_param_grid.items():
			if isinstance(spec, list):
				model_params[name] = trial.suggest_categorical(name, spec)
			elif isinstance(spec, tuple) and len(spec) == 2:
				low, high = spec
				if isinstance(low, int) and isinstance(high, int):
					model_params[name] = trial.suggest_int(name, low, high)
				else:
					model_params[name] = trial.suggest_float(name, low, high)
			else:
				raise ValueError(f'Bad spec for {name}: {spec}')

		if model_cls is xgb.XGBClassifier:
			model_params.setdefault('tree_method', 'hist')
			model_params.setdefault('device', 'cuda')
			to_gpu = ToGpuTransformer()
			# to_gpu = 'passthrough'
		elif model_cls is lgb.LGBMClassifier:
			model_params.setdefault('device', 'gpu')
			model_params.setdefault('gpu_platform_id', 0)
			model_params.setdefault('gpu_device_id', 0)
			to_gpu = 'passthrough'
		else:
			to_gpu = 'passthrough'

		try:
			default_params = model_cls().get_params()
			if 'n_jobs' in default_params:
				model.set_params(n_jobs=n_jobs)
		except Exception:
			pass
		model = model_cls(**model_params)

		pipe = pipeline.Pipeline([
			('scaler', scaler),
			('variance', var_sel),
			('pca', pca),
			('sampler', sampler),
			('to_gpu', to_gpu),
			('model', model),
		], memory=cache_dir)

		cv = StratifiedKFold(
			n_splits=5, shuffle=True,
			random_state=random_state
		)

		# AuC to measure model's ability rank true cases above non-cases across all thresholds
		# Similar to F1 but emphasises on Recall, since for heart disease detection, false negatives are costlier than false positives
		# MCC accounts for all four confusion-matrix cells and remains robust on imbalanced datasets
		roc_auc_scores = []
		fbeta_scores = []
		mcc_scores = []
		for tr_idx, va_idx in cv.split(X, y):
			pipe.fit(X.iloc[tr_idx], y.iloc[tr_idx])
			pred = pipe.predict(X.iloc[va_idx])

			fbeta_scores.append(fbeta_score(y.iloc[va_idx], pred, beta=2))

		# return float(np.mean(roc_auc_scores)), float(np.mean(fbeta_scores)), float(np.mean(mcc_scores))
		return float(np.mean(fbeta_scores))

	study.optimize(objective, n_trials=n_trials, show_progress_bar=True, n_jobs=n_jobs)
	return study


def build_pipeline_from_params(model_cls, params, random_state=RANDOM_STATE, n_jobs=2):
	# -- scaler --
	sc = params.get('scaler', 'none')
	if sc == 'standard':
		scaler = StandardScaler()
	elif sc == 'minmax':
		scaler = MinMaxScaler()
	else:
		scaler = 'passthrough'

	# -- variance --
	if 'var_thresh' in params:
		var_sel = VarianceThreshold(threshold=params.get('var_thresh', 0.0))
	else:
		var_sel = 'passthrough'

	# -- PCA --
	if params.get('use_pca', False):
		pca = PCA(n_components=params.get('pca_ratio', 1.0),
							random_state=random_state)
	else:
		pca = 'passthrough'

	# -- sampler --
	samp = params.get('sampler', 'none')
	if samp == 'smote':
		sampler = SMOTE(
			random_state=random_state,
			k_neighbors=params.get('smote_k', 5),
		)
	elif samp == 'undersample':
		sampler = RandomUnderSampler(
			random_state=random_state,
		)
	elif samp == 'smoteenn':
		sm = SMOTE(
			random_state=random_state,
			k_neighbors=params.get('smoteenn_k', 5),
		)
		sampler = SMOTEENN(random_state=random_state, smote=sm)
	else:
		sampler = 'passthrough'

	# model params
	pre_keys = {
		'scaler','var_thresh','use_pca','pca_ratio',
		'sampler','smote_k', 'smoteenn_k',
	}
	model_params = {
		k: v for k, v in params.items() if k not in pre_keys
	}

	if model_cls is xgb.XGBClassifier:
		model_params.setdefault('tree_method', 'hist')
		model_params.setdefault('device', 'cuda')
		to_gpu = ToGpuTransformer()
	else:
		to_gpu = 'passthrough'

	model = model_cls(**model_params)
	try:
		default_params = model_cls().get_params()
		if 'n_jobs' in default_params:
			model.set_params(n_jobs=n_jobs)
	except Exception:
		pass

	return pipeline.Pipeline([
		('scaler', scaler),
		('variance', var_sel),
		('pca', pca),
		('sampler', sampler),
		# ('to_gpu', to_gpu),
		('model', model),
	])


def update_study_map(
	classifier_name, study_name, sqlite_file,
	csv_path='classifier_study_map.csv',
):
	if os.path.exists(csv_path):
		df = pd.read_csv(csv_path)
	else:
		df = pd.DataFrame(columns=['classifier','study','sqlite'])

	entry = {
		'classifier': classifier_name,
		'study': study_name,
		'sqlite': sqlite_file
	}
	df = pd.concat([df, pd.DataFrame([entry])], ignore_index=True)
	df.to_csv(csv_path, index=False)

def evaluate_classification(
	y_true,
	y_pred,
	y_score=None,
	output_csv="metrics.csv"
):
	labels = np.unique(y_true)
	multiclass = labels.size > 2

	# Basic Metrics
	acc = accuracy_score(y_true, y_pred)
	if multiclass:
		f1 = f1_score(y_true, y_pred, average="weighted")
		rec = recall_score(y_true, y_pred, average="weighted")
		prec = precision_score(y_true, y_pred, average="weighted")
	else:
		f1 = f1_score(y_true, y_pred)
		rec = recall_score(y_true, y_pred)
		prec = precision_score(y_true, y_pred)

	# specificity and sensitivity
	if not multiclass:
		tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
		spec = tn / (tn + fp)
		sens = tp / (tp + fn) 
	else:
		cm = confusion_matrix(y_true, y_pred)
		spec_list = []
		for i in range(labels.size):
			tn = cm.sum() - (cm[i, :].sum() + cm[:, i].sum() - cm[i, i])
			fp = cm[:, i].sum() - cm[i, i]
			spec_list.append(tn / (tn + fp))
			sens_list.append(tp / (tp + fn))
		spec = float(np.mean(spec_list))
		sens = float(np.mean(sens_list))

	mcc = matthews_corrcoef(y_true, y_pred)

	# Auc
	if y_score is None:
		auc = np.nan
	else:
		if multiclass:
			auc = roc_auc_score(
				y_true,
				y_score,
				multi_class="ovr",
				average="weighted",
			)
		else:
			# y_score[:, 1] for the positive class
			auc = roc_auc_score(y_true, y_score[:, 1])

	df = pd.DataFrame({
		"Metrics": [
			"Accuracy", "F1-Score", "Recall",
			"Precision", "Sensitivity", "Specificity",
			"MCC", "AUC"
		],
		"Results": [
			acc, f1, rec,
			prec, sens, spec,
			mcc, auc
		]
	})

	df.to_csv(output_csv, index=False)
	return df

if __name__ == '__main__':
	mapping = {
		# -- Transformer --
		'tabfn': (
			TabPFNClassifier,
			{
				'n_estimators': (1, 20),
				'softmax_temperature': (0.1, 2.0),
				'balance_probabilities': [False, True],
				'average_before_softmax': [False, True],
			}
		),
		# -- Linear --
		'logistic_regression': (
			LogisticRegression,
			{
				'C': (0.01, 1000.0),
				'penalty': ['l1', 'l2'],
				'solver': ['liblinear', 'saga'],
				'class_weight': [None, 'balanced'],
				'max_iter': [10000],
			}
		),
		'ridge': (
			RidgeClassifier,
			{
				'alpha': (0.001, 100.0),
				'solver': ['auto', 'svd', 'cholesky'],
				'tol': (1e-4, 1e-1),
				'max_iter': [10000],
			}
		),
		'linear_svm': (
			LinearSVC,
			{
				'C': (0.01, 1000.0),
				'loss': ['hinge', 'squared_hinge'],
				'tol': (1e-4, 1e-1),
				'max_iter': [10000],
			}
		),
		# -- Tree --
		'decision_tree': (
			DecisionTreeClassifier,
			{
				'max_depth': (2, 50),
				'min_samples_split': (2, 32),
				'min_samples_leaf': (1, 16),
				'criterion': ['gini', 'entropy'],
				'class_weight': [None, 'balanced']
			}
		),
		'random_forest': (
			RandomForestClassifier,
			{
				'n_estimators': (50, 2500),
				'max_depth': (2, 50),
				'min_samples_split': (2, 32),
				'min_samples_leaf': (1, 16),
				'max_features': ['sqrt', 'log2', 0.5, 0.25, 0.75],
				'criterion': ['gini', 'entropy'],
				'class_weight': [None, 'balanced']
			}
		),
		'xgboost': (
			xgb.XGBClassifier,
			{
				'n_estimators': (50, 2500),
				'max_depth': (2, 50),
				'learning_rate': (0.001, 0.3),
				'subsample': (0.5, 1.0),
				'colsample_bytree': (0.5, 1.0),
				'gamma': (0.0, 5.0),
				'reg_alpha': (0.0, 1.0),
				'reg_lambda': (0.0, 1.0),
			}
		),
		'lightgbm': (
			lgb.LGBMClassifier,
			{
				'num_leaves': (31, 512),
				'learning_rate': (0.01, 0.3),
				'n_estimators': (50, 2000),
				'subsample': (0.5, 1.0),
				'colsample_bytree': (0.5, 1.0),
				'reg_alpha': (0.0, 1.0),
				'reg_lambda': (0.0, 1.0),
				'verbose': [-2],
			}
		),
		'catboost': (
			cat.CatBoostClassifier,
			{
				'iterations': (100, 2000),
				'learning_rate': (0.01, 0.3),
				'depth': (2, 16),
				'l2_leaf_reg': (1, 30),
				'bagging_temperature': (0.0, 1.0),
				'border_count':  (32, 256),
				'verbose': [0],
			}
		),
		'adaboost': (
			AdaBoostClassifier,
			{
				'n_estimators':   (50, 1000),
				'learning_rate':  (0.01, 1.5),
			}
		),
		# -- Generative --
		'gaussian_nb': (
			GaussianNB,
			{
				'var_smoothing': (1e-12, 5e-8)
			}
		),
		'qda': (
			QuadraticDiscriminantAnalysis,
			{
				'reg_param': (0.0, 1.0),
				'tol': (1e-6, 1e-1),
			}
		),
		'lda': (
			LinearDiscriminantAnalysis,
			{
				'solver': ['svd', 'lsqr'],
				'tol': (1e-6, 1e-1),
			}
		),
		# -- Kernel/Neural: non-linear decision boundaries  --
		'knn': (
			KNeighborsClassifier,
			{
				'n_neighbors': (2, 32),
				'weights': ['uniform', 'distance'],
				'metric': ['euclidean', 'manhattan']
			}
		),
		'mlp': (
			MLPClassifier,
			{
				'hidden_layer_sizes':   [(64,), (128,), (64, 64), (128, 64), (64, 128)],
				'activation':           ['relu', 'tanh'],
				'solver':               ['adam', 'sgd'],
				'alpha':                (1e-5, 1e-2),
				'learning_rate_init':   (1e-4, 1e-1),
				'max_iter':             [1000],
			}
		),
		'svm': (
			SVC,
			{
				'C':            (0.01, 1000.0),
				'kernel':       ['linear', 'rbf', 'poly'],
				'gamma':        ['scale', 'auto'],
				'degree':       (2, 5),
				'probability':  [True],
			}
		),
		# ------------------------------------------------------------------------
		'stacking': (
			StackingClassifier,
			{
				'estimators': [
					[
						('lr', LogisticRegression()), 
						('dt', DecisionTreeClassifier()), 
						('rf', RandomForestClassifier()), 
						('xgb', xgb.XGBClassifier()), 
						('nb', GaussianNB()), 
						('knn', KNeighborsClassifier())
					]
				],
				'final_estimator': [LogisticRegression(), DecisionTreeClassifier()],
				'cv': [5],
				'passthrough': [False, True]
			}
		)
	}

	parser = argparse.ArgumentParser(
		description='Train model with Optuna hyperparameter search'
	)
	parser.add_argument(
		'--model', required=True,
		choices=mapping.keys(),
	)
	parser.add_argument('--trials', type=int, default=50)
	parser.add_argument('--n-jobs', type=int, default=2)
	parser.add_argument('--data-leak', type=bool, default=True)
	args = parser.parse_args()

	df = pd.read_csv(DATA_PATH)
	X = df.drop(columns=["target"])
	y = df["target"]
	y = (y > 0).astype(int)

	num_imputer = IterativeImputer(max_iter=100, random_state=RANDOM_STATE)
	cat_imputer = SimpleImputer(strategy="most_frequent")

	scaler = StandardScaler()
	encoder = OneHotEncoder(sparse_output=False)

	num_cols = [
		'age', 'trestbps', 'chol', 'thalach', 'oldpeak'
	]
	cat_cols = [
		'sex', 'cp', 'fbs', 'restecg', 'exang', 'slope', 'ca', 'thal'
	]

	if not args.data_leak:
		X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=RANDOM_STATE, stratify=y)

		X_train_num_imp = pd.DataFrame(
			num_imputer.fit_transform(X_train[num_cols]), columns=num_cols
		)
		X_test_num_imp = pd.DataFrame(
			num_imputer.transform(X_test[num_cols]), columns=num_cols
		)

		X_train_cat_imp = pd.DataFrame(
			cat_imputer.fit_transform(X_train[cat_cols]), columns=cat_cols
		)
		X_test_cat_imp = pd.DataFrame(
			cat_imputer.transform(X_test[cat_cols]), columns=cat_cols
		)

		X_train_encoded = pd.DataFrame(
			encoder.fit_transform(X_train_cat_imp),
			columns=encoder.get_feature_names_out(cat_cols),
		)
		X_test_encoded = pd.DataFrame(
			encoder.transform(X_test_cat_imp),
			columns=encoder.get_feature_names_out(cat_cols),
		)

		X_train_proc = pd.concat([X_train_num_imp, X_train_encoded], axis=1)
		X_test_proc = pd.concat([X_test_num_imp, X_test_encoded], axis=1)
	else:
		X_num_imp = pd.DataFrame(
			num_imputer.fit_transform(X[num_cols]), columns=num_cols
		)

		X_cat_imp = pd.DataFrame(
			cat_imputer.fit_transform(X[cat_cols]), columns=cat_cols
		)

		X_encoded = pd.DataFrame(
			encoder.fit_transform(X_cat_imp),
			columns=encoder.get_feature_names_out(cat_cols),
		)

		X_proc = pd.concat([X_num_imp, X_encoded], axis=1)
		X_train_proc, X_test_proc, y_train, y_test = train_test_split(X_proc, y, test_size=0.2, random_state=RANDOM_STATE, stratify=y)

	model_cls, grid = mapping[args.model]
	study_name = f'{args.model}_study'
	sqlite_file = f'studies.sqlite'

	model_name = args.model + '_leak' if args.data_leak else ''

	study = optimise_pipeline(
		model_cls,
		grid,
		X_train_proc, y_train,
		study_name,
		sqlite_file,
		n_trials=args.trials,
		n_jobs=args.n_jobs,
	)
	print(f"\nBest Params: {study.best_params}")

	joblib.dump(study.best_params, f'models/best_{model_name}_params.joblib')
	best_pipe = build_pipeline_from_params(
		model_cls,
		study.best_params,
		n_jobs=args.n_jobs,
	)
	best_pipe.fit(X_train_proc, y_train)
	joblib.dump(best_pipe, f'models/best_{model_name}_model.joblib')

	update_study_map(args.model, study_name, sqlite_file)

	preds = best_pipe.predict(X_test_proc)
	try:
		scores = best_pipe.predict_proba(X_test_proc)
	except (AttributeError, ValueError):
		scores = None

	results = evaluate_classification(
		y_test,
		preds,
		scores,
		output_csv=f"metrics/{model_name}_metrics.csv",
	)
	print(results)

	print(f'Done. Best params in models/best_{model_name}_params.joblib, model in models/best_{model_name}_model.joblib, metrics in metrics/{model_name}_metrics.csv')
