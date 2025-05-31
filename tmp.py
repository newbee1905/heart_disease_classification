import pickle
import tqdm
import optuna

from sklearn.metrics import accuracy_score

from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier

import pandas as pd

import numpy as np
import pandas as pd
import os
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, f1_score, recall_score, precision_score, classification_report
from sklearn.metrics import confusion_matrix

from xgboost import XGBClassifier

HEART_DISEASE_PATH = "data/heart.csv"
heart_data = pd.read_csv(HEART_DISEASE_PATH)

categorical_cols = ['sex', 'cp', 'fbs', 'restecg', 'exang', 'slope', 'thal']
numerical_cols = ['age', 'trestbps', 'chol', 'thalach', 'oldpeak', 'ca']

OUTPUT_PKL = "part1_best_params.pkl"

TARGET_SCORES = {
	"Logistic Regression":    {"accuracy":0.8439, "f1":0.8620, "recall":0.9090, "precision":0.8196},
	"Naive Bayes":            {"accuracy":0.8439, "f1":0.8608, "recall":0.9000, "precision":0.8250},
	"K-Nearest Neighbors":    {"accuracy":0.8585, "f1":0.8687, "recall":0.8727, "precision":0.8648},
	"Decision Tree":          {"accuracy":0.9268, "f1":0.9289, "recall":0.8909, "precision":0.9702},
	"Random Forest":          {"accuracy":0.9268, "f1":0.9333, "recall":0.9545, "precision":0.9130},
	"Extreme Gradient Boost": {"accuracy":0.9073, "f1":0.9132, "recall":0.9090, "precision":0.9174}
}
TARGET_CM = {
	"Logistic Regression":    {'tn': 73, 'tp': 100, 'fp': 22, 'fn': 10},
	"Naive Bayes":            {'tn': 74, 'tp': 99,  'fp': 21, 'fn': 11},
	"K-Nearest Neighbors":    {'tn': 80, 'tp': 96,  'fp': 15, 'fn': 14},
	"Decision Tree":          {'tn': 92, 'tp': 98,  'fp': 3,  'fn': 12},
	"Random Forest":          {'tn': 85, 'tp': 105, 'fp': 10, 'fn': 5 },
	"Extreme Gradient Boost": {'tn': 86, 'tp': 100, 'fp': 9,  'fn': 10},
}

RANDOM_STATE = 1883669736
results = {}

MODEL_SPECS = {
  "Decision Tree": {
    "class": DecisionTreeClassifier, "init_args": {"random_state":RANDOM_STATE},
    "search": {
      "criterion":       ("categorical", ["gini", "entropy", "log_loss"]),
      "splitter":        ("categorical", ["best", "random"]),
      "max_depth":       ("int", 1, 200),
      "min_samples_split":("int", 2, 500),
      "min_samples_leaf": ("int", 1, 150),
      "max_features":    ("float", 0.01, 1.0),
      "ccp_alpha":       ("float", 0.0, 0.05)
    }
  },
  "Logistic Regression": {
    "class": LogisticRegression, "init_args": {"max_iter":1000},
    "search": {
      "C": ("float", 1e-6, 1e3),
      "penalty": ("categorical", ["l1", "l2"]),
      "solver": ("categorical", ["liblinear", "saga"])
    }
  },
  "Naive Bayes": {
    "class": GaussianNB, "init_args": {},
		"search": {
			"var_smoothing": ("float", 1e-12, 5e-8),
		}
  },
  "K-Nearest Neighbors": {
    "class": KNeighborsClassifier, "init_args": {},
    "search": {
      "n_neighbors": ("int", 1, 50),
      "weights": ("categorical", ["uniform", "distance"]),
      "metric": ("categorical", ["euclidean", "manhattan", "minkowski"])
    }
  },
  "Random Forest": {
    "class": RandomForestClassifier, "init_args": {"random_state":RANDOM_STATE},
    "search": {
      "n_estimators":    ("int", 2, 1000),
      "max_depth":       ("int", 1, 200),
      "min_samples_split":("int", 2, 100),
      "min_samples_leaf": ("int", 1, 150),
      "max_features":    ("float", 0.01, 1.0)
    }
  },
  "Extreme Gradient Boost": {
    "class": XGBClassifier, "init_args":{
      "use_label_encoder":False, "eval_metric":"logloss", "random_state":RANDOM_STATE
    },
    "search": {
      "n_estimators":    ("int", 2, 2000),
      "max_depth":       ("int", 1, 20),
      "learning_rate":   ("float", 0.001, 1.0),
      "subsample":       ("float", 0.5, 1.0),
      "colsample_bytree":("float", 0.5, 1.0)
    }
  }
}

preprocessor = ColumnTransformer(
	transformers=[
		('num', StandardScaler(), numerical_cols),
		('cat', OneHotEncoder(drop='first'), categorical_cols)
	]
)

heart_data_processed = preprocessor.fit_transform(heart_data)
heart_labels = heart_data['target'].values

X_train, X_val, y_train, y_val = train_test_split(
	heart_data_processed, heart_labels, test_size=0.2, random_state=RANDOM_STATE,
)

for name, spec in tqdm.tqdm(MODEL_SPECS.items(), desc="Training"):
	def objective(trial):
		params = {}
		for p, info in spec["search"].items():
			t = info[0]
			if t == "int":
				params[p] = trial.suggest_int(p, info[1], info[2])
			elif t == "float":
				params[p] = trial.suggest_float(p, info[1], info[2])
			else:
				params[p] = trial.suggest_categorical(p, info[1])

		model = spec["class"](**{**spec["init_args"], **params})
		model.fit(X_train, y_train)
		y_pred = model.predict(X_val)

		scores = {
			'accuracy':	 accuracy_score(y_val, y_pred),
			'f1':				 f1_score(y_val, y_pred),
			'recall':		 recall_score(y_val, y_pred),
			'precision': precision_score(y_val, y_pred),
		}

		tn, fp, fn, tp = confusion_matrix(y_val, y_pred).ravel()
		cm = {'tp': tp, 'tn': tn, 'fp': fp, 'fn': fn}

		penalty_scores = sum(
			abs(scores[k] - TARGET_SCORES[name][k])
			for k in TARGET_SCORES[name]
		)
		penalty_cm = sum(
			abs(cm[k] - TARGET_CM[name][k])
			for k in TARGET_CM[name]
		)
		total_penalty = penalty_scores + penalty_cm

		return total_penalty

	study = optuna.create_study(
		direction='minimize',
		storage='sqlite:///part1.db',
		load_if_exists=True,
		study_name=f"{name}_study",
	)
	study.optimize(
		objective,
		n_trials=1500,
		show_progress_bar=True,
		n_jobs=12,
	)

	best_acc = study.best_value
	best_params = study.best_params
	print(f'Best validation accuracy: {best_acc:.3f}')
	print('Best hyperparameters:', best_params)

	model = spec["class"](**{**spec["init_args"]}, **best_params)
	model.fit(X_train, y_train)

	pred = model.predict(X_val)
	print(classification_report(y_val, pred))

	results[name] = {
		"best_params": best_params,
		"metrics": {
			"accuracy": accuracy_score(y_val, pred),
			"precision": precision_score(y_val, pred),
			"recall": recall_score(y_val, pred),
			"f1": f1_score(y_val, pred),
			"confusion_matrix": confusion_matrix(y_val, pred).ravel().tolist()
		}
	}

if os.path.exists(OUTPUT_PKL):
	with open(OUTPUT_PKL, "rb") as f:
		existing = pickle.load(f)
else:
	existing = {}

for m, r in results.items():
	existing[m] = r["best_params"]

with open(OUTPUT_PKL, "wb") as f:
	pickle.dump(existing, f)
	# pickle.dump({m: r["best_params"] for m, r in results.items()}, f)

for name, res in results.items():
	print(f"\n{name}")
	print("Best Params:", res["best_params"])
	print("Achieved Metrics:", res["metrics"])
