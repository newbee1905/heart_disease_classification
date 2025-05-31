import os
import pandas as pd

DATA_DIR = 'data'
INPUT_FILES = [
	'processed.cleveland.data',
	'processed.hungarian.data',
	'processed.switzerland.data',
	'processed.va.data',
]
COLUMN_NAMES = [
	'age', 'sex', 'cp', 'trestbps', 'chol', 'fbs', 'restecg',
	'thalach', 'exang', 'oldpeak', 'slope', 'ca', 'thal', 'target',
]
OUTPUT_PATH = os.path.join(DATA_DIR, 'heart_raw.csv')

def load_and_merge():
	frames = []
	for fname in INPUT_FILES:
		path = os.path.join(DATA_DIR, fname)
		df = pd.read_csv(
			path, header=None, names=COLUMN_NAMES, na_values='?'
		)
		frames.append(df)
	return pd.concat(frames, ignore_index=True)

def main():
	os.makedirs(DATA_DIR, exist_ok=True)
	df = load_and_merge()
	df.to_csv(OUTPUT_PATH, index=False)
	print(f'Raw data merged and saved to {OUTPUT_PATH}')

if __name__ == '__main__':
	main()

