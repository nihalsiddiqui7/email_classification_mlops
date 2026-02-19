import argparse
import json
import logging
import os
import pickle

import pandas as pd
from sklearn import metrics


# logs
log_dir = "logs"
os.makedirs(log_dir, exist_ok=True)

logger = logging.getLogger("model_eval")
logger.setLevel(logging.DEBUG)
if not logger.handlers:
	console_handler = logging.StreamHandler()
	console_handler.setLevel(logging.DEBUG)

	log_file_path = os.path.join(log_dir, "model_eval.log")
	file_handler = logging.FileHandler(log_file_path)
	file_handler.setLevel(logging.DEBUG)

	formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
	console_handler.setFormatter(formatter)
	file_handler.setFormatter(formatter)

	logger.addHandler(console_handler)
	logger.addHandler(file_handler)


def load_data(path: str) -> pd.DataFrame:
	df = pd.read_csv(path, encoding="utf-8")
	logger.info(f"Loaded test data from {path} with shape {df.shape}")
	return df


def load_model(path: str):
	with open(path, "rb") as f:
		model = pickle.load(f)
	logger.info(f"Loaded model from {path}")
	return model


def evaluate(model, df: pd.DataFrame) -> dict:
	X_test = df.drop("label", axis=1)
	y_true = df["label"]
	y_pred = model.predict(X_test)

	# Overall scores
	scores = {
		"accuracy": float(metrics.accuracy_score(y_true, y_pred)),
		"precision_weighted": float(metrics.precision_score(y_true, y_pred, average="weighted", zero_division=0)),
		"recall_weighted": float(metrics.recall_score(y_true, y_pred, average="weighted", zero_division=0)),
		"f1_weighted": float(metrics.f1_score(y_true, y_pred, average="weighted", zero_division=0)),
	}

	# Per-class details
	labels = sorted(pd.unique(y_true))
	scores["confusion_matrix"] = metrics.confusion_matrix(y_true, y_pred, labels=labels).tolist()
	scores["labels"] = [int(l) for l in labels]
	class_report = metrics.precision_recall_fscore_support(y_true, y_pred, labels=labels, zero_division=0)
	scores["per_class"] = [
		{
			"label": int(label),
			"precision": float(prec),
			"recall": float(rec),
			"f1": float(f1),
			"support": int(supp),
		}
		for label, prec, rec, f1, supp in zip(labels, *class_report)
	]

	logger.info(
		"Evaluation completed | accuracy=%.4f precision=%.4f recall=%.4f f1=%.4f",
		scores["accuracy"],
		scores["precision_weighted"],
		scores["recall_weighted"],
		scores["f1_weighted"],
	)
	return scores


def save_metrics(metrics_dict: dict, path: str) -> None:
	os.makedirs(os.path.dirname(path), exist_ok=True)
	with open(path, "w", encoding="utf-8") as f:
		json.dump(metrics_dict, f, indent=2)
	logger.info(f"Saved metrics to {path}")


def main(model_path: str, test_data_path: str, metrics_path: str) -> None:
	test_df = load_data(test_data_path)
	model = load_model(model_path)
	scores = evaluate(model, test_df)
	save_metrics(scores, metrics_path)


if __name__ == "__main__":
	parser = argparse.ArgumentParser(description="Evaluate a trained spam classifier and save metrics")
	parser.add_argument("--model_path", default="./models/random_forest.pkl", help="Path to the trained model file")
	parser.add_argument("--test_data_path", default="./data/processed/test_tfidf.csv", help="Path to the TF-IDF test data")
	parser.add_argument("--metrics_path", default="./logs/metrics.json", help="Where to write the metrics JSON")
	args = parser.parse_args()

	main(args.model_path, args.test_data_path, args.metrics_path)
