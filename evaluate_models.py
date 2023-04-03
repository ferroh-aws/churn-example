import argparse
import pickle
import os
import json
import tarfile
import pandas as pd
import numpy as np
from sklearn.metrics import precision_recall_curve
import logging

logging.basicConfig(format='%(asctime)s %(levelname)s - %(name)s - %(message)s', level=logging.INFO)
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


def load_model(file, model_file='model.pkl'):
    if file.endswith('tar.gz'):
        with tarfile.open(file, 'r:gz') as tar:
            for name in tar.getnames():
                if name == model_file:
                    f = tar.extractfile(name)
                    return pickle.load(f)
            return None
    elif file.endswith('pkl'):
        with open(file, 'rb') as f:
            return pickle.load(f)
    else:
        return None


if __name__ == '__main__':
    script_name = os.path.basename(__file__)
    logger.info('Starting model evaluation.')

    parser = argparse.ArgumentParser()
    parser.add_argument('--algos', type=str, required=True)
    parser.add_argument('--min-precision', type=float, required=True)
    parser.add_argument('--test-data-file', type=str, required=True)
    parser.add_argument('--test-target-file', type=str, required=True)
    parser.add_argument('--thresholds-file', type=str, required=True)
    parser.add_argument('--metrics-report-file', type=str, required=True)

    args, _ = parser.parse_known_args()

    logger.info(f'Received parameters {args}')

    input_path = '/opt/ml/processing/input'
    output_path = '/opt/ml/processing/output'

    # Load test datasets
    test_target_path = os.path.join(input_path, 'target', args.test_target_file)
    test_target = pd.read_csv(test_target_path)

    test_data_path = os.path.join(input_path, 'data', args.test_data_file)
    test_data = pd.read_csv(test_data_path)

    # Thresholds for models evaluation
    algo_metrics = {'Algorithm': [], 'Threshold': [], 'Precision': [], 'Recall': []}

    metrics_report = {}

    algos = args.algos.split(',')
    for algo in algos:
        model_path = os.path.join(input_path, algo, 'model.tar.gz')

        # Deserialize model into memory
        logger.info(f'Loading model: {model_path}')
        clf = load_model(model_path)

        # Generate predictions from test dataset
        predictions = clf.predict_proba(test_data)[:, 1]

        # Retrieves the decision threshold
        precision, recall, thresholds = precision_recall_curve(test_target, predictions)
        operating_point_idx = np.argmax(precision >= args.min_precision)

        algo_metrics['Threshold'].append(thresholds[operating_point_idx])
        algo_metrics['Precision'].append(precision[operating_point_idx])
        algo_metrics['Recall'].append(recall[operating_point_idx])
        algo_metrics['Algorithm'].append(algo)

        metrics_report[algo] = {
            'precision': {'value': precision[operating_point_idx], 'standard_deviation': 'NaN'},
            'recall': {'value': recall[operating_point_idx], 'standard_deviation': 'NaN'}}

    # Store the thresholds
    metrics = pd.DataFrame(algo_metrics)
    logger.info('Found metrics')
    logger.info(metrics)
    metrics.to_csv(os.path.join(output_path, args.thresholds_file), index=False)

    # Stores the metrics for each model
    for algo in metrics_report:
        with open(os.path.join(output_path, f'{algo}_metrics.json'), 'w') as f:
            json.dump({'binary_classification_metrics': metrics_report[algo]}, f)

    logger.info('Finished model evaluation')
