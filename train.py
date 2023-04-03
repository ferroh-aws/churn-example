import argparse
import pickle
import os
import pandas as pd
from sklearn.model_selection import cross_validate, StratifiedKFold
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import ExtraTreesClassifier
import logging

logging.basicConfig(format='%(asctime)s %(levelname)s - %(name)s - %(message)s', level=logging.INFO)
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


def serialize(data, file):
    with open(file, 'wb') as f:
        pickle.dump(data, f)


def random_forest(**hyperparameters):
    return RandomForestClassifier(
        n_jobs=-1,
        min_samples_split=hyperparameters['min_samples_split'],
        n_estimators=hyperparameters['n_estimators'],
        max_depth=hyperparameters['max_depth'],
        max_features=hyperparameters['max_features']
    )


def gradient_boosting(**hyperparameters):
    return GradientBoostingClassifier(
        learning_rate=hyperparameters['learning_rate'],
        min_samples_split=hyperparameters['min_samples_split'],
        n_estimators=hyperparameters['n_estimators'],
        max_depth=hyperparameters['max_depth'],
        max_features=hyperparameters['max_features']
    )


def extra_trees(**hyperparameters):
    return ExtraTreesClassifier(
        n_jobs=-1,
        min_samples_split=hyperparameters['min_samples_split'],
        n_estimators=hyperparameters['n_estimators'],
        max_depth=hyperparameters['max_depth'],
        max_features=hyperparameters['max_features']
    )


def invalid_algorithm(**hyperparameters):
    raise Exception('Invalid Algorithm')


def algorithm_selector(algorithm, **hyperparameters):
    algorithms = {
        'RandomForest': random_forest,
        'GradientBoosting': gradient_boosting,
        'ExtraTrees': extra_trees
    }
    estimator = algorithms.get(algorithm, invalid_algorithm)
    return estimator(**hyperparameters)


if __name__ == '__main__':
    logger.info('Starting model training.')

    parser = argparse.ArgumentParser()

    parser.add_argument('--output-data-dir', type=str, default=os.environ.get('SM_OUTPUT_DATA_DIR'))
    parser.add_argument('--model-dir', type=str, default=os.environ.get('SM_MODEL_DIR'))
    parser.add_argument('--train-data', type=str, default=os.environ.get('SM_CHANNEL_TRAIN_DATA'))
    parser.add_argument('--train-target', type=str, default=os.environ.get('SM_CHANNEL_TRAIN_TARGET'))

    parser.add_argument('--algorithm', type=str)
    parser.add_argument('--splits', type=int, default=10)
    parser.add_argument('--target-metric', type=str)

    parser.add_argument('--learning-rate', type=float)
    parser.add_argument('--min-samples-split', type=int)
    parser.add_argument('--n-estimators', type=int)
    parser.add_argument('--max-depth', type=int)
    parser.add_argument('--max-features', type=int)

    args, _ = parser.parse_known_args()

    logger.info(f'Received parameters {args}')

    files = os.listdir(args.train_data)
    if len(files) == 1:
        train_data = pd.read_csv(os.path.join(args.train_data, files[0]))
    else:
        raise Exception('This script does not support more than one file for training.')

    files = os.listdir(args.train_target)
    if len(files) == 1:
        train_target = pd.read_csv(os.path.join(args.train_target, files[0]))
        train_target = train_target['Churn'].tolist()
    else:
        raise Exception('This script does not support more than one file for target.')

    estimator = algorithm_selector(
        args.algorithm,
        learning_rate=args.learning_rate,
        min_samples_split=args.min_samples_split,
        n_estimators=args.n_estimators,
        max_depth=args.max_depth,
        max_features=args.max_features
    )

    skf = StratifiedKFold(n_splits=args.splits)
    cv_scores = cross_validate(estimator, train_data, train_target, cv=skf, scoring=args.target_metric, n_jobs=-1)
    score = cv_scores['test_score'].mean().round(4) * 100
    # This output to the log is important, Amazon SageMaker will search for this value.
    logger.info(f'{args.target_metric} = {score}')

    # Model training
    estimator.fit(train_data, train_target)

    # Serialize model
    serialize(estimator, os.path.join(args.model_dir, 'model.pkl'))

    logger.info('Finished training of model')
