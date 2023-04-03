import pickle
import os
import io
import json
import pandas as pd
import numpy as np
from sagemaker_containers.beta.framework import content_types
import logging

logger = logging.getLogger(__name__)


def deserialize(model_path):
    with open(model_path, 'rb') as f:
        return pickle.load(f)


def model_fn(model_dir):
    logger.info('Loading model.')
    estimator = deserialize(os.path.join(model_dir), 'model.pkl')
    return estimator


def input_fn(content, content_type):
    if content_type == content_types.JSON:
        data = pd.DataFrame.from_dict(json.loads(content))
    elif content_type == content_types.CSV:
        data = pd.read_csv(io.StringIO(content), header=None)
    else:
        raise ValueError(f'Unsupported content type {content_type}.')
    return data


def predict_fn(data, model):
    predict = getattr(model, 'predict_proba', None)
    if callable(predict):
        prediction = predict(data)[:, 1]
    else:
        prediction = model.predict(data)
    return prediction


def output_fn(predictions, content_type):
    if content_type == content_types.JSON:
        output = json.dumps(predictions)
    elif content_type == content_types.CSV:
        string_io = io.StringIO()
        np.savetxt(string_io, predictions, delimiter=',')
        output = string_io.getvalue()
    return output
