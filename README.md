# Sample scripts used in Amazon SageMaker

The examples on this repository are meant to be run using the custom images build with the source in
https://github.com/ferroh-aws/sklearn-sagemaker.

## Data processing

The script ```data_processing.py``` is an example of how we can write scripts for processing and using them in a custom
image.

Example:
```python
from sagemaker import Session, get_execution_role
from sagemaker.processing import Processor, ProcessingInput, ProcessingOutput

processor = Processor(
    image_uri='XXXX/sklearn-registry:latest', # URI to the custom container.
    role=get_execution_role(), # Role used during execution.
    instance_count=1,
    instance_type='ml.m5.4xlarge',
    entrypoint=['python', '/opt/ml/processing/input/code/data_processing.py'], # Override default entry point.
    volume_size_in_gb=5,
    max_runtime_in_seconds=60 * 60
)

train_data_file = 'train_data.csv'
train_target_file = 'train_target.csv'
test_data_file = 'test_data.csv'
test_target_file = 'test_target.csv'
encoder_file = 'encoder.pkl'

session = Session()
sagemaker_role = get_execution_role()

bucket = session.default_bucket()
data_file = 'data/churn/churn.txt'

region = session.boto_region_name
account_id = session.account_id()

prefix = 'churn-clf'
datasets_prefix = f'{prefix}/datasets'
processed_data_prefix = f'{prefix}/processed'
eval_prefix = f'{prefix}/eval'
transformed_data_prefix = f'{prefix}/transformed'
code_prefix = f'{prefix}/code'
model_prefix = f'{prefix}/models'

data_prep_parameters = {
    'inputs':[ProcessingInput(input_name='input',
                    source=f's3://{bucket}/{datasets_prefix}',
                    destination='/opt/ml/processing/input'),
              ProcessingInput(input_name='code',
                    source=f's3://{bucket}/code/data_processing.py',
                    destination='/opt/ml/processing/input/code')],
    'outputs':[ProcessingOutput(output_name='train_data',
                    source=f'/opt/ml/processing/output/train_data',
                    destination=f's3://{bucket}/{processed_data_prefix}/train_data'),
               ProcessingOutput(output_name='train_target',
                    source=f'/opt/ml/processing/output/train_target',
                    destination=f's3://{bucket}/{processed_data_prefix}/train_target'),
               ProcessingOutput(output_name='test_data',
                    source=f'/opt/ml/processing/output/test_data',
                    destination=f's3://{bucket}/{processed_data_prefix}/test_data'),
               ProcessingOutput(output_name='test_target',
                    source=f'/opt/ml/processing/output/test_target',
                    destination=f's3://{bucket}/{processed_data_prefix}/test_target'),
               ProcessingOutput(output_name='encoder',
                    source=f'/opt/ml/processing/output/encoder',
                    destination=f's3://{bucket}/{processed_data_prefix}/encoder')],
    'arguments':['--test-size', '0.1',
                 '--data-file', 'churn.txt',
                 '--train-data-file', train_data_file,
                 '--train-target-file', train_target_file,
                 '--test-data-file', test_data_file,
                 '--test-target-file', test_target_file,
                 '--encoder-file', encoder_file]}

processor.run(**data_prep_parameters)
```
**NOTE: Review ```data_processing.py``` script for parameter details.**

## Training

The script ```train.py``` is a customization for the training job.

Example:
```python
from sagemaker import Session, get_execution_role
from sagemaker.estimator import Estimator

bucket = Session.default_bucket()
model_prefix = 'churn-example/model'

estimators = {'GradientBoosting':{}, 'RandomForest':{}, 'ExtraTrees':{}}
metric_name = 'cross-val:recall'
metric_regex = 'recall = (\d+\.\d{1,2})?'

for algorithm in estimators:   
    estimators[algorithm] = Estimator(
        image_uri = 'XXXX/sklearn-registry:latest', # URI to the custom container.
        entry_point = 'train.py',
        source_dir = f's3://{bucket}/code/train.tar.gz',
        role = get_execution_role(),
        instance_count = 1,
        instance_type = 'ml.m5.xlarge',
        output_path = f's3://{bucket}/{model_prefix}',
        metric_definitions = [{'Name': metric_name, 'Regex': metric_regex}],
        volume_size = 5,
        max_run = 60*60*2,
        hyperparameters={
            'algorithm':algorithm,
            'splits':5,
            'target-metric':'recall',
            'learning-rate': 0.1, 
            'min-samples-split': 3, 
            'n-estimators': 300,
            'max-depth': 25,
            'max-features':20})
    
    estimators[algorithm].fit({
        'train_data': f's3://{bucket}/churn-clf/processed/train_data/train_data.csv',
        'train_target': f's3://{bucket}/churn-clf/processed/train_target/train_target.csv'
    }, wait=False)

```

## Evaluation processing job

To create an evaluation job using the same container.

```python
from sagemaker import Session, get_execution_role
from sagemaker.processing import ProcessingInput, ProcessingOutput, Processor

session = Session()
bucket = session.default_bucket()

evaluation_processor = Processor(
    image_uri='XXXX/sklearn-registry:latest',
    role=get_execution_role(),
    instance_count=1,
    instance_type='ml.m5.large',
    entrypoint=['python3','/opt/ml/processing/input/code/evaluate_models.py'],
    volume_size_in_gb=5,
    max_runtime_in_seconds=60*60*2
)

thresholds_file = 'thresholds.csv'
metrics_report_file = 'metrics_report.json'
eval_prefix = '/evaluation'

eval_parameters = {
    'inputs':[ProcessingInput(
                  input_name='code',
                  source=f's3://{bucket}/code/evaluate_models.py',
                  destination='/opt/ml/processing/input/code'),
              ProcessingInput(
                  source=f's3://sagemaker-us-east-1-253323635394/churn-clf/processed/test_target/', 
                  destination='/opt/ml/processing/input/target'),
              ProcessingInput(
                  source=f's3://sagemaker-us-east-1-253323635394/churn-clf/processed/test_data/', 
                  destination='/opt/ml/processing/input/data'),
              ProcessingInput(
                  source=f's3://sagemaker-us-east-1-253323635394/churn-clf/models/churn-clf-GradientBoosting-32-42-015-4003e9fb/output/model.tar.gz', 
                  destination='/opt/ml/processing/input/GradientBoosting'),
              ProcessingInput(
                  source=f's3://sagemaker-us-east-1-253323635394/churn-clf/models/churn-clf-RandomForest-32-42-013-fd733c1e/output/model.tar.gz',
                  destination='/opt/ml/processing/input/RandomForest'),
              ProcessingInput(
                  source=f's3://sagemaker-us-east-1-253323635394/churn-clf/models/churn-clf-ExtraTrees-32-42-014-f0689f68/output/model.tar.gz', 
                  destination='/opt/ml/processing/input/ExtraTrees')],
    'outputs':[ProcessingOutput(
                   output_name='eval',
                   source='/opt/ml/processing/output',
                   destination=f's3://{bucket}/{eval_prefix}')],
    'arguments':['--algos', ','.join(estimators.keys()),
                 '--min-precision', '0.85',
                 '--test-data-file', test_data_file,
                 '--test-target-file', test_target_file,
                 '--thresholds-file', thresholds_file,
                 '--metrics-report-file', metrics_report_file]}

evaluation_processor.run(**eval_parameters)
```

## Endpoint Hosting

To host an already trained model.

```python
from sagemaker import Session, get_execution_role
from sagemaker.model import Model

bucket = Session.default_bucket()
model_prefix = 'churn-example/model'

model = Model(
    image_uri='XXXX/sklearn-registry:latest', # URI to the custom container.
    model_data=f's3://{bucket}/{model_prefix}/path/model.tar.gz',
    role=get_execution_role(),
    entry_point = 'serve.py',
    source_dir = f's3://{bucket}/code/serve.tar.gz'
)
model.deploy(
    initial_instance_count=1,
    instance_type='ml.m5.large',
    endpoint_name='ChurnExample',
    volume_size=5
)

```