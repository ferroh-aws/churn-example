import argparse
import pickle
import os
import pandas as pd
from sklearn.preprocessing import OneHotEncoder
from sklearn.model_selection import train_test_split
import logging

logging.basicConfig(format='%(asctime)s %(levelname)s - %(name)s - %(message)s', level=logging.INFO)
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


def serialize(data, file):
    with open(file, 'wb') as f:
        pickle.dump(data, f)


if __name__=='__main__':
    script_name = os.path.basename(__file__)

    logger.info('Starting data preparation.')

    parser = argparse.ArgumentParser()
    parser.add_argument('--test-size', type=float, default=0.1)
    parser.add_argument('--data-file', type=str, default='train.csv')
    parser.add_argument('--train-data-file', type=str)
    parser.add_argument('--train-target-file', type=str)
    parser.add_argument('--test-data-file', type=str)
    parser.add_argument('--test-target-file', type=str)
    parser.add_argument('--encoder-file', type=str)

    args, _ = parser.parse_known_args()

    logger.info(f'Received arguments: {args}')

    input_path = '/opt/ml/processing/input'
    output_path = '/opt/ml/processing/output'

    data_path = os.path.join(input_path, args.data_file)
    data = pd.read_csv(data_path)

    data.columns = [''.join(c if c.isalnum() else '_' for c in str(column)) for column in data.columns]

    columns = ['State', 'Account_Length', 'Area_Code', 'Int_l_Plan', 'VMail_Plan', 'VMail_Message',
               'Day_Mins', 'Day_Calls', 'Eve_Mins', 'Eve_Calls', 'Night_Mins', 'Night_Calls',
               'Intl_Mins', 'Intl_Calls', 'CustServ_Calls', 'Churn_']
    data = data[columns]

    data['Churn_'] = data['Churn_'].str.replace('.', '')
    data.rename(columns={'Churn_': 'Churn'}, inplace=True)

    # One hot encoding for categorical values
    columns = ['State', 'Area_Code']
    encoder = OneHotEncoder().fit(data[columns])

    transformed = encoder.transform(data[columns]).toarray()

    data.drop(columns, axis=1, inplace=True)
    data = pd.concat([data, pd.DataFrame(transformed, columns=encoder.get_feature_names_out())], axis=1)

    # Substitute 1/0 for yes/no values
    data['Int_l_Plan'] = data['Int_l_Plan'].map(dict(yes=1, no=0))
    data['VMail_Plan'] = data['VMail_Plan'].map(dict(yes=1, no=0))

    # Substitute 1/0 for boolean values
    data['Churn'] = data['Churn'].map({'True': 1, 'False': 0})

    # Separate target values from dataset
    target = data[['Churn']]
    data.drop(['Churn'], axis=1, inplace=True)

    # Divide the dataset into train and test groups
    train_data, test_data, train_target, test_target = train_test_split(data, target, stratify=target,
                                                                        test_size=args.test_size)

    # Save datasets and encoder
    train_data.to_csv(os.path.join(output_path, 'train_data', args.train_data_file), index=False)
    train_target.to_csv(os.path.join(output_path, 'train_target', args.train_target_file), index=False)
    test_data.to_csv(os.path.join(output_path, 'test_data', args.test_data_file), index=False)
    test_target.to_csv(os.path.join(output_path, 'test_target', args.test_target_file), index=False)
    serialize(encoder, os.path.join(output_path, 'encoder', args.encoder_file))

    logger.info('Finished data preparation')
