"""
Chunking Dataset
"""

import numpy as np
import sklearn
from .definition import Dataset, select_training_and_test, to_tf_dataset_fn

def _init_z(train_inputs, num_inducing):
    # Initialize inducing points using clustering.
    print('start clustering')
    mini_batch = sklearn.cluster.MiniBatchKMeans(num_inducing)
    mini_batch.fit_predict(train_inputs)
    inducing_locations = mini_batch.cluster_centers_
    print('done clustering')
    return inducing_locations

def chunking(_):
    """
    accept processed CHUNKING dataset
    """
    num_inducing = 100
    print('loading data...')
    X = np.load('./datasets/data/chunking/x_train.npz')
    Y = np.load('./datasets/data/chunking/y_train.npz')
    print(f'data loaded: shape of x = {X.shape}, shape of y = {Y.shape}')

#    X = np.linspace(0, 5, num=300, endpoint=False, dtype=np.float32)[:, np.newaxis]    # toy dataset
#    Y = np.full((300,3), 0, dtype=np.float32)  # toy dataset
#    Y[:100][:, 0] = 1      # toy dataset
#    Y[100:200][:, 1] = 1   # toy dataset
#    Y[200:][:, 2] = 1      # toy dataset
    
    num_train = round(.8*X.shape[0])
    print(f'training data splited into: train = {num_train}, validation = {X.shape[0]-num_train}')
    (x_train, y_train), (x_test, y_test) = select_training_and_test(num_train, X, Y)
    
    return Dataset(
            num_train=num_train,
            input_dim=X.shape[1],
            output_dim = Y.shape[1],
#            input_dim=1,   # toy dataset
#            output_dim=3,  # toy dataset
            train_fn=to_tf_dataset_fn(x_train, y_train),
            test_fn=to_tf_dataset_fn(x_test, y_test),
            xtest=x_test, # must be set if prediction is enabled
            ytest=y_test, # must be set if prediction is enabled
            inducing_inputs=_init_z(x_train, num_inducing),
#            inducing_inputs=x_train[::num_train//num_inducing], # toy dataset
            lik="LikelihoodSoftmax",
            metric='soft_accuracy'
    )
