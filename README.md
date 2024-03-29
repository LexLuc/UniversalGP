# UniversalGP

UniversalGP: A basic generic implementation for Gaussian process models

The code was tested on Python 3.6 and Tensorflow 1.10.

## Installation

Dependencies: Tensorflow, SciPy, sklearn, matplotlib

After installing all dependencies clone the repository and do (inside 
the top directory):
```sh
pip install .
```
This will install a copy of the code as a package. If you want to 
install a package that links to the cloned code, run
```sh
pip install --editable .
```
This makes changes to the source files in the cloned directory 
immediately available to everything that imports the package.

Anywhere on the system you can then import the package:
```python
import universalgp as ugp
```

## Usage

### Train with included dataset

To train the GP with one of the included datasets, run for example
```sh
python gaussian_process.py --data=mnist
```
This will train on the MNIST dataset. For other included datasets see 
the files in `universalgp/datasets/`.

By default, the code uses *Variational Inference*. In order to use a 
different method, specify the `--inf` parameter. For example
```sh
python gaussian_process.py --data=mnist --inf=Exact
```
uses *Exact Inference* (which can only be used for regression tasks).  
Other allowed values for `--inf` are `Variational` and `Loo`.

Other useful parameters:

* `--train_steps`: the number of training steps (default: 500)
* `--plot`: if the result is supposed to be plotted after training, 
  specify a plotting function here or leave empty for no plotting
* `--tf_mode`: either "eager" to use Tensorflow in Eager Execution mode 
  or "graph" to use Tensorflow in the normal mode based on computational 
  graphs

To see all available parameters with explanations and default values, 
run
```sh
python gaussian_process.py --helpfull
```

#### Call training function from Python code (not recommended)

In general, it is recommended to run the training in a standalone 
process (i.e. executing `python train_gp.py` from command line). But 
occasionally it can be useful to call the training function from other 
Python code.

In this case, all the parameters for the training function have to be 
given in a dictionary. The code would look something like this:
```python
import universalgp as ugp

data = ...
gp = ugp.train_eager.train_gp(
        data,
        {'inf': 'Variational', 'cov': 'SquaredExponential', 'plot': '', 
        'train_steps': 500, 'lr': 0.005, 'length_scale': 1.0,
        ...  # many more...
        }
    )
gp.predict(test_data)
```

A list of all necessary parameters can be seen by running `python 
train_gp.py --helpfull`.

### Add a new dataset

To add a new dataset, create a new python file in 
`universalgp/datasets/`. In this file, create a function which returns a 
dataset according to the definition in 
`universalgp/datasets/definition.py`.  The resulting file should look 
something like this:
```python
from .definition import Dataset

def my_dataset():

    # ...
    # read data from files
    # ...

    return Dataset(train_fn=lambda: train_set,
                   test_fn=lambda: test_set,
                   num_train=100,
                   input_dim=28,
                   inducing_inputs=inducing_inputs_numpy,
                   output_dim=1,
                   lik="LikelihoodGaussian",
                   metric="RMSE")
```

The last step is to import the new function in 
`universalgp/datasets/__init__.py`.  If you named the file `my_dataset` 
and the function also `my_dataset()` then add the following line to 
`universalgp/datasets/__init__.py`:
```python
from .my_dataset import my_dataset
```

Now you can use this data by using the parameter `--data=my_dataset`.

### Call a trained model from your code

When you are done with training, you might want to used the trained 
model for something. Here is how to do that.

First, we need to make sure the trained model is saved. This is done by 
specifying the `--save_dir` parameter. This parameter expects the path 
to a directory where the model can be saved. UniversalGP will create a 
new directory in this directory with the name given by `--model_name` 
(default name is "local"). If there is already a saved model in there, 
UniversalGP will use this as the initial values and continue training 
from there.

When training with these parameters, UniversalGP creates checkpoints in 
`save_dir/model_name`. The frequency of checkpoint creation is 
determined by `--chkpnt_steps`. The name of the checkpoint file includes 
the training step at which it was created. For example `ckpt-500` for a 
checkpoint created at step 500.

To use this trained model do:
```python
import numpy as np
import tensorflow.contrib.eager as tfe  # eager execution
from universalgp import inf, cov, lik

# restore all variables from checkpoint
with tfe.restore_variables_on_create("save_dir/model_name/chkpt-500"):
    gp = inf.Exact(cov.SquaredExponential(input_dim, {'iso': False}),
                   lik.LikelihoodGaussian(),
                   num_train=3756)

prediction = gp.predict({'input': np.array([[3.2], [4.6]])})
print(prediction)
print(gp.get_all_variables())  # print the values of all train variables
```


## Reference
Kehrenberg, Thomas, et al. "Interpretable Fairness via Target Labels in Gaussian Process Models." arXiv preprint arXiv:1810.05598 (2018).

Krauth, Karl, et al. "AutoGP: Exploring the capabilities and limitations of Gaussian Process models." arXiv preprint arXiv:1610.05392 (2016).

Bonilla, Edwin V., Karl Krauth, and Amir Dezfouli. "Generic Inference in Latent Gaussian Process Models." arXiv preprint arXiv:1609.00577 (2016).
