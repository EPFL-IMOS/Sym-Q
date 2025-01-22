# Sym-Q V2

In this version, the pipeline has been enhanced to support inputs with varying dimensions. Additionally, the model has been modularized, allowing for easy customization of the encoder and loss functions.

## Dataset Creation

Firstly, define the dataset configuration in `NeuralSymbolicRegressionThatScales/dataset_configuration.json`, for this project we use following settings:

- 3 Var

```json
{
    "max_len": 20,
    "operators": "add:10,mul:10,sub:5,div:5,sqrt:4,pow2:4,pow3:2,pow4:1,pow5:1,ln:4,exp:4,sin:4,cos:4,tan:4",
    "max_ops": 5,
    "rewrite_functions": "",
    "variables": ["x_1","x_2","x_3"], 
    "eos_index": 1,
    "pad_index": 0
}
```

- 2 Var

```json
{
    "max_len": 20,
    "operators": "add:10,mul:10,sub:5,div:5,sqrt:4,pow2:4,pow3:2,pow4:1,pow5:1,ln:4,exp:4,sin:4,cos:4,tan:4",
    "max_ops": 5,
    "rewrite_functions": "",
    "variables": ["x_1","x_2"], 
    "eos_index": 1,
    "pad_index": 0
}
```

We use 10 million skeletons as training dataset and 1000 skeletons as validation dataset. Generate the corresponding datasets with following commands:

- 3 Var

```bash
cd NeuralSymbolicRegressionThatScales
python scripts/data_creation/dataset_creation.py --number_of_equations 10000000 --no-debug
python scripts/data_creation/dataset_creation.py --number_of_equations 1000 --no-debug
```

- 2 Var

```bash
cd NeuralSymbolicRegressionThatScales
python scripts/data_creation/dataset_creation.py --number_of_equations 1000000 --no-debug
python scripts/data_creation/dataset_creation.py --number_of_equations 500 --no-debug
```

Then you will have your raw dataset under `NeuralSymbolicRegressionThatScales/data/raw_datasets/NumberOfEquations`. We will sample 5 different constants to substitute the ones in the skeltons, you can configure the number in `.yaml` files。 Also set the folder path to place your dataset in the `.yaml` files. Run:

```bash
cd Sym-Q/symq_v2
# Whichever you need
./run_parallel_3var.sh
./run_parallel_2var.sh
```

This will save `.h5` files under the folder you defined and each file contains the data based on 100 skeletons.

## Quick Start

Download the pretrain weights under 10M skeletons from [NeSymReS]<https://github.com/SymposiumOrganization/NeuralSymbolicRegressionThatScales/tree/main?tab=readme-ov-file> under `Sym-Q/weights`.
Train the Sym-Q model:

```cmd
# Whichever you need
python train.py --n_var 3
python train.py --n_var 2
```

## Example

An example for inference is given in `example.ipynb`. To use model with different input dimensions, please modify the `num_vars` variable in `regressor.py` and specify the `weights_path` in `.yaml` files.
