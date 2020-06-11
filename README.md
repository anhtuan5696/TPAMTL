# Clinical Risk Prediction with Temporal Probabilistic Asymmetric Multi-Task Learning

This repository is the official implementation of Clinical Risk Prediction with Temporal Probabilistic Asymmetric Multi-Task Learning.

## Requirements

To install requirements:

```setup
pip install -r requirements.txt
```

## Training

To train the models in the paper, run this command:

```train
python run.py
```
Also for P-AMTL experiment on MNIST-variant dataset, you can run the same command inside [MNIST](/MNIST) directory. Please see MNIST Toy experiment section below for details.

Please look up and modify [config.py](config.py) for selecting base model and datasets. For example, if you want to run tp_amtl model on physionet2012 dataset, you can modify [config.py](config.py) as follows:


```config
config.mtl_model = "tp_amtl"
config.task_code = "physionet2012"
```

Please put model name on config.mtl_model and dataset on config.task_code (both in lower-case).

For single-task version of UA,LSTM and RETAIN model, you can simply select config.tasks = [i] where i is the task you want to run (0,1,2,etc.). Also, for RETAIN-Kendall model, you need to change config.tasks [i for i in range(len(tasks))] where i is task id (0,1,2, etc.). Below is example for physionet2012 (also can be applied to MIMIC-III Heart Failure).

```config
config.tasks = [0,1,2,3]
```

## MNIST toy experiment

Requirement:

 python3.6.9

 torch

 tqdm

 matplotlib
 
 sklearn

Training:
 Please download mnist_rotation_back_image_new.zip (https://sites.google.com/a/lisa.iro.umontreal.ca/public_static_twiki/variations-on-the-mnist-digits) into a folder named mnist_rotation_back inside the MNIST directory and unzip. Then run this command:

```train_mnist
python run.py
```

## Datasets
We experiment on four datasets that we compile for clinical risk prediction from two open-source electronic health records (EHR) datasets.

[MIMIC-III Infection](mimic_infection/): [code for train/test/valid separation](task_codes/mimic_infection.py)

[Physionet2012](physionet2012/): [code for train/test/valid separation](task_codes/physionet2012.py)

[MIMIC-III Heart Failure](mimic_heart_failure/): [code for train/test/valid separation](task_codes/mimic_heart_failure.py)

[MIMIC-III Respiratory Failure](mimic_respiratory_failure/): [code for train/test/valid separation](task_codes/mimic_respiratory_failure.py)

For MIMIC-III Respiratory Failure dataset, you can set as below on [task_codes/mimic_respiratory_failure.py](task_codes/mimic_respiratory_failure.py) to experiment on dataset with full 37,818 instances.

```task_codes
path = 'mimic_respiratory_failure/30000/'
```

## Results

Our model achieves the following performance on four datasets:
<img src="/imgs/table2.png" alt="mimic_infection"/>
<img src="/imgs/table3.png" alt="physionet"/>
<img src="/imgs/supple_table3.png" alt="mimic_heart_failure"/>
<img src="/imgs/supple_table4.png" alt="mimic_respiratory_failure"/>
<img src="/imgs/supple_table5.png" alt="mimic_respiratory_30000"/>