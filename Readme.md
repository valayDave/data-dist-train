## Running PT DDP 

## `distributed_trainer`
TODO : Document the Module 

## Setting Up Repo 
1. 
    ```
    git submodule update --init
    ```
2. `pip install -r requirements.txt`

## Running Credit Card Fraud Training Experiments

```
sh monlith_experiments.sh # Runs training for Monolith Network with DataParallel
```

```
sh uniform_experiments.sh # Runs training with DistributedDataparallel Network on Credit card fraud dataset
```


## Running Individual Scripts
```
python distributed_cred.py distributed --help
```

```
python distributed_cred.py monolith --help
```

## Running Results Visualisation

```
streamlit run visual_explorer.py
```


## Import Point While Training 
1. Batchsize Will influence the number of minibatches of distributed workers. Gradient syncing happens for works so imbalanced minibatches can cause a deadlock for gradient syncing due to Pytorch's DistributedDataParallel Implementation