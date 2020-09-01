## Running PT DDP 

## `distributed_trainer`
TODO : Document the Module 

## Setting Up Repo 
1. 
    ```
    git submodule update --init
    ```
2. `pip install -r requirements.txt`
3. `cd dataset-repo && tar -xzvf dataset.tar.gz`

## Running Credit Card Fraud Training Experiments

```
sh monlith_experiments.sh # Runs training for Monolith Network with DataParallel
```

```
sh uniform_experiments.sh # Runs training with DistributedDataparallel Network on Credit card fraud dataset
```


## Running Individual Scripts
```sh
python distributed_cred.py distributed --help
```
```sh
python distributed_cifar.py distributed --help
```
```sh
python distributed_cred.py monolith --help
```
```sh
python parameter_map.py # Converts Grid params to objects for running experiment. 
```

## Running Results Visualisation

```
streamlit run visual_explorer.py
```


## Running With Global Shuffling of Distributed Sampler 
`distributed_trainer.data_dispather.DistributedIndexSamplerServer` which is `rpyc.Service`. This will run a TCP index sampling server. 

The `distributed_trainer.data_dispather.DistributedSampler` is an implementation of the client for this server. We recover `connection_id` before spawning processes so that we can reference the same sampler session with multiple processes accessing the same server.

**1 Instance of `DistributedIndexSamplerServer` is required to run Global Shuffle training jobs**

Running Sampler Service
```sh
$ python sampler_service.py will start the distributed sampler server.
```

Running CIFAR Training Job:
```sh
$ python distributed_cifar.py distributed-global-shuffle --sample 52 --model ResNet18 --note "testing model with no Usecaase" --epochs 4 --world_size 2 --batch_size 2 Will run the CIFAR10 distributed training module.
```
The number of Minibatches needs to be equal across all the workers otherwise there will be a synchronization issue.

Runnning Credit Card Fraud Detection Job: 
```sh
python distributed_cred.py distributed-global-shuffle --sample 200 --model FF --note "testing model with no Usecaase" --epochs 4 --world_size 2 --batch_size 2
```


## Import Point While Training 
1. Batchsize Will influence the number of minibatches of distributed workers. Gradient syncing happens for works so imbalanced minibatches can cause a deadlock for gradient syncing due to Pytorch's DistributedDataParallel Implementation

2. Metaflow can be used a the version management and infrastructure bundling layer. 
