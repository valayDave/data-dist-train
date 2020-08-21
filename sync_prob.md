
# Dispatching Block Size and Starvation Problems. 

Lets say we have a 7 workers to whom we dispatch data in a RR and Random Fashion. 

The following are how the number datapoints dispatched across the 7 workers look with Different block sizes:

## RANDOM

- BLOCK SIZE : 2  DISTRIBUTION : 7054, 6904, 7180, 7274, 7280, 7222, 7086
- BLOCK SIZE : 10  DISTRIBUTION : 6870, 7060, 7190, 7200, 7010, 7570, 7100 
- BLOCK SIZE : 100  DISTRIBUTION : 9100, 7100, 7900, 6300, 7800, 6900, 4900
- BLOCK SIZE : 1000  DISTRIBUTION : 6000, 9000, 10000, 6000, 6000, 4000, 9000

## ROUND_ROBIN 
- BLOCK SIZE : 2  DISTRIBUTION : 7144, 7144, 7144, 7142, 7142, 7142, 7142
- BLOCK SIZE : 10  DISTRIBUTION : 7150, 7150, 7140, 7140, 7140, 7140, 7140
- BLOCK SIZE : 100  DISTRIBUTION : 7200, 7200, 7200, 7100, 7100, 7100, 7100

Round robin works perfectly to evenly distribute the data among worker until we go to a larger Block size of 100. But random dispatching creates an imbalance in the number of data items per worker. 

## Why is Data Imbalance a Problem ? 

Let take the worker with Random dispatch and block size of 2 from the above dispatched schemes. As Every worker will have the same batch size. The number of Minibatches created in each worker will be : 7054/bs, 6904/bs, 7180/bs, 7274/bs, 7280/bs, 7222/bs, 7086/bs . `bs` is batch_size. Based no the batchsize the number of minibatches will created. And these may not be equal based on the size of `bs`. If `bs` is 4096 then we will have equal number of minibatches([2, 2, 2, 2, 2, 2, 2]). But If `bs` is 64 the the number of minibatches is not equal([110, 108, 112, 114, 114, 113, 111]). To understand the problem with this imbalance lets first see how Pytorch works in a distributed way : 
- Gradient averaging happens after the end of every minibatch. This is mathematically equivalent so one can always use it. Gradient averaging in pytorch  `DistributedDataParallel` is done using the `dist.all_reduce` premitive. 
- Pytorch also leverages `mp.spawn` which will basically invoke the same training loop on all processes and they will synchronise using the `dist.all_reduce` which will block until completion for that process. All of this works because of "Process groups" in pytorch. This ensures that all the workers know who they are going to communicate with. Ideally we do everyone communicating with everyone. 
- As one AllReduce operation cannot start until all processes join, it is considered to be a synchronized communication, as opposed to the P2P communication usedin parameter servers[1](https://arxiv.org/pdf/2006.15704.pdf)

- When a minibatch imbalance situation happens some of the processes finish execution while rest are starving for resources from those processes because they might have made a `dist.all_reduce` call and are waiting for processes from their process group. This creates the starvation problem(or deadlock ?)

- Parameter averaging is ideally not recommended because can produce vastly diferent results compared to local training, which, sometimes, can be detrimental to model accuracy. The root cause is that parameter averaging is not mathematically equivalent to processing all input data locally, especially when the optimizer relies on past local gradients values (e.g., momentum)[1](https://arxiv.org/pdf/2006.15704.pdf)

## Conclusions 

- Block size and dispatching strategy influence how balanced the data is for distributed training. Balanced here means the number of items on each worker is equal or so close that the number minibatches turn out to the equal. 
- Round robin works well to evenly distribute data so synchronisation issues wouldn't happen with round robin. But random creates data imbalance which leads so synchronisation issues with all reduce. 
-  With larger blocksizes in round robi dispatching, its ideal to have a larger batch size. But larger batchsize for large models wouldn't fit in memory. So figuring that tune of batch size is some that will be figured next to complete the experiment. Larger block size with round robin can also cause an imbalance problem. 


## Some Hacky Ways I think To Fix it. 
Run a synchronisation to only run min([mini_batches]) for all processes ? But is this correct for experimentation conditions ? 

