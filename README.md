# multi-device-and-gpu-train-with-torch

### Distributed Data Parallelism

Distributed Data Parallel is a PyTorch module that enables distributed training of deep learning models. It is designed to work in a multi-node, multi-GPU environment, where each node has one or more GPUs. DDP enables each GPU to train a subset of the data, and then aggregates the gradients from each GPU to update the model parameters. This method allows for faster training times and the ability to train on larger datasets

## Installation
Clone the repository:
``` bash
git clone https://github.com/yahyoxonqwe/multi-device-and-gpu-train-with-torch.git
```
Change into the project directory:
``` bash
cd multi-device-and-gpu-train-with-torch
```
Install the required dependencies:
``` bash
pip install -r requirements.txt
```
## Usage
``` bash
 torchrun --nnodes=i --nproc_per_node=i --node_rank=i --master_addr=ip_address --master_port=i train.py 
```

    --nnodes                     Number of devices 
    --nproc_per_node             Number of GPUs of each device.
    --node_rank                  Rank of device of you run command.{ 0, 1, 2, ...}
    --master_addr                Master device ip address.
    --master_port                Master device's any free port.
    
this command run each nodes (with other nodes_rank)
