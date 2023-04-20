# Value Memory Graph: A Graph-Structured World Model for Offline Reinforcement Learning
Official Repository of the ICLR2023 paper Value Memory Graph.

## Install the Environment
Please run the following commands to install the environment via conda.

```
conda create -n vmg python=3.9
conda activate vmg
conda install pytorch torchvision torchaudio cudatoolkit=11.3 -c pytorch
conda install pytorch-scatter -c pyg

conda install -c conda-forge mesalib
conda install -c conda-forge patchelf

pip install -r requirement.txt
```

## Usage
To train a model, run
```
python main.py --dataset dataset-you-want --gpu desired-gpu-idx
```


To evaluate the model, run and set the hyperparameters in the arguments in the following way 
```
python eval_script/eval_policy.py  --dataset dataset-you-want  --action_mode top  --cluster_thresh gamma_{m}  --discount discount  --min_future_step N_{sg}  --ckpt CheckPointYouWant  --gpu 0
```

The hyperparameters we use can be found in the Appx.D of the paper. 
Set the argument '--action_mode' to 'top' when the search step N_s is infinit. Otherwise, set it to 'neighbor'. 
Note that different checkpoints in a single training case may perform differently 
and you need to search for the checkpoint for the best performance.
