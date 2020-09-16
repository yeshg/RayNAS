PBT-NAS
=======

Library for running NAS with the distributed machine learning framework [Ray](https://docs.ray.io/en/latest/).

## Running experiments

### Basics
Can search for CNN and RNN architectures from the same entry-point.

To search for CNN architecture for cifar10,

```bash
python main.py cnn --dataset cifar10 --layers 20 --cuda
```

### Logging details / Monitoring live training progress
Monitor runs in ray dashboard, or by launching tensorboard in the experiment directory created by Ray Tune:

```bash
tensorboard --logdir="~/ray_results/exp/"
```

### Visualizing Searched Architectures
Find the location of the run which generated the desired architecture and run the entry-point in visualize mode:

```bash
python main.py viz --load <path_to_trial> --viz
```

### To Do
- [ ] Verify checkpointing on experiments across multiple GPUs
- [ ] Extend to image datasets other than cifar10
- [ ] Create `nas.rnn_nas` module as entry point for RNN model search


## Acknowledgements

NAS Algorithm used here is a fork of DARTS (Differentiable Architecture Search):
- [paper](https://arxiv.org/abs/1806.09055)
- [official code](https://github.com/quark0/darts)

