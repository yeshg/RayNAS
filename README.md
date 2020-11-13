Ray-NAS
=======

Scaling and Distributing Neural Architecture Search (NAS) Algorithms with the distributed machine learning framework [Ray](https://docs.ray.io/en/latest/).

## Algorithms

- DARTS
  - [uses official pytorch implementation](https://github.com/quark0/darts)
  - CNN only
- ENAS
  - [uses unofficial pytorch reimplimentation](https://github.com/carpedm20/ENAS-pytorch)
  - RNN only
- Random NAS
  - Same search space as ENAS, no RNN controller
  - CNN only

## Running experiments

### 1. Basics
Can search for CNN and RNN architectures from the same entry-point: `main.py`

First choose algorithm from `darts`, `enas`, or `random`, which correspond to differentiable architecture search, efficient neural architecture search, and a simple random sample based approach which simplifies NAS to hyperparameter tuning.

To search for CNN architecture for cifar10 with DARTS,

```bash
python main.py darts cnn --dataset cifar10 --layers 2 --cuda
```

To search for RNN architecture for ptb with ENAS,

```bash
python main.py enas rnn --num_blocks 4 --cuda
```

To search for CNN architecture for cifar10 with simple random NAS,

```bash
python main.py random cnn --dataset cifar10 --cuda
```

### 2. Logging details / Monitoring live training progress

Monitor runs in ray dashboard, or by launching tensorboard in the experiment directory created by Ray Tune:

```bash
tensorboard --logdir="~/ray_results/exp/"
```

### 3. Visualizing Searched Architectures

Find the location of the run which generated the desired architecture and run the entry-point in visualize mode:

```bash
python main.py viz --load <path_to_trial> --viz
```

### General Arguments for All NAS algorithms

TBD

### To Do
- [ ] DARTS
  - [ ] RNN implementation
  - [ ] Results
- [ ] ENAS
  - [ ] CNN implementation
  - [ ] Results
- [ ] RandomNAS
  - [ ] Search Space expansion for CNN
  - [ ] RNN implementation
  - [ ] Results

