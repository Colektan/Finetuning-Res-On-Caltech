### Finetuning Pretrained ResNet on Caltech-101

This experiment compares the finetuned performance of a pretrained ResNet-18 and a from-scratch one on a small dataset.

### Environment

This Experiment uses a standard `Pytorch` environment with `torchvision` and `matplotlib` installed.

### Dataset

Download from https://data.caltech.edu/records/mzrjq-6wc02，and unzip it to the folder `data`.

This experiment uses `torchvision.datasets.Caltech101` to read in data. Create and instance using the following code:

```python
dataset = Caltech101(root="./data")
```

where `root` refers to the parent directory of `caltech101`.

### Train the model

Modify the `config.py`, then run `main.py`. A new training log folder will be created in the logs folder named with your EXPERI_NAME in `config.py`. And your current config will be saved to that folder.

### View the metrics

Run in terminal:

```bash
tensorboard --logdir=./logs --port=[YOUR_DESIRED_PORT]
```

and visit `localhost:[YOUR_DESIRED_PORT]` in Explorer, your training metrics will be exhibited.

### Test the model

Specify the model to be tested in `test_model.py`，and then run the same file.

### Visiaulization and Utilities

`split_dataset.py` helps to split the dataset into train set and eval set.

`test_model.ipynb` contains some codes to visualize some results.

`data_explore.ipynb` contains codes to checkout the dataset.

