import torch
from cnn_mnist.data import corrupt_mnist
import os.path
import pytest

def test_data():

    if not os.path.exists("/Users/rosalouisemarker/Desktop/cnn_cookiecutter/cnn_cookiecutter/data"):  # Replace with the actual path to your data files
        pytest.skip("Data files not found")

    print("Testing data")
    train, test = corrupt_mnist()
    assert len(train) == 30000, "Dataset did not have the correct number of samples"
    assert len(test) == 5000, "Dataset did not have the correct number of samples"
    for dataset in [train, test]:
        for x, y in dataset:
            assert x.shape == (1, 28, 28)
            assert y in range(10)
    train_targets = torch.unique(train.tensors[1])
    assert (train_targets == torch.arange(0,10)).all()
    test_targets = torch.unique(test.tensors[1])
    assert (test_targets == torch.arange(0,10)).all()

if __name__ == "__main__":
    test_data()
