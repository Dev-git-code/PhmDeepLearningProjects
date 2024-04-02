import numpy as np
import sklearn.metrics as me

from torch.utils.data import Dataset


class Sampler:
    """
    This class is used for customize yourself sampling method used for dataset.

    This class is first used in cmapss.CMAPSS dataset, and will be supported more custom dataset sampling.

    When customizing your own sampler, you should:

    1. override the sample(index) method. The sample(index) method should return the sample and label similar to
    torch.utils.data.Dataset class.

    2. Making sure your __init__(dataset) method containing the sampling target argument "dataset".
    The argument "dataset" should be a torch.utils.data.Dataset instance.
    And call the super.__init__(dataset) at the first line in your __init__ method.
    """
    def __init__(self, dataset: Dataset):
        self.dataset = dataset

    def sample(self, index: int):
        raise NotImplementedError("You must define the Sampler.sample(index) method.")


def compute_metrics(path):
    out = np.load(path+r"/model_test_output_part1.npy")
    label = np.load(path+r"/model_test_labels_part1.npy")
    mse = me.mean_squared_error(out, label)
    rmse = np.sqrt(mse)
    mape = me.mean_absolute_percentage_error(out, label)
    print("MSE:{}".format(mse))
    print("RMSE:{}".format(rmse))
    print("MAPE:{}".format(mape))
    print("R2:{}".format(me.r2_score(out, label)))
    return (out, label), mse, rmse, mape


def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def gaussian_distribution(x: int or np.ndarray, mean, std):
    l1 = 1/((2*np.pi)**0.5 * std)
    l2 = np.exp(-((x-mean)**2)/(2*std**2))
    return l1*l2

