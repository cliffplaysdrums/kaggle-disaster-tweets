import pandas as pd
from functools import partial
from torch.utils.data import Dataset, DataLoader
from torch import FloatTensor
from torchtext.functional import to_tensor


class DisasterTweetsDataset(Dataset):
    """Custom Dataset for loading entries from this Kaggle challenge's training data."""

    def __init__(self, csv_file):
        """
        Args:
            csv_file (str): Path to the CSV file containing the training data
        """
        self.data_df = pd.read_csv(csv_file)

    def __getitem__(self, index):
        """
        Args:
            index (int): index of the row to retrieve from the CSV file

        Returns:
            tuple(str, list): The tweet from the given row ("text" field) and the label as a one-hot encoded list
        """
        tweet = self.data_df["text"][index]
        label = self.data_df["target"][index]
        encoded_label = [1 - label, label]

        return tweet, encoded_label

    def __len__(self):
        """
        Returns:
            int: The number of records in the dataset
        """
        return self.data_df.shape[0]


class DisasterTweetsTestDataset(Dataset):
    """Custom Dataset for loading entries from this Kaggle challenge's test data."""

    def __init__(self, csv_file):
        """
        Args:
            csv_file (str): Path to the CSV file containing the test data
        """
        self.data_df = pd.read_csv(csv_file)

    def __getitem__(self, index):
        """
        Args:
            index (int): index of the row to retrieve from the CSV file

        Returns:
            numpy.ndarray: Array containing the "id" field of the specified row and the "text" field
        """
        features = self.data_df.iloc[index, self.data_df.columns.get_indexer(["id", "text"])].values
        return features

    def __len__(self):
        """
        Returns:
            int: The number of records in the dataset
        """
        return self.data_df.shape[0]


def get_collate_fn(text_transform, labeled_data=True):
    """Return an appropriate collate function for the given transform and data.

    Args:
        text_transform (torchtext.transforms.Sequential): Transform to preprocess input data
        labeled_data (bool, optional): Whether or not the data contains labels (train vs test). Defaults to True.

    Returns:
        function: The appropriate collate function
    """
    if labeled_data:
        return partial(collate_train_batch, text_transform=text_transform)
    else:
        return partial(collate_test_batch, text_transform=text_transform)


def collate_train_batch(batch_data, text_transform):
    """Groups batched training data in a way that is more convenient for training.

    Args:
        batch_data (list): List of records from our custom Dataset
        text_transform (torchtext.transforms.Sequential): Transform to preprocess input data

    Returns:
        tuple(torch.Tensor, torch.FloatTensor): A tensor containing all the preprocessed tweets from this batch and a
            tensor containing all the labels for this batch
    """
    tweets = [row[0] for row in batch_data]
    labels = FloatTensor([row[1] for row in batch_data])
    processed_tweets = to_tensor(text_transform(tweets), padding_value=1)

    return processed_tweets, labels


def collate_test_batch(batch_data, text_transform):
    """Groups batched test data in a way that is more convenient for evaluation.

    Args:
        batch_data (list): List of records from our custom Dataset
        text_transform (torchtext.transforms.Sequential): Transform to preprocess input data

    Returns:
        tuple(list, torch.Tensor): A list of the record IDs for this batch and a tensor of the preprocessed tweets from
            this batch
    """
    ids = [row[0] for row in batch_data]
    tweets = [row[1] for row in batch_data]
    processed_tweets = to_tensor(text_transform(tweets), padding_value=1)

    return ids, processed_tweets


def get_disaster_tweets_dataloader(csv_file, text_transform, batch_size=32):
    """Construct a custom DataLoader for this Kaggle challenge's training data.

    Args:
        csv_file (str): Path to the CSV file containing the training data
        text_transform (torchtext.transforms.Sequential): Transform to preprocess input data
        batch_size (int, optional): How many samples to process at once. Defaults to 32.

    Returns:
        torch.utils.data.DataLoader: Custom DataLoader for loading batches of preprocessed data and labels
    """
    dataset = DisasterTweetsDataset(csv_file=csv_file)
    dl = DataLoader(
        dataset,
        batch_size=batch_size,
        collate_fn=get_collate_fn(text_transform=text_transform, labeled_data=True),
    )
    return dl


def get_disaster_tweets_test_dataloader(csv_file, text_transform, batch_size=32):
    """Construct a custom DataLoader for this Kaggle challenge's test data.

    Args:
        csv_file (str): Path to the CSV file containing the test data
        text_transform (torchtext.transforms.Sequential): Transform to preprocess input data
        batch_size (int, optional): How many samples to process at once. Defaults to 32.

    Returns:
        torch.utils.data.DataLoader: Custom DataLoader for loading batches of preprocessed data
    """
    dataset = DisasterTweetsTestDataset(csv_file=csv_file)
    dl = DataLoader(
        dataset,
        batch_size=batch_size,
        collate_fn=get_collate_fn(text_transform=text_transform, labeled_data=False),
    )
    return dl
