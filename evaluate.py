import csv
import os
import torch
import data


def test_model(test_dataloader, model):
    """Evaluate model accuracy on the given dataset

    Args:
        test_dataloader (torch.utils.data.DataLoader): DataLoader for generating batches of input and their labels
        model (torch.nn.Module): Model to evaluate

    Returns:
        float: Model's accuracy on the provided data
    """
    DEVICE = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

    model.eval()
    model.to(DEVICE)

    with torch.no_grad():
        correct = 0
        total = 0
        for tweet_batch, label_batch in test_dataloader:
            tweet_batch = tweet_batch.to(DEVICE)
            label_batch = label_batch.to(DEVICE)
            preds = torch.argmax(model(tweet_batch), dim=1)
            labels = torch.argmax(label_batch, dim=1)
            correct += torch.sum(preds == labels)
            total += tweet_batch.size()[0]

        accuracy = float(correct) / total
        print(f"Accuracy: {accuracy: .4f}%")
        return accuracy


def kaggle_submission(model, text_transform, csv_save_path):
    """Generate a CSV file of predictions for this Kaggle competition

    Args:
        model (torch.nn.Module): The model for making predictions
        text_transform (torchtext.transforms.Sequential): Transform for preprocessing input
        csv_save_path (str): Path to which the submission file will be written
    """
    DEVICE = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    model.to(DEVICE)
    model.eval()

    test_dataloader = data.get_disaster_tweets_test_dataloader(
        csv_file=os.path.join(".", "data", "test.csv"), text_transform=text_transform
    )

    with open(csv_save_path, "w", newline="") as fp:
        submission_writer = csv.writer(fp, delimiter=",")
        submission_writer.writerow(["id", "target"])

        with torch.no_grad():
            for id_batch, tweet_batch in test_dataloader:
                preds = torch.argmax(model(tweet_batch.to(DEVICE)), dim=1).cpu().numpy()
                submission_writer.writerows(zip(id_batch, preds))
