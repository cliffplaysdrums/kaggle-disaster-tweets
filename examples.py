import os
from data import get_disaster_tweets_dataloader

from evaluate import kaggle_submission
from time import time
from model import train_model, get_default_model

if __name__ == "__main__":
    model, text_transform = get_default_model(
        # saved_params_path=os.path.join(".", "models", "disaster-tweets.pt")
    )

    train_dataloader = get_disaster_tweets_dataloader(
        csv_file=os.path.join(".", "data", "train.csv"),
        text_transform=text_transform,
        batch_size=32,
    )

    model = train_model(
        train_dataloader,
        model=model,
        epochs=10,
        save_path=os.path.join(".", "models", "disaster-tweets.pt"),
    )

    submission_file = os.path.join(".", "data", "submission.csv")
    start = time()
    kaggle_submission(model, text_transform=text_transform, csv_save_path=submission_file)
    end = time()
    print(f"Generated submission in {end - start: .4f}")
