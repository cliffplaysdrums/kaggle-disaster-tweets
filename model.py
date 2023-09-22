import os.path
import torch
from datetime import datetime
from torchtext.models import (
    RobertaClassificationHead,
    ROBERTA_LARGE_ENCODER,
)


def save_model(model, path, id=None):
    """Save a torch model at the given path, appending the ID prior to the file extension.

    Args:
        model (torch.nn.Module): The model to save
        path (str): Path at which to save model
        id (str, optional): Additional string to append to model name (prior to any file extension). Defaults to None.

    Returns:
        str: The path at which the model was saved including the ID (if any)
    """
    if id is not None:
        path, ext = os.path.splitext(path)
        path = path + "-" + id + ext

    torch.save(model.state_dict(), path)
    return path


def get_default_model(saved_params_path=None, encoder=ROBERTA_LARGE_ENCODER):
    """Get a model with sensible defaults and corresponding transform.

    Args:
        saved_params_path (str, optional): Path to previously saved model parameters. Defaults to None.
        encoder (torchtext.models.roberta.bundler.RobertaBundle, optional): Specific encoder to use. Defaults to ROBERTA_LARGE_ENCODER.

    Returns:
        tuple(torchtext.models.roberta.bundler.RobertaBundle, torchtext.transforms.Sequential): Model and transform to prepare input for model
    """
    classifier_head = RobertaClassificationHead(num_classes=2, input_dim=encoder.encoderConf.embedding_dim)
    model = encoder.get_model(head=classifier_head)
    text_transform = encoder.transform()

    if saved_params_path is not None:
        model.load_state_dict(torch.load(saved_params_path))

    return model, text_transform


def train_model(train_dataloader, model=None, epochs=100, save_path=None):
    """Train a model.

    Args:
        train_dataloader (torch.utils.data.DataLoader): DataLoader for generating batches of training input and labels
        model (torch.nn.Module, optional): Model to train. Defaults to None, meaning a sensible default model is loaded
        epochs (int, optional): Number of times to iterate over the entire dataset. Defaults to 100.
        save_path (str, optional): Path at which to save model when training is complete. Also saves each epoch in case
            training is halted. Defaults to None.

    Returns:
        torch.nn.Module: The trained model
    """
    DEVICE = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    if model is None:
        model = get_default_model()
    model.to(DEVICE)

    save_id = datetime.today().strftime("%Y%m%d-%H%M%S")
    previous_save_path = None

    print(f"Training on device {DEVICE}")

    learning_rate = 1e-5
    loss_fn = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(params=model.parameters(), lr=learning_rate)

    accumulated_loss = 0
    for current_epoch in range(epochs):
        batch = 0
        for tweet_batch, label_batch in train_dataloader:
            batch += 1
            tweet_batch = tweet_batch.to(DEVICE)
            label_batch = label_batch.to(DEVICE)
            preds = model(tweet_batch)
            loss = loss_fn(preds, label_batch)
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
            accumulated_loss = accumulated_loss * 0.9 + 0.1 * loss.item()
            if batch % 10 == 0:
                print(f"Batch {batch}, running average loss per batch: {accumulated_loss: .6f}")

        print(f"Finished epoch {current_epoch}, running average loss: {accumulated_loss: .6f}")
        if save_path is not None:
            model_saved_at = save_model(model, path=save_path, id=f"{save_id}-epoch{current_epoch + 1}")
            if current_epoch > 0:
                os.remove(previous_save_path)
            previous_save_path = model_saved_at

    if save_path is not None:
        save_model(model, save_path, id=save_id)

    return model
