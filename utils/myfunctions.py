
import matplotlib.pyplot as plt
import torch
from time import perf_counter
from tqdm.auto import tqdm #progress bar
from torch.utils.data import DataLoader


def print_image(img):
    img_permute = img.permute(1, 2, 0)
    plt.figure(figsize=(6, 6))
    plt.imshow(img_permute)
    plt.show()


def get_time(func):
    def wrapper(*args, **kwargs):

        start = perf_counter()
        func(*args, **kwargs)
        end = perf_counter()
        total_time = round(end - start,2)

        print(f"time: {total_time:.2f} seconds")

    return wrapper

def model_eval(model: torch.nn.Module,
               test_data: torch.utils.data.DataLoader,
               loss_fn:torch.nn.Module,
               acc1):
    torch.manual_seed(42)

    loss, accuracy = 0, 0
    model.eval()
    with torch.inference_mode():
        for x, y in tqdm(test_data):
            test_logits = model(x)
            loss += loss_fn(test_logits, y)
            accuracy += acc1(y_true=y,y_pred= test_logits.argmax(dim=1))

        loss /= len(test_data)
        accuracy /= len(test_data)

    return {"model_name": model.__class__.__name__,
            "model_loss": loss.item(), #returns the value of tensor
            "model_acc": accuracy}

def train_step(model: torch.nn.Module,
               train_data: torch.utils.data.DataLoader,
               loss_fn:torch.nn.Module,
               optimize,
               acc):

    tloss:float = 0
    tacc:float = 0
    model.train()

    for batch, (x, y) in enumerate(train_data):
        y_logits = model(x)
        loss = loss_fn(y_logits, y)
        optimize.zero_grad()
        loss.backward()
        optimize.step()
        model.eval()

        tloss += loss.item()
        tacc += acc(y_true=y, y_pred=y_logits.argmax(dim=1))
    tloss /= len(train_data)
    tacc /= len(train_data)

    return tloss, tacc


def test_step(model: torch.nn.Module,
              test_data: torch.utils.data.DataLoader,
              loss_fn: torch.nn.Module,
              acc):
    tloss: float = 0
    tacc: float = 0
    model.eval()

    for x, y in test_data:
        with torch.inference_mode():
            test_logits = model(x)
            loss = loss_fn(test_logits, y)

            tloss += loss.item()
            tacc += acc(y_true=y,y_pred=test_logits.argmax(dim=1))
    tloss /= len(test_data)
    tacc /= len(test_data)

    return tloss, tacc


def training_loop(model: torch.nn.Module,
                  train: torch.utils.data.DataLoader,
                  # test: torch.utils.data.DataLoader,
                  optimizer,
                  loss_fn,
                  acc,
                  epochs: int = 5):

    results = {"train_loss": [],
               "train_acc": [],
               "test_loss": [],
               "test_acc": []}

    for epoch in tqdm(range(epochs)):

        train_loss, train_acc = train_step(model=model,
                                           train_data = train,
                                           optimize= optimizer,
                                           loss_fn = loss_fn,
                                           acc = acc)

        # test_loss, test_acc = test_step(model=model,
        #                                 test_data=test,
        #                                 loss_fn=loss_fn,
        #                                 acc=acc)

        # print(f"Epoch: {epoch} train loss: {train_loss}, train acc: {train_acc}, test loss: {test_loss}, test acc: {test_acc}")
        #
        # results["train_loss"].append(train_loss)
        # results["test_loss"].append(test_loss)
        # results["train_acc"].append(train_acc)
        # results["test_acc"].append(test_acc)

        # plot_loss_curves(results=results)

    return results

