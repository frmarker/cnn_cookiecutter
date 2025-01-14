import matplotlib.pyplot as plt
import numpy as np
import torch
import typer
import wandb
from data import corrupt_mnist
from model import MyAwesomeModel
from sklearn.metrics import RocCurveDisplay, accuracy_score, f1_score, precision_score, recall_score

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")


def train(lr: float = 1e-3, batch_size: int = 32, epochs: int = 10) -> None:
    """Train a model on MNIST."""
    print("Training day and night")
    print(f"{lr=}, {batch_size=}, {epochs=}")

    config = {"lr": lr, "batch_size": batch_size, "epochs": epochs}

    wandb.init(
        project="corrupt_mnist",
        config=config,
    )

    model = MyAwesomeModel().to(DEVICE)
    train_set, _ = corrupt_mnist()

    train_dataloader = torch.utils.data.DataLoader(train_set, batch_size=wandb.config.batch_size)

    loss_fn = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    statistics = {"train_loss": [], "train_accuracy": []}
    for epoch in range(epochs):
        model.train()
        preds = []
        targets = []
        for i, (img, target) in enumerate(train_dataloader):
            img, target = img.to(DEVICE), target.to(DEVICE)
            optimizer.zero_grad()
            y_pred = model(img)
            preds.append(y_pred)
            targets.append(target)
            loss = loss_fn(y_pred, target)
            loss.backward()
            optimizer.step()
            statistics["train_loss"].append(loss.item())

            accuracy = (y_pred.argmax(dim=1) == target).float().mean().item()
            statistics["train_accuracy"].append(accuracy)

            wandb.log({"train_loss": loss.item(), "train_accuracy": accuracy})

            if i % 100 == 0:
                print(f"Epoch {epoch}, iter {i}, loss: {loss.item()}")

                # add a plot of the input images
                images = wandb.Image(img[:5].detach().cpu(), caption="Input images")
                wandb.log({"images": images})

                # add a plot of histogram of the gradients
                grads = torch.cat([p.grad.flatten() for p in model.parameters() if p.grad is not None], 0)
                wandb.log({"gradients": wandb.Histogram(grads.detach().cpu().numpy())})

        # add a custom matplotlib plot of the ROC curves
        preds = torch.cat(preds, 0).detach().cpu().numpy()
        targets = torch.cat(targets, 0).detach().cpu().numpy()

        for class_id in range(10):
            one_hot = np.zeros_like(targets)
            one_hot[targets == class_id] = 1
            _ = RocCurveDisplay.from_predictions(
                one_hot,
                preds[:, class_id],
                name=f"ROC curve for {class_id}",
                plot_chance_level=(class_id == 2),
            )

        wandb.log({"roc": wandb.Image(plt)})
        # alternatively the wandb.plot.roc_curve function can be used

    final_accuracy = accuracy_score(targets, preds.argmax(axis=1))
    final_precision = precision_score(targets, preds.argmax(axis=1), average="weighted")
    final_recall = recall_score(targets, preds.argmax(axis=1), average="weighted")
    final_f1 = f1_score(targets, preds.argmax(axis=1), average="weighted")


    print("Training complete")
    torch.save(model.state_dict(), "models/model.pth")

    artifact = wandb.Artifact(
        name="corrupt_mnist_model",
        type="model",
        description="A model trained to classify corrupt MNIST images",
        metadata={"accuracy": final_accuracy, "precision": final_precision, "recall": final_recall, "f1": final_f1},
    )
    artifact.add_file("models/model.pth")
    #run.log_artifact(artifact)

    #fig, axs = plt.subplots(1, 2, figsize=(15, 5))
    #axs[0].plot(statistics["train_loss"])
    #axs[0].set_title("Train loss")
    #axs[1].plot(statistics["train_accuracy"])
    #axs[1].set_title("Train accuracy")
    #fig.savefig("reports/figures/training_statistics.png")

    


if __name__ == "__main__":
    typer.run(train)