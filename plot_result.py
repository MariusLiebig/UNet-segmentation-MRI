import os
import json
import matplotlib.pyplot as plt
from config import base_path

def plot_training_curves(train_history_path, val_history_path, save_path="loss_plot.png"):
    # Load saved training and validation history
    with open(train_history_path, "r") as f:
        train_history = json.load(f)

    with open(val_history_path, "r") as f:
        val_history = json.load(f)

    # Extract losses
    train_losses = list(train_history["loss"].values())
    # val_losses = list(val_history["loss"].values())
    train_accuracies = list(train_history["accuracy"].values())
    val_accuracies = list(val_history["accuracy"].values())

    epochs = range(len(train_losses))

    # Plot
    plt.figure(figsize=(10,6))
    plt.plot(epochs, train_losses, label="Training Loss", marker="o")
    # plt.plot(epochs, val_losses, label="Validation Loss", marker="o")
    plt.plot(epochs, train_accuracies, label="Training Accuracy", marker="o")
    plt.plot(epochs, val_accuracies, label="Validation Accuracy", marker="o")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Training and Validation Loss Curves")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(save_path)
    plt.show()

    print(f"Saved plot to {save_path}")

    class_counts = list(train_history["dice_per_class"].values())
    class_0 = [count[0] for count in class_counts]
    class_1 = [count[1] for count in class_counts]
    class_2 = [count[2] for count in class_counts]

    epochs = range(len(train_losses))

    # Plot
    plt.figure(figsize=(10,6))
    plt.plot(epochs, class_0, label="class 0", marker="o")
    plt.plot(epochs, class_1, label="class 1", marker="o")
    plt.plot(epochs, class_2, label="class 2", marker="o")
    plt.xlabel("Epoch")
    plt.ylabel("accuracy")
    plt.title("Training and Validation Loss Curves")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(save_path.replace(".png", "_class_counts.png"))
    plt.show()

    print(f"Saved plot to {save_path}")




if __name__ == "__main__":
    # Paths to training and validation history
    train_history_path = os.path.join("checkpoints/train_history.json")
    val_history_path = os.path.join("checkpoints/validation_history.json")

    # Plot training curves
    plot_training_curves(train_history_path, val_history_path)