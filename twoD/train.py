import torch
import torch.nn as nn
from torch.amp import autocast
from albumentations.pytorch import ToTensorV2
import time
import torch.optim as optim
import collections
import os
import json
import gc
from tqdm import tqdm

from utils import(
    to_cuda,
    save_checkpoint,
    save_predictions_as_img,
    save_predictions_as_img_3d
)
from metric import (
    dice_coefficient,
    )

class Trainer:

    def __init__(self, batch_size, learning_rate ,epochs, model, dataloaders, loss_fn, optimizer,scaler, early_stop_count = 5):
        """
            Initialize our trainer class.
        """
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.epochs = epochs    
        self.loss_fn = loss_fn
        self.model = model
        self.model = to_cuda(self.model)
        self.optimizer = optimizer
        self.dataloader_train, self.dataloader_val = dataloaders
        self.scaler = scaler

        self.best_loss = float("inf")
        self.num_steps_per_val = len(self.dataloader_train) // 10
        self.global_step = 0
        self.start_time = time.time()

        self.early_stop_count = early_stop_count 

        self.validation_history = dict(
            loss=collections.OrderedDict(),
            accuracy=collections.OrderedDict()
        )

        self.train_history = dict(
            loss=collections.OrderedDict(),
            accuracy=collections.OrderedDict()

        )

    def train_batch(self):
        """
            Train the model for one epoch.
        """
        loop = tqdm(self.dataloader_train, leave=True)
        running_loss = 0
        num_batches = 0
        for img_batch, mask_batch in loop:
            img_batch, mask_batch= img_batch.to("cuda"), mask_batch.to("cuda")
            #predictions = self.model(img_batch)
            #loss = self.loss_fn(predictions, mask_batch)
            #loss.backward()
            #self.optimizer.step()

            # Faster training on GPU
            with autocast(device_type='cuda'): #Reduces floating point precision to 16 bits when its ok
                predictions = self.model(img_batch)
                if mask_batch.ndim == 4 and mask_batch.shape[1] == 1:
                    mask_batch = mask_batch.squeeze(1)
                mask_batch = mask_batch.long()

                loss = self.loss_fn(predictions, mask_batch)

            self.optimizer.zero_grad(set_to_none=True) #set_to_none=True can save a bit of memory
            self.scaler.scale(loss).backward()
            # Clip gradients
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            self.scaler.step(self.optimizer)
            self.scaler.update()

            running_loss += loss.item()
            num_batches += 1
            loop.set_postfix(loss=loss.item())



        avg_loss = running_loss / num_batches
        return avg_loss

            #Adding learning rate scheduler?

    
    def train(self):
        
        for epoch in range(self.epochs):
            print(f"Epoch {epoch + 1}/{self.epochs}") #Epoch plus 1 because of 0 indexing
            intermidiate_time = time.time()
            avg_loss = self.train_batch()
            print(f"Train loss: {avg_loss:.4f}")
            dice_coefficient(self.dataloader_val, self.model)
            save_predictions_as_img(self.dataloader_train, self.model, folder="saved_images/")
            if (epoch +1) % 5 == 0:
                self.save_checkpoint(epoch+1)

    #Putting in UTILS?

    
    def early_stop(self):
        """
            Checks if validation loss doesn't improve over early_stop_count epochs.
        """
        # Check if we have more than early_stop_count elements in our validation_loss list.
        val_loss = self.validation_history["loss"]
        if len(val_loss) < self.early_stop_count:
            return False
        # We only care about the last [early_stop_count] losses.
        relevant_loss = list(val_loss.values())[-self.early_stop_count:]
        first_loss = relevant_loss[0]
        if first_loss == min(relevant_loss):
            print("Early stop criteria met")
            return True
        return False
    
    def save_checkpoint(self, epoch):
        save_dict = {
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'train_history': self.train_history,
            'validation_history': self.validation_history,
            'time': time.time() - self.start_time,
        }
        torch.save(save_dict, f"checkpoints/checkpoint_epoch_{epoch}_2D.pth")
        print(f"Checkpoint saved at epoch {epoch}.")

    def save_history(self, filename="train_history.json"):
        os.makedirs(os.path.dirname(filename) or ".", exist_ok=True)

        serializable = {}
        for phase in ("train", "validation"):
            hist = getattr(self, f"{phase}_history")
            serializable[phase] = {}

            for metric, metric_dict in hist.items():
                serializable[phase][metric] = {
                    str(k): float(v.item() if isinstance(v, torch.Tensor) else v)
                    for k, v in metric_dict.items()
                }

        with open(filename, "w") as f:
            json.dump(serializable, f, indent=4)
        print(f"=> Saved history to {filename}")


