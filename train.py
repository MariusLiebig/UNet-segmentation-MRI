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
import nibabel as nib
import numpy as np
import torch.nn.functional as F
import signal



from utils import(
    to_cuda,
    save_checkpoint,
    save_predictions_as_img,
)
from metric import (
    dice_coefficient,
    )

class Trainer:

    def __init__(self, batch_size, learning_rate ,epochs, model, dataloaders, loss_fn, optimizer,scaler, scheduler, inferer, early_stop_count = 5):
        """
            Initialize our trainer class.
        """
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.epochs = epochs    
        self.loss_fn = loss_fn
        self.model = model
        self.scheduler =scheduler
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
            accuracy=collections.OrderedDict(),
            loss_per_step=collections.OrderedDict(),
        )

        self.train_history = dict(
            loss=collections.OrderedDict(),
            accuracy=collections.OrderedDict(),
            pred_counts=collections.OrderedDict(),
            class_counts=collections.OrderedDict(),
        )
        self.inferer = inferer
        self.best_val_loss = float('inf')
        self.epochs_since_improvement = 0

    def train_batch(self):
        """
        Train the model for one epoch.
        """
        self.model.train()
        loop = tqdm(self.dataloader_train, leave=True)
        running_loss = 0
        running_correct = 0
        running_total = 0
        running_dice = 0

        num_batches = 0
        class_counts = torch.zeros(3, dtype=torch.long)
        pred_counts = torch.zeros(3, dtype=torch.long)  # <-- add this


        for img_batch, mask_batch in loop:
            img_batch, mask_batch = to_cuda(img_batch), to_cuda(mask_batch)
            # Comes when cropping around the tumor

            with autocast(device_type='cuda'):
                predictions = self.model(img_batch)
                preds = predictions.argmax(dim=1)
                pred_counts += torch.bincount(preds.flatten().cpu(), minlength=3)


                if mask_batch.ndim == 4 and mask_batch.shape[1] == 1:#For 2D
                    mask_batch = mask_batch.squeeze(1)

                if mask_batch.ndim == 5 and mask_batch.shape[1] == 1: #For 3D
                    mask_batch = mask_batch.squeeze(1)
                    
                mask_batch = mask_batch.long()

                counts = torch.bincount(mask_batch.flatten().cpu(), minlength=3)
                class_counts += counts


                loss = self.loss_fn(predictions, mask_batch)

 


            self.optimizer.zero_grad(set_to_none=True)
            self.scaler.scale(loss).backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            self.scaler.step(self.optimizer)
            self.scaler.update()
            self.scheduler.step()



            num_batches += 1
            running_loss += loss.item()

#------------------------------- Dice accuracy -----------------------------------
            dice_batch, _, _ = dice_coefficient([(img_batch, mask_batch)], self.model, inferer = self.inferer, num_classes=predictions.shape[1], device=img_batch.device)
            running_dice += dice_batch
#--------------------------------------------------------------------------------

            loop.set_postfix(loss=loss.item())

        print(f"Epoch class distribution: {class_counts.tolist()}")
        print(f"Predicted label counts: {pred_counts.tolist()}")


        avg_loss = running_loss / num_batches
        avg_dice = running_dice / num_batches

        return avg_loss, avg_dice



    
    def train(self):
        signal.signal(signal.SIGUSR1, self.handle_usr1)

        os.makedirs("checkpoints", exist_ok=True)
        os.makedirs("saved_nifti", exist_ok=True)
        for epoch in range(self.epochs):
            self.global_step += 1
            print(f"Epoch {epoch + 1}/{self.epochs}") #Epoch plus 1 because of 0 indexing

            avg_loss, avg_dice = self.train_batch()
            print(f"Train Dice: {avg_dice:.4f}")
            print(f"Train loss: {avg_loss:.4f}")
            self.train_history["accuracy"][self.global_step] = avg_dice
            self.train_history["loss"][self.global_step] = avg_loss
            val_dice, val_loss, dice_per_class = dice_coefficient(self.dataloader_val, self.model, inferer = self.inferer, num_classes=3, loss_fn=None)
            print(f"Validation Dice: {val_dice:.4f}")
            self.validation_history["accuracy"][self.global_step] = val_dice

            for i, d in enumerate(dice_per_class):
                print(f"  Class {i} Dice: {d:.4f}")
            self.train_history["dice_per_class"][self.global_step] = dice_per_class

            # print(f"Validation loss: {val_loss:.4f}")
            # self.validation_history["loss"][self.global_step] = val_loss
            self.validation_history["accuracy"][self.global_step] = val_dice

            lr = self.optimizer.param_groups[0]['lr']
            print(f" LR reduced?  new lr = {lr:.2e}")

            if (epoch + 1) % 10 == 0:
                self.save_checkpoint(epoch + 1)
            
                self.save_training_history()

    
    def early_stop(self, current_val_loss):
        """
        Early stopping: checks if validation loss improves, otherwise stops after patience epochs.
        """
        if current_val_loss < self.best_val_loss:
            self.best_val_loss = current_val_loss
            self.epochs_since_improvement = 0
            print(f"Validation loss improved to {current_val_loss:.6f}")
            return False
        else:
            self.epochs_since_improvement += 1
            print(f"No improvement for {self.epochs_since_improvement}/{self.early_stop_count} epochs.")

            if self.epochs_since_improvement >= self.early_stop_count:
                print(f"Early stopping triggered. Best validation loss: {self.best_val_loss:.6f}")
                return True

            return False




    def save_checkpoint(self, epoch):
        save_dict = {
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict(),
            'scaler_state_dict': self.scaler.state_dict(),
            'train_history': self.train_history,
            'validation_history': self.validation_history,
            'time': time.time() - self.start_time,
        }
        torch.save(save_dict, f"checkpoints/checkpoint_epoch_{epoch}.pth")
        print(f"Checkpoint saved at epoch {epoch}.")


    def save_training_history(self):
        os.makedirs("checkpoints", exist_ok=True)
        with open("checkpoints/train_history.json", "w") as f:
            json.dump(self.train_history, f)
        with open("checkpoints/validation_history.json", "w") as f:
            json.dump(self.validation_history, f)
        print("Training and validation history saved.")

    
    def handle_usr1(self, signum, frame):
        print("Received SIGUSR1 — saving checkpoint.")
        self.save_checkpoint("checkpoints/manual_checkpoint.pth")  # adapt as needed

    # Register the handler at the start of the script