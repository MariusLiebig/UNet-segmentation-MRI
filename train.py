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
import heapq
import os
from torch.optim.lr_scheduler import ReduceLROnPlateau

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

    def __init__(self, batch_size, learning_rate ,epochs, model, dataloaders, loss_fn, optimizer,scaler, early_stop_count = 1):
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
        self.scheduler = ReduceLROnPlateau(
            optimizer=self.optimizer,
            mode='min',               # minimize val_loss
            factor=0.5,               # reduce LR by half
            patience=3,               # wait for 3 epochs with no improvement
            verbose=True              # print when LR is reduced
        )

        self.best_loss = float("inf")
        self.num_steps_per_val = len(self.dataloader_train) // 10
        self.global_step = 0
        self.start_time = time.time()

        self.early_stop_count = early_stop_count 

        self.validation_history = dict(
            loss=collections.OrderedDict(),
            accuracy=collections.OrderedDict(),
            dice_per_class=collections.OrderedDict()

        )

        self.train_history = dict(
            loss=collections.OrderedDict(),
            accuracy=collections.OrderedDict(),
        )

        self.best_checkpoints = []  # list of tuples: (val_accuracy, checkpoint_path)
        self.max_saved_checkpoints = 1

    def train_batch(self):
        """
            Train the model for one epoch.
        """
        loop = tqdm(self.dataloader_train, leave=True)
        running_loss = 0
        num_batches = 0
        for img_batch, mask_batch in loop:
            img_batch, mask_batch= to_cuda(img_batch), to_cuda(mask_batch)
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

            running_loss += loss.detach()
            num_batches += 1
            loop.set_postfix(loss=loss.item())



        avg_loss = running_loss.sum().item() / num_batches
        return avg_loss

            #Adding learning rate scheduler?

    
    def train(self):
        
        for epoch in range(self.epochs):
            print(f"Epoch {epoch + 1}/{self.epochs}") #Epoch plus 1 because of 0 indexing
            intermidiate_time = time.time()
            avg_loss = self.train_batch()
            print(f"Train loss: {avg_loss:.4f}")
            self.train_history["loss"][epoch] = avg_loss


            avg_dice, avg_loss, dice_per_class = dice_coefficient(self.dataloader_val, self.model, loss_fn=self.loss_fn)
            print(f"Validation loss: {avg_loss:.4f}")
            print(f"Validation accuracy: {avg_dice:.4f}")
            print(f"Dice per class: {dice_per_class}")
            self.validation_history["loss"][epoch] = avg_loss
            self.validation_history["accuracy"][epoch] = avg_dice
            self.validation_history["dice_per_class"][epoch] = dice_per_class



            save_predictions_as_img(self.dataloader_train, self.model, epoch, folder="saved_images/")
            # if epoch < 4 or (epoch + 1) % 5 == 0:
            self.save_checkpoint(epoch, avg_dice)


            self.scheduler.step(avg_loss) #ReduceLROnPlateau

            
            lr = self.optimizer.param_groups[0]['lr']
            print(f" LR reduced?  new lr = {lr:.2e}")

            if self.early_stop():
                break
        self.save_training_history()
        
    #Putting in UTILS?

    
    def early_stop(self):
        """
        Check if validation loss has not improved in the last `early_stop_count` epochs.
        """
        val_loss = self.validation_history["loss"]
        
        # Not enough epochs to consider early stopping
        if len(val_loss) < self.early_stop_count + 1:
            return False

        # Get the last N + 1 losses
        relevant_losses = list(val_loss.values())[-(self.early_stop_count + 1):]
        print(f"Relevant losses for early stopping: {relevant_losses}")
        best_loss = min(relevant_losses[:-1])  # best before last

        if relevant_losses[-1] >= best_loss:
            print("Early stopping triggered.")
            return True
        return False


    def save_checkpoint(self, epoch, val_accuracy):
        save_path = f"checkpoints/checkpoint.pth"
        save_dict = {
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scaler_state_dict': self.scaler.state_dict(),
            'train_history': self.train_history,
            'validation_history': self.validation_history,
            'time': time.time() - self.start_time,
            'val_accuracy': val_accuracy
        }

        # Always save current checkpoint
        torch.save(save_dict, save_path)
        print(f"Checkpoint saved at epoch {epoch} with val_acc={val_accuracy:.4f}.")

        # Store new entry and keep top 3 by accuracy
        self.best_checkpoints.append((val_accuracy, save_path))
        self.best_checkpoints.sort(reverse=True, key=lambda x: x[0])

        if len(self.best_checkpoints) > self.max_saved_checkpoints:
            # Remove worst one
            _, to_remove = self.best_checkpoints.pop()
            if os.path.exists(to_remove):
                os.remove(to_remove)
                print(f"Removed old checkpoint")



    def save_training_history(self):
        os.makedirs("checkpoints", exist_ok=True)
        with open("checkpoints/train_history.json", "w") as f:
            json.dump(self.train_history, f)
        with open("checkpoints/validation_history.json", "w") as f:
            json.dump(self.validation_history, f)
        print("Training and validation history saved.")

    def load_checkpoint(self, path, learning_rate=None):
        checkpoint = torch.load(path, map_location="cuda")

        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])

        self.validation_history = checkpoint.get('validation_history', {})
        self.train_history = checkpoint.get('train_history', {})
        self.global_step = checkpoint.get('epoch', 0)

        if learning_rate is not None:
            for param_group in self.optimizer.param_groups:
                param_group['lr'] = learning_rate

