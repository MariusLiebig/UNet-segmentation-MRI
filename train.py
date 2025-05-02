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

    def __init__(self, batch_size, learning_rate ,epochs, model, dataloaders, loss_fn, optimizer,scaler, keep_background_fraction, early_stop_count = 3, checkpoint_path=None): 
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
        self.patience_counter = 0

        self.validation_history = dict(
            loss=collections.OrderedDict(),
            accuracy=collections.OrderedDict(),
            dice_per_class=collections.OrderedDict()

        )

        self.train_history = dict(
            loss=collections.OrderedDict(),
            accuracy=collections.OrderedDict(),
            time=collections.OrderedDict(),
        )

        self.best_checkpoints = []  # list of tuples: (val_accuracy, checkpoint_path)
        self.max_saved_checkpoints = 1
        self.best_val_loss = float("inf")
        self.keep_background_fraction = keep_background_fraction

        if checkpoint_path is not None:
            self.load_checkpoint(checkpoint_path, learning_rate=learning_rate)
            print(f"Checkpoint loaded from {checkpoint_path}")

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
        self.save_training_history()

        for epoch in range(self.epochs):
            print(f"Epoch {epoch + 1}/{self.epochs}") #Epoch plus 1 because of 0 indexing
            intermidiate_time = time.time()
            avg_loss = self.train_batch()
            lr = self.optimizer.param_groups[0]['lr']
            print(f" LR reduced?  new lr = {lr:.2e}")

            print(f"Train loss: {avg_loss:.4f}")
            self.train_history["loss"][epoch] = avg_loss
            avg_dice, avg_loss, dice_per_class = dice_coefficient(self.dataloader_val, self.model, loss_fn=self.loss_fn)
            print(f"Validation loss: {avg_loss:.4f}")
            print(f"Validation accuracy: {avg_dice:.4f}")
            print(f"Dice per class: {dice_per_class}")
            self.validation_history["loss"][epoch] = avg_loss
            self.validation_history["accuracy"][epoch] = avg_dice
            self.validation_history["dice_per_class"][epoch] = dice_per_class
            self.train_history["time"][epoch] = time.time() - intermidiate_time

            if self.early_stop(avg_loss):
                break
            save_predictions_as_img(self.dataloader_train, self.model, epoch, folder="saved_images/")
            self.save_checkpoint(epoch, avg_loss)

            self.scheduler.step(avg_loss) #ReduceLROnPlateau

            self.global_step += 1



        self.save_training_history()
        
    #Putting in UTILS?

    
    def early_stop(self, val_loss):
        """
        Check if validation loss has not improved in the last `early_stop_count` epochs.
        """
        


        # Get the last N + 1 losses
        best_loss = self.best_val_loss
        print(f"Relevant losses for early stopping: {best_loss}")

        #Give it three epochs to improve
        if val_loss>= best_loss:
            self.patience_counter += 1
        else:
            self.patience_counter = 0
        print(f"Patience counter: {self.patience_counter}/{self.early_stop_count}")

        if self.patience_counter > self.early_stop_count:
            print("Early stopping triggered.")
            return True
        return False


    def save_checkpoint(self, epoch, val_loss):
        # Create checkpoints directory if it doesn't exist
        os.makedirs("checkpoints", exist_ok=True)

        if val_loss < self.best_val_loss:
            self.best_val_loss = val_loss

            save_path = f"checkpoints/best_checkpoint_{self.keep_background_fraction}_{epoch}.pth"
            save_dict = {
                'epoch': self.global_step,
                'model_state_dict': self.model.state_dict(),
                'optimizer_state_dict': self.optimizer.state_dict(),
                'scaler_state_dict': self.scaler.state_dict(),
                'train_history': self.train_history,
                'validation_history': self.validation_history,
                'time': time.time() - self.start_time,
                'val_loss': val_loss
            }

            torch.save(save_dict, save_path)
            print(f"Best checkpoint saved at epoch {epoch} with val_acc={val_loss:.4f}.")

            if hasattr(self, "last_best_checkpoint") and os.path.exists(self.last_best_checkpoint):
                os.remove(self.last_best_checkpoint)
                print(f"Removed previous checkpoint: {self.last_best_checkpoint}")

            self.last_best_checkpoint = save_path





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
        self.global_step = checkpoint.get('epoch', 4)

        self.best_val_loss = float("inf")

        if learning_rate is not None:
            for param_group in self.optimizer.param_groups:
                param_group['lr'] = learning_rate

