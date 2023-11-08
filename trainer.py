import os
# os.environ["CUDA_VISIBLE_DEVICES"] = 2
# os.environ['OPENBLAS_NUM_THREADS'] = '1'

import torch
torch.cuda.empty_cache()
import torch.nn as nn
from torch.utils.data import DataLoader, random_split
# Load dataset
from dataset import PTDataset
from unet_3D import UNet3D

import logging, traceback
import matplotlib.pyplot as plt
import gc

class Trainer:
    def __init__(self, n_epochs=None, batch_size=None, lr=None, task_num=None, num_classes=None, validation_split=0.2):
        task_num = task_num if "TASK_NUM" not in os.environ else int(os.environ["TASK_NUM"])
        
        dict_task_num2dataset = {
            1: "BrainTumour",
            2: "Heart",
            6: "Lung"
        }
        dataset = dict_task_num2dataset[task_num]
        data_dir = f"Task{task_num:02d}_{dataset}"
        folderpath_model = os.path.join("models", f"task{task_num:02d}")
        
        dict_task_num2data_prefix = {
            1: "BRATS",
            2: "la",
            6: "lung"
        }
        filename_prefix = dict_task_num2data_prefix[task_num]
        
        self.n_epochs = n_epochs if "N_EPOCHS" not in os.environ else int(os.environ["N_EPOCHS"])
        self.batch_size = batch_size if "BATCH_SIZE" not in os.environ else int(os.environ["BATCH_SIZE"])
        self.lr = lr if "LR" not in os.environ else float(os.environ["LR"])
        num_classes = num_classes if "NUM_CLASSES" not in os.environ else int(os.environ["NUM_CLASSES"])
        
        self.folderpath_model = folderpath_model
        
        # Initiate logging
        self.init_logging()
        logging.info(f"Task number: {task_num}, dataset: {dict_task_num2dataset[task_num]}")
        logging.info(f"Beginning training. Hyperparameters: n_epochs: {self.n_epochs}, batch_size: {self.batch_size}, lr: {self.lr}, data_dir: {data_dir}, num_classes: {num_classes}, validation_split: {validation_split}")
        
        # Set hyperparameters
        is_cuda_available = torch.cuda.is_available()
        logging.info(f"Is CUDA available? {'Yes' if is_cuda_available else 'No'}")
        cuda_visible_devices = os.environ.get("CUDA_VISIBLE_DEVICES")
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        # Initialize model
        self.model = UNet3D(in_channels=1, num_classes=num_classes)
        self.model.to(self.device)
        
        # Define loss and optimizer
        self.criterion = nn.CrossEntropyLoss()
        # self.criterion = nn.
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=self.lr)
        
        # Create Pytorch dataset and dataloader
        dataset_full = PTDataset(data_dir, filename_prefix=filename_prefix)
        # Split dataset into training and validation
        n_samples = len(dataset_full)
        n_val_samples = int(validation_split * n_samples)
        n_train_samples = n_samples - n_val_samples
        self.train_dataset, self.val_dataset = random_split(dataset_full, [n_train_samples, n_val_samples])
        
        self.train_loader = DataLoader(self.train_dataset, batch_size=self.batch_size, shuffle=True)
        self.val_loader = DataLoader(self.val_dataset, batch_size=self.batch_size, shuffle=False)
        self.train_losses = []
        self.val_losses = []
    
    def execute(self):
        try:
            for epoch in range(self.n_epochs):
                self.model.train()
                total_loss = 0.0
                
                # Training loop
                for i, batch in enumerate(self.train_loader):
                    logging.info(f"Batch number: {i+1}/{len(self.train_loader)}")
                    inputs, labels = batch
                    inputs = inputs.to(self.device)
                    labels = labels.to(self.device, dtype=torch.long)
                    self.optimizer.zero_grad()
                    with torch.autocast(device_type="cuda"):
                        outputs = self.model(inputs)
                        loss = self.criterion(outputs, labels)
                    gc.collect()
                    loss.backward()
                    self.optimizer.step()
                    total_loss += loss.item()
                
                average_loss = total_loss / (i + 1)
                logging.info(f"Epoch [{epoch + 1}/{self.n_epochs}] - Training Loss: {average_loss:.4f}")
                self.train_losses.append(average_loss)

                # Validation loop
                self.model.eval()
                total_val_loss = 0.0

                with torch.no_grad():
                    for i, batch in enumerate(self.val_loader):
                        inputs, labels = batch
                        inputs = inputs.to(self.device)
                        labels = labels.to(self.device, dtype=torch.long)
                        val_outputs = self.model(inputs)
                        val_loss = self.criterion(val_outputs, labels)
                        total_val_loss += val_loss.item()

                average_val_loss = total_val_loss / (i + 1)
                logging.info(f"Epoch [{epoch + 1}/{self.n_epochs}] - Validation Loss: {average_val_loss:.4f}")
                self.val_losses.append(average_val_loss)
            
                # Plot and save the training and validation loss
                self.plot_progress(os.path.join(self.folderpath_model, "progress_plot.png"))
                logging.info(f"Progress plot saved to {os.path.join(self.folderpath_model, 'progress_plot.png')}")

            # Save the model after all epochs
            torch.save(self.model.state_dict(), f"{self.folderpath_model}/model_final.pt")
            logging.info(f"Model saved to {self.folderpath_model}")
        except Exception:
            logging.error(traceback.format_exc())
            
    def plot_progress(self, plot_filename):
        epochs = list(range(1, len(self.train_losses) + 1))
        plt.figure()
        plt.plot(epochs, self.train_losses, label='Training Loss')
        plt.plot(epochs, self.val_losses, label='Validation Loss')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.title('Training and Validation Loss Progress')
        plt.legend()
        plt.savefig(plot_filename)
        plt.close()
        
    def init_logging(self):
        log_format = "%(asctime)s [%(levelname)s]: %(message)s"
        logging.basicConfig(filename=os.path.join(self.folderpath_model, "training.log"), level=logging.INFO, format=log_format)
        console = logging.StreamHandler()
        console.setLevel(logging.INFO)
        formatter = logging.Formatter(log_format)
        console.setFormatter(formatter)
        logging.getLogger("").addHandler(console)
        
if __name__ == "__main__":
    # Create and run the trainer
    trainer = Trainer()
    trainer.execute()