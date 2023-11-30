import torch
import seaborn as sns
import matplotlib.pyplot as plt
from typing import List, Optional, Any
from torch import nn
from torch.utils.data import DataLoader
from IPython.display import clear_output
from tqdm.notebook import tqdm
from model import LanguageModel
import numpy as np

sns.set_style('whitegrid')
plt.rcParams.update({'font.size': 15})


def plot_losses(train_losses: List[float], val_losses: List[float]):

    clear_output()
    fig, axs = plt.subplots(1, 2, figsize=(13, 4))
    axs[0].plot(range(1, len(train_losses) + 1), train_losses, label='train')
    axs[0].plot(range(1, len(val_losses) + 1), val_losses, label='val')
    axs[0].set_ylabel('loss')

    train_perplexities, val_perplexities = [], []
    train_perplexities = np.power([2]*len(train_losses), train_losses)
    val_perplexities = np.power([2]*len(val_losses), val_losses)

    axs[1].plot(range(1, len(train_perplexities) + 1), train_perplexities, label='train')
    axs[1].plot(range(1, len(val_perplexities) + 1), val_perplexities, label='val')
    axs[1].set_ylabel('perplexity')

    for ax in axs:
        ax.set_xlabel('epoch')
        ax.legend()

    plt.show()


def training_epoch(model: LanguageModel, optimizer: torch.optim.Optimizer, criterion: nn.Module,
                   loader: DataLoader, tqdm_desc: str, scheduler):

    device = next(model.parameters()).device
    train_loss = 0.0

    model.train()
    for indices, lengths in tqdm(loader, desc=tqdm_desc):
        
        optimizer.zero_grad()
        indices = indices[:,:lengths.max()].to(device)
        logits = model(indices[:, :-1], lengths)
        
        loss = criterion(logits.transpose(1,2), indices[:, 1:])
        
        
        loss.backward()

        optimizer.step()

        train_loss+=loss.item()      
            
    train_loss /= len(loader.dataset)
    return train_loss*loader.batch_size


@torch.no_grad()
def validation_epoch(model: LanguageModel, criterion: nn.Module,
                     loader: DataLoader, tqdm_desc: str):

    device = next(model.parameters()).device
    val_loss = 0.0

    model.eval()
    for indices, lengths in tqdm(loader, desc=tqdm_desc):
        
        tokens = indices[:,:lengths.max()].to(device)
        logits = model(tokens[:, :-1], lengths)
        
        loss = criterion(logits.transpose(1,2), tokens[:, 1:])
        val_loss += loss.item()

    val_loss /= len(loader.dataset)
    return val_loss*loader.batch_size


def train(model: LanguageModel, optimizer: torch.optim.Optimizer, scheduler: Optional[Any],
          train_loader: DataLoader, val_loader: DataLoader, num_epochs: int, num_examples=2, temp=1):

    train_losses, val_losses = [], []
    criterion = nn.CrossEntropyLoss(ignore_index=train_loader.dataset.pad_id)

    for epoch in range(1, num_epochs + 1):
        train_loss = training_epoch(
            model, optimizer, criterion, train_loader,
            tqdm_desc=f'Training {epoch}/{num_epochs}', scheduler=scheduler
        )
        val_loss = validation_epoch(
            model, criterion, val_loader,
            tqdm_desc=f'Validating {epoch}/{num_epochs}'
        )

        if scheduler:
            scheduler.step()

        train_losses += [train_loss]
        val_losses += [val_loss]
        plot_losses(train_losses, val_losses)

        print('Generation examples:')
        for _ in range(1):
            print(model.inference(temp=temp))
            print()
        for prefix in [ 'купил мужик шляпу,', 'сел медведь в машину и', 'подумал штирлиц']:

            generated = model.inference(prefix, temp=1)
            print(generated)
            print()