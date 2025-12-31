import torch
import torch.nn as nn
import lightning as L
from torchmetrics import Accuracy


class CachedCLIPClassifier(L.LightningModule):
    """
    Не содержит CLIP модель - только классификационную голову.
    """
    
    def __init__(self, hidden_size: int = 768, num_classes: int = 4, lr: float = 1e-3):
        """
            hidden_size: Размерность CLIP-признаков
            num_classes: Количество классов
            lr: Learning rate
        """
        super().__init__()
        self.save_hyperparameters()

        self.classifier = nn.Sequential(
            nn.Linear(hidden_size, 256),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(256, num_classes)
        )
        
        self.criterion = nn.CrossEntropyLoss()
        self.train_acc = Accuracy(task="multiclass", num_classes=num_classes)
        self.val_acc = Accuracy(task="multiclass", num_classes=num_classes)

    def forward(self, features):
        return self.classifier(features)

    def training_step(self, batch, batch_idx):
        features, labels = batch
        logits = self(features)
        loss = self.criterion(logits, labels)
        
        self.train_acc(logits, labels)
        self.log('train_loss', loss, on_step=True, on_epoch=True, prog_bar=True)
        self.log('train_acc', self.train_acc, on_step=False, on_epoch=True, prog_bar=True)
        return loss

    def validation_step(self, batch, batch_idx):
        features, labels = batch
        logits = self(features)
        loss = self.criterion(logits, labels)
        
        self.val_acc(logits, labels)
        self.log('val_loss', loss, prog_bar=True)
        self.log('val_acc', self.val_acc, prog_bar=True)

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=self.hparams.lr)
