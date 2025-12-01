"""
Modulo per addestrare una rete Siamese con PyTorch Lightning.
Include:
- Dataset Siamese per generare coppie di immagini
- FeatureExtractor (CNN)
- Modello Lightning con Contrastive Loss
- Training loop con checkpoint automatici
"""

#  TRAINING SIAMESE NETWORK 
#  Versione per tesi triennale - codice chiaro e commentato
#  - Eliminato uso GPU
#  - Aggiunta training loop con Lightning
#  - Contrastive Loss "classica" con margin

import os
import time
import random
import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint


#  VARIABILI GLOBALI
CARTELLA_TRAIN = r"Dataset\Train"   #  train
#CARTELLA_VAL   = "archiveemotions_test"     # validation

BATCH_SIZE = 64
NUM_WORKERS = 4
dimensione_immagine = (224, 224)
EPOCHS = 10
LEARNING_RATE = 1e-3
MARGINE = 2.0

#   ottimizzazione
OPTIMIZER_CLS = torch.optim.Adam  # classe di ottimizzazione (SGD, Adam, AdamW, RMSprop...)
OPTIMIZER_KWARGS = {
    # "momentum": 0.9, # per SGD
    "weight_decay": 1e-4,   # SGD, Adam, AdamW, RMSprop...
    "betas": (0.9, 0.999),  # valori di default di Adam
    "eps": 1e-8              # valore di stabilità numerica
}
# 1) Dataset Siamese
# Ogni elemento è una coppia (img1, img2, label)
# label = 0 (stessa classe) oppure 1 (classi diverse)
# Funziona con un dataset personalizzato di immagini. Ogni sottocartella = una classe.
class SiameseDataset(Dataset):

    def __init__(self, root_dir):
        self.root_dir = root_dir

        # Lista delle classi (sottocartelle)
        self.class_folders = sorted(
            [os.path.join(root_dir, d) for d in os.listdir(root_dir)
             if os.path.isdir(os.path.join(root_dir, d))]
        )

        # Dizionario: per ogni classe trova la lista immagini
        self.images_by_class = {
            cls: [os.path.join(cls, f) for f in os.listdir(cls)
                  if f.lower().endswith((".jpg", ".jpeg", ".png"))]
            for cls in self.class_folders
        }

        # Trasformazioni immagini 
        self.transform = transforms.Compose([
            transforms.Resize(dimensione_immagine),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5,0.5,0.5], std=[0.5,0.5,0.5])
        ])

    def __len__(self):
        return 20000  # numero virtuale di coppie

    
    def __getitem__(self, idx):
        #  genera una coppia di immagini randomicamente (coppie sempre nuove)
        # ogni epoca sempre coppie  diverse
        coppia_positiva = random.random() < 0.5

        if coppia_positiva:
            class_path = random.choice(self.class_folders)
            img_list = self.images_by_class[class_path]
            img1, img2 = random.sample(img_list, 2)
            label = 0
        else:
            cls1, cls2 = random.sample(self.class_folders, 2)
            img1 = random.choice(self.images_by_class[cls1])
            img2 = random.choice(self.images_by_class[cls2])
            label = 1

        img1 = Image.open(img1).convert("RGB")
        img2 = Image.open(img2).convert("RGB")

        return self.transform(img1), self.transform(img2), torch.tensor(label, dtype=torch.float32)


# 2) Feature extractor (CNN)
# Estrae un vettore di embedding normalizzato da un’immagine
# accetta tre  canali RGB in input
# ha 3 blocchi di convoluzione + 2 layer fully connected

class FeatureExtractor(nn.Module):

    def __init__(self):
        super().__init__()
        # BatchNorm lo iseriamo? si, datset molto eterogeneo e grande

        self.conv = nn.Sequential(
            nn.Conv2d(3, 16, 5), #in_channels=3 numero canali, out_channels=16 numero di filtri convoluzionali Più filtri → più capacità di rappresentazione, ma più parametri, kernel_size=5 dimensione del filtro 5×5
            nn.BatchNorm2d(16), # Deve avere lo stesso numero di canali di output della conv precedente
            nn.ReLU(),
            nn.MaxPool2d(2), #riduce la risoluzione di 2×2 (stride = 2 di default)

            nn.Conv2d(16, 32, 5), #in_channels = numero di feature map in ingresso (output della conv precedente) e cosi via 
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(2),

            nn.Conv2d(32, 64, 5),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(2),
        )

        # Calcolo automatico del flatten size
        with torch.no_grad():
            dummy = torch.zeros(1, 3, *dimensione_immagine)
            self.flatten_dim = self.conv(dummy).view(1, -1).size(1)

        self.fc = nn.Sequential(
            nn.Flatten(), #Serve sapere quante unità arrivano al Linear.
            nn.Linear(self.flatten_dim, 256), #in_features = canali_finali * altezza * larghezza
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Linear(256, 64)
        )

    def forward(self, x):
        x = self.conv(x)
        x = self.fc(x)
        return nn.functional.normalize(x, p=2, dim=1)


# 3) Loss contrastiva

class ContrastiveLoss(nn.Module):
    def __init__(self, margin=2.0):
        super().__init__()
        self.margin = margin
    # Contrastive Loss: L=(1−y)d^2+ymax(0,m−d)^2
    def forward(self, out1, out2, label):
        distanza = torch.norm(out1 - out2, p=2, dim=1)
        loss_simili = (1 - label) * distanza**2
        loss_diversi = label * torch.clamp(self.margin - distanza, min=0.0)**2
        return torch.mean(loss_simili + loss_diversi)

#4) Modello Siamese

class SiameseNetwork(pl.LightningModule):
    def __init__(self, embedding_net, 
                 loss_fn=None,
                 optimizer_cls=torch.optim.Adam,
                 optimizer_kwargs=None,
                 lr=LEARNING_RATE,
                 margin=MARGINE):
        super().__init__()
        self.save_hyperparameters()
        self.embedding_net = embedding_net
        self.criterion = loss_fn if loss_fn is not None else ContrastiveLoss(margin)
        self.lr = lr
        self.optimizer_cls = optimizer_cls
        self.optimizer_kwargs = optimizer_kwargs or {}

    def forward(self, x):
        return self.embedding_net(x)

    def training_step(self, batch, batch_idx):
        start_time = time.time()  # inizio temporizzazione
        img1, img2, label = batch
        emb1 = self.forward(img1)
        emb2 = self.forward(img2)
        loss = self.criterion(emb1, emb2, label)

        batch_time = time.time() - start_time  # calcolo tempo batch

        # log della loss e del tempo batch
        self.log("train_loss", loss)
        self.log("batch_time", batch_time, prog_bar=True)

        return loss

    def validation_step(self, batch, batch_idx):
        img1, img2, label = batch
        emb1 = self.forward(img1)
        emb2 = self.forward(img2)
        loss = self.criterion(emb1, emb2, label)
        self.log("val_loss", loss, prog_bar=True)

    def configure_optimizers(self):
        return self.optimizer_cls(self.parameters(), lr=self.lr, **self.optimizer_kwargs)



# 5) TRAINING

if __name__ == "__main__":

    dataset = SiameseDataset(CARTELLA_TRAIN)
    loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=NUM_WORKERS, persistent_workers=(NUM_WORKERS>0))

    #val_ds = SiameseDataset(CARTELLA_VAL)
    #val_loader = DataLoader(
    #    val_ds, batch_size=BATCH_SIZE, shuffle=False,
    #    num_workers=NUM_WORKERS, persistent_workers=(NUM_WORKERS>0)
    #)

     

    model = SiameseNetwork(
        embedding_net=FeatureExtractor(),
        loss_fn=ContrastiveLoss(margin=MARGINE),
        optimizer_cls=OPTIMIZER_CLS,
        optimizer_kwargs=OPTIMIZER_KWARGS,
        lr=LEARNING_RATE,
        margin=MARGINE)
    

    checkpoint = ModelCheckpoint(
        dirpath="checkpoints",
        filename="snet_siamese",
        save_top_k=1,
        # monitor="val_loss",   # ora monitoriamo la metrica giusta
        monitor="train_loss",
        mode="min"
    )

    trainer = pl.Trainer(
        max_epochs=EPOCHS,
        accelerator="cpu",
        callbacks=[checkpoint],
        log_every_n_steps=10
    )

     # Addestramento completo con validazione
    #trainer.fit(model, loader, val_loader)
    trainer.fit(model, loader)