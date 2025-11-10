# siamese_train_lucas.py
# Addestramento di una rete Siamese in PyTorch

import os
import random
import time
from PIL import Image
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as T

# CONFIGURAZIONE

# Impostiamo i parametri principali dell’esperimento.
dataset_root = "dataset_lucas_usda_resized"     # cartella con le immagini organizzate per classe
selected_classes = ["clay", "loam", "sand", "silt"]  # classi usate in questo test
image_size = (300, 300)     # dimensione uniforme delle immagini
batch_size = 64             # numero di coppie per batch
epochs = 5                  # quante volte il modello vede tutto il dataset
learning_rate = 1e-3        # passo di aggiornamento dell’ottimizzatore
margin = 1.0                # margine per la contrastive loss

print("Addestramento su CPU.\n")

# TRASFORMAZIONI
# Ridimensioniamo le immagini e le convertiamo in tensori normalizzati [0,1].
transform = T.Compose([
    T.Resize(image_size),
    T.ToTensor()
])


# CREAZIONE DEL DATASET DI COPPIE

# Ogni elemento del dataset è una coppia (img1, img2, label)
# label = 0 (stessa classe) oppure 1 (classi diverse)
class SiameseDataset(Dataset):
    def __init__(self, root, classes, transform=None):
        self.root = root
        self.classes = classes
        self.transform = transform
        self.class_to_images = {}

        # nome_classe -> lista di immagini in quella cartella
        for cls in self.classes:
            path = os.path.join(root, cls)
            imgs = [os.path.join(path, f) for f in os.listdir(path) if f.lower().endswith('.jpg')]
            if imgs:
                self.class_to_images[cls] = imgs

        # Lista piatta con tutte le immagini e la loro classe
        self.all_images = [(cls, img) for cls in self.class_to_images for img in self.class_to_images[cls]]

    def __len__(self):
        return len(self.all_images)

    def __getitem__(self, idx):
        cls1, img1_path = self.all_images[idx]

        # 50% coppia positiva, 50% coppia negativa
        if random.random() < 0.5:
            cls2 = cls1
            label = 0.0
        else:
            cls2 = random.choice([c for c in self.class_to_images if c != cls1])
            label = 1.0

        img2_path = random.choice(self.class_to_images[cls2])

        img1 = Image.open(img1_path).convert("RGB")
        img2 = Image.open(img2_path).convert("RGB")

        if self.transform:
            img1 = self.transform(img1)
            img2 = self.transform(img2)

        return img1, img2, torch.tensor(label, dtype=torch.float32)


# DEFINIZIONE DELLA RETE SIAMESE

# La rete è formata da due rami identici che condividono gli stessi pesi (embedding network).
class EmbeddingNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(3, 8, kernel_size=3, padding=1),
            nn.BatchNorm2d(8),
            nn.ReLU(),
            nn.MaxPool2d(2),

            nn.Conv2d(8, 16, kernel_size=3, padding=1),
            nn.BatchNorm2d(16),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d(1)
        )
        self.fc = nn.Linear(16, 64)

    def forward(self, x):
        x = self.conv(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return nn.functional.normalize(x, p=2, dim=1)

# La rete Siamese applica l'embedding alle due immagini in parallelo.
class SiameseNet(nn.Module):
    def __init__(self, embedding_net):
        super().__init__()
        self.embedding_net = embedding_net

    def forward(self, x1, x2):
        z1 = self.embedding_net(x1)
        z2 = self.embedding_net(x2)
        return z1, z2


# CONTRASTIVE LOSS

# Funzione di perdita usata per reti Siamese: penalizza distanze piccole per coppie diverse
# e distanze grandi per coppie uguali.
def contrastive_loss(z1, z2, label, margin=1.0):
    d = torch.norm(z1 - z2, p=2, dim=1)
    loss = torch.mean((1 - label) * d.pow(2) + label * torch.clamp(margin - d, min=0).pow(2))
    return loss


# PREPARAZIONE DEL DATASET E DEL MODELLO

dataset = SiameseDataset(dataset_root, selected_classes, transform)
train_loader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=0)

model = SiameseNet(EmbeddingNet())  # nessun .to(device), rimane su CPU
optimizer = optim.Adam(model.parameters(), lr=learning_rate)

print(f"Numero di immagini totali: {len(dataset)}")
print(f"Batch totali per epoca: {len(train_loader)}\n")


# TRAINING LOOP

# Per ogni epoca: passiamo su tutti i batch, calcoliamo loss e aggiorniamo i pesi.
start_total = time.time()
for epoch in range(1, epochs + 1):
    model.train()
    epoch_loss = 0
    print(f"=== Epoca {epoch}/{epochs} ===")

    start_epoch = time.time()

    for i, (x1, x2, y) in enumerate(train_loader, start=1):
        start_batch = time.time()

        optimizer.zero_grad()
        z1, z2 = model(x1, x2)
        loss = contrastive_loss(z1, z2, y, margin)
        loss.backward()
        optimizer.step()

        batch_time = time.time() - start_batch
        epoch_loss += loss.item()

        # Stampa (loss e tempo batch)
        print(f"Batch {i}/{len(train_loader)} | loss={loss.item():.4f} | tempo batch={batch_time:.2f} sec")

    epoch_time = time.time() - start_epoch
    print(f"Fine epoca {epoch} | tempo epoca={epoch_time:.2f} sec | loss media={epoch_loss/len(train_loader):.4f}\n")

print(f"\nTempo totale di addestramento: {time.time() - start_total:.2f} sec")


# SALVATAGGIO MODELLO

torch.save(model.state_dict(), "snet_lucas_siamese.pth")
print(" Modello salvato come 'snet_lucas_siamese.pth'")
