# test_siamese_python.py 
# Script di test per la rete Siamese realizzata in PyTorch
# Vengono considerate solo le classi: clay, loam, sand, silt.

import os
import random
import torch
import torch.nn as nn
from PIL import Image
import torchvision.transforms as T
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
import matplotlib.pyplot as plt
import numpy as np

# === CONFIGURAZIONE ===
# Definizione delle principali impostazioni del test:
# le cartelle di addestramento e di test, le classi da utilizzare,
# la dimensione delle immagini e il file contenente il modello salvato.
train_folder = "dataset_lucas_usda_resized"
test_folder  = "dataset_lucas_usda_resized_test"
selected_classes = ["clay", "loam", "sand", "silt"]
image_size = (300, 300)
model_file = "snet_lucas_siamese.pth"
K = 7   # numero di vicini considerati per il classificatore k-NN
num_test = 50  # numero di immagini di test

#CPU.
device = "cpu"
print(f"Dispositivo in uso: {device}")

#TRASFORMAZIONI
# Le immagini vengono ridimensionate e convertite in tensori normalizzati.
transform = T.Compose([
    T.Resize(image_size),
    T.ToTensor()
])

# DEFINIZIONE DELLA RETE
# La rete di embedding estrae caratteristiche compatte da ciascuna immagine.
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
        # L’output viene normalizzato per ottenere vettori unitari
        return nn.functional.normalize(x, p=2, dim=1)

# La rete Siamese confronta due immagini calcolando la distanza tra i rispettivi embedding.
class SiameseNet(nn.Module):
    def __init__(self, embedding_net):
        super().__init__()
        self.embedding_net = embedding_net
    def forward(self, x1, x2):
        return self.embedding_net(x1), self.embedding_net(x2)

# CARICAMENTO DEL MODELLO ADDDESTRATO 
snet = SiameseNet(EmbeddingNet()).to(device)
snet.load_state_dict(torch.load(model_file, map_location=device))
snet.eval()
print("Modello caricato correttamente.")

#  FUNZIONI DI SUPPORTO 
# compute_embedding(): calcola l’embedding di una singola immagine
def compute_embedding(img_path):
    img = Image.open(img_path).convert("RGB")
    img = transform(img).unsqueeze(0) 
    with torch.no_grad():
        emb = snet.embedding_net(img)
    emb = emb.numpy().flatten()
    return emb / np.linalg.norm(emb + 1e-8)

# build_gallery(): costruisce una “galleria” di embedding dalle immagini di addestramento
def build_gallery(folder):
    gallery = []
    for cls in selected_classes:
        cls_path = os.path.join(folder, cls)
        if not os.path.isdir(cls_path):
            continue
        files = [f for f in os.listdir(cls_path) if f.lower().endswith(".jpg")]
        for f in files:
            emb = compute_embedding(os.path.join(cls_path, f))
            gallery.append({"class": cls, "emb": emb, "file": os.path.join(cls_path, f)})
    return gallery

# build_centroids(): calcola un vettore medio (centroide) per ciascuna classe
def build_centroids(gallery):
    centroids = {}
    for cls in selected_classes:
        cls_embs = [g["emb"] for g in gallery if g["class"] == cls]
        if cls_embs:
            m = np.mean(cls_embs, axis=0)
            m = m / (np.linalg.norm(m)+1e-8)
            centroids[cls] = m
    return centroids

# predict_knn(): effettua la classificazione con metodo k-Nearest Neighbors
def predict_knn(v, gallery, K=7):
    dists = [1 - np.dot(v, g["emb"]) for g in gallery]
    idx = np.argsort(dists)[:min(K, len(dists))]
    neighbor_classes = [gallery[i]["class"] for i in idx]
    # viene restituita la classe più frequente tra i K vicini più simili
    return max(set(neighbor_classes), key=neighbor_classes.count)

# COSTRUZIONE GALLERIA E CENTROIDI
gallery = build_gallery(train_folder)
centroids = build_centroids(gallery)
print(f"Galleria costruita con {len(gallery)} embedding totali.")

# COSTRUZIONE DEL SET DI TEST
all_test_files = []
for cls in selected_classes:
    cls_path = os.path.join(test_folder, cls)
    if not os.path.isdir(cls_path):
        continue
    files = [os.path.join(cls_path, f) for f in os.listdir(cls_path) if f.lower().endswith(".jpg")]
    for f in files:
        all_test_files.append({"class": cls, "file": f})

# Viene selezionato un numero limitato di immagini per il test
if len(all_test_files) < num_test:
    num_test = len(all_test_files)
test_set = random.sample(all_test_files, num_test)

# CICLO DI TEST
# Per ogni immagine vengono calcolati:
# - embedding dell’immagine
# - classe predetta con metodo dei centroidi
# - classe predetta con k-NN
true_labels = []
pred_centroid = []
pred_knn = []

for t in test_set:
    v = compute_embedding(t["file"])
    true_labels.append(t["class"])

    # Classificazione tramite distanza dai centroidi
    dists_c = [1 - np.dot(v, centroids[cls]) for cls in centroids]
    pred_c = list(centroids.keys())[np.argmin(dists_c)]
    pred_centroid.append(pred_c)

    # Classificazione tramite k-NN
    pred_k = predict_knn(v, gallery, K)
    pred_knn.append(pred_k)

    print(f"Img: {os.path.basename(t['file']):30} True={t['class']:10} | Centroid={pred_c:10} | kNN={pred_k:10}")

# RISULTATI
# Calcolo dell’accuratezza media per entrambi i metodi
acc_centroid = np.mean([t==p for t,p in zip(true_labels, pred_centroid)])*100
acc_knn = np.mean([t==p for t,p in zip(true_labels, pred_knn)])*100

print(f"\nAccuratezza Centroidi: {acc_centroid:.2f}%")
print(f"Accuratezza k-NN: {acc_knn:.2f}%")

# MATRICE DI CONFUSIONE
# mostrano le prestazioni della classificazione
true_cat = np.array(true_labels)
centroid_cat = np.array(pred_centroid)
knn_cat = np.array(pred_knn)

fig, axs = plt.subplots(1,2, figsize=(12,5))
ConfusionMatrixDisplay(confusion_matrix(true_cat, centroid_cat, labels=selected_classes),
                       display_labels=selected_classes).plot(ax=axs[0], cmap='Blues')
axs[0].set_title(f"Confusion Matrix - Centroidi (Acc={acc_centroid:.2f}%)")
ConfusionMatrixDisplay(confusion_matrix(true_cat, knn_cat, labels=selected_classes),
                       display_labels=selected_classes).plot(ax=axs[1], cmap='Greens')
axs[1].set_title(f"Confusion Matrix - k-NN (K={K}, Acc={acc_knn:.2f}%)")
plt.show()
