import os
import random
import time
import torch

from PIL import Image
import torchvision.transforms as T
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
from tqdm import tqdm

# Importa le classi dal tuo script di training
from siamese_train_lucas_emotions import FeatureExtractor, SiameseNetwork, ContrastiveLoss

# ----------------------------------------------------------------------
# CONFIGURAZIONE
# ----------------------------------------------------------------------
CARTELLA_TRAIN = r"Dataset\Gallery"      # cartella con le classi per costruire galleria/centroidi
CARTELLA_TEST  = r"Dataset\Test" # cartella con immagini di test (non viste dal training)
dimensione_immagine = (224, 224)
FILE_MODELLO = r"checkpoints/snet_siamese-v4.ckpt"
NUM_TEST = 50 
K = 7
DEVICE = "cpu"  # usa "cuda" se vuoi GPU

ESTENSIONI_IMG = (".jpg", ".jpeg", ".png")


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

trasformazione = T.Compose([
    T.Resize(dimensione_immagine),
    T.ToTensor(),
    T.Normalize(mean=[0.5,0.5,0.5], std=[0.5,0.5,0.5])
])

# ----------------------------------------------------------------------
# FEATURE EXTRACTOR (DEVE MATCHARE IL MODELLO USATO IN TRAIN)
# ----------------------------------------------------------------------
# Qui usiamo l'embedding_net salvato all'interno del LightningModule
feature_net = FeatureExtractor()

# ----------------------------------------------------------------------
# CARICAMENTO MODELLO LIGHTNING
# ----------------------------------------------------------------------
model = SiameseNetwork.load_from_checkpoint(FILE_MODELLO)

model.eval()
feature_net = model.embedding_net.to(DEVICE)

print("Modello caricato correttamente.")

# ----------------------------------------------------------------------
# FUNZIONI DI SUPPORTO
# ----------------------------------------------------------------------
def compute_embedding(percorso_file, rete):
    """Calcola embedding normalizzato per una singola immagine."""
    img = Image.open(percorso_file).convert("RGB")
    img = trasformazione(img).unsqueeze(0).to(DEVICE)
    with torch.no_grad():
        emb = rete(img).cpu().numpy().flatten()
    return emb / (np.linalg.norm(emb) + 1e-8)

def costruisci_galleria(cartella, classi, rete):
    """Costruisce la galleria di embedding di tutte le immagini della cartella."""
    galleria = []
    for cls in classi:
        cls_path = os.path.join(cartella, cls)
        if not os.path.isdir(cls_path):
            continue
        files = [f for f in os.listdir(cls_path) if f.lower().endswith(ESTENSIONI_IMG)]
        for f in tqdm(files, desc=f"Embedding classe {cls}", ncols=80):
            file_path = os.path.join(cls_path, f)
            emb = compute_embedding(file_path, rete)
            galleria.append({"classe": cls, "emb": emb, "file": file_path})
    return galleria

def costruisci_centroidi(galleria, classi):
    """Calcola il vettore medio normalizzato per ogni classe."""
    centroidi = {}
    for cls in classi:
        emb_cls = [g["emb"] for g in galleria if g["classe"] == cls]
        if emb_cls:
            m = np.mean(emb_cls, axis=0)
            centroidi[cls] = m / (np.linalg.norm(m) + 1e-8)
    return centroidi

def predici_knn(v, galleria, K=7):
    """Predizione classe con k-NN basato su cosine similarity."""
    dists = [1 - np.dot(v, g["emb"]) for g in galleria]
    idx = np.argsort(dists)[:K]
    vicini = [galleria[i]["classe"] for i in idx]
    return max(set(vicini), key=vicini.count)

# ----------------------------------------------------------------------
# CARICAMENTO CLASSI
# ----------------------------------------------------------------------
classi_selezionate = sorted([
    d for d in os.listdir(CARTELLA_TRAIN)
    if os.path.isdir(os.path.join(CARTELLA_TRAIN, d))
])
print("Classi trovate:", classi_selezionate)

# ----------------------------------------------------------------------
# COSTRUZIONE GALLERIA E CENTROIDI
# ----------------------------------------------------------------------
print("\nCostruzione galleria...")
galleria = costruisci_galleria(CARTELLA_TRAIN, classi_selezionate, feature_net)
centroidi = costruisci_centroidi(galleria, classi_selezionate)
print(f"Galleria costruita con {len(galleria)} embedding totali.\n")

# ----------------------------------------------------------------------
# COSTRUZIONE TEST SET
# ----------------------------------------------------------------------
test_files = []
for cls in classi_selezionate:
    path = os.path.join(CARTELLA_TEST, cls)
    if not os.path.isdir(path):
        continue
    files = [os.path.join(path, f) for f in os.listdir(path) if f.lower().endswith(ESTENSIONI_IMG)]
    for f in files:
        test_files.append({"classe": cls, "file": f})

NUM_TEST = min(NUM_TEST, len(test_files))
test_set = random.sample(test_files, NUM_TEST)

# ----------------------------------------------------------------------
# CICLO DI TEST
# ----------------------------------------------------------------------
true_labels = []
pred_centroid = []
pred_knn = []

print("Inizio test...\n")
start_time = time.time()

for t in tqdm(test_set, ncols=80):
    v = compute_embedding(t["file"], feature_net)
    true_labels.append(t["classe"])

    # centroidi
    d_c = [1 - np.dot(v, centroidi[c]) for c in centroidi]
    pred_c = list(centroidi.keys())[np.argmin(d_c)]
    pred_centroid.append(pred_c)

    # k-NN
    p_knn = predici_knn(v, galleria, K)
    pred_knn.append(p_knn)

test_time = time.time() - start_time

# ----------------------------------------------------------------------
# RISULTATI FINALI
# ----------------------------------------------------------------------
acc_centroid = np.mean([t == p for t, p in zip(true_labels, pred_centroid)]) * 100
acc_knn = np.mean([t == p for t, p in zip(true_labels, pred_knn)]) * 100

print(f"\n---- RISULTATI ----")
print(f"Campioni testati: {NUM_TEST}")
print(f"Tempo totale test: {test_time:.2f} sec")
print(f"Tempo medio per immagine: {test_time/NUM_TEST:.3f} sec")
print(f"Accuratezza Centroidi: {acc_centroid:.2f}%")
print(f"Accuratezza k-NN (K={K}): {acc_knn:.2f}%\n")

# ----------------------------------------------------------------------
# MATRICE DI CONFUSIONE
# ----------------------------------------------------------------------
true_cat = np.array(true_labels)
centroid_cat = np.array(pred_centroid)
knn_cat = np.array(pred_knn)

fig, axs = plt.subplots(1, 2, figsize=(12, 5))

ConfusionMatrixDisplay(
    confusion_matrix(true_cat, centroid_cat, labels=classi_selezionate),
    display_labels=classi_selezionate
).plot(ax=axs[0], cmap='Blues')
axs[0].set_title(f"Centroidi (Acc={acc_centroid:.1f}%)")

ConfusionMatrixDisplay(
    confusion_matrix(true_cat, knn_cat, labels=classi_selezionate),
    display_labels=classi_selezionate
).plot(ax=axs[1], cmap='Greens')
axs[1].set_title(f"k-NN K={K} (Acc={acc_knn:.1f}%)")

plt.tight_layout()
plt.show()