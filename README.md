# Rete-Siamese-per-il-Confronto-Visivo-del-Suolo
Vogliamo sviluppare e addestrare una rete neurale Siamese per il riconoscimento e il confronto di immagini di suolo.
L’obiettivo principale è stato verificare la capacità della rete di distinguere tra immagini simili e diverse, partendo dal dataset LUCAS Topsoil


# Struttura
LUCAS_Text_All_10032025.csv     # File del dataset con le informazioni utili per la rete.

coppie_siamese.csv              # Coppie generate (non filtrate)

coppie_siamese_filtrate.csv     # Coppie pulite e pronte al training

genera_coppie_siamese.m         # Script per generare coppie dal dataset

pulizia_coppie_lucas.m          # Script di pre-pulizia coppie

resize.m                        # Resize immagini 

scarica_immagini_lucas.m        # Download immagini dataset LUCAS

siamese_train_lucas.m           # Training principale della rete Siamese

snet_lucas_final.mat            # Modello addestrato

dataset_lucas_usda/             # Dataset originale

dataset_lucas_usda_resized/     # Dataset preprocessato (resize immagini)

lucas_images/                   # Immagini scaricate


# Timeline del progetto

1) Acquisizione e organizzazione dati

scarica_immagini_lucas.m

scaricare automaticamente le foto LUCAS dal server EC, con selezione per Paese/anno e gestione del codice PointID + lettera (C,E,N,P,S,W).

organizza_per_usda.m

riorganizzare le immagini per classe USDA (es. loam, siltLoam, …) a partire da un CSV con mappatura POINTID -> CLASSE.

resize.m

uniformare le dimensioni delle immagini per il training (300×300).

2) Generazione e pulizia coppie

genera_coppie_siamese.m

produrre un set di coppie positive (stessa classe) e negative (classi diverse) per la rete Siamese.

pulizia_coppie_lucas.m

pre-pulizia delle coppie per velocizzare l’addestramento e ridurre errori in run.

Controlli: esistenza file, leggibilità, dimensioni corrette, immagini non corrotte.

3) Addestramento

siamese_train_lucas.m

training della rete Siamese.

Modifiche chiave rispetto all’esempio MATLAB:

Datastore personalizzato sostituito con batch loader manuale.

Mini-batch grande (es. 128) per sfruttare meglio CPU/GPU e ridurre le iterazioni/epoch.

Logging tempi per batch (per le prestazioni).

Output: snet_lucas_final.mat.

4) Valutazione e risultati

.....

snet_lucas_final.mat – primo modello addestrato.

