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




