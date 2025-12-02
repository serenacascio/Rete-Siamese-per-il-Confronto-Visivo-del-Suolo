# Rete-Siamese-per-il-Confronto-Visivo-del-Suolo

Vogliamo sviluppare e addestrare una rete neurale Siamese per il
riconoscimento e il confronto di immagini di suolo.

L'obiettivo principale è stato verificare la capacità della rete di
distinguere tra immagini simili e diverse

## File

-   <https://www.kaggle.com/datasets/saurabhshahane/soil-texture-dataset/data>
    :Dataset utilizzato per l\'addestramento finale della rete

-   trainsiamese.py : Training principale della rete Siamese

-   snet_siamese-v4.ckpt :Modello addestrato

-   test.py : test del modello



## Timeline e struttura del progetto

### 1\) __Ricerca del dataset e adattamento alla mia rete__

ricerca di un Dataset che potesse adattarsi bene al mio progetto
utilizzando dapprima dei dataset generici sulle texture e poi il dataset
scelto

Trovato il dataset in uso ho poi creato tre cartelle con le immagini:

-   Train: cartella principale con alll\'interno la quasi totalita di
    immagini divise in classi

-   Gallery: cartella con dentro le sottocartelle delle classi con circa
    25 immagini copiate da ogni classe

-   Test: cartella con dentro le cartelle delle classi con immagini non
    presenti nel train per poter testare la mia rete 



    
### 2\) __Addestramento__

*trainsiamese.py*

training della rete Siamese.

Questo modulo implementa l'addestramento di una Rete Siamese utilizzando
PyTorch e PyTorch Lightning, con l'obiettivo di creare uno spazio in cui
immagini appartenenti alla stessa classe risultino vicine, mentre
immagini di classi diverse risultino distanti.

La rete è progettata per problemi di metric learning, similarità visiva
e classificazione basata su embedding, particolarmente adatta per
dataset complessi come texture di terreni o immagini con forte
variabilità interna.

struttura:

#### 1\. Dataset Siamese personalizzato

Un Dataset che genera automaticamente coppie di immagini:

\- coppie positive: due immagini appartenenti alla stessa classe (label:0)

\- coppie negative: due immagini di classi diverse (label:1)

Le coppie vengono generate in modo casuale ad ogni epoca

Output: *snet_siamese-v4.ckpt*


#### 2\. Feature Extractor (CNN)

modello per estrarre le feature dalle immagini è una CNN compatta,
progettata appositamente per:

riconoscere pattern locali (tipico delle texture)

La rete contiene:

- 3 blocchi Convolution + BatchNorm + ReLU + MaxPool

- Flatten

- 2 Fully Connected layers

- normalizzazione L2 degli embedding

Il vettore finale a 64 dimensioni è utilizzato per confrontare immagini
tramite distanza euclidea.

#### 3\. Contrastive Loss

La Rete Siamese è addestrata con Contrastive Loss, definita come:

𝐿=(1−𝑦)⋅𝑑\^2+𝑦⋅max⁡(0,𝑚−𝑑)\^2

dove:

- 𝑦=0 : immagini simili

- 𝑦=1 : immagini diverse

- 𝑑 = la distanza euclidea tra gli embedding

- 𝑚 = il margin (distanza minima desiderata tra classi diverse)

Questa loss avvicina le immagini simili e allontana quelle diverse.

#### 4\. Modello Siamese Lightning

La classe SiameseNetwork è un modulo PyTorch Lightning che gestisce:

- forward pass

- computazione della loss

- logging automatico

- configurazione dell'ottimizzatore

Il modello supporta ottimizzatori flessibili tramite:

- optimizer_cls=OPTIMIZER_CLS

- optimizer_kwargs=OPTIMIZER_KWARGS

Nel progetto viene utilizzato __Adam__, ottimo per dataset con grande
variabilità come texture e immagini da campo.

#### 5\. Training con PyTorch Lightning

L'addestramento è gestito tramite Trainer, con:

- checkpoint automatico del modello migliore

- logging della loss

- supporto CPU

- generazione continua di coppie sempre nuove

Il modello finale è salvato in:

*/checkpoints/snet_siamese.ckpt*

ed è successivamente utilizzabile per:

- estrarre embedding

- fare classificazione via k-NN

- costruire una gallery di riferimento

- valutare immagini di test tramite nearest neighbor o centroidi


### 3\) __Valutazione e risultati__

*test.py*

Lo script di test carica un modello Siamese addestrato e lo utilizza per
confrontare un'immagine di query con una galleria di immagini,
restituendo la più simile sulla base della distanza nel embedding space.

Valutare la capacità del modello di riconoscere immagini simili,
generare gli embedding e confrontare la query con la galleria tramite
distanza euclidea o cosine similarity.

Schema:

-   carica il modello addestrato

-   costruisce l'embedding space della galleria

-   calcola l'embedding della query

-   confronta query vs galleria

-   restituisce la migliore corrispondenza

