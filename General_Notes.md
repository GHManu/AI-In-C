## ESEMPIO(per capire in generale): Diagnostica Medica (Dataset di pazienti)
Immagina di voler classificare se un paziente è sano o malato in base a delle analisi.

| Campione (Riga) | Feature 1 (Età) | Feature 2 (Pressione) | Feature 3 (Colesterolo) | Classe (Target) |
| :--- | :--- | :--- | :--- | :--- |
| **Paziente A** | 45 | 120 | 200 | **Sano** |
| **Paziente B** | 67 | 150 | 240 | **Rischio Alto** |
| **Paziente C** | 30 | 115 | 180 | **Sano** |

* **Campioni:** I singoli pazienti.
* **Features:** Età, Pressione sanguigna, Livello di colesterolo.
* **Classe:** "Sano" o "Rischio Alto".

---

* **Campione**: L'oggetto fisico o l'evento reale che stai analizzando (una riga).

* **Feature**: Una caratteristica misurabile di quell'oggetto (l'indice di colonna).

* **Classe**: La categoria a cui appartiene l'oggetto (l'etichetta target).

* **sample** --> un sample (campione) è un'istanza specifica (un'osservazione).

---

- i samples come le singole righe, dove ogni riga rappresenta un esperimento o un oggetto distinto. Una riga identifica un sample, quindi un vettore di features a cui viene associato una classe; 
- La matrice X ha grandezza n_samples × n_features, contiene tutti i samples del mio dataset dove ognuno di esso è un insieme di features; il vettore y ha grandezza n_sample ed è l'insieme di classi che mappano i sample del mio dataset.

- Il mio modello sarà suddiviso in 2 fasi aventi diversi dataset ma nati dallo stesso dataset:
1. TRAINING: il modello analizza questi dati per individuare pattern, relazioni e regole, ottimizzando i propri parametri interni per minimizzare gli errori.
2. TESTING: Una volta addestrato, il modello viene utilizzato in scenari reali per fare previsioni, classificazioni o generare contenuti su dati nuovi e mai visti prima. In questa fase, il modello applica ciò che ha imparato durante l'addestramento.

- COSA VUOL DIRE IN PRATICA: praticamente con nuovi samples in input, il mio modello deve saper classificare correttamente, ovvero predirre la classe a cui appartiene, ad esempio aggiungo un nuovo paziente e guardando i sintomi ovvero le features mi deve dire se è sano o malato.I nuovi samples e features vengono presi dal dataset nuovo.

### STATISTICA E PROBABILITÀ
Senza di esse, un modello non saprebbe "imparare" perché non avrebbe un modo matematico per misurare l'incertezza o trovare pattern nei dati.
## 1. La Distribuzione dei Dati
Una distribuzione di probabilità descrive come i valori di una variabile sono sparsi e quanto spesso si presentano.

* Nel Training: Il modello cerca di capire la "forma" dei tuoi dati di addestramento. Se addestri un'IA a riconoscere pesci usando solo foto di salmoni (una distribuzione specifica), non saprà cosa fare con uno squalo (una distribuzione diversa).
* L'Assunzione IID: Quasi tutti i modelli assumono che i dati siano IID (Independently and Identically Distributed). Significa che ogni sample è indipendente dagli altri e che i dati di test provengono dalla stessa distribuzione del training. Se questa assunzione salta (ad esempio, se i dati di test sono molto diversi da quelli di training), il modello fallisce. 

## 2. Parametri Statistici e Pesi
In statistica, un parametro è un valore che descrive una caratteristica di una popolazione (come la media $\mu$ o la varianza $\sigma^2$). 

* Nella Logistic Regression, i pesi ($w$) che il modello impara sono tecnicamente dei parametri statistici stimati dai dati.
* L'addestramento è un processo di inferenza statistica: usi un campione di dati per stimare i parametri che meglio rappresentano la realtà.

## 3. Valutazione e Incertezza
La statistica ti serve nel Testing per capire se i risultati sono reali o frutto del caso. 

* Metriche: L'accuratezza, la precisione o l'errore quadratico medio ($MSE$) sono concetti puramente statistici usati per misurare la distanza tra ciò che il modello prevede e la realtà.
* Ti permette di dire: "Il mio modello ha un'accuratezza del 95%, e sono sicuro che non sia fortuna".
