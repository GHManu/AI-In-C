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
