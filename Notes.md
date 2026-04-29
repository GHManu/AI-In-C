# AI with the C programming language
## Logistic Regression : Supervised machine learning algorithm
Logistic regression is a supervised machine learning algorithm widely used for binary classification tasks, such as identifying whether an email is spam or not and diagnosing diseases by assessing the presence or absence of specific conditions based on patient test results. This approach utilizes the logistic (or sigmoid) function to transform a linear combination of input features into a probability value ranging between 0 and 1. This probability indicates the likelihood that a given input corresponds to one of two predefined categories. The essential mechanism of logistic regression is grounded in the logistic function's ability to model the probability of binary outcomes accurately. With its distinctive S-shaped curve, the logistic function effectively maps any real-valued number to a value within the 0 to 1 interval. This feature renders it particularly suitable for binary classification tasks, such as sorting emails into "spam" or "not spam". By calculating the probability that the dependent variable will be categorized into a specific group, logistic regression provides a probabilistic framework that supports informed decision-making. --> [wikipedia](https://en.wikipedia.org/wiki/Logistic_regression)

La regressione logistica è un algoritmo di apprendimento automatico supervisionato ampiamente utilizzato per attività di classificazione binaria, come identificare se un'e-mail è spam o meno e diagnosticare malattie valutando la presenza o l'assenza di condizioni specifiche sulla base dei risultati dei test del paziente. Questo approccio utilizza la funzione logistica (o sigmoide) per trasformare una combinazione lineare di caratteristiche di input in un valore di probabilità compreso tra 0 e 1. Questa probabilità indica la probabilità che un dato input corrisponda a una delle due categorie predefinite. Il meccanismo essenziale della regressione logistica si basa sulla capacità della funzione logistica di modellare accuratamente la probabilità di risultati binari. Con la sua caratteristica curva a S, la funzione logistica mappa efficacemente qualsiasi numero reale a un valore compreso nell'intervallo da 0 a 1. Questa caratteristica la rende particolarmente adatta per attività di classificazione binaria, come la suddivisione delle e-mail in "spam" o "non spam". Calcolando la probabilità che la variabile dipendente venga categorizzata in un gruppo specifico, la regressione logistica fornisce un quadro probabilistico che supporta un processo decisionale informato.

---

* Nel Training:
* I Parametri: Sono i pesi ($w$) e il bias ($b$). L'obiettivo del training è trovare i valori di questi coefficienti che descrivono meglio la relazione tra i dati di input e le etichette (0 o 1).
   * Cosa succede: Il modello usa una funzione (chiamata Sigmoide) per trasformare l'input in una probabilità. Attraverso una funzione di costo (come la Log Loss), l'algoritmo misura quanto "sbaglia" e corregge i pesi usando l'Ottimizzazione (es. Gradient Descent).
* Nel Test:
* Cosa succede: I parametri ($w$ e $b$) sono ormai "congelati". Il modello riceve dati nuovi, applica la formula matematica con i pesi imparati e sputa fuori un risultato.
   * L'obiettivo: Misurare l'accuratezza (quante volte indovina la classe corretta) o altre metriche (come Precision e Recall).


---

## 1. Architettura del Modello
* **Struttura Dati:** Utilizza una `struct Data` per gestire le matrici delle caratteristiche ($X$) e le etichette target ($y$).
* **Inizializzazione:** I pesi vengono inizializzati con valori casuali molto piccoli (`randn() * 0.001`) per rompere la simmetria e favorire la convergenza iniziale, mentre il bias parte da un valore predefinito.

## 2. Componenti Matematiche Fondamentali
Il flusso di lavoro segue i pilastri dell'apprendimento supervisionato:
* **Funzione Sigmoide:** Trasforma il risultato della combinazione lineare ($z = Wx + b$) in una probabilità compresa tra **0 e 1**.
* **Predict Probs:** Calcola la probabilità che un esempio appartenga alla classe "1".
* **Binary Cross-Entropy (Loss):** È la funzione di costo che misura l'errore del modello. Più la probabilità predetta è lontana dall'etichetta reale, più la Loss aumenta (usando i logaritmi).
* **Gradient Descent:** Il cuore dell'addestramento. Il codice calcola le derivate della Loss rispetto ai pesi e al bias per aggiornarli nella direzione che minimizza l'errore.

## 3. Flusso del Programma
Il `main` esegue una simulazione completa simile a un workflow professionale di Data Science:
1.  **Generazione Dataset:** Invece di usare dati casuali puri (che non porterebbero a nulla), il codice genera dati $X$ e assegna $y$ in base a un prodotto scalare con dei "pesi reali" segreti. Questo crea un **pattern logico** che il modello può effettivamente imparare.
2.  **Split Train/Test:** I dati vengono divisi (80% per l'addestramento, 20% per la verifica). Questo serve a capire se il modello ha imparato a generalizzare o se ha solo memorizzato i dati.
3.  **Training (Fit):** Il modello cicla per un numero di epoche pari a `n_samples`, aggiornando i parametri e stampando la perdita (Loss) che diminuisce nel tempo.
4.  **Testing & Accuracy:** Alla fine, il modello predice le classi sui dati "mai visti" del Test Set e calcola la percentuale di risposte corrette.

## 4. Risultati Aspettati
* **Convergenza della Loss:** Durante l'esecuzione, vedrai nei log che il valore "Loss" cala costantemente (es. da 0.69 verso valori molto più bassi come 0.1 o 0.05).
* **Accuracy Elevata:** Poiché i dati sono stati generati con una logica lineare (`dot > 0.0`), la regressione logistica dovrebbe essere in grado di raggiungere un'accuracy molto alta, spesso superiore al **90-95%**, a patto che il `learning_rate` e le epoche siano ben bilanciati.


