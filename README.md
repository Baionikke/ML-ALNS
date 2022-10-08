# ML-ALNS
Implementation of Random Forest models into ALNS algorithm for reaching bettere performance

# INTRO ALNS ED OBIETTIVO PROGETTO
Il funzionamento della metaeuristica ALNS è basato sull’utilizzo di un framework di ricerca locale, in cui un insieme di diversi algoritmi competono per modificare una data soluzione corrente. In letteratura, questi algoritmi sono noti con l’appellativo destroy and repair. L’idea fondamentale che conduce all’utilizzo di questi algoritmi è quella di individuare soluzioni miglioranti, a partire da una data soluzione, applicando su di essa prima un metodo di distruzione, e successivamente un metodo di ricostruzione, che porti alla determinazione di una nuova soluzione. Distruggere una data soluzione significa rimuovere da essa un certo numero di vertici, per poi ricostruire, tramite un operatore di repair, una nuova soluzione, cercando di inserire al suo interno i vertici eliminati, seguendo un criterio prestabilito.
Nel caso specifico si vuole ottimizzare la consegna di merce da un deposito a dei clienti tenendo in considerazione parametri quali le tempistiche di consegna, il numero di veicoli impiegati, la distanza, la carica dei veicoli ecc… Dunque, sono presenti dei file all’interno delle sottocartelle *data* ciascuno dei quali contiene informazioni relative ai vertici, che sono di tre tipologie: D (deposito), S (stazione di ricarica), e C (cliente). 
Per modificare gli archi, dunque i tragitti da compiere, l’ALNS utilizza le seguenti coppie destroy-repair:

CUSTOMER REPAIR:
- GreedyRepairCustomer
- ProbabilisticGreedyRepairCustomer
- ProbabilisticGreedyConfidenceRepairCustomer

CUSTOMER DESTROY:
- GreedyDestroyCustomer
- WorstDistanceDestroyCustomer
- WorstTimeDestroyCustomer
- RandomRouteDestroyCustomer
- ZoneDestroyCustomer
- DemandBasedDestroyCustomer
- TimeBasedDestroyCustomer
- ProximityBasedDestroyCustomer
- ShawDestroyCustomer
- GreedyRouteRemoval
- ProbabilisticWorstRemovalCustomer

STATION REPAIR:
- DeterministicBestRepairStation
- ProbabilisticBestRepairStation

STATION DESTROY:
- RandomDestroyStation
- LongestWaitingTimeDestroyStation

L’obiettivo del Progetto è inserire algoritmi machine learning che riescano a decidere, in maniera migliore, che coppia destroy-repair impiegare, basandosi su alcune feature che veranno raccolte nei vari DB.

# DESCRIZIONE PROGETTO 
## COSTRUZIONE DB
La cartella *EVRPTW-main-DBProduction* è funzionale proprio alla raccolta di tale feature. Si è deciso di scrivere un file csv chiamato *DB-Output* in cui ad ogni iterazione vengono inseriti i seguenti dati:
- Instance's Name: nome del file di input
- Initial Solution: array rappresentante l'insieme delle rotte
- OFIS: funzione obiettivo iniziale
- Moves: algoritmi destroy-repair utilizzati (station destroy, station repair, customer destroy, customer repair) 
- OFFS: funzione obiettivo a seguito delle mosse applicate
- OF_Diff: differenza tra funzione obiettivo finale ed iniziale per l'iterazione corrente
- Exe_Time_d-r: tempo di elaborazione per applicare le mosse
- Avg_Battery_Status: media del consumo di batteria dei veicoli
- Avg_SoC: media degli stati della batteria con i quali i veicoli tornano in deposito
- Avg_Num_Charge: media del numero di cariche effettuate dai veicoli
- Avg_Vehicle_Capacity: media della capacità dei veicoli
- Avg_Customer_Demand: media delle domande dei customer in termini di package weight
- Num_Vehicles: numero di veicoli utilizzati
- Avg_Service_Time: media con la quale ogni customer viene servito
- Avg_Customer_TimeWindow: media delle differenze tra DueDate e ReadyTime dei customer
- Var_Customer_TimeWindow: varianza delle differenze tra DueDate e ReadyTime dei customer
- Avg_Customer_customer_min_dist: media delle distanze tra ciascun cliente e quello più vicino
- Var_Customer_customer_min_dist: varianza delle distanze tra ciascun cliente e quello più vicino
- Avg_Customer_station_min_dist: media delle distanze tra ciascun cliente e la stazione più vicina
- Var_Customer_station_min_dist: varianza delle distanze tra ciascun cliente e la stazione più vicina
- Avg_Customer_deposit_dist: media delle distanze tra ciascun cliente e il deposito
- Var_Customer_deposit_dist: varianza delle distanze tra ciascun cliente e il deposito
- CounterD_R: vettore composto dai 18 possibili algoritmi che si possono applicare (*), in cui si conta quante volte ciascun algoritmo ha permesso di migliorare il valore della soluzione precedente
- CounterD_Rlast: vettore composto dai 18 possibili algoritmi che si possono applicare (*), in cui si conta quante volte ciascun algoritmo è stato l'ultimo, in ciascun ciclo, ad aver migliorato la soluzione (quindi ad aver portato alla funzione obiettivo migliore)

Per ottenere una mole di dati maggiore è stato implementare un algoritmo python chiamato execution che una volta lanciato va a modificare di volta in volta, all’interno del file setting.json, il nome del file di input contenente i vertici del grafo. Ciascun file viene richiamato 10 volte e ogni chiamata sarà costituita a sua volta da 10 iterazioni. A causa dell’ azzeramento dei dati relativi a CounterD_R e CounterD_Rlast, ogni qualvolta viene inizializzato un nuovo file di input, è stato necessario l’utilizzo di due file csv di appoggio chiamati Counter e Counterlast.

## IMPLEMENTAZIONE RANDOM FOREST
Adesso che il dataset composto da più di 20k righe è stato ottenuto si passa alla scelta e all’implementazione dell’algoritmo machine learning. Tale algoritmo ha come finalità la comprensione della migliore coppia destroy-repair, dati alcuni parametri di ingresso. In particolare, si è optato per l’algoritmo RANDOM FOREST (classificatore d'insieme ottenuto dall'aggregazione tramite bagging di alberi di decisione). 
La presenza di tre cartelle (*EVRPTW-main-modelled-binary*, *EVRPTW-main-modelled-3way* e *EVRPTW-main-modelled-4way*) è motivata dalla volontà di confrontare tre tipi di classificazione. Per poter procedere a tali classificazioni sono stati scartati i campi *'Seed','CounterD_R','CounterD_Rlast','Initial Solution',"Instance's Name",'Moves'* in quanto non funzionali al task e non di tipo numerico. È stato introdotto una nuovo campo *DIFF4CLAS* che costituirà l’etichetta di classificazione, ma che per forza di cose è differente in base al tipo di classificazione:

Classificazione binaria:
-	Bad <= 1
-	Good >1

Classificazione a tre classi:
-	Bad <= 0
-	Good >0 && <=1500
-	Very Good >1500

Classificazione a quattro classi:
-	Bad <= 0
-	Normal >0 && <=1500
-	Good >1500 && <=3000
-	Very Good >3000

Vengono quindi trainati i rispettivi random forest (uno per ciascun algoritmo customer/repair) ed eseguito un piccolo test su un campione di dati, pari al 30%, dal quale è possibile visualizzare statistiche quali accuratezza, precision, recall, f1-score, support ecc.. all’interno del file *model_assessment_output*. I modelli vengono salvati in dei file joblib che però non rendono possibile la visualizzazione degli alberi. 

## VISUALIZZAZIONE ALBERI RAPID MINER
Per sopperire a tale mancanza, e a scopo dimostrativo, è stato inserito all’interno del progetto il file *RandomForestVisualization.rmp*. Come la tipologia suggerisce, si tratta di un file rapid miner che a partire dal *DB-Output* esegue tutte le operazioni descritte in precedenza e addestra modelli random forest con le stesse variabili di max profondità dell’albero, criterio di split, numero di alberi, ecc.. del codice python.

## IMPLEMENTAZIONE MODELLI ALL’INTERNO DEL CODICE ALNS
L’ultima operazione che è stata svolta riguarda l’implementazione di tali modelli all’interno del codice originale ALNS. Dunque, la scelta di destroy-repair viene demandata ai suddetti sulla base di alcuni parametri che vengono estratti di volta in volta. Le cartelle in cui viene svolto ciò sono *EVRPTW-main-modelled-binary*, *EVRPTW-main-modelled-3way*, *EVRPTW-main-modelled-4way*. 

## RISULTATI
Per poter confrontare i risultati vengono scritti i relativi file csv denominati *model_results[nome classificatore]*. All’interno di tale DB vengono salvati i seguenti dati: *File_Name,Seed,InitOF,FinalOF,DiffOF,Time,Iterations_Diff*.
Per raggruppare tali info e mettere a confronto i risultati ottenuti è stato prima creato un file csv *Comparison*, con i seguenti campi *File_Name,Seed,DiffOF_Original,Time_Original,DiffOF_Binary,Time_Binary,Diff_3Way,Time_3Way,DiffOF_4Way,Time_4Way*, per poi creare un notebook jupiter denominato *statistics* in cui viene analizzata la distribuzione del DB di partenza rispetto al campo *OF_Diff* e vengono confrontati e graficati i risultati dei modelli rispetto all’algoritmo ALNS originale.

## CONCLUSIONI
I risultati ottenuti sono altalenanti in quanto vengono fortemente influenzati dal numero di dati prodotti nel DB di partenza, dal seme impiegato, da come viene creato il modello random forest (che essendo random per natura ottiene risultati diversi ad esecuzioni successive). Si passa così da risultati comparabili all’ ALNS originale, come nel caso dei file della repo corrente,  a risultati decisamente positivi e promettenti.

