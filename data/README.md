https://github.com/HMProenca/RobustSubgroupDiscovery/tree/main/data/single-nominal

hier gibts interessante .csv files für subgroup discovery 

Aufgabenstellung im Allgemeinen: Man kann jetzt auch messen, wie sich unterschiedliche Preprocessing-Methoden auf die Scalability auswirken denke ich mal. Ich habe auf jeden Fall beide Probleme in Pysubgroup und einmal in Gurobi geschrieben, und jetzt ist nur noch die Frage, wie man die Daten ordentlich aufbereitet. Frage Professor, ob es eher um Skalierbarkeit der Daten geht und Optimierung der Performance per se und Testen der unterschiedlichen Preprocessing Methoden oder ob Preprocessing in jeder Instanz gemacht werden soll und man die Laufzeit beider Algorithmen gegeneinander laufen lassen soll. 

Fortschritt: 
- Modellieren des SD Problems in Constraint Programming
- Programmieren von SD & Gurobi, sodass beide dieselben Resultate (Subgruppen) bereiten
- Schreiben von Tests zur Validierung
- bisherige Preprocessing Steps: Outliers & Missing Values 