# Bachelor Projekt

## Projektübersicht

Dieses Repository enthält den Code und die Ressourcen für mein Bachelorprojekt zum Thema Subgroup Discovery. Das Ziel des Projekts ist es, verschiedene Algorithmen zur Untergruppenentdeckung zu vergleichen und ihre Effizienz sowie Genauigkeit zu bewerten.

### Verzeichnis- und Dateierläuterungen

1. data/
Dieses Verzeichnis enthält alle Datensätze, die für die Analyse und das Training der Modelle verwendet werden.

2. notebooks/

In diesem Verzeichnis befinden sich alle Jupyter Notebooks, die für die explorative Datenanalyse (EDA) und die Modellvergleiche verwendet werden.

01_data_exploration.ipynb: Notebook für die erste explorative Analyse der Datensätze, Visualisierung der Verteilungen und grundlegende statistische Analysen.
02_model_comparison.ipynb: Notebook zum Vergleich der Subgroup Discovery-Modelle (z.B. Gurobi und Pysubgroup), einschließlich Leistungsmessungen und Visualisierungen der Ergebnisse.

3. scripts/

Hier befinden sich alle Python-Skripte, die für die Datenverarbeitung, Modelltraining und -evaluierung sowie die Anwendung der Subgroup Discovery-Algorithmen verwendet werden.

data_preprocessing.py: Skript zur Vorbereitung der Datensätze, einschließlich der Handhabung fehlender Werte und der Kodierung kategorialer Variablen.
model_training.py: Skript zum Training und zur Evaluierung verschiedener Subgroup Discovery-Modelle.
gurobi_optimization.py: Skript zur Implementierung der Subgroup Discovery unter Verwendung des Gurobi-Optimierers.
pysubgroup_analysis.py: Skript zur Implementierung der Subgroup Discovery unter Verwendung der PySubgroup-Bibliothek.
README.md: Erklärt die einzelnen Skripte und wie sie verwendet werden.

4. tests/

Dieses Verzeichnis enthält alle Unit-Tests für die verschiedenen Funktionen und Skripte im Projekt.

test_data_preprocessing.py: Testet die Datenvorbereitungsfunktionen auf korrekte Handhabung fehlender Werte und Kodierung.
test_model_training.py: Testet die Trainingsprozesse der Modelle, einschließlich der Validierung und der Genauigkeit der Vorhersagen.
test_gurobi_optimization.py: Testet die Subgroup Discovery-Implementierung unter Verwendung des Gurobi-Optimierers auf korrektes Verhalten und optimale Ergebnisse.
README.md: Beschreibt die Tests und wie sie ausgeführt werden.