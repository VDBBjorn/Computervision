_______________________________________________________________________________
README voor code bij paper

Speed estimation for vehicles using SVM classification and edge detection
-------------------------------------------------------------------------
	door Mathias Dierickx, Tim Ranson, Maarten Tindemans en Bjorn Vandenbussche

_______________________________________________________________________________

Maximum snelheden berekenen voor evaluate.py:
---------------------------------------------
$ ./maxSpeed [datasetFolder]
1 argument datasetFolder: de map die de dataset van toepassing bevat
	bvb: Dataset/01

Looptijd van maxspeed is enkele minuten (~2min. voor 150 frames).
Uitvoer wordt weggeschreven naar [datasetFolder]/results.txt in formaat zoals vereist voor evaluate.py .



Andere binaries:
----------------
$ ./svm
Testprogramma om output van SVM te testen.

Configuratie te vinden in constanten bovenaan io.hpp .
Dataset bepaald onderaan svm.cpp (standaard dataset 1).
>>Verwacht dataset in zelfde map, en van de vorm ./Dataset/01


$./trainingsdata
Itereert alle parameter combinaties, traint SVM op verschillende combinaties van datasets en geeft logging met resultaten en F-scores.
Genereert featurevectoren voor alle iteraties.

>>Verwacht dataset in zelfde map, en van de vorm ./Dataset/01



Headers:
--------
Headers/LineDetection.hpp


Headers/io.hpp
Bevat configuratiewaarden voor maxSpeed.cpp, svm.cpp en trainingsdata.cpp.
Bevat ondersteunende code voor andere deelapplicaties, zoals inlezen van datafiles, een map verifiëren,.. en dergelijke.

Headers/lbpfeaturevector.hpp
Bevat implementatie van LBP.
Belangrijkste functie is processFrame: berekent verzameling van featurevectoren volgens gegeven parameters.

Headers/svm.hpp
Bevat implementatie van SVM.
Bevat implementatie voor berekenen van F-scores en tonen van classificatie resultaten op een frame.

_______________________________________________________________________________