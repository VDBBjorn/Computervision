# Speed estimation for vehicles using SVM classification and edge detection
### Courtesy @sigisang, @MathiasDierickx, @maarten13

Project "Computervisie" (P. Veelaert) at Ghent University. Tries to determine the maximum safe speed for a vehicule, given images taken with the vehicules top camera. 

## Maximum snelheden berekenen voor evaluate.py:
$ ./maxSpeed [datasetFolder]
1 argument [datasetFolder]: pad naar de map die de dataset van toepassing bevat
	bvb: Dataset/01/

Looptijd van maxspeed is enkele minuten (~2min. voor 150 frames).
Uitvoer wordt weggeschreven naar [datasetFolder]/results.txt in formaat zoals
vereist voor evaluate.py .
Genereert ook een map outputframes/ die per dataset voor elk frame
het resultaat van edge detection en road detection samen toont. 



## Andere binaries:
$ ./svm
Testprogramma om output van de SVM classifier te testen.

Configuratie te vinden in constanten bovenaan io.hpp .
Dataset bepaald onderaan svm.cpp (standaard dataset 1).
>>Verwacht dataset in zelfde map, en van de vorm ./Dataset/01


$./trainingsdata
Itereert alle parameter combinaties, traint SVM op verschillende combinaties
van datasets en geeft logging met resultaten en F-scores.
Genereert featurevectoren voor alle iteraties, data wordt weggeschreven naar
een bestand per combinatie van parameters, in Data/.
>>Verwacht dataset in zelfde map, en van de vorm ./Dataset/01



## Headers:
Headers/LineDetection.hpp
Bevat de imlementatie voor het uitvoeren van Canny Edge Detection,
waarbij eerst de details op de rijweg weggefilterd zijn door erode en dilate.

Headers/io.hpp
Bevat configuratiewaarden voor maxSpeed.cpp, svm.cpp en trainingsdata.cpp.
Bevat ondersteunende code voor andere deelapplicaties, zoals inlezen van
datafiles, een map verifiëren,.. en dergelijke.

Headers/lbpfeaturevector.hpp
Bevat implementatie van LBP.
Belangrijkste functie is processFrame: berekent verzameling van featurevectoren
volgens gegeven parameters.

Headers/svm.hpp
Bevat implementatie van SVM.
Bevat implementatie voor berekenen van F-scores en tonen van classificatie
resultaten op een frame.



## Map Data/
Bevat per frame de nodige labels om op te trainen.
Bevat per frame en per configuratie de featurevectoren van de blokken in dat
frame.
Op moment van indienen bevat Data/ enkel de nodige labels en featurevectoren
voor de huidige configuratie van maxSpeed,
zoals besloten in de paper.
Data/ bevat ook Backup_alle_labels.zip: bevat alle mogelijk labels voor de
conguraties besproken in de paper.
Deze labels in Data/ plaatsen en ./trainingsdata runnen zal alle bijbehorende
featurevectoren genereren.
