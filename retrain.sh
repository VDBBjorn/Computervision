sudo rm -rf CMake* && sudo bash openCvCmake.bash trainingsdata && sudo chmod a+x trainingsdata && ./trainingsdata
cp output/*frame00*_feature* Trainingsdata/
sudo rm -rf CMake* && sudo bash openCvCmake.bash svm && sudo chmod a+x svm && ./svm 