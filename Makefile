# C++ compiler
CXX=g++
# GSL >=2.0 required
GSLCFLAGS=`gsl-config --cflags`
GSLLIBS=`gsl-config --libs`
# C++ compiler flags
CXXFLAGS =-Wall -std=c++11 -O3 -lstdc++ -fopenmp
# ini file parser files
INISRC =\
		 inih/ini.c \
		 inih/INIReader.cpp
INIINC=-I./inih	
# source files of KoMPoST
KOMPOSTSRC =\
		 src/Main.cpp \
		 src/EventInput.cpp \
		 src/BackgroundEvolution.cpp \
		 src/GreensFunctions.cpp \
		 src/KineticEvolution.cpp \
		 src/ScalingVariable.cpp

# KoMPoST3D.exe: KoMPoST on every eta_s slice of a 3D T^{mu nu} (src/Main3D.cpp)
KOMPOST3DSRC = $(filter-out src/Main.cpp,$(KOMPOSTSRC)) src/Main3D.cpp

all: KoMPoST.exe KoMPoST3D.exe

KoMPoST.exe: $(KOMPOSTSRC) $(wildcard src/*.h src/*.inc)
	$(CXX) -o KoMPoST.exe $(KOMPOSTSRC) $(INISRC) $(INIINC) $(CXXFLAGS) $(GSLCFLAGS) $(GSLLIBS) 

KoMPoST3D.exe: $(KOMPOST3DSRC) $(wildcard src/*.h src/*.inc)
	$(CXX) -o KoMPoST3D.exe $(KOMPOST3DSRC) $(INISRC) $(INIINC) $(CXXFLAGS) $(GSLCFLAGS) $(GSLLIBS) 

.PHONY: clean
clean:
	rm -f KoMPoST.exe KoMPoST3D.exe
