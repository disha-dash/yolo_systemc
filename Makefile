SYSTEMC = /usr/local/systemc

CXX = g++
CXXFLAGS = -I$(SYSTEMC)/include -L$(SYSTEMC)/lib-linux64 -lsystemc -lm

all:
	$(CXX) main.cpp yolo.cpp -o yolo $(CXXFLAGS)

run:
	./yolo