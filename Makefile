main.exe: main.o CTRNN.o TSearch.o random.o LeggedAgent.o FlexWalk.o
	g++ -pthread -o main.exe main.o CTRNN.o TSearch.o random.o LeggedAgent.o FlexWalk.o
random.o: random.cpp random.h VectorMatrix.h
	g++ -pthread -c -O3 random.cpp
CTRNN.o: CTRNN.cpp random.h VectorMatrix.h CTRNN.h
	g++ -pthread -c -O3 CTRNN.cpp
LeggedAgent.o: LeggedAgent.cpp CTRNN.h random.h LeggedAgent.h
	g++ -pthread -c -O3 LeggedAgent.cpp
FlexWalk.o: FlexWalk.cpp CTRNN.h LeggedAgent.h VectorMatrix.h FlexWalk.h
	g++ -pthread -c -O3 FlexWalk.cpp
TSearch.o: TSearch.cpp TSearch.h
	g++ -pthread -c -O3 TSearch.cpp
main.o: main.cpp CTRNN.h TSearch.h FlexWalk.h
	g++ -pthread -c -O3 main.cpp
clean:
	rm *.o main.exe
