// Test the fitness of a neural pattern with constant up and down states that have given durations

#include "CTRNN.h"
#include "LeggedAgent.h"
#include "VectorMatrix.h"
#include "random.h"

//CTRNN settings
const int N = 2;

//Simulation Settings
const double StepSize = 0.01;
const double TransientDur = 700; //transient in this case runs the body (ensuring we hit a boundary)
const double TestDur = 280;

//Pattern Characteristics
double upval = .7;
double downval = .1;
double updur = 30;
double downdur = 40;
bool foot_coord_fw = false;

int main(void){
    LeggedAgent Agent;
    Agent.NervousSystem.SetCircuitSize(N);
    //Transient
    Agent.PolicyRun(upval,downval,updur,downdur,TransientDur,foot_coord_fw,StepSize);

    //cancel out distance traveled
    double init = Agent.cx;

    //Velocity test
    Agent.PolicyRun(upval,downval,updur,downdur,TestDur,foot_coord_fw,StepSize);

    double fit = (Agent.cx - init)/TestDur;
    cout << "Fitness: " << fit << endl;

    return 0;
}