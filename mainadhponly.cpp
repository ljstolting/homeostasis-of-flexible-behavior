//---------------------------------------
//  test a genome from a bestind file
//---------------------------------------
#include "CTRNN.h"
#include "LeggedAgent.h"
#include "FlexWalk.h"
#include "TSearch.h"
#include "VectorMatrix.h"
#include "random.h"

//CTRNN settings
const double StepSize = 0.01;
const int N = 2;          

// ADHP settings
const double plasticitydur = 5000;   //how long to allow adhp to act before measure fitness
const double testdur = 500;        //how long to run the agent for fitness evaluation

//input files
char walkerfname[] = "./Forward Walkers/3/bestind.dat";
char adhpfname[] = "./ADHP Mechanisms/best3_38.dat";

//output files
char trajectoryfname[] = "./trajectory.dat";

int main(){
    ifstream walkerfile;
    walkerfile.open(walkerfname);
    if (!walkerfile){
        cerr << "walker file not found" << endl;
        exit(EXIT_FAILURE);
    }   
    ifstream adhpfile;
    adhpfile.open(adhpfname);
    if (!adhpfile){
        cerr << "adhp file not found" << endl;
        exit(EXIT_FAILURE);
    }

    ofstream trajectoryfile;
    trajectoryfile.open(trajectoryfname);

    LeggedAgent Agent;
    Agent.NervousSystem.SetCircuitSize(N);
    walkerfile >> Agent.NervousSystem;
    cout << "Initial Biases: " << Agent.NervousSystem.biases << endl;

    Agent.NervousSystem.ShiftedRho(true);
    Agent.NervousSystem.SetHPPhenotype(adhpfile, StepSize, true); 
    cout << Agent.NervousSystem.l_boundary << " " << Agent.NervousSystem.u_boundary << endl;

    // check agent's initial walking speed before allowing ADHP to act
    double init_x = Agent.PositionX();
    Agent.Walk(testdur, StepSize,false,trajectoryfile);
    double fit = (Agent.PositionX() - init_x)/testdur;
    cout << "Initial fitness before ADHP: " << fit << endl;

    for (double t = 0; t < plasticitydur; t += StepSize){
        Agent.Step2CPG(StepSize,true);
    }

    init_x = Agent.PositionX();
    Agent.Walk(testdur, StepSize,true,trajectoryfile);
    fit = (Agent.PositionX() - init_x)/testdur;
    
    cout << "Biases after ADHP: " << Agent.NervousSystem.biases << endl;
    cout << "Fitness after ADHP: " << fit << endl;

    walkerfile.close();
    adhpfile.close();
    trajectoryfile.close();

    return 0;
}