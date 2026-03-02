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
const double plasticitydur = 5500;   //how long to allow adhp to act before measure fitness
const double testdur = 650;           //how long to run the agent for fitness evaluation
const double transientdur = 150;      //transient before adhp activates, also quick check to see if circuit state moving

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

    ofstream paramtrackfile;
    paramtrackfile.open("./paramtrack.dat");

    ofstream foottrackfile;
    foottrackfile.open("./foottrack.dat");

    LeggedAgent Agent;
    Agent.NervousSystem.SetCircuitSize(N);
    walkerfile >> Agent.NervousSystem;
    cout << "Initial Parameters: " << Agent.NervousSystem.taus << endl << Agent.NervousSystem.biases << endl << Agent.NervousSystem.weights << endl;

    Agent.NervousSystem.ShiftedRho(true);
    Agent.NervousSystem.SetHPPhenotype(adhpfile, StepSize, true); 
    cout << Agent.NervousSystem.l_boundary << " " << Agent.NervousSystem.u_boundary << " " << Agent.NervousSystem.windowsize << " " << Agent.NervousSystem.tausBiases << " " << Agent.NervousSystem.plasticitypars <<endl;

    // transient bc idfk
    for(double t = 0; t < transientdur; t += StepSize){
		Agent.NervousSystem.EulerStep(StepSize,false);
	}

    // // check agent's initial walking speed before allowing ADHP to act
    // double init_x = Agent.PositionX();
    // for(double t = 0; t < testdur; t += StepSize){
	// 	Agent.Step2CPG(StepSize,false);
	// }
    // double dist = Agent.PositionX() - init_x;
    // double fit = (Agent.PositionX() - init_x)/testdur;
    // cout << "Initial fitness before ADHP: " << fit << endl;

    // allow ADHP to act for designated time, then measure fitness

    for (double t = 0; t < plasticitydur; t += StepSize){
        Agent.Step2CPG(StepSize,true);
        paramtrackfile << " " << Agent.NervousSystem.biases << endl;
    }

    double init_x = Agent.PositionX();
    cout << "initial position for test: " << init_x << endl;

    int fmr_ft = 1;
    int new_ft = 1;
    double cycle_time = 0;
    bool first_lil_bit = false;

    for(double t = 0; t < testdur; t += StepSize){
        fmr_ft = new_ft;
        Agent.Step2CPG(StepSize,true);
        foottrackfile << " " << Agent.Leg.FootState << endl;
        new_ft = Agent.Leg.FootState;
        cycle_time += StepSize;
        if(fmr_ft - new_ft == -1){
            if(first_lil_bit){
                cout << "cycle: " << cycle_time << endl;
            }
            first_lil_bit = true;
            cycle_time = 0;
        }
		
		trajectoryfile << Agent.NervousSystem.outputs << endl;
	}
    double dist = Agent.PositionX() - init_x;
    cout << "Distance traveled during test: " << dist << endl;
    double fit = (Agent.PositionX() - init_x)/testdur;
    
    cout << "Biases after ADHP: " << Agent.NervousSystem.biases << endl;
    cout << "Fitness after ADHP: " << fit << endl;

    walkerfile.close();
    adhpfile.close();
    paramtrackfile.close();
    foottrackfile.close();
    trajectoryfile.close();

    return 0;
}