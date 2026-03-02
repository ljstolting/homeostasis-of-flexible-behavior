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

//ADHP settings
const int num =          2; //future: SUM OF SOME INPUT FILE
const double Btau =      100;  //setting the time constant of regulation to the lowest value from before
const double SW =        0;    //setting the sliding window averaging to zero

const double plasticitydur = 5500; //in seconds
// const double transient = 150;   

//Neuromodulation settings
// const int num_NM =       3; //future: SUM OF SOME INPUT FILE, change genphenmapping function to be more like arbdparam

//Define Evolution Ranges
//ctrnn
const double WR =        10.0; //-10 -> +10 (smaller range used so that maximum ADHP traversal time is lower)
const double BR =        10.0; //(WR*N)/2; //<-for allowing center crossing
const double T_min =      0.1; 
const double T_max =      2.0; 
//adhp
const double LB_min =    0.0;
const double Range_min = 0.0;
const double LB_max =    1.0;
const double Range_max = 1.0; 
//neuromodulation
const double Wnm_R =     10.0; //traverse up to +/-(half the allowed range) for each parameter with neuromodulation
const double Bnm_R =     10.0;
const double Tnm_R =     0.95;

// Vector Size Calculation
const int ctrnnvectsize = (2*N)+(N*N);
const int VectSize = 2*(ctrnnvectsize)+2*N;

//USING BESTIND FILE
char bestindfname[] = "./best_test.dat";
//USING PHENOTYPE FILE
// char bestindfname[] = "./best_phen.dat";
//USING INDIVIDUAL WALKER FILE - DEBUGGING
// char bestindfname[] = "./Forward Walkers/3/bestind.dat";

char adhpfname[] = "./ADHP Mechanisms/best3_38.dat";

int main(){
    ifstream bestindfile;
    bestindfile.open(bestindfname);

    //debugging
    ifstream adhpfile;
    adhpfile.open(adhpfname);

    ofstream trajectoryfile;
    trajectoryfile.open("./trajectory.dat");

    LeggedAgent Agent;
    Agent.NervousSystem.SetCircuitSize(N);
    TVector<double> neuromodvec(1,ctrnnvectsize);

    // USING BESTIND FILE
    Setup(bestindfile,Agent,neuromodvec);

    //USING PHENOTYPE FILE
    // TVector<double> phenotype(1,VectSize);
    // bestindfile >> phenotype;
    // cout << "Phenotype read from file: " << phenotype << endl;

    // Setup(phenotype,Agent,neuromodvec);

    //USING INDIVIDUAL WALKER FILE - DEBUGGING
    // bestindfile >> Agent.NervousSystem;
    // Agent.NervousSystem.ShiftedRho(true);
    // Agent.NervousSystem.SetHPPhenotype(adhpfile, StepSize, true);

    cout << "Initial Parameters: " << Agent.NervousSystem.taus << endl << Agent.NervousSystem.biases << endl << Agent.NervousSystem.weights << endl;
    cout << "ADHP params:" << endl << Agent.NervousSystem.l_boundary << endl << Agent.NervousSystem.u_boundary << endl << Agent.NervousSystem.windowsize << endl << Agent.NervousSystem.tausBiases << endl << Agent.NervousSystem.plasticitypars << endl << endl;
    cout << "neuromodulatory params:" << endl << neuromodvec << endl << endl;

    double fit = FlexibleWalking(Agent,neuromodvec,plasticitydur,0,true);

    // for (double t = 0; t < transient; t += StepSize){
    //     Agent.Step2CPG(StepSize,false);
    // }

    // double init_x = Agent.PositionX();

    // Agent.Walk(plasticitydur, StepSize, false, trajectoryfile);

    // double fit = (Agent.PositionX() - init_x)/plasticitydur;
    
    // cout << Agent.cx << " " << Agent.Leg.FootX << endl;
    // cout << fit << endl;
    // -4.49679 -1.90087

    bestindfile.close();
    adhpfile.close();
    trajectoryfile.close();

    return 0;
}