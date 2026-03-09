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
const int N = 2;         

//ADHP settings
const int num =          2; //future: SUM OF SOME INPUT FILE

const double plasticitydur = 5500; //in seconds 

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
char bestindfname[] = "./bestind.dat";
//USING PHENOTYPE FILE
// char bestindfname[] = "./best_phen.dat";
//USING INDIVIDUAL WALKER FILE - DEBUGGING
// char bestindfname[] = "./Forward Walkers/3/bestind.dat";

char adhpfname[] = "./ADHP Mechanisms/best3_38.dat";
char trajfname[] = "./Forward Walkers/3/testtrajectory0.dat";

int main(){
    ifstream bestindfile;
    bestindfile.open(bestindfname);

    //debugging
    ifstream adhpfile;
    adhpfile.open(adhpfname);

    ifstream trajectoryfile;
    trajectoryfile.open(trajfname);

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

    TVector<double> neural_lc(1,50001);
    trajectoryfile >> neural_lc;
    // cout << neural_lc; 

    TVector<double> sorted_lc(1,50001);
    SortTraj(sorted_lc,neural_lc);
    // cout << sorted_lc << endl;
    double ub = CalcUB(sorted_lc, .6);
    cout << "Calculated UB: " << ub << endl;

    double fit = FlexibleWalking(Agent,neuromodvec,plasticitydur,2,true);

    bestindfile.close();
    adhpfile.close();
    trajectoryfile.close();

    return 0;
}