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

const double plasticitydur = 5000; //in seconds 
const int rounds = 3;

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
char bestindfname[] = "./designedbestind.dat";
//USING PHENOTYPE FILE
// char bestindfname[] = "./Evolutions_2N/8/phen.dat";
//USING INDIVIDUAL WALKER FILE - DEBUGGING
// char bestindfname[] = "./Backward Walkers/38/bestind.dat";

// char adhpfname[] = "./ADHP Mechanisms/test_Forward.dat";
// char trajfname[] = "./Backward Walkers/38/trajectory.dat";

double FitnessFunction(ifstream &bestindfile){
    // cout << "Fitness func started " << endl;
    // TVector<double> phenotype(1,genotype.UpperBound());
    // GenPhenMapping(genotype,phenotype);
    // cout << "mapped" << endl;
    LeggedAgent Agent; //should return with 3 neurons by default....
    Agent.NervousSystem.SetCircuitSize(N);
    cout << Agent.NervousSystem.CircuitSize() << endl;
    TVector<double> neuromodvec(1,ctrnnvectsize);
    Setup(bestindfile, Agent, neuromodvec);
    // for(int i = 1; i <= Agent.NervousSystem.CircuitSize(); i ++){
    //     Agent.NervousSystem.SetNeuronBiasTimeConstant(i,Btau);
    // }
    cout << "Initial Parameters: " << Agent.NervousSystem.taus << endl << Agent.NervousSystem.biases << endl << Agent.NervousSystem.weights << endl;
    cout << "ADHP params:" << endl << Agent.NervousSystem.l_boundary << endl << Agent.NervousSystem.u_boundary << endl << Agent.NervousSystem.windowsize << endl << Agent.NervousSystem.tausBiases << endl << Agent.NervousSystem.plasticitypars << endl << endl;
    cout << "neuromodulatory params:" << endl << neuromodvec << endl << endl;
    double fit = FlexibleWalking(Agent,neuromodvec,plasticitydur,rounds,true);

    return fit;
}

int main(){
    ifstream bestindfile;
    bestindfile.open(bestindfname);

    //debugging
    // ifstream adhpfile;
    // adhpfile.open(adhpfname);

    // ofstream paramfile;
    // paramfile.open("./paramtrack.dat");

    // ofstream ubsfile;
    // ubsfile.open("./calculatedubs_bw38.dat");

    // ifstream trajectoryfile;
    // trajectoryfile.open(trajfname);

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

    // cout << "Initial Parameters: " << Agent.NervousSystem.taus << endl << Agent.NervousSystem.biases << endl << Agent.NervousSystem.weights << endl;
    // cout << "ADHP params:" << endl << Agent.NervousSystem.l_boundary << endl << Agent.NervousSystem.u_boundary << endl << Agent.NervousSystem.windowsize << endl << Agent.NervousSystem.tausBiases << endl << Agent.NervousSystem.plasticitypars << endl << endl;
    // cout << "neuromodulatory params:" << endl << neuromodvec << endl << endl;

    // TVector<double> neural_lc(1,50001);
    // trajectoryfile >> neural_lc;
    // // cout << neural_lc; 

    // TVector<double> sorted_lc(1,50001);
    // SortTraj(sorted_lc,neural_lc);
    // // cout << sorted_lc << endl;
    // for(double lb=0.1;lb<=0.9;lb+=0.01){
    //     double ub = CalcUB(sorted_lc, lb);
    //     ubsfile << ub << endl;
    // }
    

    // for(double t=StepSize;t<=20000;t+=StepSize){
    //     Agent.Step2CPG(StepSize,true);
    //     paramfile << Agent.NervousSystem.biases << endl;
    // }

    // cout << "after ADHP " << Agent.NervousSystem.biases << endl;

    // for (int i = 1; i <= 10; i++){
    //     double fit = meas_velocity(Agent,trajectoryfile,paramfile, true);
        
    //     cout << fit << endl;
    // }

    // cout << "after testing " << Agent.NervousSystem.biases << endl;

    double fit = FlexibleWalking(Agent,neuromodvec,plasticitydur,rounds,true,true);
    // double fit = FitnessFunction(bestindfile);
    // cout << fit << endl;

    bestindfile.close();
    // adhpfile.close();
    // paramfile.close();
    // trajectoryfile.close();
    // ubsfile.close();

    return 0;
}