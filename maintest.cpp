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
const int N = 3;          //this repository will not be general to higher neural dimensions - must be 3 neuron CTRNN.
                         //you CAN change the number of neurons that are controlled by ADHP
//ADHP settings
const bool shiftedrho = true;
const int num =          3; //future: SUM OF SOME INPUT FILE
const double Btau =      100;  //setting the time constant of regulation to the lowest value from before
const double SW =        0;    //setting the sliding window averaging to zero

const double plasticitydur = 5000; //in seconds

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

char bestindfname[] = "./bestind.dat";
// char bestindfname[] = "./best_phen.dat";
// char bestindfname[] = "walker.ns";

int main(){
    ifstream bestindfile;
    bestindfile.open(bestindfname);

    LeggedAgent Agent;

    TVector<double> neuromodvec(1,ctrnnvectsize);

    TVector<double> phenotype(1,VectSize);
    // bestindfile >> phenotype;

    // Setup(phenotype,Agent,neuromodvec);

    Setup(bestindfile,Agent,neuromodvec);

    cout << "ADHP params" << endl << Agent.NervousSystem.l_boundary << endl << Agent.NervousSystem.u_boundary << endl << Agent.NervousSystem.windowsize << endl << Agent.NervousSystem.tausBiases << endl;
    cout << "neuromodulatory params" << endl << neuromodvec << endl;

    double fit = FlexibleWalking(Agent,neuromodvec,plasticitydur,true);
    cout << fit << endl;
    bestindfile.close();

    return 0;
}