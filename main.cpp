//Evolve Flexible Single-Legged Walkers with Homeostatic plasticity 

#include "CTRNN.h"
#include "LeggedAgent.h"
#include "FlexWalk.h"
#include "TSearch.h"
#include "VectorMatrix.h"
#include "random.h"

//EA settings
const int POPSIZE =      370;
const int GENS =         250;
// const int trials = 1;    // number of times to run the EA from random starting pop
const double MUTVAR =    0.1;
const double CROSSPROB = 0.0;
const double EXPECTED =  1.1;
const double ELITISM =   0.1;
const bool seed_CC =     false; //seed with center crossing circuits?

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
// const int num_NM =       15; //future: SUM OF SOME INPUT FILE, change genphenmapping function to be general like arbdparam

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

// Evolution Vector Size Calculation
const int ctrnnvectsize = (2*N)+(N*N);
const int VectSize = (2*ctrnnvectsize) + (2*N);

// ------------------------------------
// Genotype-Phenotype Mapping Functions  - GENERALIZED to use parameter vectors for both adhp and neuromodulators
// ------------------------------------
void GenPhenMapping(TVector<double> &gen, TVector<double> &phen)
{
    int k = 1;
    // CTRNN
    // Taus
    for (int i = 1; i <= N; i++) {
        phen(k) = MapSearchParameter(gen(k), T_min, T_max);
        k ++;
    }
    // Biases
    for (int i = 1; i <= N; i++) {
        phen(k) = MapSearchParameter(gen(k), -BR, BR);
        k ++;
    }
    // Weights
    for (int i = 1; i <= N; i++) {
        for (int j = 1; j <= N; j++){
            phen(k) = MapSearchParameter(gen(k), -WR, WR);
            k ++;
        }
    }

    // ADHP
	// Lower Bounds
	for (int i = 1; i <= num; i++) {
		phen(k) = MapSearchParameter(gen(k), LB_min, LB_max);
		k++;
	}
	// Ranges
	for (int i = 1; i <= num; i++) {
		phen(k) = MapSearchParameter(gen(k), Range_min, Range_max);
		k++;
	}
    
    //NEUROMODULATORY VECTOR
    // Taus
    for (int i = 1; i <= N; i++) {
        phen(k) = MapSearchParameter(gen(k), -Tnm_R, Tnm_R);
        k ++;
    }
    // Biases
    for (int i = 1; i <= N; i++) {
        phen(k) = MapSearchParameter(gen(k), -Bnm_R, Bnm_R);
        k ++;
    }
    // Weights
    for (int i = 1; i <= N; i++) {
        for (int j = 1; j <= num; j++){
            phen(k) = MapSearchParameter(gen(k), -Wnm_R, Wnm_R);
            k ++;
        }
    } 

}

// ------------------------------------
// Display functions
// ------------------------------------
ofstream Evolfile;
ofstream BestIndividualFile;

int trial = 1;
void ResultsDisplay(TSearch &s)
{
	TVector<double> bestVector;
	TVector<double> phenotype;
	phenotype.SetBounds(1, VectSize);

	// Save the genotype of the best individual
	bestVector = s.BestIndividual();
	GenPhenMapping(bestVector, phenotype);

    TVector<double> neuromodvec(1,ctrnnvectsize);
    LeggedAgent Agent;
    Setup(phenotype,Agent,neuromodvec); //fills empty neuromodulatory vector
    //default should take care of the plasticity time constants but it doesn't...
    // for(int i = 1; i <= Agent.NervousSystem.CircuitSize(); i ++){
    //     Agent.NervousSystem.SetNeuronBiasTimeConstant(i,Btau);
    // }
    //CTRNN output
    TakeDown(Agent,BestIndividualFile,neuromodvec);
    // Other things I might consider wanting later... or splitting into separate files, of course. 
	// cout << plasticitypars << endl;
	// BestIndividualFile << trial << endl;
	// BestIndividualFile << plasticitypars << endl;

	// cout << trial << "finished" << endl;
    FlexibleWalking(Agent,neuromodvec,plasticitydur,true);

	trial ++;
}

void EvolutionaryRunDisplay(TSearch &s)
{
	//cout << Generation << " " << BestPerf << " " << AvgPerf << " " << PerfVar << endl;
	Evolfile << s.Generation() << " " << s.BestPerformance() << " " << s.AvgPerformance() << " " << s.PerfVariance() << endl;

	TVector<double> bestVector;
	TVector<double> phenotype;
	phenotype.SetBounds(1, VectSize);

	// Save the genotype of the best individual
	bestVector = s.BestIndividual();
	GenPhenMapping(bestVector, phenotype);

	Evolfile << phenotype << endl;
}

//actual fitness function in form GA needs
double FitnessFunction(TVector<double>& genotype){
    // cout << "Fitness func started " << endl;
    TVector<double> phenotype(1,genotype.UpperBound());
    GenPhenMapping(genotype,phenotype);
    // cout << "mapped" << endl;
    LeggedAgent Agent; //should return with 3 neurons by default....
    // cout << Agent.NervousSystem.CircuitSize();
    TVector<double> neuromodvec(1,ctrnnvectsize);
    Setup(phenotype, Agent, neuromodvec);
    // for(int i = 1; i <= Agent.NervousSystem.CircuitSize(); i ++){
    //     Agent.NervousSystem.SetNeuronBiasTimeConstant(i,Btau);
    // }
    double fit = FlexibleWalking(Agent,neuromodvec,plasticitydur);

    return fit;
}

int main(int argc, const char* argv[]){
    Evolfile.open("./evol.dat");
	BestIndividualFile.open("./bestind.dat");

    long randomseed = static_cast<long>(time(NULL));
    if (argc == 2){randomseed += atoi(argv[1]);}

    TSearch s(VectSize,FitnessFunction);
    s.SetRandomSeed(randomseed);
    s.SetSearchResultsDisplayFunction(ResultsDisplay);
    s.SetPopulationStatisticsDisplayFunction(EvolutionaryRunDisplay);
    s.SetSelectionMode(RANK_BASED);
    s.SetReproductionMode(GENETIC_ALGORITHM);
    s.SetPopulationSize(POPSIZE);
    s.SetMaxGenerations(GENS);
    s.SetCrossoverProbability(CROSSPROB);
    s.SetCrossoverMode(UNIFORM);
    s.SetMutationVariance(MUTVAR);
    s.SetMaxExpectedOffspring(EXPECTED);
    s.SetElitistFraction(ELITISM);
    s.SetSearchConstraint(1);
    s.SetReEvaluationFlag(0);

    s.ExecuteSearch(seed_CC);
    

    Evolfile.close();
	BestIndividualFile.close();
    
    return 0;
}