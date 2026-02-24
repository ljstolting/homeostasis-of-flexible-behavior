//Evolve Single-Legged Walkers without Flexibility or Homeostatic plasticity 
//I will then hand-design an ADHP mechanism that accommodates the most similar forward and backward walker

#include "CTRNN.h"
#include "LeggedAgent.h"
#include "FlexWalk.h"
#include "TSearch.h"
#include "VectorMatrix.h"
#include "random.h"

//EA settings
const int POPSIZE =      5000;
const int GENS =         100;
const double MUTVAR =    0.1;
const double CROSSPROB = 0.0;
const double EXPECTED =  1.1;
const double ELITISM =   0.1;
const bool seed_CC =     true; //seed with center crossing circuits?

//CTRNN settings
const double StepSize = 0.01;
const int N = 2;   
// Evolution Vector Size Calculation
const int VectSize = (2*N)+(N*N);       

//Define Evolution Ranges
const double WR =        10.0; //-10 -> +10 (smaller range used so that maximum ADHP traversal time is lower)
const double BR =        10.0; //(WR*N)/2; //<-enable to for allowing center crossing
const double T_min =      0.1; 
const double T_max =      2.0; 

//Task Params
const bool backwards = true;
const double testdur = 500; //how long to run the agent for fitness evaluation
const double transient = 150; //how long to run the agent before starting to evaluate fitness (allows it to get into a limit cycle)

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
}

// ------------------------------------
// Display functions
// ------------------------------------
ofstream Evolfile;
ofstream BestIndividualFile;
ofstream TrajectoryFile;

void ResultsDisplay(TSearch &s)
{
	TVector<double> bestVector;
	TVector<double> phenotype;
	phenotype.SetBounds(1, VectSize);

	// Save the genotype of the best individual
	bestVector = s.BestIndividual();
	GenPhenMapping(bestVector, phenotype);

    LeggedAgent Agent;
    Agent.NervousSystem.SetCircuitSize(N);
    phenotype >> Agent.NervousSystem;

    for (double t = 0; t < transient; t += StepSize){
        Agent.Step2CPG(StepSize,false);
    }
    Agent.DragBack();
    cout << Agent.cx << " " << Agent.cy << endl;

    double init_x = Agent.PositionX();

    Agent.Walk(testdur, StepSize,TrajectoryFile);

    double fitness = (Agent.PositionX() - init_x)/testdur;
    
    cout << "Best fitness:" << fitness << endl;
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

    TVector<double> phenotype(1,genotype.UpperBound());
    GenPhenMapping(genotype,phenotype);

    LeggedAgent Agent;
    Agent.NervousSystem.SetCircuitSize(N);
    phenotype >> Agent.NervousSystem;

    for (double t = 0; t < transient; t += StepSize){
        Agent.Step2CPG(StepSize,false);
    }
    double init_x = Agent.PositionX();

    Agent.Walk(testdur, StepSize);

    double fit = (Agent.PositionX() - init_x)/testdur;
    fit = fit * (-1*backwards); //if backwards is true, then we want to minimize forward distance, which is the same as maximizing backward distance

    return fit;
}

int main(int argc, const char* argv[]){
    Evolfile.open("./evol.dat");
	BestIndividualFile.open("./bestind.dat");
    TrajectoryFile.open("./trajectory.dat");

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
    TrajectoryFile.close();
    
    return 0;
}