// ---------------------------------------------------------------
//  Family of functions for evaluating CTRNNs on their pyloricness
//
//  Lindsay Stolting 4/6/24
// ---------------------------------------------------------------

#include "CTRNN.h"
#include "random.h"
#include "VectorMatrix.h"
#include <iostream>
#include <cmath>

using namespace std;

// Task params
// const double TransientDuration = 500; //seconds without HP
// const double PlasticDuration1 = 5000; //seconds allowing HP to act
// const double PlasticDuration2 = 5000; //number of seconds to wait before testing again, to make sure not relying on precise timing
const double TestDuration = 100; //maximum number of seconds allowed to locate 3 cycles of the rhythm
const bool HPequilibrate = false; //is HP on during the transient/equilibration period? 
const bool HPtest = true;       //is HP on during test (shouldn't matter if platicity time constants are slow enough, *****but seems to be mattering in select cases****)
// const double StepSize = 0.1;
const double StepSize = 0.01;
const int TestSteps = TestDuration/StepSize; // in steps

// Detection params
const double burstthreshold = .5; //threshold that must be crossed for detecting bursts
const double tolerance = 3*StepSize; //for detecting double periodicity

// Evaluation params (adjust the fitness function)
const double scaling_factor = 0.05;   //how much is awarded for each binary criteria met (changes importance relative to timing award)
const bool timing_award = true;	  //award extra points for timing (true) or cap at reaching all 6 criteria (false)?
const bool legacy = false;            //use the old set of timing criteria from the Prinz paper using PD end as anchor (LPstart<PYstart, LPend<PYend, PDend<LPstart)
									  //or the new one using LPstart as anchor (LP starts in silence, PYstart<LPend, LPend<PYend)

//Take an output history matrix and return (# oscillating neurons, LPstart, LPend, PYstart, PYend, PDstart, PDend, period) in that order, in time units, relative to PDstart
void BurstTimesfromOutputHist(TMatrix<double> &OutputHistory, TVector<double> &features){
	features.FillContents(0);

	int N = OutputHistory.ColumnUpperBound();

	TVector<double> maxoutput(1,N);
	maxoutput.FillContents(0.0);
	TVector<double> minoutput(1,N);
	minoutput.FillContents(1.0);

	// Step through time series until you identify 3 PD starts (capping 2 full cycles), keeping track of whether each other neuron crossed the threshold or not
	int tstep = 1;
	double t = StepSize;
	int PDstartcount = 0;
	TVector<int> PDstarts(1,3);
	PDstarts.FillContents(0);

	while (tstep < OutputHistory.RowUpperBound() && PDstartcount < 3) {
		for (int i = 1; i <= N; i += 1) {
			if (OutputHistory[tstep][i] > maxoutput(i)) {maxoutput(i)=OutputHistory[tstep][i];}
			if (OutputHistory[tstep][i] < minoutput(i)) {minoutput(i)=OutputHistory[tstep][i];}
		}
		// trajfile << Agent.NeuronOutput(1) << " " << Agent.NeuronOutput(2) << " " << Agent.NeuronOutput(3) << endl;

		//Check for PD start
		if (OutputHistory(tstep,3) < burstthreshold && OutputHistory(tstep+1,3) > burstthreshold){
			PDstartcount += 1;
			PDstarts[PDstartcount] = tstep;
		}
		tstep += 1;
		t += StepSize;
	}

	for (int i = 1; i <= N; i += 1) {
		// SHORT HAND FOR ALL NEURONS OSCILLATING APPRECIABLY
		if (minoutput[i]<(burstthreshold)) {
			if (maxoutput[i]>burstthreshold) {
				features[1] += 1;
			}
		}
	}
	
	//list of conditions that return zero fitness because they preclude accurately calculating the burst start and end times (assumptions)
	if (features[1] < 3){
		// cout << "not all neurons cross burst threshold";
		return;
	}
	if (PDstartcount < 3){
		cout << "unable to find two full cycles; may want to increase transient, lengthen runtime, or speed up slowest timescale" << endl;
		// cout << features << endl;
		return;
	}
	// at the two points where PD crosses up, are the other two neurons approximately in the same place?
	if ((abs(OutputHistory(PDstarts[1],1) - OutputHistory(PDstarts[2],1))>tolerance)||(abs(OutputHistory(PDstarts[1],2) - OutputHistory(PDstarts[2],2))>tolerance)){
		cout << "Too many PD bursts found in one cycle - suspected multiperiodicity"<<endl;
		return;
	}
	
	int PDend = 0;
	int PDendcount = 0;
	int LPstart = 0;
	int LPstartcount = 0;
	int LPend = 0;
	int PYstart = 0;
	int PYstartcount = 0;
	int PYend = 0;
	for (int step=PDstarts(1); step<=PDstarts(2); step ++){
		if (PDendcount == 0){
			if (OutputHistory(step,3)>burstthreshold){
				if (OutputHistory(step+1,3)<burstthreshold){
					PDend = step;
					PDendcount ++;
					//cout << "PDend";
				}
			}
		}
		if (LPstartcount == 0){
			if (OutputHistory(step,1)<burstthreshold){
				if (OutputHistory(step+1,1)>burstthreshold){
					LPstart = step;
					LPstartcount ++;
					// cout << "LPstart: " << LPstart;
				}
			}
		}
		if (PYstartcount == 0){
			if (OutputHistory(step,2)<burstthreshold){
				if (OutputHistory(step+1,2)>burstthreshold){
					PYstart = step;
					PYstartcount ++;
					//cout << "PYstart";
				}
			}
		}
	}
	if (LPstartcount == 1){
		for (int step=LPstart;step<=PDstarts(3);step++){
			if (OutputHistory(step,1)>burstthreshold){
				if (OutputHistory(step+1,1)<burstthreshold){
					LPend = step;
					//cout << "LPend";
					break;
				}
			}
		}
	}
	else{cout << "Too few or too many LP bursts found in one cycle" << endl; return;}

	if (PYstartcount == 1){
		for (int step=PYstart;step<=PDstarts(3);step++){
			if (OutputHistory(step,2)>burstthreshold){
				if (OutputHistory(step+1,2)<burstthreshold){
					PYend = step;
					//cout << "PYend" << endl;
					break;
				}
			}
		}
	}
	else {cout << "Too few or too many PY bursts found in one cycle" << endl; return;}

	double period = (PDstarts[2] - PDstarts[1])*StepSize;

	// convert all features to be relative to PDstart, and in time units, and collect in solution vector
	features[2] = std::fmod((LPstart-PDstarts[1])*StepSize,period);
	features[3] = std::fmod((LPend-PDstarts[1])*StepSize,period);
	features[4] = std::fmod((PYstart-PDstarts[1])*StepSize,period);
	features[5] = std::fmod((PYend-PDstarts[1])*StepSize,period);
	features[6] = std::fmod((PDstarts[1]-PDstarts[1])*StepSize,period); // equals 0, but just in case there is ever another relative start point
	features[7] = std::fmod((PDend-PDstarts[1])*StepSize,period);
	features[8] = period;

	return;
}

//take the vector of rhythm features that are output by the previous function and return pyloric fitness, with a particular fitness awarding system
double PyloricFitFromFeatures(TVector<double> &FeatureVect){
	double fitness = 0.0;

	double num_oscillating = FeatureVect[1];
	double LPstart = FeatureVect[2];
	double LPend = FeatureVect[3];
	double PYstart = FeatureVect[4];
	double PYend = FeatureVect[5];
	double PDstart = FeatureVect[6];
	double PDend = FeatureVect[7];
	double period = FeatureVect[8];

	double LPdelay = LPstart-PDstart;
	double PYdelay = PYstart-PDstart;

	// cout << "before " << FeatureVect << endl;

	//number neurons oscillating
	int criteria = int(num_oscillating);
	
	if(legacy){
		// Recorrect to ensure start < end
		for (int i=2; i <= 6; i+=2){
			if (FeatureVect(i) > FeatureVect(i+1)){
				FeatureVect(i+1) = FeatureVect(i+1) + FeatureVect(8);
			}
		}
		double LPstart = FeatureVect[2];
		double LPend = FeatureVect[3];
		double PYstart = FeatureVect[4];
		double PYend = FeatureVect[5];
		double PDstart = FeatureVect[6];
		double PDend = FeatureVect[7];

		// 	ORDERING CRITERIA
		// 
		if (LPstart <= PYstart){
			// cout << "order1" << endl;
			criteria += 1;
		}
		if (LPend <= PYend){
			// cout << "order2" << endl;
			criteria += 1;
		}
		if (PDend <= LPstart){
			// cout << "order3" << endl;
			criteria += 1;
		}
	}

	else{
		//first adjust relative to LP
		for(int i = 7;i >= 2;i--){
			FeatureVect[i] = FeatureVect[i] - FeatureVect[2];
			if (FeatureVect[i] < 0){
				FeatureVect[i] = FeatureVect[i] + period;
			}
		}
		LPstart = FeatureVect[2];
		LPend = FeatureVect[3];
		PYstart = FeatureVect[4];
		PYend = FeatureVect[5];
		PDstart = FeatureVect[6];
		PDend = FeatureVect[7];

		// cout << "PDstart " << PDstart << " PDend " << PDend << endl << ((PDstart<PDend) && (PYstart<PYend)) << endl;

		//Check whether PYstart<PYend and PDstart<PDend (none of them span LPstart)
		if ((PDstart<PDend) && (PYstart<PYend)){
			criteria += 1;
			// cout << criteria << endl;
		}
		if (PYstart<LPend){
			criteria += 1;
			// cout << criteria << endl;
		}
		if (LPend<PYend){
			criteria += 1;
			// cout << criteria << endl;
		}
	}

	fitness += (criteria * scaling_factor);
	// cout << fitness << endl;

	//additional fitness for conforming to timing averages
	if (timing_award && (criteria == 6)){
		double LPburstlen = LPend - LPstart; 
		double LPdutycycle = LPburstlen/period; //burstduration/period
		double LPdutycyclezscore = abs(LPdutycycle - .264)/.059;
		double PYburstlen = PYend-PYstart;
		double PYdutycycle = PYburstlen/period; //burstduration/period
		double PYdutycyclezscore = abs(PYdutycycle - .348)/.054;
		double PDburstlen = PDend-PDstart;
		double PDdutycycle = PDburstlen/period; //burstduration/period
		double PDdutycyclezscore = abs(PDdutycycle - .385)/.040;
		//taking the delay from the original, pd-centric timing for alignment with the paper
		double LPstartphase = LPdelay/period; //delay/period
		double LPstartphasezscore = abs(LPstartphase - .533)/.054;
		double PYstartphase = PYdelay/period; //delay/period
		double PYstartphasezscore = abs(PYstartphase - .758)/.060;

		double average = (LPdutycyclezscore+PYdutycyclezscore+PDdutycyclezscore+LPstartphasezscore+PYstartphasezscore)/5;
		fitness += 1/(average);
	}
	
	// cout << "fitness" << fitness;
	return fitness;
}

//Test pyloric performance of a circuit, which is assumed to be already equilibrated
double PyloricPerformance(CTRNN &Agent)
{
	// cout << "6" << Agent.biases << endl;
	int N = Agent.CircuitSize();
	TMatrix<double> OutputHistory;
	OutputHistory.SetBounds(1,TestSteps,1,N);
	OutputHistory.FillContents(0.0);

	double fitness = 0.0;

	TVector<double> maxoutput(1,N);
	maxoutput.FillContents(0.0);
	TVector<double> minoutput(1,N);
	minoutput.FillContents(1.0);
	// cout << "7" << Agent.biases << endl;

	// Run the circuit to calculate Pyloric fitness while HP either on or off depending on global setting.
	int t = 0;
	for (double time = StepSize; time <= TestDuration; time += StepSize) {
		t += 1;
		for (int i = 1; i <= N; i += 1) {
			OutputHistory[t][i] = Agent.NeuronOutput(i);
			if (Agent.NeuronOutput(i) > maxoutput[i]) {maxoutput[i]=Agent.NeuronOutput(i);}
			if (Agent.NeuronOutput(i) < minoutput[i]) {minoutput[i]=Agent.NeuronOutput(i);}
		}
		Agent.EulerStep(StepSize,HPtest);
	}

	TVector<double> features(1,8);
	BurstTimesfromOutputHist(OutputHistory, features);
	// cout << features << endl;

	fitness = PyloricFitFromFeatures(features);

	return fitness;
}

//Overload which allows the circuit to equilibrate for some length of time before testing performance
double PyloricPerformance(CTRNN &Agent, double TransientDur)
{
	// Initialize the outputs at 0.5 for all neurons in the circuit
	// Agent.RandomizeCircuitOutput(0.5, 0.5);

	// Run the circuit for an initial transient; HP is on or off (depending on global setting) and fitness is not evaluated
	for (double t = StepSize; t <= TransientDur; t += StepSize) {
		Agent.EulerStep(StepSize,HPequilibrate);
	}

	double fitness = PyloricPerformance(Agent);

	return fitness;
}

//overload but output the neuron trajectories to a file and the burst start/endpoints to another file. Assumes network already equilibrated
double PyloricPerformance(CTRNN &Agent, ofstream &trajfile, ofstream &burstfile){
	int N = Agent.CircuitSize();
	TMatrix<double> OutputHistory;
	OutputHistory.SetBounds(1,TestSteps,1,N);
	OutputHistory.FillContents(0.0);

	double fitness = 0.0;

	TVector<double> maxoutput(1,N);
	maxoutput.FillContents(0.0);
	TVector<double> minoutput(1,N);
	minoutput.FillContents(1.0);

	// Run the circuit to calculate Pyloric fitness while HP either on or off depending on global setting.
	int t = 0;
	for (double time = StepSize; time <= TestDuration; time += StepSize) {
		t += 1;
		for (int i = 1; i <= N; i += 1) {
			OutputHistory[t][i] = Agent.NeuronOutput(i);
			if (Agent.NeuronOutput(i) > maxoutput[i]) {maxoutput[i]=Agent.NeuronOutput(i);}
			if (Agent.NeuronOutput(i) < minoutput[i]) {minoutput[i]=Agent.NeuronOutput(i);}
			trajfile << Agent.NeuronOutput(i) << " ";
		}
		trajfile << endl;
		Agent.EulerStep(StepSize,HPtest);
	}

	TVector<double> features(1,8);
	BurstTimesfromOutputHist(OutputHistory, features);
	burstfile << features;

	fitness = PyloricFitFromFeatures(features);
	// cout << "higher fit:" << fitness << endl;

	return fitness;
}

//TRIO OF UTILITIES FOR GENERATING A GRID OF POINTS, WHICH HAD TO BE IN THIS FILE FOR SOME REASON I DON'T REMEMBER...
void converttobase(int N,int resolution,TVector<int> &converted){
	//recursive function to convert to base of choice
	int dimension = converted.UpperBound();
	while(N>0){
		int r = N % resolution;
		N = (N-r) / resolution;
		converted[dimension] = r;
		dimension -= 1;
	}
	return;
}

void PointCombos(TMatrix<int> &answer,int resolution){
	//find all the point combinations for a given dimension count and resolution
	int num_points = answer.RowUpperBound();
	int dimension = answer.ColumnUpperBound();
	for (int i = 1; i <= num_points; i ++){
		TVector<int> row(1,dimension);
		row.FillContents(0);
		converttobase(i-1,resolution,row);
		for(int j = 1; j <= dimension; j++){
			answer(i,j) += row(j); //allows for 1 indexing if matrix initialized with ones
		}
	}
	return;
}

void PointGrid(TMatrix<double> &points, TVector<double> &parVals){
	
	//calculate initial conditions on a predefined grid -- gets too unruly when many parameter dimensions are considered
	int resolution = parVals.Size(); //number of points per dimenison

	int num = points.ColumnUpperBound();

	TMatrix<double> par_ICs(1,num,1,resolution);
	for (int i = 1; i <= num; i++){
		for (int j = 1; j <= resolution; j++){
			par_ICs(i,j) = parVals(j); // assumes that all dimensions have the same resolution of points spaced in the same way
		}
	}

	TMatrix<int>par_idxs(1,points.RowUpperBound(),1,points.ColumnUpperBound());
	par_idxs.FillContents(1);
	
	PointCombos(par_idxs,resolution);
	
	for (int i = 1; i <= points.RowUpperBound(); i ++){
		for (int b=1;b<=num;b++){
			points(i,b) = par_ICs(b,par_idxs(i,b));
		}
	}
	return;	
}

