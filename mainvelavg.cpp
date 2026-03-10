//-----------------------------------------------------------------------
// For each point on the plane surrounding evolved walker solutions
// without ADHP, compute the walking velocity, average values taken on by 
// the neurons (as calculated in the pyloric ADHP paper), max and min, and 
// the set of ranges that best balance the rhythm, for each lower bound on
// [0.025,.975,0.025] that is contained within [min,max] for each neuron. Other 
// lb's will correspond to a value of 0 as a placeholder in the data matrix

// This will allow us to determine what ADHP mechanisms would be successful
// and be more permissive of ADHP mechanisms with range. For every lb in
// [min,max], there is exactly one range that balances every rhythm. This forms
// a curve. Given an ADHP mechanism with a certain LB, if its UB lies above
// the curve, drive will be negative. If its UB lies below the curve, then
// drive will be positive
//-----------------------------------------------------------------------

#include "CTRNN.h"
#include "LeggedAgent.h"
#include "FlexWalk.h"
#include "VectorMatrix.h"
#include "random.h"

// Circuit size
const int N = 2;

// LB sampling resolution - tries all the ones that are within the max and min of the cycle
const double LB_step = 0.025;
const double LB_start = LB_step;
const double LB_stop = 1-LB_step;
int num_LB_steps = int((LB_stop-LB_start)/LB_step)+1;

//INPUT FILES
// Specifies circuit sampling resolution (each dimension in a row with start, stop ,step)
char resfname[] = "./circuitsamplingres.dat";
// Specifies a circuit around which to center the plane (specifies non-homeostatic parameters)
char circuitfname[] = "./Evolutions_2N/8/extracted_bw.dat";

//OUTPUT FILES
// Holds average values for each neuron for every circuit
char avgsfname[] = "./Evolutions_2N/8/extracted_bwavgs.dat";
// Holds walking velocity (fitness) values for every circuit
char velfname[] = "./Evolutions_2N/8/extracted_bwvel.dat";
// Holds boundary curve for each neuron for every circuit
char boundcurvefname[] = "./Evolutions_2N/8/extracted_bwboundcurve.dat";

void CalcAvgandBoundCurve(TMatrix<double> &boundcurve, TVector<double> &avgval, double &velocity, LeggedAgent Agent){
    // set boundcurve vector, avg, and fitness to zeros if not already
    boundcurve.FillContents(0);
    avgval.FillContents(0);
    velocity = 0;

    // Set up matrix to hold neural time series if necessary 
    int max_steps = int(test_dur/StepSize);
    TMatrix<double> bloated_timeseries(1,max_steps,1,N);

    double distance_traveled = 0;
    double cycle_time = 1;
    distance_and_time(Agent,distance_traveled,cycle_time,bloated_timeseries,true);
    // cout << "dist: " << distance_traveled << " time: " << cycle_time << endl;
    velocity = distance_traveled/cycle_time;
    int used_steps = int(cycle_time/StepSize);
    TMatrix<double> full_timeseries; //place to hold the curve values if cycle detected 

    if(cycle_time==1){
        avgval = Agent.NervousSystem.outputs;
    }
    
    else{
        //copy only the used part of the timeseries over
        full_timeseries.SetBounds(1,used_steps,1,N);
        full_timeseries.FillContents(0);
        for (int i=1;i<=used_steps;i++){
            for (int j=1;j<=N;j++){
                full_timeseries[i][j] = bloated_timeseries[i][j];
            }
        }

        for(int i=1;i<=N;i++){
            //extract each neuron's timeseries in turn
            TVector<double> neuron_timeseries(1,used_steps);
            full_timeseries.ExtractColumn(i,neuron_timeseries);

            //record its average
            avgval(i) = neuron_timeseries.Sum()/used_steps;

            //and sort it
            TVector<double> sorted_neuron_timeseries(1,used_steps);
            SortTraj(sorted_neuron_timeseries,neuron_timeseries);
            // cout << sorted_neuron_timeseries << endl;

            double min_val = sorted_neuron_timeseries[used_steps];
            double avg_val = sorted_neuron_timeseries.Sum()/used_steps;
            // cout << "N" << N << " avg val: " << avg_val << endl;

            //Call the CalcUB function in a loop to get corresponding UB values
            int LB_idx = 1;
            for (int k = 0; k < num_LB_steps; k++) {
                double LB = LB_start + k * LB_step;
                if ((LB>min_val)&&(LB<avg_val)){ 
                    boundcurve[i][k+1] = CalcUB(sorted_neuron_timeseries,LB);
                }
            }
        }
    }
}

int main(){
    // open output files
    ofstream avgsfile;
    avgsfile.open(avgsfname);
    ofstream velfile;
    velfile.open(velfname);
    ofstream boundcurvefile;
    boundcurvefile.open(boundcurvefname);

    // initialize the agent and its nervous system
    ifstream circuitfile;
    circuitfile.open(circuitfname);

    if (!circuitfile){
        cerr << "circuit file not found: " << circuitfname << endl;
        exit(EXIT_FAILURE);
    }

    LeggedAgent Agent;
    Agent.NervousSystem.SetCircuitSize(N);

    circuitfile >> Agent.NervousSystem;
    circuitfile.close();
    
    // Organize information about the grid of points, gathered from the res file
	ifstream resfile;
    resfile.open(resfname);

    //figure out how many dimensions we are searching (probably will always be 2 but it was general so i'll leave it)
    int num_dims=0;
    double testvar = 1;
    while (resfile >> testvar){
        num_dims ++;
    }
    num_dims = int(num_dims)/3;
    resfile.close();
    resfile.open(resfname);

	TMatrix<double> resmat(1,num_dims,1,3);
	TVector<double> parvec(1,num_dims);

	for(int i=1;i<=num_dims;i++){
		for(int j=1;j<=3;j++){
			resfile >> resmat(i,j);
		}
	}
    resfile.close();

    for(int i=1;i<=num_dims;i++){
		parvec(i) = resmat(i,1); //initialize the parameter values at the lowest given bound
    }

    TMatrix<double> bound_curve(1,N,1,num_LB_steps);
    TVector<double> avgvals(1,N);
    double fitness;
    bool finished = false;

    while(!finished){
        //clear vectors that hold information
        bound_curve.FillContents(0);
        avgvals.FillContents(0);
        fitness = 0;

        //set the proper parameters and initialize (right now not generalized to weights, always biases)
        for (int i=1;i<=N;i++){
            // Agent.NervousSystem.SetNeuronOutput(i,.5);
            Agent.NervousSystem.SetNeuronBias(i,parvec(i));
        }
        cout << Agent.NervousSystem.biases << endl;

        CalcAvgandBoundCurve(bound_curve,avgvals,fitness,Agent);

        velfile << fitness << " ";
        avgsfile << avgvals << endl;
        boundcurvefile << bound_curve << endl;

        parvec(num_dims)+=resmat(num_dims,3); //step the last dimension
		for (int i=(num_dims-1); i>=1; i-=1){ //start at the second to last dimension and count backwards to see if the next dimension has completed a run
			if(parvec(i+1)>resmat(i+1,2)){   //if the next dimension is over its max
				parvec(i+1) = resmat(i+1,1); //set it to its min
				parvec(i) += resmat(i,3);    //and step the current dimension
                //to do: add blank lines to outfiles
                velfile << endl;
                avgsfile << endl;
                boundcurvefile << endl;
			}
		}
		if (parvec(1)>resmat(1,2)){
			finished = true;
        }
    }
    avgsfile.close();
    velfile.close();
    boundcurvefile.close();
    return 0;
}