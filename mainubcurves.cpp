// Calculate the UB curves for every evolved walker (even the ones that didn't evolve, will delete later)

#include "CTRNN.h"
#include "LeggedAgent.h"
#include "FlexWalk.h"
#include "VectorMatrix.h"

// LB sampling resolution - tries all the ones that are within the max and min of the cycle
const double LB_step = 0.025;
const double LB_start = LB_step;
const double LB_stop = 1-LB_step;
int num_LB_steps = int((LB_stop-LB_start)/LB_step)+1;

//copied from mainvelavg
void CalcAvgandBoundCurve(TMatrix<double> &boundcurve, TVector<double> &avgval, double &velocity, LeggedAgent Agent){
    // set boundcurve vector, avg, and fitness to zeros if not already
    boundcurve.FillContents(0);
    avgval.FillContents(0);
    velocity = 0;

    int N = Agent.NervousSystem.CircuitSize();

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
    char trajfname[] = "./trajectory.dat";
    ifstream trajectoryfile;
    trajectoryfile.open(trajfname);

    char circuitfname[] = "./bestind.dat";
    ifstream circuitfile;
    circuitfile.open(circuitfname);

    char ubsfname[] = "./calculatedubs.dat";
    ofstream ubsfile;
    ubsfile.open(ubsfname);

    LeggedAgent Agent;
    circuitfile >> Agent.NervousSystem;
    int N = Agent.NervousSystem.CircuitSize();

    TMatrix<double> bound_curve(1,N,1,num_LB_steps);
    TVector<double> avgvals(1,N);
    double vel;

    CalcAvgandBoundCurve(bound_curve,avgvals,vel,Agent);

    ubsfile << bound_curve << endl << endl;

    return 0;
}