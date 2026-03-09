//****************************/
//  Set of methods for Legged
//  Agents with homeostatic
//  nervous systems
//  LJS 10/6/25
//****************************/

#include "CTRNN.h"
#include "LeggedAgent.h"
#include "VectorMatrix.h"

//Constants
const double test_dur = 200;     //maximum duration to run the embodied agent to evaluate the velocity - at max spans two cycles
const double StepSize = 0.01;

void step_function(LeggedAgent& Agent);
void distance_and_time(LeggedAgent& Agent, double &dist_traveled, double &cycle_time, ofstream &timeseriesfile, ofstream &paramsfile, bool recordoutputs, bool recordparams);
void distance_and_time(LeggedAgent& Agent, double &distancetraveled, double &cycletime, TMatrix<double>& timeseries, bool recordoutputs);
void SortTraj(TVector<double>& sortedtraj, TVector<double>& neurontraj);
double CalcUB(TVector<double>& sortedneurontraj, double lb);
void quick_osc_check_sync(LeggedAgent& Agent, bool &osc, bool &footupdown, double transientdur=500.0, double quick_check_dur=10.0);
double meas_velocity(LeggedAgent& Agent, ofstream &timeseriesfile, ofstream &paramsfile, bool recordoutputs=false, bool recordparams=false);
double meas_velocity(LeggedAgent& Agent, TMatrix<double>& timeseries, bool recordoutputs=true); //override to output the trajectory to a local file
void Setup(TVector<double>& phen, LeggedAgent& Agent, TVector<double>& neuromodvec);//read agent from phenotype vector
void Setup(ifstream& bestind, LeggedAgent& Agent, TVector<double>& neuromodvec);  //overload to read from file
void TakeDown(LeggedAgent& Agent, ostream& indivout, TVector<double>& neuromodvec);
void Modulate(LeggedAgent& Agent, TVector<double>& neuromodvec);
void Modulate(LeggedAgent& Agent, TVector<double>& neuromodvec, TVector<int>& modulatedpars); //overload to allow specification of different modulated parameters
// void Reverse_NM(TVector<double>& neuromodvec);
void Shift_NM(TVector<double>& neuromodvec,int shift_num);
double FlexibleWalking(LeggedAgent& Agent,TVector<double> neuromodvec, double plasticitydur, int rounds, bool debug=false);