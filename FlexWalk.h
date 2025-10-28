//****************************/
//  Set of methods for Legged
//  Agents with homeostatic
//  nervous systems
//  LJS 10/6/25
//****************************/

#include "CTRNN.h"
#include "LeggedAgent.h"
#include "VectorMatrix.h"

double meas_velocity(LeggedAgent& Agent, bool record = false, double synchronization_time = 100);
void Setup(TVector<double>& phen, LeggedAgent& Agent, TVector<double>& neuromodvec);//read agent from phenotype vector
void Setup(ifstream& bestind, LeggedAgent& Agent, TVector<double>& neuromodvec);  //overload to read from file
void TakeDown(LeggedAgent& Agent, ostream& indivout, TVector<double>& neuromodvec);
void Modulate(LeggedAgent& Agent, TVector<double>& neuromodvec);
void Modulate(LeggedAgent& Agent, TVector<double>& neuromodvec, TVector<int>& modulatedpars); //overload to allow specification of different modulated parameters
void Reverse_NM(TVector<double>& neuromodvec);
void Shift_NM(TVector<double>& neuromodvec,int shift_num);
double FlexibleWalking(LeggedAgent& Agent,TVector<double> neuromodvec, double plasticitydur, bool debug=false);