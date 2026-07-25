// ***********************
// Methods for LeggedAgent
//
// RDB 2/16/96
// ***********************

#include "LeggedAgent.h"
#include "random.h"

// Constants
const int    LegLength = 15;
const double MaxLegForce = 0.05;
const double ForwardAngleLimit = Pi/6;
const double BackwardAngleLimit = -Pi/6;
const double MaxVelocity = 6.0;
const double MaxTorque = 0.5;
const double MaxOmega = 1.0;

// *******
// Control
// *******

// Reset the state of the agent

void LeggedAgent::Reset(double ix, double iy, int randomize)
{
	cx = ix; cy = iy; vx = 0.0;
	Leg.FootState = 0;
	if (randomize) Leg.Angle = UniformRandom(BackwardAngleLimit,ForwardAngleLimit);
	else Leg.Angle = ForwardAngleLimit;
	Leg.Omega = Leg.ForwardForce = Leg.BackwardForce = 0.0;
	Leg.JointX = cx; Leg.JointY = cy + 12.5;
	Leg.FootX = Leg.JointX + LegLength * sin(Leg.Angle);
	Leg.FootY = Leg.JointY + LegLength * cos(Leg.Angle);
	if (randomize){
		NervousSystem.RandomizeCircuitState(-0.1,0.1);
		cout << "circuit state randomized" << endl;
	}
}

void LeggedAgent::Reset(double ix, double iy, int randomize, RandomState &rs)
{
	cx = ix; cy = iy; vx = 0.0;
	Leg.FootState = 0;
	if (randomize) Leg.Angle = rs.UniformRandom(BackwardAngleLimit,ForwardAngleLimit);
	else Leg.Angle = ForwardAngleLimit;
	Leg.Omega = Leg.ForwardForce = Leg.BackwardForce = 0.0;
	Leg.JointX = cx; Leg.JointY = cy + 12.5;
	Leg.FootX = Leg.JointX + LegLength * sin(Leg.Angle);
	Leg.FootY = Leg.JointY + LegLength * cos(Leg.Angle);
	if (randomize) NervousSystem.RandomizeCircuitState(-0.1,0.1,rs);
}

void LeggedAgent::DragBack(void)
{ //teleports agent back at the x starting line without disrupting its body or nervous system
	cx = 0.0;
	Leg.JointX = cx; Leg.JointY = cy + 12.5;
	Leg.FootX = Leg.JointX + LegLength * sin(Leg.Angle);
	Leg.FootY = Leg.JointY + LegLength * cos(Leg.Angle);
}

//Body update is the same physics for all functions once the force is defined, so keep parity
void LeggedAgent::UpdateBody(double StepSize){
	double force = 0.0;
	// Compute the force applied to the body
	// *** This is a CHANGE from the original body model that allows a supporting leg that has
	// *** passed outside of the mechanical limits to apply force in a direction that moves it
	// *** back toward that mechanical limit but not in a direction that would move it further
	// *** away.  In effect, the mechanical limits become 1-way constraints for a supporting leg.
	double f = Leg.ForwardForce - Leg.BackwardForce;
	if (Leg.FootState == 1.0){
		if ((Leg.Angle >= BackwardAngleLimit && Leg.Angle <= ForwardAngleLimit) ||
		    (Leg.Angle < BackwardAngleLimit && f < 0) ||
		    (Leg.Angle > ForwardAngleLimit && f > 0)){
			force = f;
		}
	}
	// *** The original code
//		if (Leg.FootState == 1.0 && Leg.Angle >= BackwardAngleLimit &&  Leg.Angle <= ForwardAngleLimit)
//			force = Leg.ForwardForce - Leg.BackwardForce;
	// ***

	// Update the position of the body
	vx = vx + StepSize * force;
	if (vx < -MaxVelocity) vx = -MaxVelocity;
	if (vx > MaxVelocity) vx = MaxVelocity;
	cx = cx + StepSize * vx;
	// Update the leg geometry
	Leg.JointX = Leg.JointX + StepSize * vx;
	if (Leg.FootState == 1.0) {
		double angle = atan2(Leg.FootX - Leg.JointX,Leg.FootY - Leg.JointY);
		Leg.Omega = (angle - Leg.Angle)/StepSize;
		Leg.Angle = angle;
	}
	else {
		vx = 0.0;
		Leg.Omega	= Leg.Omega + StepSize * MaxTorque * (Leg.BackwardForce - Leg.ForwardForce);
		if (Leg.Omega < -MaxOmega) Leg.Omega = -MaxOmega;
		if (Leg.Omega > MaxOmega) Leg.Omega = MaxOmega;
		Leg.Angle = Leg.Angle + StepSize * Leg.Omega;
		if (Leg.Angle < BackwardAngleLimit) {Leg.Angle = BackwardAngleLimit; Leg.Omega = 0;}
		if (Leg.Angle > ForwardAngleLimit) {Leg.Angle = ForwardAngleLimit; Leg.Omega = 0;}
		Leg.FootX = Leg.JointX + LegLength * sin(Leg.Angle);
		Leg.FootY = Leg.JointY + LegLength * cos(Leg.Angle);
	}
	// If the foot is too far back, the body becomes "unstable" and forward motion ceases
	if (fabs(cx - Leg.FootX) > 20) vx = 0.0;
}

// Step the insect using a general CTRNN CPG
void LeggedAgent::StepCPG(double StepSize,bool adaptpars)
{
	// Update the nervous system
	NervousSystem.EulerStep(StepSize,adaptpars);
	// Update the leg effectors
	if (NervousSystem.NeuronOutput(1) > 0.5) {Leg.FootState = 1; Leg.Omega = 0;}
	else Leg.FootState = 0;
	Leg.ForwardForce = NervousSystem.NeuronOutput(2) * MaxLegForce;
	Leg.BackwardForce = NervousSystem.NeuronOutput(3) * MaxLegForce;
	
	// body update is same through all functions once the force is defined
	UpdateBody(StepSize);
}


// Step the insect using a general CTRNN RPG
void LeggedAgent::StepRPG(double StepSize, bool adaptpars)
{
	// Update the sensory input
	for (int i = 1; i <= NervousSystem.CircuitSize(); i++){
		NervousSystem.SetNeuronExternalInput(i, Leg.Angle * 5.0/ForwardAngleLimit);
	}

	//rest of this function is identical to CPG after input is updated. Keep parity
	StepCPG(StepSize, adaptpars);
}

// Step the LeggedAgent using a 2-neuron CTRNN CPG

void LeggedAgent::Step2CPG(double StepSize, bool adaptpars)
{
	// Update the nervous system
	NervousSystem.EulerStep(StepSize,adaptpars);
	// Update the leg effectors
	if (NervousSystem.NeuronOutput(1) > 0.5) {Leg.FootState = 1; Leg.Omega = 0;}
	else Leg.FootState = 0;
	
    // Updated to be a foot neuron and a force neuron instead of FT and only FS neuron. (allows backward walking)
    double o = NervousSystem.NeuronOutput(2);
	// also set the other direction to zero when necessary. Not present in other places where coordination problem is wrapped into one neuron
	// Theoretically it would smoothly transition, but not necessarily with discrete steps

	// // Update the leg effectors
	if (o > 0.5) {
		Leg.ForwardForce = 2 * (o - 0.5) * MaxLegForce;
		Leg.BackwardForce = 0.0;
	}
	else {
		Leg.BackwardForce = 2 * (0.5 - o) * MaxLegForce;
		Leg.ForwardForce = 0.0;
	}
	// in the update body function, the force is defined as follows:
 	// double f = Leg.ForwardForce - Leg.BackwardForce; 

	//I believe the above is equivalent to saying that f = 2*(o - 0.5)*MaxLegForce, but this way makes it more clear that the force is in the direction of the stronger output
	// double f = 2*(o - 0.5)*MaxLegForce;
	
    UpdateBody(StepSize);
}

// Step the LeggedAgent using a 2-neuron CTRNN CPG

void LeggedAgent::Step2RPG(double StepSize, bool adaptpars)
{
	// Update the sensory input
	for (int i = 1; i <= NervousSystem.CircuitSize(); i++){
		NervousSystem.SetNeuronExternalInput(i, Leg.Angle * 5.0/ForwardAngleLimit);
	}
	Step2CPG(StepSize, adaptpars);
}

// Step the LeggedAgent using a 1-neuron CTRNN CPG

void LeggedAgent::Step1CPG(double StepSize, bool adaptpars)
{
	double force = 0.0;

	// Update the nervous system
	NervousSystem.EulerStep(StepSize, adaptpars);
	double o = NervousSystem.NeuronOutput(1);
	// Update the leg effectors
	if (o > 0.5) {
		Leg.FootState = 1;
		Leg.Omega = 0;
		Leg.ForwardForce = 2 * (o - 0.5) * MaxLegForce;
	}
	else {
		Leg.FootState = 0;
		Leg.BackwardForce = 2 * (0.5 - o) * MaxLegForce;
	}
	UpdateBody(StepSize);
}


// Step the LeggedAgent using a 1-neuron CTRNN RPG

void LeggedAgent::Step1RPG(double StepSize, bool adaptpars)
{
	// Update the sensory input
	for (int i = 1; i <= NervousSystem.CircuitSize(); i++){
		NervousSystem.SetNeuronExternalInput(i, Leg.Angle * 5.0/ForwardAngleLimit);
	}
	Step1CPG(StepSize, adaptpars);
}


// Step the LeggedAgent using the optimal pattern generator

void LeggedAgent::PerfectStep(double StepSize)
{
	double force = 0.0;

	// Update the leg effectors
	if (Leg.FootState == 0.0 && Leg.Angle >= ForwardAngleLimit) {Leg.FootState = 1; Leg.Omega = 0;}
	else if (Leg.FootState == 1.0 && (cx - Leg.FootX > 20)) Leg.FootState = 0;
	// Compute the force applied to the body
	if (Leg.FootState == 1.0 && Leg.Angle >= BackwardAngleLimit && Leg.Angle <= ForwardAngleLimit)
		force = MaxLegForce;
	// Update the position of the body
	vx = vx + StepSize * force;
	if (vx < -MaxVelocity) vx = -MaxVelocity;
	if (vx > MaxVelocity) vx = MaxVelocity;
	cx = cx + StepSize * vx;
	// Update the leg geometry
	Leg.JointX = Leg.JointX + StepSize * vx;
	if (Leg.FootState == 1.0) {
		double angle = atan2(Leg.FootX - Leg.JointX,Leg.FootY - Leg.JointY);
		Leg.Omega = (angle - Leg.Angle)/StepSize;
		Leg.Angle = angle;
	}
	else {
		vx = 0.0;
		Leg.Omega	= Leg.Omega + StepSize * MaxTorque * MaxLegForce;
		if (Leg.Omega < -MaxOmega) Leg.Omega = -MaxOmega;
		if (Leg.Omega > MaxOmega) Leg.Omega = MaxOmega;
		Leg.Angle = Leg.Angle + StepSize * Leg.Omega;
		if (Leg.Angle < BackwardAngleLimit) {Leg.Angle = BackwardAngleLimit; Leg.Omega = 0;}
		if (Leg.Angle > ForwardAngleLimit) {Leg.Angle = ForwardAngleLimit; Leg.Omega = 0;}
		Leg.FootX = Leg.JointX + LegLength * sin(Leg.Angle);
		Leg.FootY = Leg.JointY + LegLength * cos(Leg.Angle);
	}
	// If the foot is too far back, the body becomes "unstable" and forward motion ceases
	if (fabs(cx - Leg.FootX) > 20) vx = 0.0;
}

// Step the 2N Agent using up and down epochs with fixed values and lengths and perfectly timed foot transitions
//     ftcoord_fw true means coordinate it for a forward walker and false means coordinate it for a backward walker
void LeggedAgent::PolicyStep(double FBSval, double footstate, double StepSize){
	// equivalent to 2N-CPG but without the nervous system update
	Leg.FootState = footstate;
	
    double o = FBSval;

	if (o > 0.5) {
		Leg.ForwardForce = 2 * (o - 0.5) * MaxLegForce;
		Leg.BackwardForce = 0.0;
	}
	else {
		Leg.BackwardForce = 2 * (0.5 - o) * MaxLegForce;
		Leg.ForwardForce = 0.0;
	}
	
    UpdateBody(StepSize);
	return;
}

void LeggedAgent::PolicyRun(double upval, double downval, double updur, double downdur, double totaldur, bool ftcoord_fw, double StepSize){
	double time = StepSize;
	double footstate_up, footstate_down;

	if (ftcoord_fw){
		footstate_up = 1;
		footstate_down = 0;
	}
	else{
		footstate_up = 0;
		footstate_down = 1;
	}

	while (time < totaldur){
		// up phase (stance for forward and swing for backward)
		for (double t = 0; t<updur; t += StepSize){
			PolicyStep(upval,footstate_up,StepSize);
			time += StepSize;
			if (time > totaldur){return;}
		}

		// down phase (swing for forward, stance for backward)
		for (double t = 0; t<downdur; t += StepSize){
			PolicyStep(downval,footstate_down,StepSize);
			time += StepSize;
			if (time > totaldur){return;}
		}
	}

	return;
}
