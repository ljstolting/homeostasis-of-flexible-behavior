// *************************
// A class for legged agents
//
// RDB 2/16/96
// *************************

#pragma once

#include "CTRNN.h"

// Global constants

const double Pi = 3.1415926;


// The Leg class declaration

class TLeg {
	public:
		// The constructor
		TLeg() {};
		// The destructor
		~TLeg() {};

		// Accessors

		double Angle, Omega, ForwardForce, BackwardForce;
		double FootX, FootY, JointX, JointY;
		double FootState;
};


// The LeggedAgent class declaration

class LeggedAgent {
	public:
		//Data members
		double cx, cy, vx;
		TLeg Leg;
		CTRNN NervousSystem; 

	public:
		// The constructor
		LeggedAgent(double ix = 0.0, double iy = 0.0)
			: NervousSystem(3)
		{
			Reset(ix,iy);
		};
		// The destructor
		~LeggedAgent() {};

		// Accessors
		double PositionX(void) {return cx;};
		// void SetPositionX(double newx) ; //was causing problems so took out

		// Control
		void Reset(double ix, double iy, int randomize = 0);
		void Reset(double ix, double iy, int randomize, RandomState &rs);
		void DragBack(void);
		void StepCPG(double StepSize, bool adaptpars); 
		void Step2CPG(double StepSize, bool adaptpars); // updated to read out foot and force neurons instead of fs, bs
		void PerfectStep(double StepSize);
		void Walk(double time, double StepSize, bool adaptpars); //measure fitness the "dumb" way as distance traveled in a fixed amount of time
		                                     //adaptpars is whether to allow adhp to act during the walk or not
		void Walk(double time, double StepSize, bool adaptpars, ofstream &outputs); //overloaded version that also outputs the trajectory to a file
};
