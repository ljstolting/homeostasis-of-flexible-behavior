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
		void StepCPG(double StepSize, bool adaptpars); //this is the only one that we're going to be using in this repo...should i delete the others? just commented out because might get suggestions (i.e. need RPG or 2 neuron)
		// void StepRPG(double StepSize);
		// void Step2CPG(double StepSize);
		// void Step2RPG(double StepSize);
		// void Step1CPG(double StepSize);
		// void Step1RPG(double StepSize);
		void PerfectStep(double StepSize);
};
