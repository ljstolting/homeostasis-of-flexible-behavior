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
		// The constructor
		LeggedAgent(double ix = 0.0, double iy = 0.0)
		{
			Reset(ix,iy,0);
		};
		// Copy constructor (new Lindsay, might not work right)
		LeggedAgent(LeggedAgent&) = default;

		// The destructor
		~LeggedAgent() {};

		// Accessors
		double PositionX(void) {return cx;};
		void SetPositionX(double newx) {cx = newx;};

		// Control
    void Reset(double ix, double iy, int randomize = 0);
    void Reset(double ix, double iy, int randomize, RandomState &rs);
	void DragBack(void);
		// stepping actuators
		void UpdateBody(double StepSize);
		void StepCPG(double StepSize, bool adaptpars);
		void StepRPG(double StepSize, bool adaptpars);
		void Step2CPG(double StepSize, bool adaptpars);
		void Step2RPG(double StepSize, bool adaptpars);
		void Step1CPG(double StepSize, bool adaptpars);
		void Step1RPG(double StepSize, bool adaptpars);
		void PerfectStep(double StepSize);
		void PolicyStep(double FBSval, double footstate, double StepSize);
		void PolicyRun(double upval, double downval, double updur, double downdur, double totaldur, bool ftcoord_fw, double StepSize);

		double cx, cy, vx;
		TLeg Leg;
		CTRNN NervousSystem;
};
