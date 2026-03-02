//****************************/
//  Set of methods for Legged
//  Agents with homeostatic
//  nervous systems
//  LJS 10/6/25
//****************************/

#include "FlexWalk.h"
#include "CTRNN.h"
#include "LeggedAgent.h"
#include "VectorMatrix.h"

//Constants
const double transient_dur = 150; //transient before adhp activates, also quick check to see if circuit state moving
const double quick_check_dur = 50; //how long to run the agent to check if the foot goes up and down before doing the full velocity measurement
const double test_dur = 200;     //maximum duration to run the embodied agent to evaluate the velocity - at max spans two cycles
const double StepSize = 0.01;
const double cross_tolerance = 0; //option to shift the threshold away from zero so that parameter jiggle can't influence, if relevant
const double bubble_tolerance = 5*StepSize; //protects against multi-peak oscillations, calculated in state space

const bool HPon = true; //turns ADHP on or off during velocity measurement

void step_function(LeggedAgent& Agent){
    // Allows you to dynamially adjust what kind of stepper you're using (mapping neurons to body actuators)
    Agent.Step2CPG(StepSize,HPon);
};

//Calcuate the instantaneous walking velocity by running the homeostatic walker for a set time, 
// then measuring the time and distance of the next full neuron cycle 
// used to be calculated from peak to peak, now is calculated from footfall to footfall
double meas_velocity(LeggedAgent& Agent, ofstream &timeseriesfile, ofstream &paramsfile, bool record, double synchronization_time){
    double vel = 0;
    //Assume circuit already equilibrated 
    // do a quick check to see if the foot neuron goes above and below 0.5 (not bothering with body yet)
    double max_ft = 0;
    double min_ft = 1;
    for (double i = StepSize;i<=quick_check_dur;i+=StepSize){
        step_function(Agent);
        if (record){
             timeseriesfile << " " << Agent.NervousSystem.outputs << endl;
             paramsfile << " " << Agent.NervousSystem.biases << endl;
        }
        if (Agent.NervousSystem.NeuronOutput(1)>max_ft){max_ft = Agent.NervousSystem.NeuronOutput(1);}
        if (Agent.NervousSystem.NeuronOutput(1)<min_ft){min_ft = Agent.NervousSystem.NeuronOutput(1);}
    }
    if ((max_ft>0.5)&&(min_ft<0.5)){ //if it does, continue
        // run the body for test duration to sync leg to the nervous system, and for body to get stuck if it's going to get stuck
        for (double i=StepSize;i<=synchronization_time;i+=StepSize){
            step_function(Agent);
            // //record or not record during the leg synchronization time
            if(record){
                timeseriesfile << " " << Agent.NervousSystem.outputs << endl;
                paramsfile << " " << Agent.NervousSystem.biases << endl;
            }
        }
        //run the body until the foot state goes from 0 to 1, then goes from 0 to 1 again
        int put_down_count = 0;
        int fmr_ft = 1;
        int new_ft = 1;

        double st_pos_x = 0;
        double test_time = 0;
        double cycle_time = 0;

        while ((put_down_count < 2)&&(test_time < test_dur)){
            if(record){
                timeseriesfile << " " << Agent.NervousSystem.outputs << endl;
                paramsfile << " " << Agent.NervousSystem.biases << endl;
                // bodyfile << " " << Agent.vx << " " << Agent.Leg.ForwardForce-Agent.Leg.BackwardForce << " " << Agent.Leg.FootX << endl;
            }
        
            test_time += StepSize;

            fmr_ft = new_ft;
            step_function(Agent);
            new_ft = Agent.Leg.FootState;
            
            if(fmr_ft - new_ft == -1){ //foot put down 
                put_down_count += 1;
                if (put_down_count == 1){
                    st_pos_x = Agent.PositionX();
                }
            }

            if (put_down_count == 1){
                cycle_time += StepSize;
            }
        }

        if (put_down_count == 2){
            double traveled = Agent.PositionX() - st_pos_x;
            cout << "traveled: " << traveled << endl;
            vel = traveled/cycle_time;
            cout << "cycle time: " << cycle_time << endl;
        }
    }
    return vel;
}

//Given a phenotype and an initialized walker, set up a homeostatic flexible legged walker individual to pass to the fitness function and populates the nm vector
void Setup(TVector<double>& phen, LeggedAgent& Agent, TVector<double>& neuromodvec){
    // cout << "full phenotpye:" << phen << endl;
    int ctrnnvecsize = neuromodvec.UpperBound();
    // cout << "ctrnnvecsize" << ctrnnvecsize << endl;
    int adhpvecsize = phen.UpperBound() - (2*ctrnnvecsize);
    // cout << "adhpvecsize" << adhpvecsize << endl;
    // cout << ctrnnvecsize << " " << adhpvecsize << endl; 
    int N = adhpvecsize / 2; // for now assuming that all neurons are being regulated and each has lb and range


    TVector<double> ctrnnvec(1,ctrnnvecsize);
    TVector<double> adhpvec(1,adhpvecsize);
    adhpvec.FillContents(0);

    TVector<int> plasticitypars(1,(N)+(N*N)); //plastic pars vector doesn't include taus
    plasticitypars.FillContents(0);
    for (int i=1;i<=N;i++){
        plasticitypars[i] = 1; //bias specifiers are in the beginning, assuming that all biases are being regulated
    }
    Agent.NervousSystem.SetCircuitSize(N);
    Agent.NervousSystem.SetPlasticityPars(plasticitypars);
    Agent.NervousSystem.ShiftedRho(true);

    int k = 1;

    for (int i = 1; i <= ctrnnvecsize; i ++){
        ctrnnvec(i) = phen(k);
        k++;
    }
    // cout << "ctrnn:" << ctrnnvec << endl;
    ctrnnvec >> Agent.NervousSystem;

    // assume default ADHP parameters are fine, so just set the target ranges based on the phenotype
    for (int i = 1; i <= adhpvecsize; i ++){
        adhpvec(i) = phen(k);
        k ++;
    }
    // cout << "adhp:" << adhpvec << endl;
    for (int i = 1; i <= N; i++){
        Agent.NervousSystem.SetPlasticityLB(i,adhpvec(i));
        Agent.NervousSystem.SetPlasticityUB(i,adhpvec(i) + adhpvec(i+N)); //range ecoding
    }
    Agent.NervousSystem.WindowReset(); //just in case

    for (int i = 1; i <= ctrnnvecsize; i ++){ //neuromod vec does contain taus, and taus are at the beginning 
        neuromodvec(i) = phen(k);
        k++;
    }
    // cout << "neuromod:" << neuromodvec << endl;
    return;
}

// Overload where we can use a bestind file -- not sure this works...
void Setup(ifstream& bestind, LeggedAgent& Agent, TVector<double>& neuromodvec){
    bestind >> Agent.NervousSystem;
    // cout << "biases after ctrnn load" << Agent.NervousSystem.biases << endl;
    int N = Agent.NervousSystem.CircuitSize();

    TVector<int> plasticitypars(1,(N)+(N*N)); //plastic pars vector doesn't include taus
    plasticitypars.FillContents(0);
    for (int i=1;i<=N;i++){
        plasticitypars[i] = 1; //bias specifiers are in the beginning, assuming that all biases are being regulated and no weights
    }

    Agent.NervousSystem.ShiftedRho(true); 
    Agent.NervousSystem.SetPlasticityPars(plasticitypars);

    // I believe the defaults for all adhp values are okay, so we will just set the target ranges
    TVector<double> adhplbs(1,N);
    bestind >> adhplbs;
    TVector<double> adhpranges(1,N);
    bestind >> adhpranges;
    for (int i = 1; i <= N; i++){
        Agent.NervousSystem.SetPlasticityLB(i,adhplbs(i));
        Agent.NervousSystem.SetPlasticityUB(i,adhplbs(i)+adhpranges(i));
    }
    Agent.NervousSystem.WindowReset(); //just to be safe

    for(int i = 1;i <= (2*N)+(N*N);i++){ //neuromod vec does contain taus, and taus are at the beginning 
        bestind >> neuromodvec[i];
        // cout << neuromodvec[i] << " ";
    }
    // cout << endl;
    return;
}


void TakeDown(LeggedAgent& Agent, ostream& indivout, TVector<double>& neuromodvec){
    indivout << Agent.NervousSystem;
    indivout << endl;
    Agent.NervousSystem.WriteHPGenome(indivout);
    indivout << endl;
    indivout << neuromodvec;
    return;
}

// Function to apply neuromodulation to a circuit
// Decision was made here that the modulation will be clipped at the boundary, allowing for the possibility
// of hidden, unexpressed, shadow genes. I.e. what you see in the file might not be what is actually expressed.
// Also, will continue to TRY to modulate that way every time, no matter the effects of ADHP. Biorealistic as a control signal
// But the reverse of the neuromodulation will only ever UN-do whatever its original effects were. This keeps the
// effects of neuromodulation linearly separate from the net effects of adhp
void Modulate(LeggedAgent& Agent, TVector<double>& neuromodvec, TVector<int>& modulatedpars){
    // neuromodvec should already have time constants shifted to the end
    if (modulatedpars.Sum()!=neuromodvec.UpperBound()){
        cerr << "Number of neuromodulatory magnitudes and desired parameter changes do not match:" << endl << modulatedpars.Sum() << " and " << neuromodvec.UpperBound();
        exit(EXIT_FAILURE);
    }

    int k = 1;
    for (int i = 1;i <= modulatedpars.UpperBound(); i++){
        if (modulatedpars[i] == 1){
            // cout << i << " " << k << endl;
            double ogpar = Agent.NervousSystem.ArbDParam(i,modulatedpars);
            double mod_par = ogpar + neuromodvec(k);
            // if a bias, clip to +- bias range
            if (i <= Agent.NervousSystem.CircuitSize()){
                mod_par = max(mod_par,-Agent.NervousSystem.br);
                mod_par = min(mod_par,Agent.NervousSystem.br);
            }
            // if a weight, clip to +- weight range
            else if (i <= modulatedpars.UpperBound()-Agent.NervousSystem.CircuitSize()){
                mod_par = max(mod_par,-Agent.NervousSystem.wr);
                mod_par = min(mod_par,Agent.NervousSystem.wr);
            }
            // if a time constant clip to tc range
            else{
                mod_par = max(mod_par,Agent.NervousSystem.tc_min);
                mod_par = min(mod_par,Agent.NervousSystem.tc_max);
            }
            Agent.NervousSystem.SetArbDParam(i,mod_par,modulatedpars);
            //updates neuromodvec to hold the effective neuromodulatory parameters rather than ideal
            neuromodvec[k] = mod_par - ogpar;
            // cout << "modulating: " << ogpar << " + " << neuromodvec(k) << " = " << Agent.NervousSystem.ArbDParam(i,modulatedpars)<<endl;
            k ++;
        }
    }
    return;
}


// Overload to give default parameter vector
void Modulate(LeggedAgent& Agent, TVector<double>& neuromodvec){
    int N = Agent.NervousSystem.CircuitSize();
    // cout << "N= " << N << endl;
    TVector<int> modulatedpars(1, (2*N)+(N*N));
    modulatedpars.FillContents(1); //default is all parameters are modulated

    Modulate(Agent,neuromodvec,modulatedpars);
    return;
}

// these utilities should really be added in vectormatrix but whatever
void Reverse_NM(TVector<double>& neuromodvec){
    // negates all elements of neuromodulatory vector as presented
    for(int i=neuromodvec.LowerBound();i<=neuromodvec.UpperBound();i++){
        neuromodvec[i] = -neuromodvec[i];
    }
    return;
}
void Shift_NM(TVector<double>& neuromodvec,int shift_num){
    //start by moving the time constants to the end, shifting everything else forward
    for (int i = 1; i <= shift_num;i++){
        double tc = neuromodvec[1];
        for (int j = 1; j <= neuromodvec.UpperBound()-1;j++){
            neuromodvec[j] = neuromodvec[j+1];
        }
        neuromodvec[neuromodvec.UpperBound()] = tc;
    }
    return;
}


//given a set up individual, calculate the fitness as prescribed
double FlexibleWalking(LeggedAgent& Agent,TVector<double> neuromodvec,double plasticitydur,int rounds, bool debug){
    ofstream timeseriesfile;
    timeseriesfile.open("neuraltimeseries.dat");
    ofstream paramsfile;
    paramsfile.open("neuralparameters.dat");
    // ofstream bodyfile;
    // bodyfile.open("bodytimeseries.dat");

    // cout << "neuromodvec " << neuromodvec << endl;
    int N = Agent.NervousSystem.CircuitSize();
    // ...and shifting the time constants to the end of the vector
    Shift_NM(neuromodvec,N);
    
    // Pass transient without ADHP
    for (double i=StepSize;i<=transient_dur;i+=StepSize){
        Agent.NervousSystem.EulerStep(StepSize,false);
    }

    if (debug){
        cout << "unmodulated, before ADHP" << endl << Agent.NervousSystem.taus << endl << Agent.NervousSystem.biases <<endl <<Agent.NervousSystem.weights << endl << endl;
    }

    double avg_unmodulated_vel = 0;
    double avg_modulated_vel = 0;
    double unmodulated_vel = 0;
    double modulated_vel = 0;
    int unmodulated_tests = 0;
    int modulated_tests = 0;

    if (rounds == 0){
        // Allow ADHP to run for designated time. Now also moves the body because I took away the sync time
        for (double t = 0; t < plasticitydur; t += StepSize){
            step_function(Agent);
            if (debug){
                timeseriesfile <<  " " << Agent.NervousSystem.outputs << endl;
                paramsfile << " " << Agent.NervousSystem.biases << endl;
                // bodyfile << " " << Agent.vx << " " << Agent.Leg.ForwardForce-Agent.Leg.BackwardForce << " " << Agent.Leg.FootX << endl;
            }
        }
        // Test unmodulated velocity at homeostatic steady state - fuckin shit up
        double init_x = Agent.PositionX();
        int fmr_ft = 1;
        int new_ft = 1;
        double cycle_time = 0;
        bool first_lil_bit = false;
        double testdur = 650;
        
        for(double t = 0; t < testdur; t += StepSize){
            fmr_ft = new_ft;
            Agent.Step2CPG(StepSize,true);
            new_ft = Agent.Leg.FootState;
            cycle_time += StepSize;
            if(fmr_ft - new_ft == -1){
                if(first_lil_bit){
                    cout << "cycle: " << cycle_time << endl;
                }
                first_lil_bit = true;
                cycle_time = 0;
            }
        }
        double dist = Agent.PositionX() - init_x;
        cout << "Distance traveled during test: " << dist << endl;
        double fit = (Agent.PositionX() - init_x)/testdur;
        unmodulated_vel = fit;

        // unmodulated_vel = meas_velocity(Agent,timeseriesfile,paramsfile,debug);
        unmodulated_tests ++;
        if (debug){
            cout << "unmodulated, after ADHP: " << unmodulated_vel << endl << Agent.NervousSystem.taus << endl << Agent.NervousSystem.biases <<endl <<Agent.NervousSystem.weights << endl << endl;
        }
        avg_unmodulated_vel += unmodulated_vel;
    }
    else{
        //store the effective neuromod vectors and reverse neuromod vectors because we'll be switching back and forth
        for (int round=1;round<=rounds;round++){
            // Allow ADHP to run for designated time. Also moves the body becasue I took away the sync time
            for (double i = StepSize; i <= plasticitydur; i += StepSize){
                step_function(Agent);
                if (debug){
                    timeseriesfile << " " << Agent.NervousSystem.outputs << endl;
                    paramsfile << " " << Agent.NervousSystem.biases << endl;
                    // bodyfile  << " " << Agent.vx << " " << Agent.Leg.ForwardForce-Agent.Leg.BackwardForce << " " << Agent.Leg.FootX << endl;
                }
            }
            if (debug){
                cout << "unmodulated, after ADHP, before test" << endl << Agent.NervousSystem.taus << endl << Agent.NervousSystem.biases <<endl <<Agent.NervousSystem.weights << endl << endl;
            }
            // Test unmodulated velocity at homeostatic steady state
            unmodulated_vel = meas_velocity(Agent,timeseriesfile,paramsfile,debug);
            unmodulated_tests ++;
            if (debug){
                cout << "unmodulated, after ADHP, after test: " << unmodulated_vel << endl << Agent.NervousSystem.taus << endl << Agent.NervousSystem.biases <<endl <<Agent.NervousSystem.weights << endl<< endl;
            }
            avg_unmodulated_vel += unmodulated_vel;

            //apply neuromodulation (default is apply to all parameters)
            Modulate(Agent,neuromodvec);
            if (debug){
                cout << "modulated, before ADHP, before test" << endl << Agent.NervousSystem.taus << endl << Agent.NervousSystem.biases <<endl <<Agent.NervousSystem.weights << endl << endl;
            }

            //and measure the immediate modulated velocity
            modulated_vel = meas_velocity(Agent,timeseriesfile,paramsfile,debug); 
            modulated_tests ++;
            if (debug){
                cout << "modulated, before ADHP, after test: " << modulated_vel << endl << Agent.NervousSystem.taus << endl << Agent.NervousSystem.biases <<endl <<Agent.NervousSystem.weights << endl << endl;
            }
            avg_modulated_vel += modulated_vel;

            //allow plasticity to occur in modulated state
            for (double i = StepSize; i <= plasticitydur; i += StepSize){
                step_function(Agent);
                if (debug){
                    timeseriesfile << " " << Agent.NervousSystem.outputs << endl;
                    paramsfile << " " << Agent.NervousSystem.biases << endl;
                    // bodyfile  << " " << Agent.vx << " " << Agent.Leg.ForwardForce-Agent.Leg.BackwardForce << " " << Agent.Leg.FootX << endl;
                }
            }
            if (debug){
                cout << "modulated, after ADHP, before test" << endl << Agent.NervousSystem.taus << endl << Agent.NervousSystem.biases <<endl <<Agent.NervousSystem.weights << endl << endl;
            }

            // measure modulated velocity at modulated homeostatic steady state
            modulated_vel = meas_velocity(Agent,timeseriesfile,paramsfile,debug);
            modulated_tests ++;
            if (debug){
                cout << "modulated, after ADHP, after test: " << modulated_vel << endl << Agent.NervousSystem.taus << endl << Agent.NervousSystem.biases <<endl <<Agent.NervousSystem.weights << endl << endl;
            }
            avg_modulated_vel += modulated_vel;

            //then negate all the elements of the effective neuromodulation
            Reverse_NM(neuromodvec);
            //and reverse neuromodulation
            Modulate(Agent,neuromodvec);
            if (debug){
                cout << "unmodulated, before ADHP, before test" << endl << Agent.NervousSystem.taus << endl << Agent.NervousSystem.biases <<endl <<Agent.NervousSystem.weights << endl << endl;
            }

            //measure the immediate unmodulated velocity
            unmodulated_vel = meas_velocity(Agent,timeseriesfile,paramsfile,debug);
            unmodulated_tests ++;
            if (debug){
                cout << "unmodulated, before ADHP, after test: " << unmodulated_vel << endl << Agent.NervousSystem.taus << endl << Agent.NervousSystem.biases <<endl <<Agent.NervousSystem.weights << endl << endl;
            }
            avg_unmodulated_vel += unmodulated_vel;
        }
    }

    avg_unmodulated_vel = avg_unmodulated_vel/max(unmodulated_tests,1);
    avg_modulated_vel = avg_modulated_vel/max(modulated_tests,1);

    double fit = abs(avg_unmodulated_vel - avg_modulated_vel); //maximum fitness is 2*6.27 --max speed both directions

    timeseriesfile.close();
    // bodyfile.close();
    paramsfile.close();
    return fit;
}