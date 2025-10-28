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
const double transient_dur = 50; //transient before adhp activates, also quick check to see if circuit state moving
const double test_dur = 200;     //maximum duration to run the embodied agent to evaluate the velocity - at max spans two cycles
const double StepSize = 0.01;
const double cross_tolerance = 0.5*StepSize; //shift the threshold away from zero so that parameter jiggle can't influence, if relevant
const double bubble_tolerance = 10*StepSize; //protects against multi-peak oscillations, calculated in state space
const int rounds = 3; //how many times to repeat the switching process

const bool HPon = false; //turns ADHP on or off during velocity measurement

//Calcuate the instantaneous walking velocity by running the homeostatic walker for some time (def = 50 seconds), 
// then measuring the time and distance of the next full neuron cycle -- now calculated from peak to peak
double meas_velocity(LeggedAgent& Agent, bool record, double synchronization_time){
    ofstream timeseriesfile;
    timeseriesfile.open("neuraltimeseries.dat");
    double vel = 0;
    int N = Agent.NervousSystem.size;
    double sync_time = 0;
    double test_time = 0;
    double cycle_time = 0;
    //Assume circuit already equilibrated 
    // do a quick check to see if the circuit state is actually moving appreciably
    double dist = 0;
    TVector<double> start(1,N);
    start = Agent.NervousSystem.states;
    for (double i = StepSize;i<=transient_dur;i+=StepSize){Agent.NervousSystem.EulerStep(StepSize,HPon);}
    for (int i = 1; i <= N; i++){
        dist += pow(start[i]-Agent.NervousSystem.NeuronState(i),2);
    }
    dist = pow(dist,.5);
    if (dist > 0.05){ //if it did move, go on
        // run the body for test duration to sync leg to the nervous system, and for body to get stuck if it's going to get stuck
        for (double i=StepSize;i<=synchronization_time;i+=StepSize){
            Agent.StepCPG(StepSize,HPon); //
            // //record or not record during the leg synchronization time
            if(record){
                timeseriesfile << sync_time << " " << Agent.NervousSystem.outputs << endl;
                timeseriesfile << sync_time + test_time << " " << Agent.vx << " " << Agent.Leg.ForwardForce-Agent.Leg.BackwardForce << " " << Agent.Leg.FootX << endl;
            }
            sync_time += StepSize;
        }
        //run the body until we get to a peak (derivative sign change + -> -) in the neuron 1 output
        int peak_count = 0;

        double new_n1_div = 0;
        double fmr_n1_div = 0;
        bool flip = false;
        bool first_step = true;

        double st_pos_x = 0;
        double max_ft = 0;
        double min_ft = 1;

        while ((peak_count<2)&(test_time < test_dur)){
            if(record){
                timeseriesfile << sync_time + test_time << " " << Agent.NervousSystem.NeuronOutput(1) << " " <<  Agent.NervousSystem.NeuronOutput(2) << " " <<  Agent.NervousSystem.NeuronOutput(3) << endl;
                timeseriesfile << sync_time + test_time << " " << Agent.vx << " " << Agent.Leg.ForwardForce-Agent.Leg.BackwardForce << " " << Agent.Leg.FootX << endl;
            }
        
            test_time += StepSize;

            fmr_n1_div = new_n1_div;
            new_n1_div = Agent.NervousSystem.NeuronState(1);
            Agent.StepCPG(StepSize,HPon);
            new_n1_div = Agent.NervousSystem.NeuronState(1) - new_n1_div; //final - initial
            
            if (!first_step){
                if((fmr_n1_div>cross_tolerance)&(new_n1_div<cross_tolerance)){
                    if (peak_count == 1){
                        dist = 0;
                        for (int i = 1; i <= N; i++){
                            dist += pow(start[i]-Agent.NervousSystem.NeuronState(i),2);
                        }
                        dist = pow(dist,.5);
                        if (dist < bubble_tolerance) {
                            peak_count += 1;
                        }
                    }
                    if (peak_count == 0){
                        start = Agent.NervousSystem.states;
                        peak_count += 1;
                        // cout << "peaked at " << sync_time + test_time << endl;
                        st_pos_x = Agent.PositionX(); //subtract off the progress that the agent made before the recorded cycle
                    }
                }
            }

            if (peak_count == 1){
                cycle_time += StepSize;
                if (Agent.NervousSystem.NeuronOutput(1)>max_ft){max_ft = Agent.NervousSystem.NeuronOutput(1);}
                if (Agent.NervousSystem.NeuronOutput(1)<min_ft){min_ft = Agent.NervousSystem.NeuronOutput(1);}
            }

            first_step = false;   
        }

        if (peak_count == 2){
            if((min_ft<.5)&(max_ft>.5)){
                vel = (Agent.PositionX()-st_pos_x)/cycle_time;
            } 
        }
        //choice to only count fitness if i was able to detect a cycle and if the foot goes up and down
        // vel = Agent.PositionX()/total_time;             //or to always count it even if there is no formal oscillation
        timeseriesfile.close();
    }
    return vel;
}

//Given a phenotype and an initialized walker, set up a homeostatic flexible legged walker individual to pass to the fitness function and populates the nm vector
// pretty sure having problems here
void Setup(TVector<double>& phen, LeggedAgent& Agent, TVector<double>& neuromodvec){
    // cout << "full phenotpye:" << phen << endl;
    int ctrnnvecsize = neuromodvec.UpperBound();
    // cout << "ctrnnvecsize" << ctrnnvecsize << endl;
    int adhpvecsize = phen.UpperBound() - (2*ctrnnvecsize);
    // cout << "adhpvecsize" << adhpvecsize << endl;
    // cout << ctrnnvecsize << " " << adhpvecsize << endl; 
    int N = adhpvecsize / 2; // for now assuming that all neurons are being regulated
    adhpvecsize = adhpvecsize  + 2*N;

    TVector<double> ctrnnvec(1,ctrnnvecsize);
    TVector<double> adhpvec(1,adhpvecsize);
    adhpvec.FillContents(0);

    TVector<int> plasticitypars(1,(2*N)+(N*N));
    plasticitypars.FillContents(0);
    for (int i=1;i<=N;i++){
        plasticitypars[i] = 1; //bias specifiers are in the beginning now?
    }
    Agent.NervousSystem.SetCircuitSize(N);
    Agent.NervousSystem.SetPlasticityPars(plasticitypars);

    int k = 1;

    for (int i = 1; i <= ctrnnvecsize; i ++){
        ctrnnvec(i) = phen(k);
        k++;
    }
    // cout << "ctrnn:" << ctrnnvec << endl;
    ctrnnvec >> Agent.NervousSystem;

    for (int i = 1; i <= N; i++){
        adhpvec(i) = 1; //gets overwritten in main function by specified Btau value, but this is to prevent divide by zero errors
    }

    for (int i = N+1; i <= adhpvecsize-N; i ++){
        adhpvec(i) = phen(k);
        k ++;
    }
    // cout << "adhpvec:" << adhpvec << endl;
    Agent.NervousSystem.SetHPPhenotype(adhpvec, StepSize, true);
    // cout << Agent.NervousSystem.l_boundary << endl;
    Agent.NervousSystem.WindowReset();

    for (int i = 1; i <= ctrnnvecsize; i ++){
        neuromodvec(i) = phen(k);
        k++;
    }
    // cout << "neuromod:" << neuromodvec << endl;
    return;
}

// Overload where we can use a bestind file -- this works pretty sure
void Setup(ifstream& bestind, LeggedAgent& Agent, TVector<double>& neuromodvec){
    bestind >> Agent.NervousSystem;
    cout << "biases after ctrnn load" << Agent.NervousSystem.biases << endl;
    Agent.NervousSystem.SetHPPhenotypebestind(bestind,StepSize,true); //TODO bestind function needs to be updated because I don't give a sliding window, but instead move on to the neuromod vector
                                                                      // either make it so that the file does contain these things in the future, or build in detection to stop it from updating at that point if goes to neuromodvec
    // cout << "after HPphen:" << Agent.NervousSystem.l_boundary << " " << Agent.NervousSystem.u_boundary << endl;
    // cout << Agent.NervousSystem.biases << endl << Agent.NervousSystem.l_boundary;
    int N = Agent.NervousSystem.CircuitSize();

    for(int i = 1;i <= (2*N)+(N*N);i++){
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
double FlexibleWalking(LeggedAgent& Agent,TVector<double> neuromodvec,double plasticitydur,bool debug){
    // cout << "neuromodvec " << neuromodvec << endl;
    int N = Agent.NervousSystem.CircuitSize();
    // ...and shifting the time constants to the end of the vector
    Shift_NM(neuromodvec,N);
    
    // Pass transient without ADHP
    for (double i=StepSize;i<=transient_dur;i+=StepSize){
        Agent.NervousSystem.EulerStep(StepSize,false);
    }

    if (debug){
        cout << "unmodulated, before ADHP" << endl << Agent.NervousSystem.biases << endl;
    }

    double avg_unmodulated_vel = 0;
    double avg_modulated_vel = 0;
    double unmodulated_vel = 0;
    double modulated_vel = 0;
    int unmodulated_tests = 0;
    int modulated_tests = 0;

    if (rounds == 0){
        // Allow ADHP to run for designated time
        for (double i = StepSize; i <= plasticitydur; i += StepSize){
            Agent.NervousSystem.EulerStep(StepSize, true);
        }
        // Test unmodulated velocity at homeostatic steady state
        unmodulated_vel = meas_velocity(Agent,debug);
        unmodulated_tests ++;
        if (debug){
            cout << "unmodulated, after ADHP" << endl << Agent.NervousSystem.biases << endl << unmodulated_vel << endl;
        }
        avg_unmodulated_vel += unmodulated_vel;
    }
    else{
        //store the effective neuromod vectors and reverse neuromod vectors because we'll be switching back and forth
        for (int round=1;round<=rounds;round++){
            // Allow ADHP to run for designated time
            for (double i = StepSize; i <= plasticitydur; i += StepSize){
                Agent.NervousSystem.EulerStep(StepSize, true);
            }
            if (debug){
                cout << "unmodulated, after ADHP, before test" << endl << Agent.NervousSystem.biases << endl;
            }
            // Test unmodulated velocity at homeostatic steady state
            unmodulated_vel = meas_velocity(Agent,debug);
            unmodulated_tests ++;
            if (debug){
                cout << "unmodulated, after ADHP, after test" << endl << Agent.NervousSystem.biases << endl << unmodulated_vel << endl;
            }
            avg_unmodulated_vel += unmodulated_vel;

            //apply neuromodulation (default is apply to all parameters)
            Modulate(Agent,neuromodvec);
            if (debug){
                cout << "modulated, before ADHP, before test" << endl << Agent.NervousSystem.biases << endl;
            }

            //and measure the immediate modulated velocity
            modulated_vel = meas_velocity(Agent,debug); 
            modulated_tests ++;
            if (debug){
                cout << "modulated, before ADHP, after test" << endl << Agent.NervousSystem.biases << endl << modulated_vel << endl;
            }
            avg_modulated_vel += modulated_vel;

            //allow plasticity to occur in modulated state
            for (double i = StepSize; i <= plasticitydur; i += StepSize){
                Agent.NervousSystem.EulerStep(StepSize, true);
            }
            if (debug){
                cout << "modulated, after ADHP, before test" << endl << Agent.NervousSystem.biases << endl;
            }

            // measure modulated velocity at modulated homeostatic steady state
            modulated_vel = meas_velocity(Agent,debug);
            modulated_tests ++;
            if (debug){
                cout << "modulated, after ADHP, after test" << endl << Agent.NervousSystem.biases << endl << modulated_vel << endl;
            }
            avg_modulated_vel += modulated_vel;

            //then negate all the elements of the effective neuromodulation
            Reverse_NM(neuromodvec);
            //and reverse neuromodulation
            Modulate(Agent,neuromodvec);
            if (debug){
                cout << "unmodulated, before ADHP, before test" << endl << Agent.NervousSystem.biases << endl;
            }

            //measure the immediate unmodulated velocity
            unmodulated_vel = meas_velocity(Agent,debug);
            unmodulated_tests ++;
            if (debug){
                cout << "unmodulated, before ADHP, after test" << endl << Agent.NervousSystem.biases << endl << unmodulated_vel << endl;
            }
            avg_unmodulated_vel += unmodulated_vel;
        }
    }

    avg_unmodulated_vel = avg_unmodulated_vel/max(unmodulated_tests,1);
    avg_modulated_vel = avg_modulated_vel/max(modulated_tests,1);

    double fit = abs(avg_unmodulated_vel - avg_modulated_vel); //maximum fitness is 2*6.27 --max speed both directions

    return fit;
}