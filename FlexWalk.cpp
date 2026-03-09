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

const double cross_tolerance = 0; //shift the threshold away from zero so that parameter jiggle can't influence, if relevant
const double bubble_tolerance = 10*StepSize; //excludes multi-peak oscillations, calculated in state space

const bool HPon = true; //turns ADHP on or off during velocity measurement

void step_function(LeggedAgent& Agent){
    // Allows you to dynamially adjust what kind of stepper you're using (mapping neurons to body actuators)
    Agent.Step2CPG(StepSize,HPon);
};

// let the circuit equilibrate check if the foot goes up and down before doing the full velocity measurement
// simultaneously synchronizes the body to the nervous system (better to separate if doing RPG)
bool quick_osc_check_sync(LeggedAgent& Agent, double transient_dur, double quick_check_dur){
    double dist = 0; //in state space
    int N = Agent.NervousSystem.CircuitSize();
    for (double i = StepSize;i<=transient_dur;i+=StepSize){
        step_function(Agent);
    }
    TVector<double> start(1,N);
    start = Agent.NervousSystem.states;
    for (double i = StepSize;i<=quick_check_dur;i+=StepSize){
        step_function(Agent);
        // cout << "stepped ";
    }
    for (int i = 1; i <= N; i++){
        dist += pow(start(i)-Agent.NervousSystem.NeuronState(i),2);
    }
    dist = pow(dist,.5);
    // cout << "dist " << dist << endl;
    // cout << "bool " << (dist > 0.05) << endl;
    return (dist>0.05);
}

void distance_and_time(LeggedAgent& Agent, double &dist_traveled, double &cycle_time, ofstream &timeseriesfile, ofstream &paramsfile, bool recordoutputs, bool recordparams){
// fix to 1 and zero if it isn't already
    cycle_time = 1;
    int cycle_step = 1;
    dist_traveled = 0;
    // do a quick check to see if oscillates
    if(quick_osc_check_sync(Agent)){ //if it does, continue
        // cout << "oscillating" << endl; 
        // local variables needed by the function 
        int peak_count = 0;
        int N = Agent.NervousSystem.CircuitSize();
        double test_time = 0;
        double new_n1_div = 0;
        double fmr_n1_div = 0;
        bool flip = false;
        bool first_step = true;

        double st_pos_x = 0;
        // double max_ft = 0;
        // double min_ft = 1;

        TVector<double> start(1,N);
        // cout << "start " << start << endl;

        double dist = 0; //state space

        while ((peak_count<2)&(test_time < test_dur)){
            if (recordoutputs){timeseriesfile << Agent.NervousSystem.outputs << endl;}
            if (recordparams){paramsfile << Agent.PositionX() << endl;}
            test_time += StepSize;

            fmr_n1_div = new_n1_div;
            new_n1_div = Agent.NervousSystem.NeuronState(1);
            step_function(Agent);
            new_n1_div = Agent.NervousSystem.NeuronState(1) - new_n1_div; //final - initial
            if (!first_step){
                if((fmr_n1_div>cross_tolerance)&(new_n1_div<-cross_tolerance)){
                    if (peak_count == 1){
                        // cout << "testing: " << Agent.NervousSystem.states << " at " << test_time << endl;
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
                        // cout << "Start: " << start << " at " << test_time << endl;
                        peak_count += 1;
                        cycle_time = 0;
                        cycle_step = 1;
                        // cout << "peaked at " << sync_time + test_time << endl;
                        st_pos_x = Agent.PositionX(); //subtract off the progress that the agent made before the recorded cycle
                    }
                }
            }

            if (peak_count == 1){
                cycle_time += StepSize;
                cycle_step ++;
                // if (Agent.NervousSystem.NeuronOutput(1)>max_ft){max_ft = Agent.NervousSystem.NeuronOutput(1);}
                // if (Agent.NervousSystem.NeuronOutput(1)<min_ft){min_ft = Agent.NervousSystem.NeuronOutput(1);}
            }

            first_step = false;   
        }
        if (peak_count==2){dist_traveled = Agent.PositionX() - st_pos_x;}
    }
    return;
}

void distance_and_time(LeggedAgent& Agent, double &dist_traveled, double &cycle_time, TMatrix<double>& timeseries, bool recordoutputs){
    // fix to 1 and zero if it isn't already
    cycle_time = 1;
    int cycle_step = 1;
    dist_traveled = 0;
    // do a quick check to see if the foot goes up and down
    if(quick_osc_check_sync(Agent)){ //if it does, continue
        // cout << "oscillating" << endl; 
        // local variables needed by the function 
        int peak_count = 0;
        int N = Agent.NervousSystem.CircuitSize();
        double test_time = 0;
        double new_n1_div = 0;
        double fmr_n1_div = 0;
        bool flip = false;
        bool first_step = true;

        double st_pos_x = 0;
        // double max_ft = 0;
        // double min_ft = 1;

        TVector<double> start(1,N);
        start = Agent.NervousSystem.states;
        // cout << "start " << start << endl;

        double dist = 0; //state space

        while ((peak_count<2)&(test_time < test_dur)){
            test_time += StepSize;

            fmr_n1_div = new_n1_div;
            new_n1_div = Agent.NervousSystem.NeuronState(1);
            step_function(Agent);
            new_n1_div = Agent.NervousSystem.NeuronState(1) - new_n1_div; //final - initial
            if (!first_step){
                if((fmr_n1_div>cross_tolerance)&(new_n1_div<cross_tolerance)){
                    // cout << "peaked at " << test_time << endl;
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
                        cycle_time = 0;
                        cycle_step = 1;
                        st_pos_x = Agent.PositionX(); //subtract off the progress that the agent made before the recorded cycle
                    }
                }
            }

            if (peak_count == 1){
                if(recordoutputs){
                    // cout << "step" << cycle_step << " ";
                    for (int i=1;i<=N;i++){
                        timeseries[cycle_step][i] = Agent.NervousSystem.NeuronOutput(i);
                    }
                }
                cycle_time += StepSize;
                cycle_step ++;
                // if (Agent.NervousSystem.NeuronOutput(1)>max_ft){max_ft = Agent.NervousSystem.NeuronOutput(1);}
                // if (Agent.NervousSystem.NeuronOutput(1)<min_ft){min_ft = Agent.NervousSystem.NeuronOutput(1);}
            }

            first_step = false;   
        }
        if (peak_count==2){dist_traveled = Agent.PositionX() - st_pos_x;}
        // cout << "peak count " << peak_count << endl;
    }
    return;
}
//Calcuate the instantaneous walking velocity by running the homeostatic walker for a set time, 
// then measuring the time and distance of the next full neuron cycle 
// used to be calculated from peak to peak, now is calculated from footfall to footfall
// I think there was an ADHP-related reason I had switched it to peak to peak, but footfall seems more robust, so fine for now
// Modularizing this so that it's easier to use bits and pieces
double meas_velocity(LeggedAgent& Agent, ofstream &timeseriesfile, ofstream &paramsfile, bool recordoutputs, bool recordparams){
    double distance = 0;
    double cycletime = 1; //initialize to 1 to avoid dividing by zero
    distance_and_time(Agent,distance,cycletime,timeseriesfile,paramsfile,recordoutputs,recordparams);
    // cout << distance << " " << cycletime << endl;
    double vel = distance/cycletime;
    return vel;
}

//override to output to a vector (not consolidating yet in case want to track the parameters at some point)
double meas_velocity(LeggedAgent& Agent, TMatrix<double>& timeseries, bool recordoutputs){
    double distance = 0;
    double cycletime = 1; //initialize to 1 to avoid dividing by zero
    if (quick_osc_check_sync(Agent)){ //if foot moves, continue
        distance_and_time(Agent,distance,cycletime,timeseries,recordoutputs);
    }
    double vel = distance/cycletime;
    return vel;
}

void SortTraj(TVector<double>& sortedtraj, TVector<double>& neurontrajectory){
    sortedtraj.FillContents(0);
    for (int i = 1; i <= neurontrajectory.UpperBound();i++){
        int j = 1;
        while (sortedtraj[j]>neurontrajectory[i]){
            j ++;
        }
        //find how many elements you need to slide (if any)
        int k = j;
        while (sortedtraj[k] > 0){
            k++;
        }
        //slide down whatever is there
        while (k > j){
            sortedtraj[k] = sortedtraj[k-1];
            k --;
        }
        sortedtraj[j]=neurontrajectory[i];
    }
    return;
}

double CalcUB(TVector<double>& sortedneurontrajectory, double lb){
    double area_under = 0;
    //step through the trajectory
    int i = sortedneurontrajectory.UpperBound();
    //add values below lb to the area underneath
    while(sortedneurontrajectory[i]<lb){
        area_under += lb - sortedneurontrajectory[i];
        i --; //iterate backwards because is sorted in descending order
    //the rest make up the list of points above LB (potential "active" points)
    }
    // cout << "C= " << area_under << " with " << i << " points" << endl; 
    //the cumulative sum of the largest points
    double cum_sum = 0;
    double potential_ub = 0;
    int j = 1;
    while ((j<=i)&&(potential_ub < sortedneurontrajectory[j])){
        // cout << "adding value: " << sortedneurontrajectory[j] << endl;
        cum_sum += sortedneurontrajectory[j];
        potential_ub = (cum_sum - area_under)/j;
        // cout << "potential UB #" << j << ": " << potential_ub << " is greater than " << sortedneurontrajectory[j] << "?" << endl;
        j++;
    }
    return potential_ub;
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
void Reverse_NM(TVector<double>& neuromodvec, TVector<double>& reversedneuromodvec){
    // negates all elements of neuromodulatory vector as presented
    for(int i=neuromodvec.LowerBound();i<=neuromodvec.UpperBound();i++){
        reversedneuromodvec[i] = -neuromodvec[i];
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

    // format NM by shifting time constants to the end
    Shift_NM(neuromodvec,N);
    // and store the original neuromod vec with the copy constructor because that's always what we're going to want to try to apply
    TVector<double> reverse_neuromodvec(1,neuromodvec.UpperBound());
    Reverse_NM(neuromodvec,reverse_neuromodvec);
    
    // Pass transient without ADHP
    for (double i=StepSize;i<=150;i+=StepSize){
        Agent.NervousSystem.EulerStep(StepSize,false);
    }

    double avg_unmodulated_vel = 0;
    double avg_modulated_vel = 0;
    double unmodulated_vel = 0;
    double modulated_vel = 0;
    int unmodulated_tests = 0;
    int modulated_tests = 0;

    //store the effective neuromod vectors and reverse neuromod vectors because we'll be switching back and forth
    for (int round=1;round<=rounds;round++){
        if (debug){
            cout << "unmodulated, before ADHP, before test"<< endl << Agent.NervousSystem.taus << endl << Agent.NervousSystem.biases <<endl <<Agent.NervousSystem.weights << endl << endl;
        }
        // Test unmodulated velocity at initial configuration
        unmodulated_vel = meas_velocity(Agent,timeseriesfile,paramsfile,debug,debug);
        unmodulated_tests ++;
        if (debug){
            cout << "unmodulated, before ADHP, after test: " << unmodulated_vel<< endl << Agent.NervousSystem.taus << endl << Agent.NervousSystem.biases <<endl <<Agent.NervousSystem.weights << endl<< endl;
        }
        avg_unmodulated_vel += unmodulated_vel;

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
            cout << "unmodulated, after ADHP, before test" << endl<< Agent.NervousSystem.taus << endl << Agent.NervousSystem.biases <<endl <<Agent.NervousSystem.weights << endl << endl;
        }
        // Test unmodulated velocity at homeostatic steady state
        unmodulated_vel = meas_velocity(Agent,timeseriesfile,paramsfile,debug,debug);
        unmodulated_tests ++;
        if (debug){
            cout << "unmodulated, after ADHP, after test: " << unmodulated_vel<< endl << Agent.NervousSystem.taus << endl << Agent.NervousSystem.biases <<endl <<Agent.NervousSystem.weights << endl<< endl;
        }
        avg_unmodulated_vel += unmodulated_vel;

        //apply neuromodulation (default is apply to all parameters)
        Modulate(Agent,neuromodvec);

        if (debug){
            cout << "modulated, before ADHP, before test"<< endl << Agent.NervousSystem.taus << endl << Agent.NervousSystem.biases <<endl <<Agent.NervousSystem.weights << endl << endl;
        }

        //and measure the immediate modulated velocity
        modulated_vel = meas_velocity(Agent,timeseriesfile,paramsfile,debug,debug); 
        modulated_tests ++;
        if (debug){
            cout << "modulated, before ADHP, after test: " << modulated_vel << endl<< Agent.NervousSystem.taus << endl << Agent.NervousSystem.biases <<endl <<Agent.NervousSystem.weights << endl << endl;
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
            cout << "modulated, after ADHP, before test"<< endl << Agent.NervousSystem.taus << endl << Agent.NervousSystem.biases <<endl <<Agent.NervousSystem.weights << endl << endl;
        }

        // measure modulated velocity at modulated homeostatic steady state
        modulated_vel = meas_velocity(Agent,timeseriesfile,paramsfile,debug,debug);
        modulated_tests ++;
        if (debug){
            cout << "modulated, after ADHP, after test: " << modulated_vel << endl<< Agent.NervousSystem.taus << endl << Agent.NervousSystem.biases <<endl <<Agent.NervousSystem.weights << endl << endl;
        }
        avg_modulated_vel += modulated_vel;

        //and reverse neuromodulation
        Modulate(Agent,reverse_neuromodvec); //caps are always applied in the modulate function, so don't need to calculate ahead of time
    }


    avg_unmodulated_vel = avg_unmodulated_vel/max(unmodulated_tests,1);
    avg_modulated_vel = avg_modulated_vel/max(modulated_tests,1);

    double fit = abs(avg_unmodulated_vel - avg_modulated_vel); //maximum fitness is 2*.627 --max speed both directions

    timeseriesfile.close();
    // bodyfile.close();
    paramsfile.close();
    return fit;
}