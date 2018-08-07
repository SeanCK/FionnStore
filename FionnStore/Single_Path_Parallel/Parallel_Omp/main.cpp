/*!
 *  \mainpage
 *  \brief Using a Trotter Approximation to calculate a quantum-classical non-adiabatic approximation to
 * separate the quantum and the quasi-classical degrees of freedom to allow a surface-hopping scheme to
 * be implemented that creates a tree-like structure.
 *
 * This code follows one of the possible paths through the tree structure. The non-adiabatic propagator
 * determines the likelihood of a jump based on importance sampling
 *
 * \author Donal MacKernan, Athina Lange, Philip McGrath and Sean Kelly
 * \date 24.7.17
 */

#include   <stdio.h>
#include   <math.h>
#include   <iostream>
#include   <complex>
#include   "density.h"
#include   "dmatrix.c"
#include   "functions.h"
#include   <omp.h>

using namespace std;
#include <gsl/gsl_rng.h>

///////////////////////////////////////////////////////////////////////////////
/// SYSTEM INPUT
///////////////////////////////////////////////////////////////////////////////

char  datafilename[80] = {"Output"};
FILE   *stream;

const gsl_rng_type * TT; /*!< Random Number Generator seed based on Gaussian Distribution */

int N_bath; /*!< Size of bath */
int N_slice; /*!< Number of time intervals */
int Ncut; /*!< Truncation parameter */
double timestep; /*!< Size of time interval */
int Nsample; /*!< Sample Size (No. of trees calculated) */
double beta; /*!< Inverse Temperature */
double delta; /*!< MD Integrating Timestep \f$ (\delta) \f$*/
double ppower; /*!< */


double ddd; /*!<  \f$ \delta^2 \f$ */
double ddd4; /*!< \f$ \delta^2/4 \f$ */
double *m; /*!< Mass of particles */
double *c; /*!< */
double *w; /*!< */

double *mww; /*!< */
double *sig; /*!< Sigma/Variance */
double *mu; /*!< */
double TSLICE; /*!< Time per time interval*/

double *abszsum1;
double *argzsum1;
double *habszsum1;
double *hargzsum1;

double **realsum;
double **imagsum;
double **hrealsum;
double **himagsum;

extern double **dmatrix(int, int, int, int);

double *a; // dummy variable for reduction testing

extern double de;

double (*dens_init[4])(double*, double*); /*!< Initial Density Matrix*/
double (*obs[4])(double*, double*); /*!< Observable Matrix*/
double (*obs1[4])(double*, double*); /*!< Another Observable Matrix*/
void (*force[4])(double *, double*); /*!< Hellman-Feynman Forces*/


// ================================================================
// MAIN
// ================================================================


int main(int argc, char *argv[]){

    double w_max, eta, T;
    int  i,init_seed;

    int Nblock = 1024;
    int t_strobe = 1;

    ///////////////////////////////////////////////////////////////////////////////
    /// SYSTEM INPUT
    ///////////////////////////////////////////////////////////////////////////////
    //cout << " Print information about new stream: " << endl;
    //cout << "Input datafilename, N_bath, N_slice, Ncut\n timestep, T, init_seed, Nsample\n w_max, eta, beta, delta,power " << endl;
    //cin >> datafilename >> N_bath >> N_slice >> Ncut >> timestep >> T >> init_seed >> Nsample >> w_max >> eta >> beta >> delta >> ppower;
//    cout << " Print information about new stream:" << endl;
//    cout << "Input datafilename" << endl;
//    cin >> datafilename;
    N_bath = 200;
    N_slice = 60;
    Ncut = 10;
    timestep = 0.05;
    T = 15;
    init_seed = 0;
    Nsample = 10000;
    w_max = 3;
    eta = 0.13;
    beta = 25;
    delta = 0.8;
    ppower = 100000;

    ///////////////////////////////////////////////////////////////////////////////
    /// ALLOCATING MEMORY
    ///////////////////////////////////////////////////////////////////////////////

    mww = new double[N_bath];
    mu = new double[N_bath];
    sig =  new double[2*N_bath];
    c = new double[N_bath];
    m = new double[N_bath];
    w = new double[N_bath];
    abszsum1  = new double[N_slice];
    argzsum1  = new double[N_slice];
    habszsum1  = new double[N_slice];
    hargzsum1  = new double[N_slice];
    realsum = dmatrix(0,N_slice,0,N_slice+1);   // Included the real/imag arrays to compare output to OldCode output
    imagsum = dmatrix(0,N_slice,0,N_slice+1);
    hrealsum = dmatrix(0,N_slice,0,N_slice+1);
    himagsum = dmatrix(0,N_slice,0,N_slice+1);

    double Dt = T/N_slice;

    a = new double[N_slice]; // dummy variable for reduction test

    ///////////////////////////////////////////////////////////////////////////////
    /// INITIALIZATION OF SYSTEM
    ///////////////////////////////////////////////////////////////////////////////


    dens_init[0] = dens_init_0; dens_init[1] = dens_init_1;
    dens_init[2] = dens_init_2; dens_init[3] = dens_init_3;
    obs[0] = obs_0; obs[1] = obs_1; obs[2] = obs_2; obs[3] = obs_3;
    obs1[0] = H_0; obs1[1] = H_1; obs1[2] = H_2; obs1[3] = H_3;

    ddd4 = delta*delta*0.25;
    ddd =  delta*delta;
    TSLICE  = T/N_slice;

    bath_para(eta, w_max);       /*!< Defining Bath Parameters */

    for (i = 0; i < N_bath; ++i)
        mu[i] = beta*w[i]*0.5;
    for (i = 0; i < N_bath; ++i){
        sig[i] = 1.0/sqrt(w[i]*2.0*tanh(mu[i]));
        mww[i] = -m[i]*w[i]*w[i];
    }
    for (i = 0; i < N_bath; ++i)
        sig[i+N_bath] = 1.0*sqrt(w[i]/(2.0*tanh(mu[i])));

    /*! Defining force field */
    force[0] = F1;
    force[1] = Fb;
    force[2] = Fb;
    force[3] = F2;


//    stream = fopen(datafilename,"w");
//    fprintf(stream,"%s\n w_max %lf eta %lf beta %lf delta %lf ppower %lf N_bath %d N_slice %d\n", argv[0], w_max, eta, beta, delta, ppower, N_bath, N_slice);
//    fclose(stream);

    /*! Initializing sum1 counters*/
    for (int i = 0; i < N_slice; ++i){
        abszsum1[i] = 0.0;
        argzsum1[i]  = 0.0;
        habszsum1[i] = 0.0;
        hargzsum1[i] = 0.0;

        a[i] = 0.0; // dummy variable for reduction testing

        for (int j = 0; j <= N_slice; j++){
            realsum[i][j] = 0.0;
            imagsum[i][j] = 0.0;
            hrealsum[i][j] = 0.0;
            himagsum[i][j] = 0.0;
            }

    }

    /// PARALLEL DIRECTIVE

    /*! Defining non-adiabatic coupling matrix */
    setwww();
    ///////////////////////////////////////////////////////////////////////////////
    /// PROCESSING TREE
    ///////////////////////////////////////////////////////////////////////////////

    int nthreads;

    // Print column headers for output file
    stream = fopen(datafilename,"a");
    fprintf(stream,"Nens\tDt\tj\tabszsum1\targzsum1\trealsum\timagsum\thabszsum1\thargzsum1\threalsum\thimagsum\n");
    fclose(stream);

    int numthreads;
    numthreads = 24;

    gsl_rng ** rrp;
    gsl_rng_env_setup();
    TT = gsl_rng_default;
    rrp = new gsl_rng * [numthreads];
    int * seed_rrp;
    seed_rrp = new int [numthreads];


    for (int i = 0; i < numthreads; ++i) {
        rrp[i] = gsl_rng_alloc(TT);
        seed_rrp[i] = i;
        gsl_rng_set(rrp[i], seed_rrp[i]);
    }

#pragma omp parallel for num_threads(numthreads) reduction(+:abszsum1[0:N_slice], argzsum1[0:N_slice], habszsum1[0:N_slice], hargzsum1[0:N_slice], a[0:N_slice])
for (int i = 0; i < Nsample; ++i){

	double *f; /*!< Force on particles */
	double  *R1, *v;

    double ranVector[10001];

    int id;
    id = omp_get_thread_num();

    /*! Sets up the use of a Gaussian Random Number Generator from GSL */

    gsl_rng * rlocal;
    rlocal = rrp[id];

    R1 = new double[N_bath];
	v = new double[N_bath];
	f = new double[N_bath];

//        printf(" calling density %d %d j %d \n", id, numthreads, i);
	density(R1, v, f, abszsum1, argzsum1, habszsum1, hargzsum1, realsum, imagsum, hrealsum, himagsum, a, rlocal, ranVector);

            if (((i+1) % Nblock) == 0){
                double l  = 1.0/(i+1);
                #pragma omp critical
                {
                    stream = fopen(datafilename, "a");
                    for (int k = 0; k < N_slice; k++)
                        if (((k + 1) % t_strobe) == 0) {

                            for (int j = 0; j <= (Ncut + 1); j++) {
                                fprintf(stream, "%d %lf %d %lf %lf %lf %lf %lf %lf %lf %lf\n", i + 1, Dt * (k + 1), j,
                                        (abszsum1[k]) * l, (argzsum1[k]) * l, realsum[k][j] * l, imagsum[k][j] * l,
                                        (habszsum1[k]) * l, (hargzsum1[k]) * l, hrealsum[k][j] * l,
                                        himagsum[k][j] * l /*, hist[k][j]*l */);
//                               printf("%d %lf %d %lf %lf %lf %lf %lf %lf %lf %lf\n", i + 1, Dt * (k + 1), j,
//                                      (abszsum1[k]) * l, (argzsum1[k]) * l, realsum[k][j] * l, imagsum[k][j] * l,
//                                       (habszsum1[k]) * l, (hargzsum1[k]) * l, hrealsum[k][j] * l,
//                                       himagsum[k][j] * l /*, hist[k][j]*l */);
                            }
                        }
                    fclose(stream);
                }
            }
	if (i == 1){nthreads = omp_get_num_threads();} // Detects number of threads in parallel region

	delete [] R1; delete [] v; delete [] f;

}

// Edited for fionn: 'printf' replaces 'fprintf' statements
#pragma omp critical
    {
        stream = fopen(datafilename, "a");
//        fprintf(stream, "dt %lf T %lf Nsample %d\n", timestep, T, Nsample);
//        fprintf(stream, "Time   abszsum1   argzsum1   realsum   habszsum1   hargzsum1   hrealsum   a\n");
	printf("dt %lf T %lf Nsample %d\n", timestep, T, Nsample);
	printf("Time   i   abszsum1   argzsum1   realsum   habszsum1   hargzsum1   hrealsum   a\n");

        for (int i = 0; i < N_slice; ++i) {
		int j = 0;
//            for (int j = 0; j < Ncut; ++j) {
                printf("%lf    %d   %lf   %lf   %lf   %lf   %lf   %lf   %lf\n", TSLICE * (i + 1), i,
                       (abszsum1[i] / Nsample), (argzsum1[i] / Nsample), realsum[i][j] / Nsample,
                       (habszsum1[i] / Nsample), (hargzsum1[i] / Nsample), hrealsum[i][j] / Nsample, a[i]);
                fprintf(stream, "%lf   %lf   %lf   %lf   %lf   %lf   %lf   %lf\n", TSLICE * (i + 1),
                        (abszsum1[i] / Nsample), (argzsum1[i] / Nsample), realsum[i][j] / Nsample,
                        (habszsum1[i] / Nsample), (hargzsum1[i] / Nsample), hrealsum[i][j] / Nsample, a[i]);
//            fprintf(stream, "\nNcut is %d\n", Ncut);
//            }
        }
//        fprintf(stream, "Number of threads was %d", nthreads);
        fclose(stream);
    }

    cout << "Number of threads was "<< nthreads << endl;

    ///////////////////////////////////////////////////////////////////////////////
    /// DEALLOCATING MEMEORY
    ///////////////////////////////////////////////////////////////////////////////

    delete [] abszsum1; delete [] argzsum1; delete [] habszsum1; delete [] hargzsum1;
    delete [] realsum; delete [] imagsum; delete [] hrealsum; delete [] himagsum;

    delete [] a; // dummy variable for reduction test

    delete [] mww; delete [] mu; delete [] sig;
    delete [] c; delete [] m; delete [] w;

    return 0;
}
