#ifndef RANDOM
#define RANDOM 


#include <complex>
#include <vector>
#include <stdlib.h>
#include <math.h>
#include <stdio.h>
#include <iostream>
#include <gsl/gsl_rng.h>

void gauss_init_W(double *R, double *v, gsl_rng * rr, double * ranVector);

#endif
