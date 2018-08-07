#ifndef DENSITY
#define DENSITY

#include <complex>
#include <vector>
#include <stdlib.h>
#include <math.h>
#include <stdio.h>
#include <iostream>
#include "random.h"
#include "functions.h"
#include   <omp.h>
#include <gsl/gsl_rng.h>

int density(double *x,double *p, double *f, double *abszsum1, double *argzsum1, double *habszsum1, double *hargzsum1, double **realsum, double **imagsum, double **hrealsum, double **himagsum, double *a, gsl_rng * rr, double * ranVector);

#endif
