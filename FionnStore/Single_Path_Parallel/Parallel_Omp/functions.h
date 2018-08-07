#ifndef FUNCTIONS
#define FUNCTIONS

#include <complex>
#include <vector>
#include <stdlib.h>
#include <math.h>
#include <stdio.h>
#include <iostream>
#include "density.h"

double gam(double *R);

double Hb(double *R, double *P);

double dgamma(double *R, int i);

void Fb(double *R, double *f);

void F1(double *R, double *f);

void F2(double *R, double *f);

double dE(double *R);

double G(double *R);

double dd(double *dhat, double *R, double *dgam);

void integ_step(double *r, double *v, double dt, int Sa, double *f);

void bath_para(double eta, double w_max);

double U(double *r,double *v, int Sa, double t, double *f);

void setwww();

double dens_init_0(double *x,double *p);

double dens_init_1(double *x,double *p);

double dens_init_2(double *x,double *p);

double dens_init_3(double *x,double *p);

double obs_0(double *x,double *p);

double obs_1(double *x,double *p);

double obs_2(double *x,double *p);

double obs_3(double *x,double *p);

double H_0(double *x,double *p);

double H_1(double *x,double *p);

double H_2(double *x,double *p);

double H_3(double *x,double *p);

#endif 
