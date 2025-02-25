#ifndef MECHANISM_H
#define MECHANISM_H

#include <AMReX_Gpu.H>
#include <AMReX_REAL.H>

/* Elements
0  H
1  C
2  O
3  N
4  Xe
*/

// Species
#define h_ID 0
#define h2_ID 1
#define o_ID 2
#define o2_ID 3
#define oh_ID 4
#define h2o_ID 5
#define n2_ID 6
#define ho2_ID 7
#define h2o2_ID 8
#define Xe_ID 9

#define NUM_GAS_ELEMENTS 5   // Elements in the homogeneous phase
#define NUM_GAS_SPECIES 10   // Species in the homogeneous phase
#define NUM_GAS_REACTIONS 21 // Reactions in the homogeneous phase

#define SITE_DENSITY 0.000000E+00 // mol/cm^2

#define NUM_SURFACE_ELEMENTS 0  // Additional elements in heterogeneous phase
#define NUM_SURFACE_SPECIES 0   // Species in the heterogeneous phase
#define NUM_SURFACE_REACTIONS 0 // Reactions in the heterogeneous phase

#define NUM_ELEMENTS (NUM_GAS_ELEMENTS + NUM_SURFACE_ELEMENTS)
#define NUM_SPECIES (NUM_GAS_SPECIES + NUM_SURFACE_SPECIES)
#define NUM_REACTIONS (NUM_GAS_REACTIONS + NUM_SURFACE_REACTIONS)

#define NUM_IONS 0

#define NUM_FIT 4

//  ALWAYS on CPU stuff -- can have different def depending on if we are CPU or
//  GPU based. Defined in mechanism.cpp
void atomicWeight(amrex::Real* awt);
//  MISC
void CKAWT(amrex::Real* awt);
void CKNCF(int* ncf);
void CKSYME_STR(amrex::Vector<std::string>& ename);
void CKSYMS_STR(amrex::Vector<std::string>& kname);
void GET_RMAP(int* _rmap);
void CKINU(const int i, int& nspec, int* ki, int* nu);
void CKKFKR(
  const amrex::Real P,
  const amrex::Real T,
  const amrex::Real* x,
  amrex::Real* q_f,
  amrex::Real* q_r);
void progressRateFR(
  amrex::Real* q_f, amrex::Real* q_r, amrex::Real* sc, amrex::Real T);
//  SPARSE INFORMATION
void SPARSITY_INFO(int* nJdata, const int* consP, int NCELLS);
void SPARSITY_INFO_SYST(int* nJdata, const int* consP, int NCELLS);
void SPARSITY_INFO_SYST_SIMPLIFIED(int* nJdata, const int* consP);
void
SPARSITY_PREPROC_CSC(int* rowVals, int* colPtrs, const int* consP, int NCELLS);
void SPARSITY_PREPROC_CSR(
  int* colVals, int* rowPtrs, const int* consP, int NCELLS, int base);
void SPARSITY_PREPROC_SYST_CSR(
  int* colVals, int* rowPtrs, const int* consP, int NCELLS, int base);
void SPARSITY_PREPROC_SYST_SIMPLIFIED_CSC(
  int* rowVals, int* colPtrs, int* indx, const int* consP);
void SPARSITY_PREPROC_SYST_SIMPLIFIED_CSR(
  int* colVals, int* rowPtr, const int* consP, int base);

// A few mechanism parameters
AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE void
CKINDX(int& mm, int& kk, int& ii, int& nfit)
{
  mm = 5;
  kk = 10;
  ii = 21;
  nfit = -1; // Why do you need this anyway ?
}

//  inverse molecular weights
#ifdef AMREX_USE_GPU
AMREX_GPU_CONSTANT const amrex::Real global_imw[10] = {
  0.9920634920634921, // h
  0.4960317460317460, // h2
  0.0625039064941559, // o
  0.0312519532470779, // o2
  0.0587993179279120, // oh
  0.0555092978073827, // h2o
  0.0356964374955379, // n2
  0.0302975216627280, // ho2
  0.0293996589639560, // h2o2
  0.0076165522914398, // Xe
};
#endif
const amrex::Real h_global_imw[10] = {
  0.9920634920634921, // h
  0.4960317460317460, // h2
  0.0625039064941559, // o
  0.0312519532470779, // o2
  0.0587993179279120, // oh
  0.0555092978073827, // h2o
  0.0356964374955379, // n2
  0.0302975216627280, // ho2
  0.0293996589639560, // h2o2
  0.0076165522914398, // Xe
};

//  molecular weights
#ifdef AMREX_USE_GPU
AMREX_GPU_CONSTANT const amrex::Real global_mw[10] = {
  1.008000,   // h
  2.016000,   // h2
  15.999000,  // o
  31.998000,  // o2
  17.007000,  // oh
  18.015000,  // h2o
  28.014000,  // n2
  33.006000,  // ho2
  34.014000,  // h2o2
  131.293000, // Xe
};
#endif
const amrex::Real h_global_mw[10] = {
  1.008000,   // h
  2.016000,   // h2
  15.999000,  // o
  31.998000,  // o2
  17.007000,  // oh
  18.015000,  // h2o
  28.014000,  // n2
  33.006000,  // ho2
  34.014000,  // h2o2
  131.293000, // Xe
};

//  inverse molecular weights
AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE void
get_imw(amrex::Real* imw_new)
{
  imw_new[0] = 0.9920634920634921; // h
  imw_new[1] = 0.4960317460317460; // h2
  imw_new[2] = 0.0625039064941559; // o
  imw_new[3] = 0.0312519532470779; // o2
  imw_new[4] = 0.0587993179279120; // oh
  imw_new[5] = 0.0555092978073827; // h2o
  imw_new[6] = 0.0356964374955379; // n2
  imw_new[7] = 0.0302975216627280; // ho2
  imw_new[8] = 0.0293996589639560; // h2o2
  imw_new[9] = 0.0076165522914398; // Xe
}

//  inverse molecular weight
AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE amrex::Real
imw(const int n)
{
#if AMREX_DEVICE_COMPILE
  return global_imw[n];
#else
  return h_global_imw[n];
#endif
}
//  molecular weights
AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE void
get_mw(amrex::Real* mw_new)
{
  mw_new[0] = 1.008000;   // h
  mw_new[1] = 2.016000;   // h2
  mw_new[2] = 15.999000;  // o
  mw_new[3] = 31.998000;  // o2
  mw_new[4] = 17.007000;  // oh
  mw_new[5] = 18.015000;  // h2o
  mw_new[6] = 28.014000;  // n2
  mw_new[7] = 33.006000;  // ho2
  mw_new[8] = 34.014000;  // h2o2
  mw_new[9] = 131.293000; // Xe
}

//  molecular weight
AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE amrex::Real
mw(const int n)
{
#if AMREX_DEVICE_COMPILE
  return global_mw[n];
#else
  return h_global_mw[n];
#endif
}

//  Returns R, Rc, Patm
AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE void
CKRP(amrex::Real& ru, amrex::Real& ruc, amrex::Real& pa)
{
  ru = 8.31446261815324e+07;
  ruc = 1.98721558317399615845;
  pa = 1.01325e+06;
}

// compute Cv/R at the given temperature
AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE void
cv_R(amrex::Real* species, const amrex::Real T)
{
  const amrex::Real T2 = T * T;
  const amrex::Real T3 = T * T * T;
  const amrex::Real T4 = T * T * T * T;

  // species with no change across T
  // species 0: h
  species[0] = +1.50000000e+00;

  // species with midpoint at T=1000 kelvin
  if (T < 1000) {
    // species 1: h2
    species[1] = +2.29812400e+00 + 8.24944200e-04 * T - 8.14301500e-07 * T2 -
                 9.47543400e-11 * T3 + 4.13487200e-13 * T4;
    // species 2: o
    species[2] = +1.94642900e+00 - 1.63816600e-03 * T + 2.42103200e-06 * T2 -
                 1.60284300e-09 * T3 + 3.89069600e-13 * T4;
    // species 3: o2
    species[3] = +2.21293600e+00 + 1.12748600e-03 * T - 5.75615000e-07 * T2 +
                 1.31387700e-09 * T3 - 8.76855400e-13 * T4;
    // species 5: h2o
    species[5] = +2.38684200e+00 + 3.47498200e-03 * T - 6.35469600e-06 * T2 +
                 6.96858100e-09 * T3 - 2.50658800e-12 * T4;
    // species 6: n2
    species[6] = +2.29867700e+00 + 1.40824000e-03 * T - 3.96322200e-06 * T2 +
                 5.64151500e-09 * T3 - 2.44485500e-12 * T4;
    // species 7: ho2
    species[7] = +3.30179801e+00 - 4.74912051e-03 * T + 2.11582891e-05 * T2 -
                 2.42763894e-08 * T3 + 9.29225124e-12 * T4;
    // species 8: h2o2
    species[8] = +2.38875400e+00 + 6.56922600e-03 * T - 1.48501300e-07 * T2 -
                 4.62580600e-09 * T3 + 2.47151500e-12 * T4;
    // species 9: Xe
    species[9] = +1.50000000e+00 - 8.99141330e-14 * T + 2.52196860e-16 * T2 -
                 2.92186662e-19 * T3 + 1.18949218e-22 * T4;
  } else {
    // species 1: h2
    species[1] = +1.99142300e+00 + 7.00064400e-04 * T - 5.63382900e-08 * T2 -
                 9.23157800e-12 * T3 + 1.58275200e-15 * T4;
    // species 2: o
    species[2] = +1.54206000e+00 - 2.75506200e-05 * T - 3.10280300e-09 * T2 +
                 4.55106700e-12 * T3 - 4.36805200e-16 * T4;
    // species 3: o2
    species[3] = +2.69757800e+00 + 6.13519700e-04 * T - 1.25884200e-07 * T2 +
                 1.77528100e-11 * T3 - 1.13643500e-15 * T4;
    // species 5: h2o
    species[5] = +1.67214600e+00 + 3.05629300e-03 * T - 8.73026000e-07 * T2 +
                 1.20099600e-10 * T3 - 6.39161800e-15 * T4;
    // species 6: n2
    species[6] = +1.92664000e+00 + 1.48797700e-03 * T - 5.68476100e-07 * T2 +
                 1.00970400e-10 * T3 - 6.75335100e-15 * T4;
    // species 7: ho2
    species[7] = +3.01721090e+00 + 2.23982013e-03 * T - 6.33658150e-07 * T2 +
                 1.14246370e-10 * T3 - 1.07908535e-14 * T4;
    // species 8: h2o2
    species[8] = +3.57316700e+00 + 4.33613600e-03 * T - 1.47468900e-06 * T2 +
                 2.34890400e-10 * T3 - 1.43165400e-14 * T4;
    // species 9: Xe
    species[9] = +1.50005322e+00 - 1.05136544e-07 * T + 6.75326897e-11 * T2 -
                 1.70944909e-14 * T3 + 1.47681049e-18 * T4;
  }

  // species with midpoint at T=1710 kelvin
  if (T < 1710) {
    // species 4: oh
    species[4] = +2.41896226e+00 + 3.19255801e-04 * T - 3.08292717e-07 * T2 +
                 3.64407494e-10 * T3 - 1.00195479e-13 * T4;
  } else {
    // species 4: oh
    species[4] = +1.85376040e+00 + 1.02994334e-03 * T - 2.32666477e-07 * T2 +
                 1.93750704e-11 * T3 - 3.15759847e-16 * T4;
  }
}

// compute Cp/R at the given temperature
AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE void
cp_R(amrex::Real* species, const amrex::Real T)
{
  const amrex::Real T2 = T * T;
  const amrex::Real T3 = T * T * T;
  const amrex::Real T4 = T * T * T * T;

  // species with no change across T
  // species 0: h
  species[0] = +2.50000000e+00;

  // species with midpoint at T=1000 kelvin
  if (T < 1000) {
    // species 1: h2
    species[1] = +3.29812400e+00 + 8.24944200e-04 * T - 8.14301500e-07 * T2 -
                 9.47543400e-11 * T3 + 4.13487200e-13 * T4;
    // species 2: o
    species[2] = +2.94642900e+00 - 1.63816600e-03 * T + 2.42103200e-06 * T2 -
                 1.60284300e-09 * T3 + 3.89069600e-13 * T4;
    // species 3: o2
    species[3] = +3.21293600e+00 + 1.12748600e-03 * T - 5.75615000e-07 * T2 +
                 1.31387700e-09 * T3 - 8.76855400e-13 * T4;
    // species 5: h2o
    species[5] = +3.38684200e+00 + 3.47498200e-03 * T - 6.35469600e-06 * T2 +
                 6.96858100e-09 * T3 - 2.50658800e-12 * T4;
    // species 6: n2
    species[6] = +3.29867700e+00 + 1.40824000e-03 * T - 3.96322200e-06 * T2 +
                 5.64151500e-09 * T3 - 2.44485500e-12 * T4;
    // species 7: ho2
    species[7] = +4.30179801e+00 - 4.74912051e-03 * T + 2.11582891e-05 * T2 -
                 2.42763894e-08 * T3 + 9.29225124e-12 * T4;
    // species 8: h2o2
    species[8] = +3.38875400e+00 + 6.56922600e-03 * T - 1.48501300e-07 * T2 -
                 4.62580600e-09 * T3 + 2.47151500e-12 * T4;
    // species 9: Xe
    species[9] = +2.50000000e+00 - 8.99141330e-14 * T + 2.52196860e-16 * T2 -
                 2.92186662e-19 * T3 + 1.18949218e-22 * T4;
  } else {
    // species 1: h2
    species[1] = +2.99142300e+00 + 7.00064400e-04 * T - 5.63382900e-08 * T2 -
                 9.23157800e-12 * T3 + 1.58275200e-15 * T4;
    // species 2: o
    species[2] = +2.54206000e+00 - 2.75506200e-05 * T - 3.10280300e-09 * T2 +
                 4.55106700e-12 * T3 - 4.36805200e-16 * T4;
    // species 3: o2
    species[3] = +3.69757800e+00 + 6.13519700e-04 * T - 1.25884200e-07 * T2 +
                 1.77528100e-11 * T3 - 1.13643500e-15 * T4;
    // species 5: h2o
    species[5] = +2.67214600e+00 + 3.05629300e-03 * T - 8.73026000e-07 * T2 +
                 1.20099600e-10 * T3 - 6.39161800e-15 * T4;
    // species 6: n2
    species[6] = +2.92664000e+00 + 1.48797700e-03 * T - 5.68476100e-07 * T2 +
                 1.00970400e-10 * T3 - 6.75335100e-15 * T4;
    // species 7: ho2
    species[7] = +4.01721090e+00 + 2.23982013e-03 * T - 6.33658150e-07 * T2 +
                 1.14246370e-10 * T3 - 1.07908535e-14 * T4;
    // species 8: h2o2
    species[8] = +4.57316700e+00 + 4.33613600e-03 * T - 1.47468900e-06 * T2 +
                 2.34890400e-10 * T3 - 1.43165400e-14 * T4;
    // species 9: Xe
    species[9] = +2.50005322e+00 - 1.05136544e-07 * T + 6.75326897e-11 * T2 -
                 1.70944909e-14 * T3 + 1.47681049e-18 * T4;
  }

  // species with midpoint at T=1710 kelvin
  if (T < 1710) {
    // species 4: oh
    species[4] = +3.41896226e+00 + 3.19255801e-04 * T - 3.08292717e-07 * T2 +
                 3.64407494e-10 * T3 - 1.00195479e-13 * T4;
  } else {
    // species 4: oh
    species[4] = +2.85376040e+00 + 1.02994334e-03 * T - 2.32666477e-07 * T2 +
                 1.93750704e-11 * T3 - 3.15759847e-16 * T4;
  }
}

// compute the g/(RT) at the given temperature
AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE void
gibbs(amrex::Real* species, const amrex::Real T)
{
  const amrex::Real T2 = T * T;
  const amrex::Real T3 = T * T * T;
  const amrex::Real T4 = T * T * T * T;
  const amrex::Real invT = 1.0 / T;
  const amrex::Real logT = log(T);

  // species with no change across T
  // species 0: h
  species[0] = +2.547163000000000e+04 * invT + 2.960117600000000e+00 -
               2.500000000000000e+00 * logT;

  // species with midpoint at T=1000 kelvin
  if (T < 1000) {
    // species 1: h2
    species[1] = -1.012521000000000e+03 * invT + 6.592218000000000e+00 -
                 3.298124000000000e+00 * logT - 4.124721000000000e-04 * T +
                 1.357169166666667e-07 * T2 + 7.896194999999999e-12 * T3 -
                 2.067436000000000e-14 * T4;
    // species 2: o
    species[2] = +2.914764000000000e+04 * invT - 1.756599999999997e-02 -
                 2.946429000000000e+00 * logT + 8.190830000000000e-04 * T -
                 4.035053333333333e-07 * T2 + 1.335702500000000e-10 * T3 -
                 1.945348000000000e-14 * T4;
    // species 3: o2
    species[3] = -1.005249000000000e+03 * invT - 2.821802000000000e+00 -
                 3.212936000000000e+00 * logT - 5.637430000000000e-04 * T +
                 9.593583333333333e-08 * T2 - 1.094897500000000e-10 * T3 +
                 4.384277000000000e-14 * T4;
    // species 5: h2o
    species[5] = -3.020811000000000e+04 * invT + 7.966090000000001e-01 -
                 3.386842000000000e+00 * logT - 1.737491000000000e-03 * T +
                 1.059116000000000e-06 * T2 - 5.807150833333333e-10 * T3 +
                 1.253294000000000e-13 * T4;
    // species 6: n2
    species[6] = -1.020900000000000e+03 * invT - 6.516950000000001e-01 -
                 3.298677000000000e+00 * logT - 7.041200000000000e-04 * T +
                 6.605369999999999e-07 * T2 - 4.701262500000001e-10 * T3 +
                 1.222427500000000e-13 * T4;
    // species 7: ho2
    species[7] = +2.948080400000000e+02 * invT + 5.851355599999999e-01 -
                 4.301798010000000e+00 * logT + 2.374560255000000e-03 * T -
                 3.526381516666666e-06 * T2 + 2.023032450000000e-09 * T3 -
                 4.646125620000001e-13 * T4;
    // species 8: h2o2
    species[8] = -1.766315000000000e+04 * invT - 3.396609000000000e+00 -
                 3.388754000000000e+00 * logT - 3.284613000000000e-03 * T +
                 2.475021666666666e-08 * T2 + 3.854838333333333e-10 * T3 -
                 1.235757500000000e-13 * T4;
    // species 9: Xe
    species[9] = -7.453750000000000e+02 * invT - 3.664419930000000e+00 -
                 2.500000000000000e+00 * logT + 4.495706650000000e-14 * T -
                 4.203281000000000e-17 * T2 + 2.434888850000000e-20 * T3 -
                 5.947460900000000e-24 * T4;
  } else {
    // species 1: h2
    species[1] = -8.350340000000000e+02 * invT + 4.346533000000000e+00 -
                 2.991423000000000e+00 * logT - 3.500322000000000e-04 * T +
                 9.389715000000000e-09 * T2 + 7.692981666666667e-13 * T3 -
                 7.913760000000000e-17 * T4;
    // species 2: o
    species[2] = +2.923080000000000e+04 * invT - 2.378248000000000e+00 -
                 2.542060000000000e+00 * logT + 1.377531000000000e-05 * T +
                 5.171338333333333e-10 * T2 - 3.792555833333334e-13 * T3 +
                 2.184026000000000e-17 * T4;
    // species 3: o2
    species[3] = -1.233930000000000e+03 * invT + 5.084119999999999e-01 -
                 3.697578000000000e+00 * logT - 3.067598500000000e-04 * T +
                 2.098070000000000e-08 * T2 - 1.479400833333333e-12 * T3 +
                 5.682175000000001e-17 * T4;
    // species 5: h2o
    species[5] = -2.989921000000000e+04 * invT - 4.190671000000000e+00 -
                 2.672146000000000e+00 * logT - 1.528146500000000e-03 * T +
                 1.455043333333333e-07 * T2 - 1.000830000000000e-11 * T3 +
                 3.195809000000000e-16 * T4;
    // species 6: n2
    species[6] = -9.227977000000000e+02 * invT - 3.053888000000000e+00 -
                 2.926640000000000e+00 * logT - 7.439885000000000e-04 * T +
                 9.474601666666666e-08 * T2 - 8.414199999999999e-12 * T3 +
                 3.376675500000000e-16 * T4;
    // species 7: ho2
    species[7] = +1.118567130000000e+02 * invT + 2.321087500000001e-01 -
                 4.017210900000000e+00 * logT - 1.119910065000000e-03 * T +
                 1.056096916666667e-07 * T2 - 9.520530833333334e-12 * T3 +
                 5.395426750000000e-16 * T4;
    // species 8: h2o2
    species[8] = -1.800696000000000e+04 * invT + 4.072030000000000e+00 -
                 4.573167000000000e+00 * logT - 2.168068000000000e-03 * T +
                 2.457815000000000e-07 * T2 - 1.957420000000000e-11 * T3 +
                 7.158270000000000e-16 * T4;
    // species 9: Xe
    species[9] = -7.453941860000000e+02 * invT - 3.664075760000000e+00 -
                 2.500053220000000e+00 * logT + 5.256827200000000e-08 * T -
                 1.125544828333333e-11 * T2 + 1.424540908333333e-15 * T3 -
                 7.384052449999999e-20 * T4;
  }

  // species with midpoint at T=1710 kelvin
  if (T < 1710) {
    // species 4: oh
    species[4] = +3.452644480000000e+03 * invT + 8.746285399999998e-01 -
                 3.418962260000000e+00 * logT - 1.596279005000000e-04 * T +
                 5.138211950000001e-08 * T2 - 3.036729116666666e-11 * T3 +
                 5.009773950000000e-15 * T4;
  } else {
    // species 4: oh
    species[4] = +3.699497200000000e+03 * invT - 2.933807850000000e+00 -
                 2.853760400000000e+00 * logT - 5.149716700000000e-04 * T +
                 3.877774616666667e-08 * T2 - 1.614589200000000e-12 * T3 +
                 1.578799235000000e-17 * T4;
  }
}

// compute the a/(RT) at the given temperature
AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE void
helmholtz(amrex::Real* species, const amrex::Real T)
{
  const amrex::Real T2 = T * T;
  const amrex::Real T3 = T * T * T;
  const amrex::Real T4 = T * T * T * T;
  const amrex::Real invT = 1.0 / T;
  const amrex::Real logT = log(T);

  // species with no change across T
  // species 0: h
  species[0] = +2.54716300e+04 * invT + 1.96011760e+00 - 2.50000000e+00 * logT;

  // species with midpoint at T=1000 kelvin
  if (T < 1000) {
    // species 1: h2
    species[1] = -1.01252100e+03 * invT + 5.59221800e+00 -
                 3.29812400e+00 * logT - 4.12472100e-04 * T +
                 1.35716917e-07 * T2 + 7.89619500e-12 * T3 -
                 2.06743600e-14 * T4;
    // species 2: o
    species[2] = +2.91476400e+04 * invT - 1.01756600e+00 -
                 2.94642900e+00 * logT + 8.19083000e-04 * T -
                 4.03505333e-07 * T2 + 1.33570250e-10 * T3 -
                 1.94534800e-14 * T4;
    // species 3: o2
    species[3] = -1.00524900e+03 * invT - 3.82180200e+00 -
                 3.21293600e+00 * logT - 5.63743000e-04 * T +
                 9.59358333e-08 * T2 - 1.09489750e-10 * T3 +
                 4.38427700e-14 * T4;
    // species 5: h2o
    species[5] = -3.02081100e+04 * invT - 2.03391000e-01 -
                 3.38684200e+00 * logT - 1.73749100e-03 * T +
                 1.05911600e-06 * T2 - 5.80715083e-10 * T3 +
                 1.25329400e-13 * T4;
    // species 6: n2
    species[6] = -1.02090000e+03 * invT - 1.65169500e+00 -
                 3.29867700e+00 * logT - 7.04120000e-04 * T +
                 6.60537000e-07 * T2 - 4.70126250e-10 * T3 +
                 1.22242750e-13 * T4;
    // species 7: ho2
    species[7] = +2.94808040e+02 * invT - 4.14864440e-01 -
                 4.30179801e+00 * logT + 2.37456025e-03 * T -
                 3.52638152e-06 * T2 + 2.02303245e-09 * T3 -
                 4.64612562e-13 * T4;
    // species 8: h2o2
    species[8] = -1.76631500e+04 * invT - 4.39660900e+00 -
                 3.38875400e+00 * logT - 3.28461300e-03 * T +
                 2.47502167e-08 * T2 + 3.85483833e-10 * T3 -
                 1.23575750e-13 * T4;
    // species 9: Xe
    species[9] = -7.45375000e+02 * invT - 4.66441993e+00 -
                 2.50000000e+00 * logT + 4.49570665e-14 * T -
                 4.20328100e-17 * T2 + 2.43488885e-20 * T3 -
                 5.94746090e-24 * T4;
  } else {
    // species 1: h2
    species[1] = -8.35034000e+02 * invT + 3.34653300e+00 -
                 2.99142300e+00 * logT - 3.50032200e-04 * T +
                 9.38971500e-09 * T2 + 7.69298167e-13 * T3 -
                 7.91376000e-17 * T4;
    // species 2: o
    species[2] = +2.92308000e+04 * invT - 3.37824800e+00 -
                 2.54206000e+00 * logT + 1.37753100e-05 * T +
                 5.17133833e-10 * T2 - 3.79255583e-13 * T3 +
                 2.18402600e-17 * T4;
    // species 3: o2
    species[3] = -1.23393000e+03 * invT - 4.91588000e-01 -
                 3.69757800e+00 * logT - 3.06759850e-04 * T +
                 2.09807000e-08 * T2 - 1.47940083e-12 * T3 +
                 5.68217500e-17 * T4;
    // species 5: h2o
    species[5] = -2.98992100e+04 * invT - 5.19067100e+00 -
                 2.67214600e+00 * logT - 1.52814650e-03 * T +
                 1.45504333e-07 * T2 - 1.00083000e-11 * T3 +
                 3.19580900e-16 * T4;
    // species 6: n2
    species[6] = -9.22797700e+02 * invT - 4.05388800e+00 -
                 2.92664000e+00 * logT - 7.43988500e-04 * T +
                 9.47460167e-08 * T2 - 8.41420000e-12 * T3 +
                 3.37667550e-16 * T4;
    // species 7: ho2
    species[7] = +1.11856713e+02 * invT - 7.67891250e-01 -
                 4.01721090e+00 * logT - 1.11991006e-03 * T +
                 1.05609692e-07 * T2 - 9.52053083e-12 * T3 +
                 5.39542675e-16 * T4;
    // species 8: h2o2
    species[8] = -1.80069600e+04 * invT + 3.07203000e+00 -
                 4.57316700e+00 * logT - 2.16806800e-03 * T +
                 2.45781500e-07 * T2 - 1.95742000e-11 * T3 +
                 7.15827000e-16 * T4;
    // species 9: Xe
    species[9] = -7.45394186e+02 * invT - 4.66407576e+00 -
                 2.50005322e+00 * logT + 5.25682720e-08 * T -
                 1.12554483e-11 * T2 + 1.42454091e-15 * T3 -
                 7.38405245e-20 * T4;
  }

  // species with midpoint at T=1710 kelvin
  if (T < 1710) {
    // species 4: oh
    species[4] = +3.45264448e+03 * invT - 1.25371460e-01 -
                 3.41896226e+00 * logT - 1.59627901e-04 * T +
                 5.13821195e-08 * T2 - 3.03672912e-11 * T3 +
                 5.00977395e-15 * T4;
  } else {
    // species 4: oh
    species[4] = +3.69949720e+03 * invT - 3.93380785e+00 -
                 2.85376040e+00 * logT - 5.14971670e-04 * T +
                 3.87777462e-08 * T2 - 1.61458920e-12 * T3 +
                 1.57879924e-17 * T4;
  }
}

// compute the e/(RT) at the given temperature
AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE void
speciesInternalEnergy(amrex::Real* species, const amrex::Real T)
{
  const amrex::Real T2 = T * T;
  const amrex::Real T3 = T * T * T;
  const amrex::Real T4 = T * T * T * T;
  const amrex::Real invT = 1.0 / T;

  // species with no change across T
  // species 0: h
  species[0] = +1.50000000e+00 + 2.54716300e+04 * invT;

  // species with midpoint at T=1000 kelvin
  if (T < 1000) {
    // species 1: h2
    species[1] = +2.29812400e+00 + 4.12472100e-04 * T - 2.71433833e-07 * T2 -
                 2.36885850e-11 * T3 + 8.26974400e-14 * T4 -
                 1.01252100e+03 * invT;
    // species 2: o
    species[2] = +1.94642900e+00 - 8.19083000e-04 * T + 8.07010667e-07 * T2 -
                 4.00710750e-10 * T3 + 7.78139200e-14 * T4 +
                 2.91476400e+04 * invT;
    // species 3: o2
    species[3] = +2.21293600e+00 + 5.63743000e-04 * T - 1.91871667e-07 * T2 +
                 3.28469250e-10 * T3 - 1.75371080e-13 * T4 -
                 1.00524900e+03 * invT;
    // species 5: h2o
    species[5] = +2.38684200e+00 + 1.73749100e-03 * T - 2.11823200e-06 * T2 +
                 1.74214525e-09 * T3 - 5.01317600e-13 * T4 -
                 3.02081100e+04 * invT;
    // species 6: n2
    species[6] = +2.29867700e+00 + 7.04120000e-04 * T - 1.32107400e-06 * T2 +
                 1.41037875e-09 * T3 - 4.88971000e-13 * T4 -
                 1.02090000e+03 * invT;
    // species 7: ho2
    species[7] = +3.30179801e+00 - 2.37456025e-03 * T + 7.05276303e-06 * T2 -
                 6.06909735e-09 * T3 + 1.85845025e-12 * T4 +
                 2.94808040e+02 * invT;
    // species 8: h2o2
    species[8] = +2.38875400e+00 + 3.28461300e-03 * T - 4.95004333e-08 * T2 -
                 1.15645150e-09 * T3 + 4.94303000e-13 * T4 -
                 1.76631500e+04 * invT;
    // species 9: Xe
    species[9] = +1.50000000e+00 - 4.49570665e-14 * T + 8.40656200e-17 * T2 -
                 7.30466655e-20 * T3 + 2.37898436e-23 * T4 -
                 7.45375000e+02 * invT;
  } else {
    // species 1: h2
    species[1] = +1.99142300e+00 + 3.50032200e-04 * T - 1.87794300e-08 * T2 -
                 2.30789450e-12 * T3 + 3.16550400e-16 * T4 -
                 8.35034000e+02 * invT;
    // species 2: o
    species[2] = +1.54206000e+00 - 1.37753100e-05 * T - 1.03426767e-09 * T2 +
                 1.13776675e-12 * T3 - 8.73610400e-17 * T4 +
                 2.92308000e+04 * invT;
    // species 3: o2
    species[3] = +2.69757800e+00 + 3.06759850e-04 * T - 4.19614000e-08 * T2 +
                 4.43820250e-12 * T3 - 2.27287000e-16 * T4 -
                 1.23393000e+03 * invT;
    // species 5: h2o
    species[5] = +1.67214600e+00 + 1.52814650e-03 * T - 2.91008667e-07 * T2 +
                 3.00249000e-11 * T3 - 1.27832360e-15 * T4 -
                 2.98992100e+04 * invT;
    // species 6: n2
    species[6] = +1.92664000e+00 + 7.43988500e-04 * T - 1.89492033e-07 * T2 +
                 2.52426000e-11 * T3 - 1.35067020e-15 * T4 -
                 9.22797700e+02 * invT;
    // species 7: ho2
    species[7] = +3.01721090e+00 + 1.11991006e-03 * T - 2.11219383e-07 * T2 +
                 2.85615925e-11 * T3 - 2.15817070e-15 * T4 +
                 1.11856713e+02 * invT;
    // species 8: h2o2
    species[8] = +3.57316700e+00 + 2.16806800e-03 * T - 4.91563000e-07 * T2 +
                 5.87226000e-11 * T3 - 2.86330800e-15 * T4 -
                 1.80069600e+04 * invT;
    // species 9: Xe
    species[9] = +1.50005322e+00 - 5.25682720e-08 * T + 2.25108966e-11 * T2 -
                 4.27362272e-15 * T3 + 2.95362098e-19 * T4 -
                 7.45394186e+02 * invT;
  }

  // species with midpoint at T=1710 kelvin
  if (T < 1710) {
    // species 4: oh
    species[4] = +2.41896226e+00 + 1.59627901e-04 * T - 1.02764239e-07 * T2 +
                 9.11018735e-11 * T3 - 2.00390958e-14 * T4 +
                 3.45264448e+03 * invT;
  } else {
    // species 4: oh
    species[4] = +1.85376040e+00 + 5.14971670e-04 * T - 7.75554923e-08 * T2 +
                 4.84376760e-12 * T3 - 6.31519694e-17 * T4 +
                 3.69949720e+03 * invT;
  }
}

// compute the h/(RT) at the given temperature (Eq 20)
AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE void
speciesEnthalpy(amrex::Real* species, const amrex::Real T)
{
  const amrex::Real T2 = T * T;
  const amrex::Real T3 = T * T * T;
  const amrex::Real T4 = T * T * T * T;
  const amrex::Real invT = 1.0 / T;

  // species with no change across T
  // species 0: h
  species[0] = +2.50000000e+00 + 2.54716300e+04 * invT;

  // species with midpoint at T=1000 kelvin
  if (T < 1000) {
    // species 1: h2
    species[1] = +3.29812400e+00 + 4.12472100e-04 * T - 2.71433833e-07 * T2 -
                 2.36885850e-11 * T3 + 8.26974400e-14 * T4 -
                 1.01252100e+03 * invT;
    // species 2: o
    species[2] = +2.94642900e+00 - 8.19083000e-04 * T + 8.07010667e-07 * T2 -
                 4.00710750e-10 * T3 + 7.78139200e-14 * T4 +
                 2.91476400e+04 * invT;
    // species 3: o2
    species[3] = +3.21293600e+00 + 5.63743000e-04 * T - 1.91871667e-07 * T2 +
                 3.28469250e-10 * T3 - 1.75371080e-13 * T4 -
                 1.00524900e+03 * invT;
    // species 5: h2o
    species[5] = +3.38684200e+00 + 1.73749100e-03 * T - 2.11823200e-06 * T2 +
                 1.74214525e-09 * T3 - 5.01317600e-13 * T4 -
                 3.02081100e+04 * invT;
    // species 6: n2
    species[6] = +3.29867700e+00 + 7.04120000e-04 * T - 1.32107400e-06 * T2 +
                 1.41037875e-09 * T3 - 4.88971000e-13 * T4 -
                 1.02090000e+03 * invT;
    // species 7: ho2
    species[7] = +4.30179801e+00 - 2.37456025e-03 * T + 7.05276303e-06 * T2 -
                 6.06909735e-09 * T3 + 1.85845025e-12 * T4 +
                 2.94808040e+02 * invT;
    // species 8: h2o2
    species[8] = +3.38875400e+00 + 3.28461300e-03 * T - 4.95004333e-08 * T2 -
                 1.15645150e-09 * T3 + 4.94303000e-13 * T4 -
                 1.76631500e+04 * invT;
    // species 9: Xe
    species[9] = +2.50000000e+00 - 4.49570665e-14 * T + 8.40656200e-17 * T2 -
                 7.30466655e-20 * T3 + 2.37898436e-23 * T4 -
                 7.45375000e+02 * invT;
  } else {
    // species 1: h2
    species[1] = +2.99142300e+00 + 3.50032200e-04 * T - 1.87794300e-08 * T2 -
                 2.30789450e-12 * T3 + 3.16550400e-16 * T4 -
                 8.35034000e+02 * invT;
    // species 2: o
    species[2] = +2.54206000e+00 - 1.37753100e-05 * T - 1.03426767e-09 * T2 +
                 1.13776675e-12 * T3 - 8.73610400e-17 * T4 +
                 2.92308000e+04 * invT;
    // species 3: o2
    species[3] = +3.69757800e+00 + 3.06759850e-04 * T - 4.19614000e-08 * T2 +
                 4.43820250e-12 * T3 - 2.27287000e-16 * T4 -
                 1.23393000e+03 * invT;
    // species 5: h2o
    species[5] = +2.67214600e+00 + 1.52814650e-03 * T - 2.91008667e-07 * T2 +
                 3.00249000e-11 * T3 - 1.27832360e-15 * T4 -
                 2.98992100e+04 * invT;
    // species 6: n2
    species[6] = +2.92664000e+00 + 7.43988500e-04 * T - 1.89492033e-07 * T2 +
                 2.52426000e-11 * T3 - 1.35067020e-15 * T4 -
                 9.22797700e+02 * invT;
    // species 7: ho2
    species[7] = +4.01721090e+00 + 1.11991006e-03 * T - 2.11219383e-07 * T2 +
                 2.85615925e-11 * T3 - 2.15817070e-15 * T4 +
                 1.11856713e+02 * invT;
    // species 8: h2o2
    species[8] = +4.57316700e+00 + 2.16806800e-03 * T - 4.91563000e-07 * T2 +
                 5.87226000e-11 * T3 - 2.86330800e-15 * T4 -
                 1.80069600e+04 * invT;
    // species 9: Xe
    species[9] = +2.50005322e+00 - 5.25682720e-08 * T + 2.25108966e-11 * T2 -
                 4.27362272e-15 * T3 + 2.95362098e-19 * T4 -
                 7.45394186e+02 * invT;
  }

  // species with midpoint at T=1710 kelvin
  if (T < 1710) {
    // species 4: oh
    species[4] = +3.41896226e+00 + 1.59627901e-04 * T - 1.02764239e-07 * T2 +
                 9.11018735e-11 * T3 - 2.00390958e-14 * T4 +
                 3.45264448e+03 * invT;
  } else {
    // species 4: oh
    species[4] = +2.85376040e+00 + 5.14971670e-04 * T - 7.75554923e-08 * T2 +
                 4.84376760e-12 * T3 - 6.31519694e-17 * T4 +
                 3.69949720e+03 * invT;
  }
}

// compute the S/R at the given temperature (Eq 21)
AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE void
speciesEntropy(amrex::Real* species, const amrex::Real T)
{
  const amrex::Real T2 = T * T;
  const amrex::Real T3 = T * T * T;
  const amrex::Real T4 = T * T * T * T;
  const amrex::Real logT = log(T);

  // species with no change across T
  // species 0: h
  species[0] = +2.50000000e+00 * logT - 4.60117600e-01;

  // species with midpoint at T=1000 kelvin
  if (T < 1000) {
    // species 1: h2
    species[1] = +3.29812400e+00 * logT + 8.24944200e-04 * T -
                 4.07150750e-07 * T2 - 3.15847800e-11 * T3 +
                 1.03371800e-13 * T4 - 3.29409400e+00;
    // species 2: o
    species[2] = +2.94642900e+00 * logT - 1.63816600e-03 * T +
                 1.21051600e-06 * T2 - 5.34281000e-10 * T3 +
                 9.72674000e-14 * T4 + 2.96399500e+00;
    // species 3: o2
    species[3] = +3.21293600e+00 * logT + 1.12748600e-03 * T -
                 2.87807500e-07 * T2 + 4.37959000e-10 * T3 -
                 2.19213850e-13 * T4 + 6.03473800e+00;
    // species 5: h2o
    species[5] = +3.38684200e+00 * logT + 3.47498200e-03 * T -
                 3.17734800e-06 * T2 + 2.32286033e-09 * T3 -
                 6.26647000e-13 * T4 + 2.59023300e+00;
    // species 6: n2
    species[6] = +3.29867700e+00 * logT + 1.40824000e-03 * T -
                 1.98161100e-06 * T2 + 1.88050500e-09 * T3 -
                 6.11213750e-13 * T4 + 3.95037200e+00;
    // species 7: ho2
    species[7] = +4.30179801e+00 * logT - 4.74912051e-03 * T +
                 1.05791445e-05 * T2 - 8.09212980e-09 * T3 +
                 2.32306281e-12 * T4 + 3.71666245e+00;
    // species 8: h2o2
    species[8] = +3.38875400e+00 * logT + 6.56922600e-03 * T -
                 7.42506500e-08 * T2 - 1.54193533e-09 * T3 +
                 6.17878750e-13 * T4 + 6.78536300e+00;
    // species 9: Xe
    species[9] = +2.50000000e+00 * logT - 8.99141330e-14 * T +
                 1.26098430e-16 * T2 - 9.73955540e-20 * T3 +
                 2.97373045e-23 * T4 + 6.16441993e+00;
  } else {
    // species 1: h2
    species[1] = +2.99142300e+00 * logT + 7.00064400e-04 * T -
                 2.81691450e-08 * T2 - 3.07719267e-12 * T3 +
                 3.95688000e-16 * T4 - 1.35511000e+00;
    // species 2: o
    species[2] = +2.54206000e+00 * logT - 2.75506200e-05 * T -
                 1.55140150e-09 * T2 + 1.51702233e-12 * T3 -
                 1.09201300e-16 * T4 + 4.92030800e+00;
    // species 3: o2
    species[3] = +3.69757800e+00 * logT + 6.13519700e-04 * T -
                 6.29421000e-08 * T2 + 5.91760333e-12 * T3 -
                 2.84108750e-16 * T4 + 3.18916600e+00;
    // species 5: h2o
    species[5] = +2.67214600e+00 * logT + 3.05629300e-03 * T -
                 4.36513000e-07 * T2 + 4.00332000e-11 * T3 -
                 1.59790450e-15 * T4 + 6.86281700e+00;
    // species 6: n2
    species[6] = +2.92664000e+00 * logT + 1.48797700e-03 * T -
                 2.84238050e-07 * T2 + 3.36568000e-11 * T3 -
                 1.68833775e-15 * T4 + 5.98052800e+00;
    // species 7: ho2
    species[7] = +4.01721090e+00 * logT + 2.23982013e-03 * T -
                 3.16829075e-07 * T2 + 3.80821233e-11 * T3 -
                 2.69771337e-15 * T4 + 3.78510215e+00;
    // species 8: h2o2
    species[8] = +4.57316700e+00 * logT + 4.33613600e-03 * T -
                 7.37344500e-07 * T2 + 7.82968000e-11 * T3 -
                 3.57913500e-15 * T4 + 5.01137000e-01;
    // species 9: Xe
    species[9] = +2.50005322e+00 * logT - 1.05136544e-07 * T +
                 3.37663448e-11 * T2 - 5.69816363e-15 * T3 +
                 3.69202622e-19 * T4 + 6.16412898e+00;
  }

  // species with midpoint at T=1710 kelvin
  if (T < 1710) {
    // species 4: oh
    species[4] = +3.41896226e+00 * logT + 3.19255801e-04 * T -
                 1.54146359e-07 * T2 + 1.21469165e-10 * T3 -
                 2.50488698e-14 * T4 + 2.54433372e+00;
  } else {
    // species 4: oh
    species[4] = +2.85376040e+00 * logT + 1.02994334e-03 * T -
                 1.16333239e-07 * T2 + 6.45835680e-12 * T3 -
                 7.89399617e-17 * T4 + 5.78756825e+00;
  }
}

// compute d(Cp/R)/dT and d(Cv/R)/dT at the given temperature
AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE void
dcvpRdT(amrex::Real* species, const amrex::Real T)
{
  const amrex::Real T2 = T * T;
  const amrex::Real T3 = T * T * T;

  // species with no change across T
  // species 0: h
  species[0] = 0.0;

  // species with midpoint at T=1000 kelvin
  if (T < 1000) {
    // species 1: h2
    species[1] = +8.24944200e-04 - 1.62860300e-06 * T - 2.84263020e-10 * T2 +
                 1.65394880e-12 * T3;
    // species 2: o
    species[2] = -1.63816600e-03 + 4.84206400e-06 * T - 4.80852900e-09 * T2 +
                 1.55627840e-12 * T3;
    // species 3: o2
    species[3] = +1.12748600e-03 - 1.15123000e-06 * T + 3.94163100e-09 * T2 -
                 3.50742160e-12 * T3;
    // species 5: h2o
    species[5] = +3.47498200e-03 - 1.27093920e-05 * T + 2.09057430e-08 * T2 -
                 1.00263520e-11 * T3;
    // species 6: n2
    species[6] = +1.40824000e-03 - 7.92644400e-06 * T + 1.69245450e-08 * T2 -
                 9.77942000e-12 * T3;
    // species 7: ho2
    species[7] = -4.74912051e-03 + 4.23165782e-05 * T - 7.28291682e-08 * T2 +
                 3.71690050e-11 * T3;
    // species 8: h2o2
    species[8] = +6.56922600e-03 - 2.97002600e-07 * T - 1.38774180e-08 * T2 +
                 9.88606000e-12 * T3;
    // species 9: Xe
    species[9] = -8.99141330e-14 + 5.04393720e-16 * T - 8.76559986e-19 * T2 +
                 4.75796872e-22 * T3;
  } else {
    // species 1: h2
    species[1] = +7.00064400e-04 - 1.12676580e-07 * T - 2.76947340e-11 * T2 +
                 6.33100800e-15 * T3;
    // species 2: o
    species[2] = -2.75506200e-05 - 6.20560600e-09 * T + 1.36532010e-11 * T2 -
                 1.74722080e-15 * T3;
    // species 3: o2
    species[3] = +6.13519700e-04 - 2.51768400e-07 * T + 5.32584300e-11 * T2 -
                 4.54574000e-15 * T3;
    // species 5: h2o
    species[5] = +3.05629300e-03 - 1.74605200e-06 * T + 3.60298800e-10 * T2 -
                 2.55664720e-14 * T3;
    // species 6: n2
    species[6] = +1.48797700e-03 - 1.13695220e-06 * T + 3.02911200e-10 * T2 -
                 2.70134040e-14 * T3;
    // species 7: ho2
    species[7] = +2.23982013e-03 - 1.26731630e-06 * T + 3.42739110e-10 * T2 -
                 4.31634140e-14 * T3;
    // species 8: h2o2
    species[8] = +4.33613600e-03 - 2.94937800e-06 * T + 7.04671200e-10 * T2 -
                 5.72661600e-14 * T3;
    // species 9: Xe
    species[9] = -1.05136544e-07 + 1.35065379e-10 * T - 5.12834727e-14 * T2 +
                 5.90724196e-18 * T3;
  }

  // species with midpoint at T=1710 kelvin
  if (T < 1710) {
    // species 4: oh
    species[4] = +3.19255801e-04 - 6.16585434e-07 * T + 1.09322248e-09 * T2 -
                 4.00781916e-13 * T3;
  } else {
    // species 4: oh
    species[4] = +1.02994334e-03 - 4.65332954e-07 * T + 5.81252112e-11 * T2 -
                 1.26303939e-15 * T3;
  }
}

// Returns the mean specific heat at CP (Eq. 33)
AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE void
CKCPBL(const amrex::Real T, const amrex::Real x[], amrex::Real& cpbl)
{
  amrex::Real result = 0;
  amrex::Real cpor[10]; //  temporary storage
  cp_R(cpor, T);

  // perform dot product
  for (int id = 0; id < 10; ++id) {
    result += x[id] * cpor[id];
  }

  cpbl = result * 8.31446261815324e+07;
}

// Returns the mean specific heat at CP (Eq. 34)
AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE void
CKCPBS(const amrex::Real T, const amrex::Real y[], amrex::Real& cpbs)
{
  amrex::Real result = 0.0;

  // compute Cp/R at the given temperature
  const amrex::Real T2 = T * T;
  const amrex::Real T3 = T * T * T;
  const amrex::Real T4 = T * T * T * T;

  // species with no change across T
  // species 0: h
  result += y[0] * (+2.50000000e+00) * 0.9920634920634921;

  // species with midpoint at T=1000 kelvin
  if (T < 1000) {
    // species 1: h2
    result += y[1] *
              (+3.29812400e+00 + 8.24944200e-04 * T - 8.14301500e-07 * T2 -
               9.47543400e-11 * T3 + 4.13487200e-13 * T4) *
              0.4960317460317460;
    // species 2: o
    result += y[2] *
              (+2.94642900e+00 - 1.63816600e-03 * T + 2.42103200e-06 * T2 -
               1.60284300e-09 * T3 + 3.89069600e-13 * T4) *
              0.0625039064941559;
    // species 3: o2
    result += y[3] *
              (+3.21293600e+00 + 1.12748600e-03 * T - 5.75615000e-07 * T2 +
               1.31387700e-09 * T3 - 8.76855400e-13 * T4) *
              0.0312519532470779;
    // species 5: h2o
    result += y[5] *
              (+3.38684200e+00 + 3.47498200e-03 * T - 6.35469600e-06 * T2 +
               6.96858100e-09 * T3 - 2.50658800e-12 * T4) *
              0.0555092978073827;
    // species 6: n2
    result += y[6] *
              (+3.29867700e+00 + 1.40824000e-03 * T - 3.96322200e-06 * T2 +
               5.64151500e-09 * T3 - 2.44485500e-12 * T4) *
              0.0356964374955379;
    // species 7: ho2
    result += y[7] *
              (+4.30179801e+00 - 4.74912051e-03 * T + 2.11582891e-05 * T2 -
               2.42763894e-08 * T3 + 9.29225124e-12 * T4) *
              0.0302975216627280;
    // species 8: h2o2
    result += y[8] *
              (+3.38875400e+00 + 6.56922600e-03 * T - 1.48501300e-07 * T2 -
               4.62580600e-09 * T3 + 2.47151500e-12 * T4) *
              0.0293996589639560;
    // species 9: Xe
    result += y[9] *
              (+2.50000000e+00 - 8.99141330e-14 * T + 2.52196860e-16 * T2 -
               2.92186662e-19 * T3 + 1.18949218e-22 * T4) *
              0.0076165522914398;
  } else {
    // species 1: h2
    result += y[1] *
              (+2.99142300e+00 + 7.00064400e-04 * T - 5.63382900e-08 * T2 -
               9.23157800e-12 * T3 + 1.58275200e-15 * T4) *
              0.4960317460317460;
    // species 2: o
    result += y[2] *
              (+2.54206000e+00 - 2.75506200e-05 * T - 3.10280300e-09 * T2 +
               4.55106700e-12 * T3 - 4.36805200e-16 * T4) *
              0.0625039064941559;
    // species 3: o2
    result += y[3] *
              (+3.69757800e+00 + 6.13519700e-04 * T - 1.25884200e-07 * T2 +
               1.77528100e-11 * T3 - 1.13643500e-15 * T4) *
              0.0312519532470779;
    // species 5: h2o
    result += y[5] *
              (+2.67214600e+00 + 3.05629300e-03 * T - 8.73026000e-07 * T2 +
               1.20099600e-10 * T3 - 6.39161800e-15 * T4) *
              0.0555092978073827;
    // species 6: n2
    result += y[6] *
              (+2.92664000e+00 + 1.48797700e-03 * T - 5.68476100e-07 * T2 +
               1.00970400e-10 * T3 - 6.75335100e-15 * T4) *
              0.0356964374955379;
    // species 7: ho2
    result += y[7] *
              (+4.01721090e+00 + 2.23982013e-03 * T - 6.33658150e-07 * T2 +
               1.14246370e-10 * T3 - 1.07908535e-14 * T4) *
              0.0302975216627280;
    // species 8: h2o2
    result += y[8] *
              (+4.57316700e+00 + 4.33613600e-03 * T - 1.47468900e-06 * T2 +
               2.34890400e-10 * T3 - 1.43165400e-14 * T4) *
              0.0293996589639560;
    // species 9: Xe
    result += y[9] *
              (+2.50005322e+00 - 1.05136544e-07 * T + 6.75326897e-11 * T2 -
               1.70944909e-14 * T3 + 1.47681049e-18 * T4) *
              0.0076165522914398;
  }

  // species with midpoint at T=1710 kelvin
  if (T < 1710) {
    // species 4: oh
    result += y[4] *
              (+3.41896226e+00 + 3.19255801e-04 * T - 3.08292717e-07 * T2 +
               3.64407494e-10 * T3 - 1.00195479e-13 * T4) *
              0.0587993179279120;
  } else {
    // species 4: oh
    result += y[4] *
              (+2.85376040e+00 + 1.02994334e-03 * T - 2.32666477e-07 * T2 +
               1.93750704e-11 * T3 - 3.15759847e-16 * T4) *
              0.0587993179279120;
  }

  cpbs = result * 8.31446261815324e+07;
}

// Returns the mean specific heat at CV (Eq. 35)
AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE void
CKCVBL(const amrex::Real T, const amrex::Real x[], amrex::Real& cvbl)
{
  amrex::Real result = 0;
  amrex::Real cvor[10]; //  temporary storage
  cv_R(cvor, T);

  // perform dot product
  for (int id = 0; id < 10; ++id) {
    result += x[id] * cvor[id];
  }

  cvbl = result * 8.31446261815324e+07;
}

// Returns the mean specific heat at CV (Eq. 36)
AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE void
CKCVBS(const amrex::Real T, const amrex::Real y[], amrex::Real& cvbs)
{
  amrex::Real result = 0.0;
  // compute Cv/R at the given temperature
  const amrex::Real T2 = T * T;
  const amrex::Real T3 = T * T * T;
  const amrex::Real T4 = T * T * T * T;

  // species with no change across T
  // species 0: h
  result += y[0] * (+1.50000000e+00) * 0.9920634920634921;

  // species with midpoint at T=1000 kelvin
  if (T < 1000) {
    // species 1: h2
    result += y[1] *
              (+2.29812400e+00 + 8.24944200e-04 * T - 8.14301500e-07 * T2 -
               9.47543400e-11 * T3 + 4.13487200e-13 * T4) *
              0.4960317460317460;
    // species 2: o
    result += y[2] *
              (+1.94642900e+00 - 1.63816600e-03 * T + 2.42103200e-06 * T2 -
               1.60284300e-09 * T3 + 3.89069600e-13 * T4) *
              0.0625039064941559;
    // species 3: o2
    result += y[3] *
              (+2.21293600e+00 + 1.12748600e-03 * T - 5.75615000e-07 * T2 +
               1.31387700e-09 * T3 - 8.76855400e-13 * T4) *
              0.0312519532470779;
    // species 5: h2o
    result += y[5] *
              (+2.38684200e+00 + 3.47498200e-03 * T - 6.35469600e-06 * T2 +
               6.96858100e-09 * T3 - 2.50658800e-12 * T4) *
              0.0555092978073827;
    // species 6: n2
    result += y[6] *
              (+2.29867700e+00 + 1.40824000e-03 * T - 3.96322200e-06 * T2 +
               5.64151500e-09 * T3 - 2.44485500e-12 * T4) *
              0.0356964374955379;
    // species 7: ho2
    result += y[7] *
              (+3.30179801e+00 - 4.74912051e-03 * T + 2.11582891e-05 * T2 -
               2.42763894e-08 * T3 + 9.29225124e-12 * T4) *
              0.0302975216627280;
    // species 8: h2o2
    result += y[8] *
              (+2.38875400e+00 + 6.56922600e-03 * T - 1.48501300e-07 * T2 -
               4.62580600e-09 * T3 + 2.47151500e-12 * T4) *
              0.0293996589639560;
    // species 9: Xe
    result += y[9] *
              (+1.50000000e+00 - 8.99141330e-14 * T + 2.52196860e-16 * T2 -
               2.92186662e-19 * T3 + 1.18949218e-22 * T4) *
              0.0076165522914398;
  } else {
    // species 1: h2
    result += y[1] *
              (+1.99142300e+00 + 7.00064400e-04 * T - 5.63382900e-08 * T2 -
               9.23157800e-12 * T3 + 1.58275200e-15 * T4) *
              0.4960317460317460;
    // species 2: o
    result += y[2] *
              (+1.54206000e+00 - 2.75506200e-05 * T - 3.10280300e-09 * T2 +
               4.55106700e-12 * T3 - 4.36805200e-16 * T4) *
              0.0625039064941559;
    // species 3: o2
    result += y[3] *
              (+2.69757800e+00 + 6.13519700e-04 * T - 1.25884200e-07 * T2 +
               1.77528100e-11 * T3 - 1.13643500e-15 * T4) *
              0.0312519532470779;
    // species 5: h2o
    result += y[5] *
              (+1.67214600e+00 + 3.05629300e-03 * T - 8.73026000e-07 * T2 +
               1.20099600e-10 * T3 - 6.39161800e-15 * T4) *
              0.0555092978073827;
    // species 6: n2
    result += y[6] *
              (+1.92664000e+00 + 1.48797700e-03 * T - 5.68476100e-07 * T2 +
               1.00970400e-10 * T3 - 6.75335100e-15 * T4) *
              0.0356964374955379;
    // species 7: ho2
    result += y[7] *
              (+3.01721090e+00 + 2.23982013e-03 * T - 6.33658150e-07 * T2 +
               1.14246370e-10 * T3 - 1.07908535e-14 * T4) *
              0.0302975216627280;
    // species 8: h2o2
    result += y[8] *
              (+3.57316700e+00 + 4.33613600e-03 * T - 1.47468900e-06 * T2 +
               2.34890400e-10 * T3 - 1.43165400e-14 * T4) *
              0.0293996589639560;
    // species 9: Xe
    result += y[9] *
              (+1.50005322e+00 - 1.05136544e-07 * T + 6.75326897e-11 * T2 -
               1.70944909e-14 * T3 + 1.47681049e-18 * T4) *
              0.0076165522914398;
  }

  // species with midpoint at T=1710 kelvin
  if (T < 1710) {
    // species 4: oh
    result += y[4] *
              (+2.41896226e+00 + 3.19255801e-04 * T - 3.08292717e-07 * T2 +
               3.64407494e-10 * T3 - 1.00195479e-13 * T4) *
              0.0587993179279120;
  } else {
    // species 4: oh
    result += y[4] *
              (+1.85376040e+00 + 1.02994334e-03 * T - 2.32666477e-07 * T2 +
               1.93750704e-11 * T3 - 3.15759847e-16 * T4) *
              0.0587993179279120;
  }

  cvbs = result * 8.31446261815324e+07;
}

// Returns the mean enthalpy of the mixture in molar units
AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE void
CKHBML(const amrex::Real T, const amrex::Real x[], amrex::Real& hbml)
{
  amrex::Real result = 0;
  amrex::Real hml[10];                       //  temporary storage
  amrex::Real RT = 8.31446261815324e+07 * T; // R*T
  speciesEnthalpy(hml, T);

  // perform dot product
  for (int id = 0; id < 10; ++id) {
    result += x[id] * hml[id];
  }

  hbml = result * RT;
}

// Returns mean enthalpy of mixture in mass units
AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE void
CKHBMS(const amrex::Real T, const amrex::Real y[], amrex::Real& hbms)
{
  amrex::Real result = 0.0;
  const amrex::Real T2 = T * T;
  const amrex::Real T3 = T * T * T;
  const amrex::Real T4 = T * T * T * T;
  const amrex::Real invT = 1.0 / T;

  // species with no change across T
  // species 0: h
  result +=
    y[0] * (+2.50000000e+00 + 2.54716300e+04 * invT) * 0.9920634920634921;

  // species with midpoint at T=1000 kelvin
  if (T < 1000) {
    // species 1: h2
    result +=
      y[1] *
      (+3.29812400e+00 + 4.12472100e-04 * T - 2.71433833e-07 * T2 -
       2.36885850e-11 * T3 + 8.26974400e-14 * T4 - 1.01252100e+03 * invT) *
      0.4960317460317460;
    // species 2: o
    result +=
      y[2] *
      (+2.94642900e+00 - 8.19083000e-04 * T + 8.07010667e-07 * T2 -
       4.00710750e-10 * T3 + 7.78139200e-14 * T4 + 2.91476400e+04 * invT) *
      0.0625039064941559;
    // species 3: o2
    result +=
      y[3] *
      (+3.21293600e+00 + 5.63743000e-04 * T - 1.91871667e-07 * T2 +
       3.28469250e-10 * T3 - 1.75371080e-13 * T4 - 1.00524900e+03 * invT) *
      0.0312519532470779;
    // species 5: h2o
    result +=
      y[5] *
      (+3.38684200e+00 + 1.73749100e-03 * T - 2.11823200e-06 * T2 +
       1.74214525e-09 * T3 - 5.01317600e-13 * T4 - 3.02081100e+04 * invT) *
      0.0555092978073827;
    // species 6: n2
    result +=
      y[6] *
      (+3.29867700e+00 + 7.04120000e-04 * T - 1.32107400e-06 * T2 +
       1.41037875e-09 * T3 - 4.88971000e-13 * T4 - 1.02090000e+03 * invT) *
      0.0356964374955379;
    // species 7: ho2
    result +=
      y[7] *
      (+4.30179801e+00 - 2.37456025e-03 * T + 7.05276303e-06 * T2 -
       6.06909735e-09 * T3 + 1.85845025e-12 * T4 + 2.94808040e+02 * invT) *
      0.0302975216627280;
    // species 8: h2o2
    result +=
      y[8] *
      (+3.38875400e+00 + 3.28461300e-03 * T - 4.95004333e-08 * T2 -
       1.15645150e-09 * T3 + 4.94303000e-13 * T4 - 1.76631500e+04 * invT) *
      0.0293996589639560;
    // species 9: Xe
    result +=
      y[9] *
      (+2.50000000e+00 - 4.49570665e-14 * T + 8.40656200e-17 * T2 -
       7.30466655e-20 * T3 + 2.37898436e-23 * T4 - 7.45375000e+02 * invT) *
      0.0076165522914398;
  } else {
    // species 1: h2
    result +=
      y[1] *
      (+2.99142300e+00 + 3.50032200e-04 * T - 1.87794300e-08 * T2 -
       2.30789450e-12 * T3 + 3.16550400e-16 * T4 - 8.35034000e+02 * invT) *
      0.4960317460317460;
    // species 2: o
    result +=
      y[2] *
      (+2.54206000e+00 - 1.37753100e-05 * T - 1.03426767e-09 * T2 +
       1.13776675e-12 * T3 - 8.73610400e-17 * T4 + 2.92308000e+04 * invT) *
      0.0625039064941559;
    // species 3: o2
    result +=
      y[3] *
      (+3.69757800e+00 + 3.06759850e-04 * T - 4.19614000e-08 * T2 +
       4.43820250e-12 * T3 - 2.27287000e-16 * T4 - 1.23393000e+03 * invT) *
      0.0312519532470779;
    // species 5: h2o
    result +=
      y[5] *
      (+2.67214600e+00 + 1.52814650e-03 * T - 2.91008667e-07 * T2 +
       3.00249000e-11 * T3 - 1.27832360e-15 * T4 - 2.98992100e+04 * invT) *
      0.0555092978073827;
    // species 6: n2
    result +=
      y[6] *
      (+2.92664000e+00 + 7.43988500e-04 * T - 1.89492033e-07 * T2 +
       2.52426000e-11 * T3 - 1.35067020e-15 * T4 - 9.22797700e+02 * invT) *
      0.0356964374955379;
    // species 7: ho2
    result +=
      y[7] *
      (+4.01721090e+00 + 1.11991006e-03 * T - 2.11219383e-07 * T2 +
       2.85615925e-11 * T3 - 2.15817070e-15 * T4 + 1.11856713e+02 * invT) *
      0.0302975216627280;
    // species 8: h2o2
    result +=
      y[8] *
      (+4.57316700e+00 + 2.16806800e-03 * T - 4.91563000e-07 * T2 +
       5.87226000e-11 * T3 - 2.86330800e-15 * T4 - 1.80069600e+04 * invT) *
      0.0293996589639560;
    // species 9: Xe
    result +=
      y[9] *
      (+2.50005322e+00 - 5.25682720e-08 * T + 2.25108966e-11 * T2 -
       4.27362272e-15 * T3 + 2.95362098e-19 * T4 - 7.45394186e+02 * invT) *
      0.0076165522914398;
  }

  // species with midpoint at T=1710 kelvin
  if (T < 1710) {
    // species 4: oh
    result +=
      y[4] *
      (+3.41896226e+00 + 1.59627901e-04 * T - 1.02764239e-07 * T2 +
       9.11018735e-11 * T3 - 2.00390958e-14 * T4 + 3.45264448e+03 * invT) *
      0.0587993179279120;
  } else {
    // species 4: oh
    result +=
      y[4] *
      (+2.85376040e+00 + 5.14971670e-04 * T - 7.75554923e-08 * T2 +
       4.84376760e-12 * T3 - 6.31519694e-17 * T4 + 3.69949720e+03 * invT) *
      0.0587993179279120;
  }

  const amrex::Real RT = 8.31446261815324e+07 * T; // R*T

  hbms = result * RT;
}

// get mean internal energy in molar units
AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE void
CKUBML(const amrex::Real T, const amrex::Real x[], amrex::Real& ubml)
{
  amrex::Real result = 0;
  amrex::Real uml[10];                       //  temporary energy array
  amrex::Real RT = 8.31446261815324e+07 * T; // R*T
  speciesInternalEnergy(uml, T);

  // perform dot product
  for (int id = 0; id < 10; ++id) {
    result += x[id] * uml[id];
  }

  ubml = result * RT;
}

// get mean internal energy in mass units
AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE void
CKUBMS(const amrex::Real T, const amrex::Real y[], amrex::Real& ubms)
{
  amrex::Real result = 0.0;
  const amrex::Real T2 = T * T;
  const amrex::Real T3 = T * T * T;
  const amrex::Real T4 = T * T * T * T;
  const amrex::Real invT = 1.0 / T;

  // species with no change across T
  // species 0: h
  result +=
    y[0] * (+1.50000000e+00 + 2.54716300e+04 * invT) * 0.9920634920634921;

  // species with midpoint at T=1000 kelvin
  if (T < 1000) {
    // species 1: h2
    result +=
      y[1] *
      (+2.29812400e+00 + 4.12472100e-04 * T - 2.71433833e-07 * T2 -
       2.36885850e-11 * T3 + 8.26974400e-14 * T4 - 1.01252100e+03 * invT) *
      0.4960317460317460;
    // species 2: o
    result +=
      y[2] *
      (+1.94642900e+00 - 8.19083000e-04 * T + 8.07010667e-07 * T2 -
       4.00710750e-10 * T3 + 7.78139200e-14 * T4 + 2.91476400e+04 * invT) *
      0.0625039064941559;
    // species 3: o2
    result +=
      y[3] *
      (+2.21293600e+00 + 5.63743000e-04 * T - 1.91871667e-07 * T2 +
       3.28469250e-10 * T3 - 1.75371080e-13 * T4 - 1.00524900e+03 * invT) *
      0.0312519532470779;
    // species 5: h2o
    result +=
      y[5] *
      (+2.38684200e+00 + 1.73749100e-03 * T - 2.11823200e-06 * T2 +
       1.74214525e-09 * T3 - 5.01317600e-13 * T4 - 3.02081100e+04 * invT) *
      0.0555092978073827;
    // species 6: n2
    result +=
      y[6] *
      (+2.29867700e+00 + 7.04120000e-04 * T - 1.32107400e-06 * T2 +
       1.41037875e-09 * T3 - 4.88971000e-13 * T4 - 1.02090000e+03 * invT) *
      0.0356964374955379;
    // species 7: ho2
    result +=
      y[7] *
      (+3.30179801e+00 - 2.37456025e-03 * T + 7.05276303e-06 * T2 -
       6.06909735e-09 * T3 + 1.85845025e-12 * T4 + 2.94808040e+02 * invT) *
      0.0302975216627280;
    // species 8: h2o2
    result +=
      y[8] *
      (+2.38875400e+00 + 3.28461300e-03 * T - 4.95004333e-08 * T2 -
       1.15645150e-09 * T3 + 4.94303000e-13 * T4 - 1.76631500e+04 * invT) *
      0.0293996589639560;
    // species 9: Xe
    result +=
      y[9] *
      (+1.50000000e+00 - 4.49570665e-14 * T + 8.40656200e-17 * T2 -
       7.30466655e-20 * T3 + 2.37898436e-23 * T4 - 7.45375000e+02 * invT) *
      0.0076165522914398;
  } else {
    // species 1: h2
    result +=
      y[1] *
      (+1.99142300e+00 + 3.50032200e-04 * T - 1.87794300e-08 * T2 -
       2.30789450e-12 * T3 + 3.16550400e-16 * T4 - 8.35034000e+02 * invT) *
      0.4960317460317460;
    // species 2: o
    result +=
      y[2] *
      (+1.54206000e+00 - 1.37753100e-05 * T - 1.03426767e-09 * T2 +
       1.13776675e-12 * T3 - 8.73610400e-17 * T4 + 2.92308000e+04 * invT) *
      0.0625039064941559;
    // species 3: o2
    result +=
      y[3] *
      (+2.69757800e+00 + 3.06759850e-04 * T - 4.19614000e-08 * T2 +
       4.43820250e-12 * T3 - 2.27287000e-16 * T4 - 1.23393000e+03 * invT) *
      0.0312519532470779;
    // species 5: h2o
    result +=
      y[5] *
      (+1.67214600e+00 + 1.52814650e-03 * T - 2.91008667e-07 * T2 +
       3.00249000e-11 * T3 - 1.27832360e-15 * T4 - 2.98992100e+04 * invT) *
      0.0555092978073827;
    // species 6: n2
    result +=
      y[6] *
      (+1.92664000e+00 + 7.43988500e-04 * T - 1.89492033e-07 * T2 +
       2.52426000e-11 * T3 - 1.35067020e-15 * T4 - 9.22797700e+02 * invT) *
      0.0356964374955379;
    // species 7: ho2
    result +=
      y[7] *
      (+3.01721090e+00 + 1.11991006e-03 * T - 2.11219383e-07 * T2 +
       2.85615925e-11 * T3 - 2.15817070e-15 * T4 + 1.11856713e+02 * invT) *
      0.0302975216627280;
    // species 8: h2o2
    result +=
      y[8] *
      (+3.57316700e+00 + 2.16806800e-03 * T - 4.91563000e-07 * T2 +
       5.87226000e-11 * T3 - 2.86330800e-15 * T4 - 1.80069600e+04 * invT) *
      0.0293996589639560;
    // species 9: Xe
    result +=
      y[9] *
      (+1.50005322e+00 - 5.25682720e-08 * T + 2.25108966e-11 * T2 -
       4.27362272e-15 * T3 + 2.95362098e-19 * T4 - 7.45394186e+02 * invT) *
      0.0076165522914398;
  }

  // species with midpoint at T=1710 kelvin
  if (T < 1710) {
    // species 4: oh
    result +=
      y[4] *
      (+2.41896226e+00 + 1.59627901e-04 * T - 1.02764239e-07 * T2 +
       9.11018735e-11 * T3 - 2.00390958e-14 * T4 + 3.45264448e+03 * invT) *
      0.0587993179279120;
  } else {
    // species 4: oh
    result +=
      y[4] *
      (+1.85376040e+00 + 5.14971670e-04 * T - 7.75554923e-08 * T2 +
       4.84376760e-12 * T3 - 6.31519694e-17 * T4 + 3.69949720e+03 * invT) *
      0.0587993179279120;
  }

  const amrex::Real RT = 8.31446261815324e+07 * T; // R*T

  ubms = result * RT;
}

// get mixture entropy in molar units
AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE void
CKSBML(
  const amrex::Real P,
  const amrex::Real T,
  const amrex::Real x[],
  amrex::Real& sbml)
{
  amrex::Real result = 0;
  // Log of normalized pressure in cgs units dynes/cm^2 by Patm
  amrex::Real logPratio = log(P / 1013250.0);
  amrex::Real sor[10]; //  temporary storage
  speciesEntropy(sor, T);

  // Compute Eq 42
  for (int id = 0; id < 10; ++id) {
    result += x[id] * (sor[id] - log((x[id] + 1e-100)) - logPratio);
  }

  sbml = result * 8.31446261815324e+07;
}

// get mixture entropy in mass units
AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE void
CKSBMS(
  const amrex::Real P,
  const amrex::Real T,
  const amrex::Real y[],
  amrex::Real& sbms)
{
  amrex::Real result = 0;
  // Log of normalized pressure in cgs units dynes/cm^2 by Patm
  amrex::Real logPratio = log(P / 1013250.0);
  amrex::Real sor[10]; //  temporary storage
  amrex::Real x[10];   //  need a ytx conversion
  amrex::Real YOW = 0; // See Eq 4, 6 in CK Manual

  // Compute inverse of mean molecular wt first
  for (int i = 0; i < 10; i++) {
    YOW += y[i] * imw(i);
  }
  // Now compute y to x conversion
  x[0] = y[0] / (1.008000 * YOW);
  x[1] = y[1] / (2.016000 * YOW);
  x[2] = y[2] / (15.999000 * YOW);
  x[3] = y[3] / (31.998000 * YOW);
  x[4] = y[4] / (17.007000 * YOW);
  x[5] = y[5] / (18.015000 * YOW);
  x[6] = y[6] / (28.014000 * YOW);
  x[7] = y[7] / (33.006000 * YOW);
  x[8] = y[8] / (34.014000 * YOW);
  x[9] = y[9] / (131.293000 * YOW);
  speciesEntropy(sor, T);
  // Perform computation in Eq 42 and 43
  for (int i = 0; i < 10; i++) {
    result += x[i] * (sor[i] - log((x[i] + 1e-100)) - logPratio);
  }
  // Scale by R/W
  sbms = result * 8.31446261815324e+07 * YOW;
}

//  get temperature given internal energy in mass units and mass fracs
AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE void
GET_T_GIVEN_EY(
  const amrex::Real e, const amrex::Real y[], amrex::Real& t, int& ierr)
{
#ifdef CONVERGENCE
  const int maxiter = 5000;
  const amrex::Real tol = 1.e-12;
#else
  const int maxiter = 200;
  const amrex::Real tol = 1.e-6;
#endif
  amrex::Real tmin = 90;   // max lower bound for thermo def
  amrex::Real tmax = 4000; // min upper bound for thermo def
  amrex::Real e1, emin, emax, cv, t1, dt;
  CKUBMS(tmin, y, emin);
  CKUBMS(tmax, y, emax);
  if (e < emin) {
    // Linear Extrapolation below tmin
    CKCVBS(tmin, y, cv);
    t = tmin - (emin - e) / cv;
    ierr = 1;
    return;
  }
  if (e > emax) {
    // Linear Extrapolation above tmax
    CKCVBS(tmax, y, cv);
    t = tmax - (emax - e) / cv;
    ierr = 1;
    return;
  }
  t1 = t;
  if (t1 < tmin || t1 > tmax) {
    t1 = tmin + (tmax - tmin) / (emax - emin) * (e - emin);
  }
  for (int i = 0; i < maxiter; ++i) {
    CKUBMS(t1, y, e1);
    CKCVBS(t1, y, cv);
    dt = (e - e1) / cv;
    if (dt > 100.) {
      dt = 100.;
    } else if (dt < -100.) {
      dt = -100.;
    } else if (fabs(dt) < tol) {
      break;
    }
    t1 += dt;
  }
  t = t1;
  ierr = 0;
}

//  get temperature given enthalpy in mass units and mass fracs
AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE void
GET_T_GIVEN_HY(
  const amrex::Real h, const amrex::Real y[], amrex::Real& t, int& ierr)
{
#ifdef CONVERGENCE
  const int maxiter = 5000;
  const amrex::Real tol = 1.e-12;
#else
  const int maxiter = 200;
  const amrex::Real tol = 1.e-6;
#endif
  amrex::Real tmin = 90;   // max lower bound for thermo def
  amrex::Real tmax = 4000; // min upper bound for thermo def
  amrex::Real h1, hmin, hmax, cp, t1, dt;
  CKHBMS(tmin, y, hmin);
  CKHBMS(tmax, y, hmax);
  if (h < hmin) {
    // Linear Extrapolation below tmin
    CKCPBS(tmin, y, cp);
    t = tmin - (hmin - h) / cp;
    ierr = 1;
    return;
  }
  if (h > hmax) {
    // Linear Extrapolation above tmax
    CKCPBS(tmax, y, cp);
    t = tmax - (hmax - h) / cp;
    ierr = 1;
    return;
  }
  t1 = t;
  if (t1 < tmin || t1 > tmax) {
    t1 = tmin + (tmax - tmin) / (hmax - hmin) * (h - hmin);
  }
  for (int i = 0; i < maxiter; ++i) {
    CKHBMS(t1, y, h1);
    CKCPBS(t1, y, cp);
    dt = (h - h1) / cp;
    if (dt > 100.) {
      dt = 100.;
    } else if (dt < -100.) {
      dt = -100.;
    } else if (fabs(dt) < tol) {
      break;
    }
    t1 += dt;
  }
  t = t1;
  ierr = 0;
}

// Compute P = rhoRT/W(x)
AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE void
CKPX(
  const amrex::Real rho,
  const amrex::Real T,
  const amrex::Real x[],
  amrex::Real& P)
{
  amrex::Real XW = 0;                      //  To hold mean molecular wt
  XW += x[0] * 1.008000;                   // h
  XW += x[1] * 2.016000;                   // h2
  XW += x[2] * 15.999000;                  // o
  XW += x[3] * 31.998000;                  // o2
  XW += x[4] * 17.007000;                  // oh
  XW += x[5] * 18.015000;                  // h2o
  XW += x[6] * 28.014000;                  // n2
  XW += x[7] * 33.006000;                  // ho2
  XW += x[8] * 34.014000;                  // h2o2
  XW += x[9] * 131.293000;                 // Xe
  P = rho * 8.31446261815324e+07 * T / XW; // P = rho*R*T/W
}

// Compute P = rhoRT/W(y)
AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE void
CKPY(
  const amrex::Real rho,
  const amrex::Real T,
  const amrex::Real y[],
  amrex::Real& P)
{
  amrex::Real YOW = 0; //  for computing mean MW

  for (int i = 0; i < 10; i++) {
    YOW += y[i] * imw(i);
  }
  P = rho * 8.31446261815324e+07 * T * YOW; // P = rho*R*T/W
}

// Compute P = rhoRT/W(c)
AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE void
CKPC(
  const amrex::Real rho,
  const amrex::Real T,
  const amrex::Real c[],
  amrex::Real& P)
{
  // See Eq 5 in CK Manual
  amrex::Real W = 0;
  amrex::Real sumC = 0;
  W += c[0] * 1.008000;   // h
  W += c[1] * 2.016000;   // h2
  W += c[2] * 15.999000;  // o
  W += c[3] * 31.998000;  // o2
  W += c[4] * 17.007000;  // oh
  W += c[5] * 18.015000;  // h2o
  W += c[6] * 28.014000;  // n2
  W += c[7] * 33.006000;  // ho2
  W += c[8] * 34.014000;  // h2o2
  W += c[9] * 131.293000; // Xe

  for (int id = 0; id < 10; ++id) {
    sumC += c[id];
  }
  P = rho * 8.31446261815324e+07 * T * sumC / W; // P = rho*R*T/W
}

// Compute rho = PW(x)/RT
AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE void
CKRHOX(
  const amrex::Real P,
  const amrex::Real T,
  const amrex::Real x[],
  amrex::Real& rho)
{
  amrex::Real XW = 0;                        //  To hold mean molecular wt
  XW += x[0] * 1.008000;                     // h
  XW += x[1] * 2.016000;                     // h2
  XW += x[2] * 15.999000;                    // o
  XW += x[3] * 31.998000;                    // o2
  XW += x[4] * 17.007000;                    // oh
  XW += x[5] * 18.015000;                    // h2o
  XW += x[6] * 28.014000;                    // n2
  XW += x[7] * 33.006000;                    // ho2
  XW += x[8] * 34.014000;                    // h2o2
  XW += x[9] * 131.293000;                   // Xe
  rho = P * XW / (8.31446261815324e+07 * T); // rho = P*W/(R*T)
}

// Compute rho = P*W(y)/RT
AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE void
CKRHOY(
  const amrex::Real P,
  const amrex::Real T,
  const amrex::Real y[],
  amrex::Real& rho)
{
  amrex::Real YOW = 0;

  for (int i = 0; i < 10; i++) {
    YOW += y[i] * imw(i);
  }

  rho = P / (8.31446261815324e+07 * T * YOW); // rho = P*W/(R*T)
}

// Compute rho = P*W(c)/(R*T)
AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE void
CKRHOC(
  const amrex::Real P,
  const amrex::Real T,
  const amrex::Real c[],
  amrex::Real& rho)
{
  // See Eq 5 in CK Manual
  amrex::Real W = 0;
  amrex::Real sumC = 0;
  W += c[0] * 1.008000;   // h
  W += c[1] * 2.016000;   // h2
  W += c[2] * 15.999000;  // o
  W += c[3] * 31.998000;  // o2
  W += c[4] * 17.007000;  // oh
  W += c[5] * 18.015000;  // h2o
  W += c[6] * 28.014000;  // n2
  W += c[7] * 33.006000;  // ho2
  W += c[8] * 34.014000;  // h2o2
  W += c[9] * 131.293000; // Xe

  for (int id = 0; id < 10; ++id) {
    sumC += c[id];
  }
  rho = P * W / (sumC * T * 8.31446261815324e+07); // rho = PW/(R*T)
}

// get molecular weight for all species
AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE void
CKWT(amrex::Real wt[])
{
  get_mw(wt);
}

// given y[species]: mass fractions
// s mean molecular weight (gm/mole)
AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE void
CKMMWY(const amrex::Real y[], amrex::Real& wtm)
{
  amrex::Real YOW = 0;

  for (int i = 0; i < 10; i++) {
    YOW += y[i] * imw(i);
  }

  wtm = 1.0 / YOW;
}

// given x[species]: mole fractions
// returns mean molecular weight (gm/mole)
AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE void
CKMMWX(const amrex::Real x[], amrex::Real& wtm)
{
  amrex::Real XW = 0;      //  see Eq 4 in CK Manual
  XW += x[0] * 1.008000;   // h
  XW += x[1] * 2.016000;   // h2
  XW += x[2] * 15.999000;  // o
  XW += x[3] * 31.998000;  // o2
  XW += x[4] * 17.007000;  // oh
  XW += x[5] * 18.015000;  // h2o
  XW += x[6] * 28.014000;  // n2
  XW += x[7] * 33.006000;  // ho2
  XW += x[8] * 34.014000;  // h2o2
  XW += x[9] * 131.293000; // Xe
  wtm = XW;
}

// given c[species]: molar concentration
// returns mean molecular weight (gm/mole)
AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE void
CKMMWC(const amrex::Real c[], amrex::Real& wtm)
{
  // See Eq 5 in CK Manual
  amrex::Real W = 0;
  amrex::Real sumC = 0;
  W += c[0] * 1.008000;   // h
  W += c[1] * 2.016000;   // h2
  W += c[2] * 15.999000;  // o
  W += c[3] * 31.998000;  // o2
  W += c[4] * 17.007000;  // oh
  W += c[5] * 18.015000;  // h2o
  W += c[6] * 28.014000;  // n2
  W += c[7] * 33.006000;  // ho2
  W += c[8] * 34.014000;  // h2o2
  W += c[9] * 131.293000; // Xe

  for (int id = 0; id < 10; ++id) {
    sumC += c[id];
  }
  //  CK provides no guard against division by zero
  wtm = W / sumC;
}

// get Cp/R as a function of T
// for all species (Eq 19)
AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE void
CKCPOR(const amrex::Real T, amrex::Real cpor[])
{
  cp_R(cpor, T);
}

// get H/RT as a function of T
// for all species (Eq 20)
AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE void
CKHORT(const amrex::Real T, amrex::Real hort[])
{
  speciesEnthalpy(hort, T);
}

// get S/R as a function of T
// for all species (Eq 21)
AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE void
CKSOR(const amrex::Real T, amrex::Real sor[])
{
  speciesEntropy(sor, T);
}

// convert y[species] (mass fracs) to x[species] (mole fracs)
AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE void
CKYTX(const amrex::Real y[], amrex::Real x[])
{
  amrex::Real YOW = 0;

  for (int i = 0; i < 10; i++) {
    YOW += y[i] * imw(i);
  }

  amrex::Real YOWINV = 1.0 / YOW;

  for (int i = 0; i < 10; i++) {
    x[i] = y[i] * imw(i) * YOWINV;
  }
}

// convert y[species] (mass fracs) to c[species] (molar conc)
AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE void
CKYTCP(
  const amrex::Real P,
  const amrex::Real T,
  const amrex::Real y[],
  amrex::Real c[])
{
  amrex::Real YOW = 0;
  amrex::Real PWORT;

  // Compute inverse of mean molecular wt first
  for (int i = 0; i < 10; i++) {
    c[i] = y[i] * imw(i);
  }
  for (int i = 0; i < 10; i++) {
    YOW += c[i];
  }

  // PW/RT (see Eq. 7)
  PWORT = P / (YOW * 8.31446261815324e+07 * T);
  // Now compute conversion

  for (int i = 0; i < 10; i++) {
    c[i] = PWORT * y[i] * imw(i);
  }
}

// convert y[species] (mass fracs) to c[species] (molar conc)
AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE void
CKYTCR(
  const amrex::Real rho,
  amrex::Real /*T*/,
  const amrex::Real y[],
  amrex::Real c[])
{

  for (int i = 0; i < 10; i++) {
    c[i] = rho * y[i] * imw(i);
  }
}

// convert x[species] (mole fracs) to y[species] (mass fracs)
AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE void
CKXTY(const amrex::Real x[], amrex::Real y[])
{
  amrex::Real XW = 0; // See Eq 4, 9 in CK Manual
  // Compute mean molecular wt first
  XW += x[0] * 1.008000;   // h
  XW += x[1] * 2.016000;   // h2
  XW += x[2] * 15.999000;  // o
  XW += x[3] * 31.998000;  // o2
  XW += x[4] * 17.007000;  // oh
  XW += x[5] * 18.015000;  // h2o
  XW += x[6] * 28.014000;  // n2
  XW += x[7] * 33.006000;  // ho2
  XW += x[8] * 34.014000;  // h2o2
  XW += x[9] * 131.293000; // Xe
  // Now compute conversion
  amrex::Real XWinv = 1.0 / XW;
  y[0] = x[0] * 1.008000 * XWinv;
  y[1] = x[1] * 2.016000 * XWinv;
  y[2] = x[2] * 15.999000 * XWinv;
  y[3] = x[3] * 31.998000 * XWinv;
  y[4] = x[4] * 17.007000 * XWinv;
  y[5] = x[5] * 18.015000 * XWinv;
  y[6] = x[6] * 28.014000 * XWinv;
  y[7] = x[7] * 33.006000 * XWinv;
  y[8] = x[8] * 34.014000 * XWinv;
  y[9] = x[9] * 131.293000 * XWinv;
}

// convert x[species] (mole fracs) to c[species] (molar conc)
AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE void
CKXTCP(
  const amrex::Real P,
  const amrex::Real T,
  const amrex::Real x[],
  amrex::Real c[])
{
  amrex::Real PORT = P / (8.31446261815324e+07 * T); // P/RT

  // Compute conversion, see Eq 10
  for (int id = 0; id < 10; ++id) {
    c[id] = x[id] * PORT;
  }
}

// convert x[species] (mole fracs) to c[species] (molar conc)
AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE void
CKXTCR(
  const amrex::Real rho,
  const amrex::Real /*T*/,
  const amrex::Real x[],
  amrex::Real c[])
{
  amrex::Real XW = 0; // See Eq 4, 11 in CK Manual
  amrex::Real ROW;
  // Compute mean molecular wt first
  XW += x[0] * 1.008000;   // h
  XW += x[1] * 2.016000;   // h2
  XW += x[2] * 15.999000;  // o
  XW += x[3] * 31.998000;  // o2
  XW += x[4] * 17.007000;  // oh
  XW += x[5] * 18.015000;  // h2o
  XW += x[6] * 28.014000;  // n2
  XW += x[7] * 33.006000;  // ho2
  XW += x[8] * 34.014000;  // h2o2
  XW += x[9] * 131.293000; // Xe
  ROW = rho / XW;

  // Compute conversion, see Eq 11
  for (int id = 0; id < 10; ++id) {
    c[id] = x[id] * ROW;
  }
}

// convert c[species] (molar conc) to x[species] (mole fracs)
AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE void
CKCTX(const amrex::Real c[], amrex::Real x[])
{
  amrex::Real sumC = 0;

  // compute sum of c
  for (int id = 0; id < 10; ++id) {
    sumC += c[id];
  }

  //  See Eq 13
  amrex::Real sumCinv = 1.0 / sumC;
  for (int id = 0; id < 10; ++id) {
    x[id] = c[id] * sumCinv;
  }
}

// convert c[species] (molar conc) to y[species] (mass fracs)
AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE void
CKCTY(const amrex::Real c[], amrex::Real y[])
{
  amrex::Real CW = 0; // See Eq 12 in CK Manual
  // compute denominator in eq 12 first
  CW += c[0] * 1.008000;   // h
  CW += c[1] * 2.016000;   // h2
  CW += c[2] * 15.999000;  // o
  CW += c[3] * 31.998000;  // o2
  CW += c[4] * 17.007000;  // oh
  CW += c[5] * 18.015000;  // h2o
  CW += c[6] * 28.014000;  // n2
  CW += c[7] * 33.006000;  // ho2
  CW += c[8] * 34.014000;  // h2o2
  CW += c[9] * 131.293000; // Xe
  // Now compute conversion
  amrex::Real CWinv = 1.0 / CW;
  y[0] = c[0] * 1.008000 * CWinv;
  y[1] = c[1] * 2.016000 * CWinv;
  y[2] = c[2] * 15.999000 * CWinv;
  y[3] = c[3] * 31.998000 * CWinv;
  y[4] = c[4] * 17.007000 * CWinv;
  y[5] = c[5] * 18.015000 * CWinv;
  y[6] = c[6] * 28.014000 * CWinv;
  y[7] = c[7] * 33.006000 * CWinv;
  y[8] = c[8] * 34.014000 * CWinv;
  y[9] = c[9] * 131.293000 * CWinv;
}

// get specific heat at constant volume as a function
// of T for all species (molar units)
AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE void
CKCVML(const amrex::Real T, amrex::Real cvml[])
{
  cv_R(cvml, T);

  // convert to chemkin units
  for (int id = 0; id < 10; ++id) {
    cvml[id] *= 8.31446261815324e+07;
  }
}

// get specific heat at constant pressure as a
// function of T for all species (molar units)
AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE void
CKCPML(const amrex::Real T, amrex::Real cpml[])
{
  cp_R(cpml, T);

  // convert to chemkin units
  for (int id = 0; id < 10; ++id) {
    cpml[id] *= 8.31446261815324e+07;
  }
}

// get internal energy as a function
// of T for all species (molar units)
AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE void
CKUML(const amrex::Real T, amrex::Real uml[])
{
  amrex::Real RT = 8.31446261815324e+07 * T; // R*T
  speciesInternalEnergy(uml, T);

  // convert to chemkin units
  for (int id = 0; id < 10; ++id) {
    uml[id] *= RT;
  }
}

// get enthalpy as a function
// of T for all species (molar units)
AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE void
CKHML(const amrex::Real T, amrex::Real hml[])
{
  amrex::Real RT = 8.31446261815324e+07 * T; // R*T
  speciesEnthalpy(hml, T);

  // convert to chemkin units
  for (int id = 0; id < 10; ++id) {
    hml[id] *= RT;
  }
}

// Returns the standard-state entropies in molar units
AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE void
CKSML(const amrex::Real T, amrex::Real sml[])
{
  speciesEntropy(sml, T);

  // convert to chemkin units
  for (int id = 0; id < 10; ++id) {
    sml[id] *= 8.31446261815324e+07;
  }
}

// Returns the specific heats at constant volume
// in mass units (Eq. 29)
AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE void
CKCVMS(const amrex::Real T, amrex::Real cvms[])
{
  cv_R(cvms, T);
  // multiply by R/molecularweight
  cvms[0] *= 8.248474819596468e+07; // h
  cvms[1] *= 4.124237409798234e+07; // h2
  cvms[2] *= 5.196863940342046e+06; // o
  cvms[3] *= 2.598431970171023e+06; // o2
  cvms[4] *= 4.888847308845322e+06; // oh
  cvms[5] *= 4.615299815794193e+06; // h2o
  cvms[6] *= 2.967966951578939e+06; // n2
  cvms[7] *= 2.519076112874398e+06; // ho2
  cvms[8] *= 2.444423654422661e+06; // h2o2
  cvms[9] *= 6.332753930638526e+05; // Xe
}

// Returns the specific heats at constant pressure
// in mass units (Eq. 26)
AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE void
CKCPMS(const amrex::Real T, amrex::Real cpms[])
{
  cp_R(cpms, T);
  // multiply by R/molecularweight
  cpms[0] *= 8.248474819596468e+07; // h
  cpms[1] *= 4.124237409798234e+07; // h2
  cpms[2] *= 5.196863940342046e+06; // o
  cpms[3] *= 2.598431970171023e+06; // o2
  cpms[4] *= 4.888847308845322e+06; // oh
  cpms[5] *= 4.615299815794193e+06; // h2o
  cpms[6] *= 2.967966951578939e+06; // n2
  cpms[7] *= 2.519076112874398e+06; // ho2
  cpms[8] *= 2.444423654422661e+06; // h2o2
  cpms[9] *= 6.332753930638526e+05; // Xe
}

// Returns internal energy in mass units (Eq 30.)
AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE void
CKUMS(const amrex::Real T, amrex::Real ums[])
{
  amrex::Real RT = 8.31446261815324e+07 * T; // R*T

  speciesInternalEnergy(ums, T);

  for (int i = 0; i < 10; i++) {
    ums[i] *= RT * imw(i);
  }
}

// Returns enthalpy in mass units (Eq 27.)
AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE void
CKHMS(const amrex::Real T, amrex::Real hms[])
{
  amrex::Real RT = 8.31446261815324e+07 * T; // R*T

  speciesEnthalpy(hms, T);

  for (int i = 0; i < 10; i++) {
    hms[i] *= RT * imw(i);
  }
}

// Returns the entropies in mass units (Eq 28.)
AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE void
CKSMS(const amrex::Real T, amrex::Real sms[])
{
  speciesEntropy(sms, T);
  // multiply by R/molecularweight
  sms[0] *= 8.248474819596468e+07; // h
  sms[1] *= 4.124237409798234e+07; // h2
  sms[2] *= 5.196863940342046e+06; // o
  sms[3] *= 2.598431970171023e+06; // o2
  sms[4] *= 4.888847308845322e+06; // oh
  sms[5] *= 4.615299815794193e+06; // h2o
  sms[6] *= 2.967966951578939e+06; // n2
  sms[7] *= 2.519076112874398e+06; // ho2
  sms[8] *= 2.444423654422661e+06; // h2o2
  sms[9] *= 6.332753930638526e+05; // Xe
}

// GPU version of productionRate: no more use of thermo namespace vectors
AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE void
comp_qfqr(
  amrex::Real* qf,
  amrex::Real* qr,
  const amrex::Real* sc,
  const amrex::Real* /*sc_qss*/,
  const amrex::Real T,
  const amrex::Real invT,
  const amrex::Real logT)
{

  // reaction 8: h + o2 (+M) <=> ho2 (+M)
  qf[0] = sc[0] * sc[3];
  qr[0] = sc[7];

  // reaction 15: h2o2 (+M) <=> 2 oh (+M)
  qf[1] = sc[8];
  qr[1] = (sc[4] * sc[4]);

  // reaction 4: h2 + M <=> 2 h + M
  qf[2] = sc[1];
  qr[2] = (sc[0] * sc[0]);

  // reaction 5: 2 o + M <=> o2 + M
  qf[3] = (sc[2] * sc[2]);
  qr[3] = sc[3];

  // reaction 6: h + o + M <=> oh + M
  qf[4] = sc[0] * sc[2];
  qr[4] = sc[4];

  // reaction 7: h + oh + M <=> h2o + M
  qf[5] = sc[0] * sc[4];
  qr[5] = sc[5];

  // reaction 0: h + o2 <=> o + oh
  qf[6] = sc[0] * sc[3];
  qr[6] = sc[2] * sc[4];

  // reaction 1: h2 + o <=> h + oh
  qf[7] = sc[1] * sc[2];
  qr[7] = sc[0] * sc[4];

  // reaction 2: h2 + oh <=> h + h2o
  qf[8] = sc[1] * sc[4];
  qr[8] = sc[0] * sc[5];

  // reaction 3: h2o + o <=> 2 oh
  qf[9] = sc[2] * sc[5];
  qr[9] = (sc[4] * sc[4]);

  // reaction 9: h + ho2 <=> h2 + o2
  qf[10] = sc[0] * sc[7];
  qr[10] = sc[1] * sc[3];

  // reaction 10: h + ho2 <=> 2 oh
  qf[11] = sc[0] * sc[7];
  qr[11] = (sc[4] * sc[4]);

  // reaction 11: ho2 + o <=> o2 + oh
  qf[12] = sc[2] * sc[7];
  qr[12] = sc[3] * sc[4];

  // reaction 12: ho2 + oh <=> h2o + o2
  qf[13] = sc[4] * sc[7];
  qr[13] = sc[3] * sc[5];

  // reaction 13: 2 ho2 <=> h2o2 + o2
  qf[14] = (sc[7] * sc[7]);
  qr[14] = sc[3] * sc[8];

  // reaction 14: 2 ho2 <=> h2o2 + o2
  qf[15] = (sc[7] * sc[7]);
  qr[15] = sc[3] * sc[8];

  // reaction 16: h + h2o2 <=> h2o + oh
  qf[16] = sc[0] * sc[8];
  qr[16] = sc[4] * sc[5];

  // reaction 17: h + h2o2 <=> h2 + ho2
  qf[17] = sc[0] * sc[8];
  qr[17] = sc[1] * sc[7];

  // reaction 18: h2o2 + o <=> ho2 + oh
  qf[18] = sc[2] * sc[8];
  qr[18] = sc[4] * sc[7];

  // reaction 19: h2o2 + oh <=> h2o + ho2
  qf[19] = sc[4] * sc[8];
  qr[19] = sc[5] * sc[7];

  // reaction 20: h2o2 + oh <=> h2o + ho2
  qf[20] = sc[4] * sc[8];
  qr[20] = sc[5] * sc[7];

  // compute the mixture concentration
  amrex::Real mixture = 0.0;
  for (int i = 0; i < 10; ++i) {
    mixture += sc[i];
  }

  // compute the Gibbs free energy
  amrex::Real g_RT[10];
  gibbs(g_RT, T);

  // reference concentration: P_atm / (RT) in inverse mol/m^3
  amrex::Real refC = 101325 / 8.31446 * invT;
  amrex::Real refCinv = 1 / refC;

  // Evaluate the kfs
  amrex::Real k_f, Corr;
  amrex::Real redP, F, logPred, logFcent, troe_c, troe_n, troe, F_troe;

  // reaction 0:  h + o2 <=> o + oh
  k_f = 191500000 * exp(-(8272.88198582959) * invT);
  qf[6] *= k_f;
  qr[6] *= k_f * exp(-(g_RT[0] - g_RT[2] + g_RT[3] - g_RT[4]));
  // reaction 1:  h2 + o <=> h + oh
  k_f = 0.0508 * exp((2.67) * logT - (3166.2392612433) * invT);
  qf[7] *= k_f;
  qr[7] *= k_f * exp(-(-g_RT[0] + g_RT[1] + g_RT[2] - g_RT[4]));
  // reaction 2:  h2 + oh <=> h + h2o
  k_f = 216 * exp((1.51) * logT - (1726.03316371019) * invT);
  qf[8] *= k_f;
  qr[8] *= k_f * exp(-(-g_RT[0] + g_RT[1] + g_RT[4] - g_RT[5]));
  // reaction 3:  h2o + o <=> 2 oh
  k_f = 2.97 * exp((2.02) * logT - (6743.10332178324) * invT);
  qf[9] *= k_f;
  qr[9] *= k_f * exp(-(g_RT[2] - 2.000000 * g_RT[4] + g_RT[5]));
  // reaction 4:  h2 + M <=> 2 h + M
  k_f = 45770000000000 * exp((-1.4) * logT - (52535.8199100127) * invT);
  Corr = mixture + (1.5) * sc[1] + (11) * sc[5];
  qf[2] *= Corr * k_f;
  qr[2] *= Corr * k_f * exp(-(-2.000000 * g_RT[0] + g_RT[1])) * (refCinv);
  // reaction 5:  2 o + M <=> o2 + M
  k_f = 6165 * exp((-0.5) * logT);
  Corr = mixture + (1.5) * sc[1] + (11) * sc[5] + (-0.17) * sc[9];
  qf[3] *= Corr * k_f;
  qr[3] *= Corr * k_f * exp(-(2.000000 * g_RT[2] - g_RT[3])) * (refC);
  // reaction 6:  h + o + M <=> oh + M
  k_f = 4714000 * exp((-1) * logT);
  Corr = mixture + (1.5) * sc[1] + (11) * sc[5] + (-0.25) * sc[9];
  qf[4] *= Corr * k_f;
  qr[4] *= Corr * k_f * exp(-(g_RT[0] + g_RT[2] - g_RT[4])) * (refC);
  // reaction 7:  h + oh + M <=> h2o + M
  k_f = 45000000000 * exp((-2) * logT);
  Corr = mixture + (-0.27) * sc[1] + (11) * sc[5] + (-0.62) * sc[9];
  qf[5] *= Corr * k_f;
  qr[5] *= Corr * k_f * exp(-(g_RT[0] + g_RT[4] - g_RT[5])) * (refC);
  // reaction 8:  h + o2 (+M) <=> ho2 (+M)
  k_f = 1475000 * exp((0.6) * logT);
  Corr = mixture + (0.3) * sc[1] + (13) * sc[5] + (-0.33) * sc[9];
  redP = Corr / k_f * 34820 * exp(-0.411 * logT - (-561.086582372262) * invT);
  F = redP / (1.0 + redP);
  logPred = log10(redP);
  logFcent = log10(
    (0.5) * exp(-T * 1e+30) + 0.5 * exp(-T * 1e-30) + exp(-1e+100 * invT));
  troe_c = -0.4 - 0.67 * logFcent;
  troe_n = 0.75 - 1.27 * logFcent;
  troe = (troe_c + logPred) / (troe_n - 0.14 * (troe_c + logPred));
  F_troe = exp(M_LN10 * logFcent / (1.0 + troe * troe));
  Corr = F * F_troe;
  qf[0] *= Corr * k_f;
  qr[0] *= Corr * k_f * exp(-(g_RT[0] + g_RT[3] - g_RT[7])) * (refC);
  // reaction 9:  h + ho2 <=> h2 + o2
  k_f = 16600000 * exp(-(414.147315957284) * invT);
  qf[10] *= k_f;
  qr[10] *= k_f * exp(-(g_RT[0] - g_RT[1] - g_RT[3] + g_RT[7]));
  // reaction 10:  h + ho2 <=> 2 oh
  k_f = 70790000 * exp(-(148.448916412392) * invT);
  qf[11] *= k_f;
  qr[11] *= k_f * exp(-(g_RT[0] - 2.000000 * g_RT[4] + g_RT[7]));
  // reaction 11:  ho2 + o <=> o2 + oh
  k_f = 32500000;
  qf[12] *= k_f;
  qr[12] *= k_f * exp(-(g_RT[2] - g_RT[3] - g_RT[4] + g_RT[7]));
  // reaction 12:  ho2 + oh <=> h2o + o2
  k_f = 28900000 * exp(-(-250.098682904946) * invT);
  qf[13] *= k_f;
  qr[13] *= k_f * exp(-(-g_RT[3] + g_RT[4] - g_RT[5] + g_RT[7]));
  // reaction 13:  2 ho2 <=> h2o2 + o2
  k_f = 420000000 * exp(-(6028.53565634054) * invT);
  qf[14] *= k_f;
  qr[14] *= k_f * exp(-(-g_RT[3] + 2.000000 * g_RT[7] - g_RT[8]));
  // reaction 14:  2 ho2 <=> h2o2 + o2
  k_f = 130000 * exp(-(-819.739948595888) * invT);
  qf[15] *= k_f;
  qr[15] *= k_f * exp(-(-g_RT[3] + 2.000000 * g_RT[7] - g_RT[8]));
  // reaction 15:  h2o2 (+M) <=> 2 oh (+M)
  k_f = 295100000000000 * exp(-(24370.7831249226) * invT);
  Corr = mixture + (1.5) * sc[1] + (11) * sc[5] + (-0.36) * sc[9];
  redP = Corr / k_f * 120200000000 * exp(-(22896.3582941147) * invT);
  F = redP / (1.0 + redP);
  logPred = log10(redP);
  logFcent = log10(
    (0.5) * exp(-T * 1e+30) + 0.5 * exp(-T * 1e-30) + exp(-1e+100 * invT));
  troe_c = -0.4 - 0.67 * logFcent;
  troe_n = 0.75 - 1.27 * logFcent;
  troe = (troe_c + logPred) / (troe_n - 0.14 * (troe_c + logPred));
  F_troe = exp(M_LN10 * logFcent / (1.0 + troe * troe));
  Corr = F * F_troe;
  qf[1] *= Corr * k_f;
  qr[1] *= Corr * k_f * exp(-(-2.000000 * g_RT[4] + g_RT[8])) * (refCinv);
  // reaction 16:  h + h2o2 <=> h2o + oh
  k_f = 24100000 * exp(-(1997.77016324474) * invT);
  qf[16] *= k_f;
  qr[16] *= k_f * exp(-(g_RT[0] - g_RT[4] - g_RT[5] + g_RT[8]));
  // reaction 17:  h + h2o2 <=> h2 + ho2
  k_f = 60250000 * exp(-(4000.57249314752) * invT);
  qf[17] *= k_f;
  qr[17] *= k_f * exp(-(g_RT[0] - g_RT[1] - g_RT[7] + g_RT[8]));
  // reaction 18:  h2o2 + o <=> ho2 + oh
  k_f = 9.55 * exp((2) * logT - (1997.77016324474) * invT);
  qf[18] *= k_f;
  qr[18] *= k_f * exp(-(g_RT[2] - g_RT[4] - g_RT[7] + g_RT[8]));
  // reaction 19:  h2o2 + oh <=> h2o + ho2
  k_f = 1000000;
  qf[19] *= k_f;
  qr[19] *= k_f * exp(-(g_RT[4] - g_RT[5] - g_RT[7] + g_RT[8]));
  // reaction 20:  h2o2 + oh <=> h2o + ho2
  k_f = 580000000 * exp(-(4809.24167509571) * invT);
  qf[20] *= k_f;
  qr[20] *= k_f * exp(-(g_RT[4] - g_RT[5] - g_RT[7] + g_RT[8]));
}

AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE void
productionRate(amrex::Real* wdot, const amrex::Real* sc, const amrex::Real T)
{
  const amrex::Real invT = 1.0 / T;
  const amrex::Real logT = log(T);

  // reference concentration: P_atm / (RT) in inverse mol/m^3
  const amrex::Real refC = 101325 / 8.31446 * invT;
  const amrex::Real refCinv = 1 / refC;

  for (int i = 0; i < 10; ++i) {
    wdot[i] = 0.0;
  }

  // compute the mixture concentration
  amrex::Real mixture = 0.0;
  for (int i = 0; i < 10; ++i) {
    mixture += sc[i];
  }

  // compute the Gibbs free energy
  amrex::Real g_RT[10];
  gibbs(g_RT, T);

  {
    // reaction 8:  h + o2 (+M) <=> ho2 (+M)
    const amrex::Real k_f = 1475000 * exp((0.6) * logT);
    amrex::Real Corr = mixture + (0.3) * sc[1] + (13) * sc[5] + (-0.33) * sc[9];
    const amrex::Real redP =
      Corr / k_f * 34820 * exp(-0.411 * logT - -561.086582372262 * invT);
    const amrex::Real F = redP / (1.0 + redP);
    const amrex::Real logPred = log10(redP);
    const amrex::Real logFcent = log10(
      0.5 * exp(-T * 1e+30) + 0.5 * exp(-T * 1e-30) + exp(-1e+100 * invT));
    const amrex::Real troe_c = -0.4 - 0.67 * logFcent;
    const amrex::Real troe_n = 0.75 - 1.27 * logFcent;
    const amrex::Real troe =
      (troe_c + logPred) / (troe_n - 0.14 * (troe_c + logPred));
    const amrex::Real F_troe = exp(M_LN10 * logFcent / (1.0 + troe * troe));
    Corr = F * F_troe;
    const amrex::Real qf = Corr * k_f * (sc[0] * sc[3]);
    const amrex::Real qr =
      Corr * k_f * exp(-(g_RT[0] + g_RT[3] - g_RT[7])) * (refC) * (sc[7]);
    const amrex::Real qdot = qf - qr;
    wdot[0] -= qdot;
    wdot[3] -= qdot;
    wdot[7] += qdot;
  }

  {
    // reaction 15:  h2o2 (+M) <=> 2 oh (+M)
    const amrex::Real k_f = 295100000000000 * exp(-(24370.7831249226) * invT);
    amrex::Real Corr = mixture + (1.5) * sc[1] + (11) * sc[5] + (-0.36) * sc[9];
    const amrex::Real redP =
      Corr / k_f * 120200000000 * exp(-22896.3582941147 * invT);
    const amrex::Real F = redP / (1.0 + redP);
    const amrex::Real logPred = log10(redP);
    const amrex::Real logFcent = log10(
      0.5 * exp(-T * 1e+30) + 0.5 * exp(-T * 1e-30) + exp(-1e+100 * invT));
    const amrex::Real troe_c = -0.4 - 0.67 * logFcent;
    const amrex::Real troe_n = 0.75 - 1.27 * logFcent;
    const amrex::Real troe =
      (troe_c + logPred) / (troe_n - 0.14 * (troe_c + logPred));
    const amrex::Real F_troe = exp(M_LN10 * logFcent / (1.0 + troe * troe));
    Corr = F * F_troe;
    const amrex::Real qf = Corr * k_f * (sc[8]);
    const amrex::Real qr = Corr * k_f * exp(-(-2.000000 * g_RT[4] + g_RT[8])) *
                           (refCinv) * ((sc[4] * sc[4]));
    const amrex::Real qdot = qf - qr;
    wdot[4] += 2.000000 * qdot;
    wdot[8] -= qdot;
  }

  {
    // reaction 4:  h2 + M <=> 2 h + M
    const amrex::Real k_f =
      45770000000000 * exp((-1.4) * logT - (52535.8199100127) * invT);
    const amrex::Real Corr = mixture + (1.5) * sc[1] + (11) * sc[5];
    const amrex::Real qf = Corr * k_f * (sc[1]);
    const amrex::Real qr = Corr * k_f * exp(-(-2.000000 * g_RT[0] + g_RT[1])) *
                           (refCinv) * ((sc[0] * sc[0]));
    const amrex::Real qdot = qf - qr;
    wdot[0] += 2.000000 * qdot;
    wdot[1] -= qdot;
  }

  {
    // reaction 5:  2 o + M <=> o2 + M
    const amrex::Real k_f = 6165 * exp((-0.5) * logT);
    const amrex::Real Corr =
      mixture + (1.5) * sc[1] + (11) * sc[5] + (-0.17) * sc[9];
    const amrex::Real qf = Corr * k_f * ((sc[2] * sc[2]));
    const amrex::Real qr =
      Corr * k_f * exp(-(2.000000 * g_RT[2] - g_RT[3])) * (refC) * (sc[3]);
    const amrex::Real qdot = qf - qr;
    wdot[2] -= 2.000000 * qdot;
    wdot[3] += qdot;
  }

  {
    // reaction 6:  h + o + M <=> oh + M
    const amrex::Real k_f = 4714000 * exp((-1) * logT);
    const amrex::Real Corr =
      mixture + (1.5) * sc[1] + (11) * sc[5] + (-0.25) * sc[9];
    const amrex::Real qf = Corr * k_f * (sc[0] * sc[2]);
    const amrex::Real qr =
      Corr * k_f * exp(-(g_RT[0] + g_RT[2] - g_RT[4])) * (refC) * (sc[4]);
    const amrex::Real qdot = qf - qr;
    wdot[0] -= qdot;
    wdot[2] -= qdot;
    wdot[4] += qdot;
  }

  {
    // reaction 7:  h + oh + M <=> h2o + M
    const amrex::Real k_f = 45000000000 * exp((-2) * logT);
    const amrex::Real Corr =
      mixture + (-0.27) * sc[1] + (11) * sc[5] + (-0.62) * sc[9];
    const amrex::Real qf = Corr * k_f * (sc[0] * sc[4]);
    const amrex::Real qr =
      Corr * k_f * exp(-(g_RT[0] + g_RT[4] - g_RT[5])) * (refC) * (sc[5]);
    const amrex::Real qdot = qf - qr;
    wdot[0] -= qdot;
    wdot[4] -= qdot;
    wdot[5] += qdot;
  }

  {
    // reaction 0:  h + o2 <=> o + oh
    const amrex::Real k_f = 191500000 * exp(-(8272.88198582959) * invT);
    const amrex::Real qf = k_f * (sc[0] * sc[3]);
    const amrex::Real qr =
      k_f * exp(-(g_RT[0] - g_RT[2] + g_RT[3] - g_RT[4])) * (sc[2] * sc[4]);
    const amrex::Real qdot = qf - qr;
    wdot[0] -= qdot;
    wdot[2] += qdot;
    wdot[3] -= qdot;
    wdot[4] += qdot;
  }

  {
    // reaction 1:  h2 + o <=> h + oh
    const amrex::Real k_f =
      0.0508 * exp((2.67) * logT - (3166.2392612433) * invT);
    const amrex::Real qf = k_f * (sc[1] * sc[2]);
    const amrex::Real qr =
      k_f * exp(-(-g_RT[0] + g_RT[1] + g_RT[2] - g_RT[4])) * (sc[0] * sc[4]);
    const amrex::Real qdot = qf - qr;
    wdot[0] += qdot;
    wdot[1] -= qdot;
    wdot[2] -= qdot;
    wdot[4] += qdot;
  }

  {
    // reaction 2:  h2 + oh <=> h + h2o
    const amrex::Real k_f =
      216 * exp((1.51) * logT - (1726.03316371019) * invT);
    const amrex::Real qf = k_f * (sc[1] * sc[4]);
    const amrex::Real qr =
      k_f * exp(-(-g_RT[0] + g_RT[1] + g_RT[4] - g_RT[5])) * (sc[0] * sc[5]);
    const amrex::Real qdot = qf - qr;
    wdot[0] += qdot;
    wdot[1] -= qdot;
    wdot[4] -= qdot;
    wdot[5] += qdot;
  }

  {
    // reaction 3:  h2o + o <=> 2 oh
    const amrex::Real k_f =
      2.97 * exp((2.02) * logT - (6743.10332178324) * invT);
    const amrex::Real qf = k_f * (sc[2] * sc[5]);
    const amrex::Real qr =
      k_f * exp(-(g_RT[2] - 2.000000 * g_RT[4] + g_RT[5])) * ((sc[4] * sc[4]));
    const amrex::Real qdot = qf - qr;
    wdot[2] -= qdot;
    wdot[4] += 2.000000 * qdot;
    wdot[5] -= qdot;
  }

  {
    // reaction 9:  h + ho2 <=> h2 + o2
    const amrex::Real k_f = 16600000 * exp(-(414.147315957284) * invT);
    const amrex::Real qf = k_f * (sc[0] * sc[7]);
    const amrex::Real qr =
      k_f * exp(-(g_RT[0] - g_RT[1] - g_RT[3] + g_RT[7])) * (sc[1] * sc[3]);
    const amrex::Real qdot = qf - qr;
    wdot[0] -= qdot;
    wdot[1] += qdot;
    wdot[3] += qdot;
    wdot[7] -= qdot;
  }

  {
    // reaction 10:  h + ho2 <=> 2 oh
    const amrex::Real k_f = 70790000 * exp(-(148.448916412392) * invT);
    const amrex::Real qf = k_f * (sc[0] * sc[7]);
    const amrex::Real qr =
      k_f * exp(-(g_RT[0] - 2.000000 * g_RT[4] + g_RT[7])) * ((sc[4] * sc[4]));
    const amrex::Real qdot = qf - qr;
    wdot[0] -= qdot;
    wdot[4] += 2.000000 * qdot;
    wdot[7] -= qdot;
  }

  {
    // reaction 11:  ho2 + o <=> o2 + oh
    const amrex::Real k_f = 32500000;
    const amrex::Real qf = k_f * (sc[2] * sc[7]);
    const amrex::Real qr =
      k_f * exp(-(g_RT[2] - g_RT[3] - g_RT[4] + g_RT[7])) * (sc[3] * sc[4]);
    const amrex::Real qdot = qf - qr;
    wdot[2] -= qdot;
    wdot[3] += qdot;
    wdot[4] += qdot;
    wdot[7] -= qdot;
  }

  {
    // reaction 12:  ho2 + oh <=> h2o + o2
    const amrex::Real k_f = 28900000 * exp(-(-250.098682904946) * invT);
    const amrex::Real qf = k_f * (sc[4] * sc[7]);
    const amrex::Real qr =
      k_f * exp(-(-g_RT[3] + g_RT[4] - g_RT[5] + g_RT[7])) * (sc[3] * sc[5]);
    const amrex::Real qdot = qf - qr;
    wdot[3] += qdot;
    wdot[4] -= qdot;
    wdot[5] += qdot;
    wdot[7] -= qdot;
  }

  {
    // reaction 13:  2 ho2 <=> h2o2 + o2
    const amrex::Real k_f = 420000000 * exp(-(6028.53565634054) * invT);
    const amrex::Real qf = k_f * ((sc[7] * sc[7]));
    const amrex::Real qr =
      k_f * exp(-(-g_RT[3] + 2.000000 * g_RT[7] - g_RT[8])) * (sc[3] * sc[8]);
    const amrex::Real qdot = qf - qr;
    wdot[3] += qdot;
    wdot[7] -= 2.000000 * qdot;
    wdot[8] += qdot;
  }

  {
    // reaction 14:  2 ho2 <=> h2o2 + o2
    const amrex::Real k_f = 130000 * exp(-(-819.739948595888) * invT);
    const amrex::Real qf = k_f * ((sc[7] * sc[7]));
    const amrex::Real qr =
      k_f * exp(-(-g_RT[3] + 2.000000 * g_RT[7] - g_RT[8])) * (sc[3] * sc[8]);
    const amrex::Real qdot = qf - qr;
    wdot[3] += qdot;
    wdot[7] -= 2.000000 * qdot;
    wdot[8] += qdot;
  }

  {
    // reaction 16:  h + h2o2 <=> h2o + oh
    const amrex::Real k_f = 24100000 * exp(-(1997.77016324474) * invT);
    const amrex::Real qf = k_f * (sc[0] * sc[8]);
    const amrex::Real qr =
      k_f * exp(-(g_RT[0] - g_RT[4] - g_RT[5] + g_RT[8])) * (sc[4] * sc[5]);
    const amrex::Real qdot = qf - qr;
    wdot[0] -= qdot;
    wdot[4] += qdot;
    wdot[5] += qdot;
    wdot[8] -= qdot;
  }

  {
    // reaction 17:  h + h2o2 <=> h2 + ho2
    const amrex::Real k_f = 60250000 * exp(-(4000.57249314752) * invT);
    const amrex::Real qf = k_f * (sc[0] * sc[8]);
    const amrex::Real qr =
      k_f * exp(-(g_RT[0] - g_RT[1] - g_RT[7] + g_RT[8])) * (sc[1] * sc[7]);
    const amrex::Real qdot = qf - qr;
    wdot[0] -= qdot;
    wdot[1] += qdot;
    wdot[7] += qdot;
    wdot[8] -= qdot;
  }

  {
    // reaction 18:  h2o2 + o <=> ho2 + oh
    const amrex::Real k_f = 9.55 * exp((2) * logT - (1997.77016324474) * invT);
    const amrex::Real qf = k_f * (sc[2] * sc[8]);
    const amrex::Real qr =
      k_f * exp(-(g_RT[2] - g_RT[4] - g_RT[7] + g_RT[8])) * (sc[4] * sc[7]);
    const amrex::Real qdot = qf - qr;
    wdot[2] -= qdot;
    wdot[4] += qdot;
    wdot[7] += qdot;
    wdot[8] -= qdot;
  }

  {
    // reaction 19:  h2o2 + oh <=> h2o + ho2
    const amrex::Real k_f = 1000000;
    const amrex::Real qf = k_f * (sc[4] * sc[8]);
    const amrex::Real qr =
      k_f * exp(-(g_RT[4] - g_RT[5] - g_RT[7] + g_RT[8])) * (sc[5] * sc[7]);
    const amrex::Real qdot = qf - qr;
    wdot[4] -= qdot;
    wdot[5] += qdot;
    wdot[7] += qdot;
    wdot[8] -= qdot;
  }

  {
    // reaction 20:  h2o2 + oh <=> h2o + ho2
    const amrex::Real k_f = 580000000 * exp(-(4809.24167509571) * invT);
    const amrex::Real qf = k_f * (sc[4] * sc[8]);
    const amrex::Real qr =
      k_f * exp(-(g_RT[4] - g_RT[5] - g_RT[7] + g_RT[8])) * (sc[5] * sc[7]);
    const amrex::Real qdot = qf - qr;
    wdot[4] -= qdot;
    wdot[5] += qdot;
    wdot[7] += qdot;
    wdot[8] -= qdot;
  }
}

// compute the production rate for each species
AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE void
CKWC(const amrex::Real T, amrex::Real C[], amrex::Real wdot[])
{

  // convert to SI
  for (int id = 0; id < 10; ++id) {
    C[id] *= 1.0e6;
  }

  // convert to chemkin units
  productionRate(wdot, C, T);

  // convert to chemkin units
  for (int id = 0; id < 10; ++id) {
    C[id] *= 1.0e-6;
    wdot[id] *= 1.0e-6;
  }
}

// Returns the molar production rate of species
// Given P, T, and mass fractions
AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE void
CKWYP(
  const amrex::Real P,
  const amrex::Real T,
  const amrex::Real y[],
  amrex::Real wdot[])
{
  amrex::Real c[10]; // temporary storage
  amrex::Real YOW = 0;
  amrex::Real PWORT;

  // Compute inverse of mean molecular wt first
  for (int i = 0; i < 10; i++) {
    YOW += y[i] * imw(i);
  }
  // PW/RT (see Eq. 7)
  PWORT = P / (YOW * 8.31446261815324e+07 * T);
  // multiply by 1e6 so c goes to SI
  PWORT *= 1e6;
  // Now compute conversion (and go to SI)
  for (int i = 0; i < 10; i++) {
    c[i] = PWORT * y[i] * imw(i);
  }

  // convert to chemkin units
  productionRate(wdot, c, T);

  // convert to chemkin units
  for (int id = 0; id < 10; ++id) {
    wdot[id] *= 1.0e-6;
  }
}

// Returns the molar production rate of species
// Given P, T, and mole fractions
AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE void
CKWXP(
  const amrex::Real P,
  const amrex::Real T,
  const amrex::Real x[],
  amrex::Real wdot[])
{
  amrex::Real c[10]; // temporary storage
  amrex::Real PORT =
    1e6 * P / (8.31446261815324e+07 * T); // 1e6 * P/RT so c goes to SI units

  // Compute conversion, see Eq 10
  for (int id = 0; id < 10; ++id) {
    c[id] = x[id] * PORT;
  }

  // convert to chemkin units
  productionRate(wdot, c, T);

  // convert to chemkin units
  for (int id = 0; id < 10; ++id) {
    wdot[id] *= 1.0e-6;
  }
}

// Returns the molar production rate of species
// Given rho, T, and mass fractions
AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE void
CKWYR(
  const amrex::Real rho,
  const amrex::Real T,
  const amrex::Real y[],
  amrex::Real wdot[])
{
  amrex::Real c[10]; // temporary storage

  // See Eq 8 with an extra 1e6 so c goes to SI
  for (int i = 0; i < 10; i++) {
    c[i] = 1e6 * rho * y[i] * imw(i);
  }

  // call productionRate
  productionRate(wdot, c, T);

  // convert to chemkin units
  for (int id = 0; id < 10; ++id) {
    wdot[id] *= 1.0e-6;
  }
}

// Returns the molar production rate of species
// Given rho, T, and mole fractions
AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE void
CKWXR(
  const amrex::Real rho,
  const amrex::Real T,
  const amrex::Real x[],
  amrex::Real wdot[])
{
  amrex::Real c[10];  // temporary storage
  amrex::Real XW = 0; // See Eq 4, 11 in CK Manual
  amrex::Real ROW;
  // Compute mean molecular wt first
  XW += x[0] * 1.008000;   // h
  XW += x[1] * 2.016000;   // h2
  XW += x[2] * 15.999000;  // o
  XW += x[3] * 31.998000;  // o2
  XW += x[4] * 17.007000;  // oh
  XW += x[5] * 18.015000;  // h2o
  XW += x[6] * 28.014000;  // n2
  XW += x[7] * 33.006000;  // ho2
  XW += x[8] * 34.014000;  // h2o2
  XW += x[9] * 131.293000; // Xe
  // Extra 1e6 factor to take c to SI
  ROW = 1e6 * rho / XW;

  // Compute conversion, see Eq 11
  for (int id = 0; id < 10; ++id) {
    c[id] = x[id] * ROW;
  }

  // convert to chemkin units
  productionRate(wdot, c, T);

  // convert to chemkin units
  for (int id = 0; id < 10; ++id) {
    wdot[id] *= 1.0e-6;
  }
}

//  species unit charge number
AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE void
CKCHRG(int kcharge[])
{
  kcharge[0] = 0; // h
  kcharge[1] = 0; // h2
  kcharge[2] = 0; // o
  kcharge[3] = 0; // o2
  kcharge[4] = 0; // oh
  kcharge[5] = 0; // h2o
  kcharge[6] = 0; // n2
  kcharge[7] = 0; // ho2
  kcharge[8] = 0; // h2o2
  kcharge[9] = 0; // Xe
}

//  species charge per unit mass
AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE void
CKCHRGMASS(amrex::Real zk[])
{

  int kchrg[10];
  CKCHRG(kchrg);

  for (int id = 0; id < 10; ++id) {
    zk[id] = 6.02214076e+23 * 1.60217663e-19 * kchrg[id] * imw(id);
  }
}

// compute an approx to the reaction Jacobian
AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE void
aJacobian_precond(
  amrex::Real* J, const amrex::Real* sc, const amrex::Real T, const int HP)
{

#if defined(PELE_COMPILE_AJACOBIAN) || !defined(AMREX_USE_HIP)
  for (int i = 0; i < 121; i++) {
    J[i] = 0.0;
  }

  amrex::Real wdot[10];
  for (auto& val : wdot) {
    val = 0.0;
  }

  const amrex::Real invT = 1.0 / T;
  const amrex::Real invT2 = invT * invT;
  const amrex::Real logT = log(T);

  // reference concentration: P_atm / (RT) in inverse mol/m^3
  amrex::Real refC = 101325 / 8.31446 / T;
  amrex::Real refCinv = 1.0 / refC;

  // compute the mixture concentration
  amrex::Real mixture = 0.0;
  for (int k = 0; k < 10; ++k) {
    mixture += sc[k];
  }

  // compute the Gibbs free energy
  amrex::Real g_RT[10];
  gibbs(g_RT, T);

  // compute the species enthalpy
  amrex::Real h_RT[10];
  speciesEnthalpy(h_RT, T);

  amrex::Real phi_f, k_f, k_r, phi_r, Kc, q, q_nocor, Corr, alpha;
  amrex::Real dlnkfdT, dlnk0dT, dlnKcdT, dkrdT, dqdT;
  amrex::Real dqdci, dcdc_fac, dqdc[10];
  amrex::Real Pr, fPr, F, k_0, logPr;
  amrex::Real logFcent, troe_c, troe_n, troePr_den, troePr, troe;
  amrex::Real Fcent1, Fcent2, Fcent3, Fcent;
  amrex::Real dlogFdc, dlogFdn, dlogFdcn_fac;
  amrex::Real dlogPrdT, dlogfPrdT, dlogFdT, dlogFcentdT, dlogFdlogPr, dlnCorrdT;
  const amrex::Real ln10 = log(10.0);
  const amrex::Real log10e = 1.0 / log(10.0);
  // reaction 8: h + o2 (+M) <=> ho2 (+M)
  // a third-body and pressure-fall-off reaction
  // 3-body correction factor
  alpha = mixture + (1.3 - 1) * sc[1] + (14 - 1) * sc[5] + (0.67 - 1) * sc[9];
  // forward
  phi_f = sc[0] * sc[3];
  k_f = 1475000 * exp(0.6 * logT);
  dlnkfdT = 0.6 * invT;
  // pressure-fall-off
  k_0 = 3.482e+16 * exp(-0.411 * logT - (-561.086582372262) * invT);
  Pr = 1e-12 * alpha / k_f * k_0;
  fPr = Pr / (1.0 + Pr);
  dlnk0dT = -0.411 * invT + (-561.086582372262) * invT2;
  dlogPrdT = log10e * (dlnk0dT - dlnkfdT);
  dlogfPrdT = dlogPrdT / (1.0 + Pr);
  // Troe form
  logPr = log10(Pr);
  Fcent1 = (1. - 0.5) * exp(-T / 1e-30);
  Fcent2 = 0.5 * exp(-T / 1e+30);
  Fcent3 = exp(-1e+100 * invT);
  Fcent = Fcent1 + Fcent2 + Fcent3;
  logFcent = log10(Fcent);
  troe_c = -.4 - .67 * logFcent;
  troe_n = .75 - 1.27 * logFcent;
  troePr_den = 1.0 / (troe_n - .14 * (troe_c + logPr));
  troePr = (troe_c + logPr) * troePr_den;
  troe = 1.0 / (1.0 + troePr * troePr);
  F = exp(M_LN10 * logFcent * troe);
  dlogFcentdT = log10e / Fcent *
                (-Fcent1 / 1e-30 - Fcent2 / 1e+30 + Fcent3 * 1e+100 * invT2);
  dlogFdcn_fac = 2.0 * logFcent * troe * troe * troePr * troePr_den;
  dlogFdc = -troe_n * dlogFdcn_fac * troePr_den;
  dlogFdn = dlogFdcn_fac * troePr;
  dlogFdlogPr = dlogFdc;
  dlogFdT = dlogFcentdT * (troe - 0.67 * dlogFdc - 1.27 * dlogFdn) +
            dlogFdlogPr * dlogPrdT;
  // reverse
  phi_r = sc[7];
  Kc = refCinv * exp(g_RT[0] + g_RT[3] - g_RT[7]);
  k_r = k_f / Kc;
  dlnKcdT = invT * (-(h_RT[0] + h_RT[3]) + (h_RT[7]) + 1.000000);
  dkrdT = (dlnkfdT - dlnKcdT) * k_r;
  // rate of progress
  q_nocor = k_f * phi_f - k_r * phi_r;
  Corr = fPr * F;
  q = Corr * q_nocor;
  dlnCorrdT = ln10 * (dlogfPrdT + dlogFdT);
  dqdT = Corr * (dlnkfdT * k_f * phi_f - dkrdT * phi_r) + dlnCorrdT * q;
  // update wdot
  wdot[0] -= q; // h
  wdot[3] -= q; // o2
  wdot[7] += q; // ho2
  // for convenience
  k_f *= Corr;
  k_r *= Corr;
  dcdc_fac = 0.0;
  dqdc[0] = dcdc_fac + k_f * sc[3];
  dqdc[1] = 1.3 * dcdc_fac;
  dqdc[2] = dcdc_fac;
  dqdc[3] = dcdc_fac + k_f * sc[0];
  dqdc[4] = dcdc_fac;
  dqdc[5] = 14 * dcdc_fac;
  dqdc[6] = dcdc_fac;
  dqdc[7] = dcdc_fac - k_r;
  dqdc[8] = dcdc_fac;
  dqdc[9] = 0.67 * dcdc_fac;
  for (int k = 0; k < 10; k++) {
    J[11 * k + 0] -= dqdc[k];
    J[11 * k + 3] -= dqdc[k];
    J[11 * k + 7] += dqdc[k];
  }
  J[110] -= dqdT; // dwdot[h]/dT
  J[113] -= dqdT; // dwdot[o2]/dT
  J[117] += dqdT; // dwdot[ho2]/dT

  // reaction 15: h2o2 (+M) <=> 2 oh (+M)
  // a third-body and pressure-fall-off reaction
  // 3-body correction factor
  alpha = mixture + (2.5 - 1) * sc[1] + (12 - 1) * sc[5] + (0.64 - 1) * sc[9];
  // forward
  phi_f = sc[8];
  k_f = 295100000000000 * exp(-(24370.7831249226) * invT);
  dlnkfdT = (24370.7831249226) * invT2;
  // pressure-fall-off
  k_0 = 1.202e+17 * exp(-(22896.3582941147) * invT);
  Pr = 1e-6 * alpha / k_f * k_0;
  fPr = Pr / (1.0 + Pr);
  dlnk0dT = (22896.3582941147) * invT2;
  dlogPrdT = log10e * (dlnk0dT - dlnkfdT);
  dlogfPrdT = dlogPrdT / (1.0 + Pr);
  // Troe form
  logPr = log10(Pr);
  Fcent1 = (1. - 0.5) * exp(-T / 1e-30);
  Fcent2 = 0.5 * exp(-T / 1e+30);
  Fcent3 = exp(-1e+100 * invT);
  Fcent = Fcent1 + Fcent2 + Fcent3;
  logFcent = log10(Fcent);
  troe_c = -.4 - .67 * logFcent;
  troe_n = .75 - 1.27 * logFcent;
  troePr_den = 1.0 / (troe_n - .14 * (troe_c + logPr));
  troePr = (troe_c + logPr) * troePr_den;
  troe = 1.0 / (1.0 + troePr * troePr);
  F = exp(M_LN10 * logFcent * troe);
  dlogFcentdT = log10e / Fcent *
                (-Fcent1 / 1e-30 - Fcent2 / 1e+30 + Fcent3 * 1e+100 * invT2);
  dlogFdcn_fac = 2.0 * logFcent * troe * troe * troePr * troePr_den;
  dlogFdc = -troe_n * dlogFdcn_fac * troePr_den;
  dlogFdn = dlogFdcn_fac * troePr;
  dlogFdlogPr = dlogFdc;
  dlogFdT = dlogFcentdT * (troe - 0.67 * dlogFdc - 1.27 * dlogFdn) +
            dlogFdlogPr * dlogPrdT;
  // reverse
  phi_r = (sc[4] * sc[4]);
  Kc = refC * exp(-2.000000 * g_RT[4] + g_RT[8]);
  k_r = k_f / Kc;
  dlnKcdT = invT * (-(h_RT[8]) + (2.000000 * h_RT[4]) - 1.000000);
  dkrdT = (dlnkfdT - dlnKcdT) * k_r;
  // rate of progress
  q_nocor = k_f * phi_f - k_r * phi_r;
  Corr = fPr * F;
  q = Corr * q_nocor;
  dlnCorrdT = ln10 * (dlogfPrdT + dlogFdT);
  dqdT = Corr * (dlnkfdT * k_f * phi_f - dkrdT * phi_r) + dlnCorrdT * q;
  // update wdot
  wdot[4] += 2 * q; // oh
  wdot[8] -= q;     // h2o2
  // for convenience
  k_f *= Corr;
  k_r *= Corr;
  dcdc_fac = 0.0;
  dqdc[0] = dcdc_fac;
  dqdc[1] = 2.5 * dcdc_fac;
  dqdc[2] = dcdc_fac;
  dqdc[3] = dcdc_fac;
  dqdc[4] = dcdc_fac - k_r * 2.000000 * sc[4];
  dqdc[5] = 12 * dcdc_fac;
  dqdc[6] = dcdc_fac;
  dqdc[7] = dcdc_fac;
  dqdc[8] = dcdc_fac + k_f;
  dqdc[9] = 0.64 * dcdc_fac;
  for (int k = 0; k < 10; k++) {
    J[11 * k + 4] += 2 * dqdc[k];
    J[11 * k + 8] -= dqdc[k];
  }
  J[114] += 2 * dqdT; // dwdot[oh]/dT
  J[118] -= dqdT;     // dwdot[h2o2]/dT

  // reaction 4: h2 + M <=> 2 h + M
  // a third-body and non-pressure-fall-off reaction
  // 3-body correction factor
  alpha = mixture + (2.5 - 1) * sc[1] + (12 - 1) * sc[5];
  // forward
  phi_f = sc[1];
  k_f = 45770000000000 * exp(-1.4 * logT - (52535.8199100127) * invT);
  dlnkfdT = -1.4 * invT + (52535.8199100127) * invT2;
  // reverse
  phi_r = (sc[0] * sc[0]);
  Kc = refC * exp(-2.000000 * g_RT[0] + g_RT[1]);
  k_r = k_f / Kc;
  dlnKcdT = invT * (-(h_RT[1]) + (2.000000 * h_RT[0]) - 1.000000);
  dkrdT = (dlnkfdT - dlnKcdT) * k_r;
  // rate of progress
  q_nocor = k_f * phi_f - k_r * phi_r;
  q = alpha * q_nocor;
  dqdT = alpha * (dlnkfdT * k_f * phi_f - dkrdT * phi_r);
  // update wdot
  wdot[0] += 2 * q; // h
  wdot[1] -= q;     // h2
  // for convenience
  k_f *= alpha;
  k_r *= alpha;
  dqdc[0] = q_nocor - k_r * 2.000000 * sc[0];
  dqdc[1] = 2.5 * q_nocor + k_f;
  dqdc[2] = q_nocor;
  dqdc[3] = q_nocor;
  dqdc[4] = q_nocor;
  dqdc[5] = 12 * q_nocor;
  dqdc[6] = q_nocor;
  dqdc[7] = q_nocor;
  dqdc[8] = q_nocor;
  dqdc[9] = q_nocor;
  for (int k = 0; k < 10; k++) {
    J[11 * k + 0] += 2 * dqdc[k];
    J[11 * k + 1] -= dqdc[k];
  }
  J[110] += 2 * dqdT; // dwdot[h]/dT
  J[111] -= dqdT;     // dwdot[h2]/dT

  // reaction 5: 2 o + M <=> o2 + M
  // a third-body and non-pressure-fall-off reaction
  // 3-body correction factor
  alpha = mixture + (2.5 - 1) * sc[1] + (12 - 1) * sc[5] + (0.83 - 1) * sc[9];
  // forward
  phi_f = (sc[2] * sc[2]);
  k_f = 6165 * exp(-0.5 * logT);
  dlnkfdT = -0.5 * invT;
  // reverse
  phi_r = sc[3];
  Kc = refCinv * exp(2.000000 * g_RT[2] - g_RT[3]);
  k_r = k_f / Kc;
  dlnKcdT = invT * (-(2.000000 * h_RT[2]) + (h_RT[3]) + 1.000000);
  dkrdT = (dlnkfdT - dlnKcdT) * k_r;
  // rate of progress
  q_nocor = k_f * phi_f - k_r * phi_r;
  q = alpha * q_nocor;
  dqdT = alpha * (dlnkfdT * k_f * phi_f - dkrdT * phi_r);
  // update wdot
  wdot[2] -= 2 * q; // o
  wdot[3] += q;     // o2
  // for convenience
  k_f *= alpha;
  k_r *= alpha;
  dqdc[0] = q_nocor;
  dqdc[1] = 2.5 * q_nocor;
  dqdc[2] = q_nocor + k_f * 2.000000 * sc[2];
  dqdc[3] = q_nocor - k_r;
  dqdc[4] = q_nocor;
  dqdc[5] = 12 * q_nocor;
  dqdc[6] = q_nocor;
  dqdc[7] = q_nocor;
  dqdc[8] = q_nocor;
  dqdc[9] = 0.83 * q_nocor;
  for (int k = 0; k < 10; k++) {
    J[11 * k + 2] += -2 * dqdc[k];
    J[11 * k + 3] += dqdc[k];
  }
  J[112] += -2 * dqdT; // dwdot[o]/dT
  J[113] += dqdT;      // dwdot[o2]/dT

  // reaction 6: h + o + M <=> oh + M
  // a third-body and non-pressure-fall-off reaction
  // 3-body correction factor
  alpha = mixture + (2.5 - 1) * sc[1] + (12 - 1) * sc[5] + (0.75 - 1) * sc[9];
  // forward
  phi_f = sc[0] * sc[2];
  k_f = 4714000 * exp(-1 * logT);
  dlnkfdT = -1 * invT;
  // reverse
  phi_r = sc[4];
  Kc = refCinv * exp(g_RT[0] + g_RT[2] - g_RT[4]);
  k_r = k_f / Kc;
  dlnKcdT = invT * (-(h_RT[0] + h_RT[2]) + (h_RT[4]) + 1.000000);
  dkrdT = (dlnkfdT - dlnKcdT) * k_r;
  // rate of progress
  q_nocor = k_f * phi_f - k_r * phi_r;
  q = alpha * q_nocor;
  dqdT = alpha * (dlnkfdT * k_f * phi_f - dkrdT * phi_r);
  // update wdot
  wdot[0] -= q; // h
  wdot[2] -= q; // o
  wdot[4] += q; // oh
  // for convenience
  k_f *= alpha;
  k_r *= alpha;
  dqdc[0] = q_nocor + k_f * sc[2];
  dqdc[1] = 2.5 * q_nocor;
  dqdc[2] = q_nocor + k_f * sc[0];
  dqdc[3] = q_nocor;
  dqdc[4] = q_nocor - k_r;
  dqdc[5] = 12 * q_nocor;
  dqdc[6] = q_nocor;
  dqdc[7] = q_nocor;
  dqdc[8] = q_nocor;
  dqdc[9] = 0.75 * q_nocor;
  for (int k = 0; k < 10; k++) {
    J[11 * k + 0] -= dqdc[k];
    J[11 * k + 2] -= dqdc[k];
    J[11 * k + 4] += dqdc[k];
  }
  J[110] -= dqdT; // dwdot[h]/dT
  J[112] -= dqdT; // dwdot[o]/dT
  J[114] += dqdT; // dwdot[oh]/dT

  // reaction 7: h + oh + M <=> h2o + M
  // a third-body and non-pressure-fall-off reaction
  // 3-body correction factor
  alpha = mixture + (0.73 - 1) * sc[1] + (12 - 1) * sc[5] + (0.38 - 1) * sc[9];
  // forward
  phi_f = sc[0] * sc[4];
  k_f = 45000000000 * exp(-2 * logT);
  dlnkfdT = -2 * invT;
  // reverse
  phi_r = sc[5];
  Kc = refCinv * exp(g_RT[0] + g_RT[4] - g_RT[5]);
  k_r = k_f / Kc;
  dlnKcdT = invT * (-(h_RT[0] + h_RT[4]) + (h_RT[5]) + 1.000000);
  dkrdT = (dlnkfdT - dlnKcdT) * k_r;
  // rate of progress
  q_nocor = k_f * phi_f - k_r * phi_r;
  q = alpha * q_nocor;
  dqdT = alpha * (dlnkfdT * k_f * phi_f - dkrdT * phi_r);
  // update wdot
  wdot[0] -= q; // h
  wdot[4] -= q; // oh
  wdot[5] += q; // h2o
  // for convenience
  k_f *= alpha;
  k_r *= alpha;
  dqdc[0] = q_nocor + k_f * sc[4];
  dqdc[1] = 0.73 * q_nocor;
  dqdc[2] = q_nocor;
  dqdc[3] = q_nocor;
  dqdc[4] = q_nocor + k_f * sc[0];
  dqdc[5] = 12 * q_nocor - k_r;
  dqdc[6] = q_nocor;
  dqdc[7] = q_nocor;
  dqdc[8] = q_nocor;
  dqdc[9] = 0.38 * q_nocor;
  for (int k = 0; k < 10; k++) {
    J[11 * k + 0] -= dqdc[k];
    J[11 * k + 4] -= dqdc[k];
    J[11 * k + 5] += dqdc[k];
  }
  J[110] -= dqdT; // dwdot[h]/dT
  J[114] -= dqdT; // dwdot[oh]/dT
  J[115] += dqdT; // dwdot[h2o]/dT

  // reaction 0: h + o2 <=> o + oh
  // a non-third-body and non-pressure-fall-off reaction
  // forward
  phi_f = sc[0] * sc[3];
  k_f = 191500000 * exp(-(8272.88198582959) * invT);
  dlnkfdT = (8272.88198582959) * invT2;
  // reverse
  phi_r = sc[2] * sc[4];
  Kc = exp(g_RT[0] - g_RT[2] + g_RT[3] - g_RT[4]);
  k_r = k_f / Kc;
  dlnKcdT = invT * (-(h_RT[0] + h_RT[3]) + (h_RT[2] + h_RT[4]));
  dkrdT = (dlnkfdT - dlnKcdT) * k_r;
  // rate of progress
  q = k_f * phi_f - k_r * phi_r;
  dqdT = (dlnkfdT * k_f * phi_f - dkrdT * phi_r);
  // update wdot
  wdot[0] -= q; // h
  wdot[2] += q; // o
  wdot[3] -= q; // o2
  wdot[4] += q; // oh
  // d()/d[h]
  dqdci = +k_f * sc[3];
  J[0] -= dqdci; // dwdot[h]/d[h]
  J[2] += dqdci; // dwdot[o]/d[h]
  J[3] -= dqdci; // dwdot[o2]/d[h]
  J[4] += dqdci; // dwdot[oh]/d[h]
  // d()/d[o]
  dqdci = -k_r * sc[4];
  J[22] -= dqdci; // dwdot[h]/d[o]
  J[24] += dqdci; // dwdot[o]/d[o]
  J[25] -= dqdci; // dwdot[o2]/d[o]
  J[26] += dqdci; // dwdot[oh]/d[o]
  // d()/d[o2]
  dqdci = +k_f * sc[0];
  J[33] -= dqdci; // dwdot[h]/d[o2]
  J[35] += dqdci; // dwdot[o]/d[o2]
  J[36] -= dqdci; // dwdot[o2]/d[o2]
  J[37] += dqdci; // dwdot[oh]/d[o2]
  // d()/d[oh]
  dqdci = -k_r * sc[2];
  J[44] -= dqdci; // dwdot[h]/d[oh]
  J[46] += dqdci; // dwdot[o]/d[oh]
  J[47] -= dqdci; // dwdot[o2]/d[oh]
  J[48] += dqdci; // dwdot[oh]/d[oh]
  // d()/dT
  J[110] -= dqdT; // dwdot[h]/dT
  J[112] += dqdT; // dwdot[o]/dT
  J[113] -= dqdT; // dwdot[o2]/dT
  J[114] += dqdT; // dwdot[oh]/dT

  // reaction 1: h2 + o <=> h + oh
  // a non-third-body and non-pressure-fall-off reaction
  // forward
  phi_f = sc[1] * sc[2];
  k_f = 0.0508 * exp(2.67 * logT - (3166.2392612433) * invT);
  dlnkfdT = 2.67 * invT + (3166.2392612433) * invT2;
  // reverse
  phi_r = sc[0] * sc[4];
  Kc = exp(-g_RT[0] + g_RT[1] + g_RT[2] - g_RT[4]);
  k_r = k_f / Kc;
  dlnKcdT = invT * (-(h_RT[1] + h_RT[2]) + (h_RT[0] + h_RT[4]));
  dkrdT = (dlnkfdT - dlnKcdT) * k_r;
  // rate of progress
  q = k_f * phi_f - k_r * phi_r;
  dqdT = (dlnkfdT * k_f * phi_f - dkrdT * phi_r);
  // update wdot
  wdot[0] += q; // h
  wdot[1] -= q; // h2
  wdot[2] -= q; // o
  wdot[4] += q; // oh
  // d()/d[h]
  dqdci = -k_r * sc[4];
  J[0] += dqdci; // dwdot[h]/d[h]
  J[1] -= dqdci; // dwdot[h2]/d[h]
  J[2] -= dqdci; // dwdot[o]/d[h]
  J[4] += dqdci; // dwdot[oh]/d[h]
  // d()/d[h2]
  dqdci = +k_f * sc[2];
  J[11] += dqdci; // dwdot[h]/d[h2]
  J[12] -= dqdci; // dwdot[h2]/d[h2]
  J[13] -= dqdci; // dwdot[o]/d[h2]
  J[15] += dqdci; // dwdot[oh]/d[h2]
  // d()/d[o]
  dqdci = +k_f * sc[1];
  J[22] += dqdci; // dwdot[h]/d[o]
  J[23] -= dqdci; // dwdot[h2]/d[o]
  J[24] -= dqdci; // dwdot[o]/d[o]
  J[26] += dqdci; // dwdot[oh]/d[o]
  // d()/d[oh]
  dqdci = -k_r * sc[0];
  J[44] += dqdci; // dwdot[h]/d[oh]
  J[45] -= dqdci; // dwdot[h2]/d[oh]
  J[46] -= dqdci; // dwdot[o]/d[oh]
  J[48] += dqdci; // dwdot[oh]/d[oh]
  // d()/dT
  J[110] += dqdT; // dwdot[h]/dT
  J[111] -= dqdT; // dwdot[h2]/dT
  J[112] -= dqdT; // dwdot[o]/dT
  J[114] += dqdT; // dwdot[oh]/dT

  // reaction 2: h2 + oh <=> h + h2o
  // a non-third-body and non-pressure-fall-off reaction
  // forward
  phi_f = sc[1] * sc[4];
  k_f = 216 * exp(1.51 * logT - (1726.03316371019) * invT);
  dlnkfdT = 1.51 * invT + (1726.03316371019) * invT2;
  // reverse
  phi_r = sc[0] * sc[5];
  Kc = exp(-g_RT[0] + g_RT[1] + g_RT[4] - g_RT[5]);
  k_r = k_f / Kc;
  dlnKcdT = invT * (-(h_RT[1] + h_RT[4]) + (h_RT[0] + h_RT[5]));
  dkrdT = (dlnkfdT - dlnKcdT) * k_r;
  // rate of progress
  q = k_f * phi_f - k_r * phi_r;
  dqdT = (dlnkfdT * k_f * phi_f - dkrdT * phi_r);
  // update wdot
  wdot[0] += q; // h
  wdot[1] -= q; // h2
  wdot[4] -= q; // oh
  wdot[5] += q; // h2o
  // d()/d[h]
  dqdci = -k_r * sc[5];
  J[0] += dqdci; // dwdot[h]/d[h]
  J[1] -= dqdci; // dwdot[h2]/d[h]
  J[4] -= dqdci; // dwdot[oh]/d[h]
  J[5] += dqdci; // dwdot[h2o]/d[h]
  // d()/d[h2]
  dqdci = +k_f * sc[4];
  J[11] += dqdci; // dwdot[h]/d[h2]
  J[12] -= dqdci; // dwdot[h2]/d[h2]
  J[15] -= dqdci; // dwdot[oh]/d[h2]
  J[16] += dqdci; // dwdot[h2o]/d[h2]
  // d()/d[oh]
  dqdci = +k_f * sc[1];
  J[44] += dqdci; // dwdot[h]/d[oh]
  J[45] -= dqdci; // dwdot[h2]/d[oh]
  J[48] -= dqdci; // dwdot[oh]/d[oh]
  J[49] += dqdci; // dwdot[h2o]/d[oh]
  // d()/d[h2o]
  dqdci = -k_r * sc[0];
  J[55] += dqdci; // dwdot[h]/d[h2o]
  J[56] -= dqdci; // dwdot[h2]/d[h2o]
  J[59] -= dqdci; // dwdot[oh]/d[h2o]
  J[60] += dqdci; // dwdot[h2o]/d[h2o]
  // d()/dT
  J[110] += dqdT; // dwdot[h]/dT
  J[111] -= dqdT; // dwdot[h2]/dT
  J[114] -= dqdT; // dwdot[oh]/dT
  J[115] += dqdT; // dwdot[h2o]/dT

  // reaction 3: h2o + o <=> 2 oh
  // a non-third-body and non-pressure-fall-off reaction
  // forward
  phi_f = sc[2] * sc[5];
  k_f = 2.97 * exp(2.02 * logT - (6743.10332178324) * invT);
  dlnkfdT = 2.02 * invT + (6743.10332178324) * invT2;
  // reverse
  phi_r = (sc[4] * sc[4]);
  Kc = exp(g_RT[2] - 2.000000 * g_RT[4] + g_RT[5]);
  k_r = k_f / Kc;
  dlnKcdT = invT * (-(h_RT[2] + h_RT[5]) + (2.000000 * h_RT[4]));
  dkrdT = (dlnkfdT - dlnKcdT) * k_r;
  // rate of progress
  q = k_f * phi_f - k_r * phi_r;
  dqdT = (dlnkfdT * k_f * phi_f - dkrdT * phi_r);
  // update wdot
  wdot[2] -= q;     // o
  wdot[4] += 2 * q; // oh
  wdot[5] -= q;     // h2o
  // d()/d[o]
  dqdci = +k_f * sc[5];
  J[24] -= dqdci;     // dwdot[o]/d[o]
  J[26] += 2 * dqdci; // dwdot[oh]/d[o]
  J[27] -= dqdci;     // dwdot[h2o]/d[o]
  // d()/d[oh]
  dqdci = -k_r * 2.000000 * sc[4];
  J[46] -= dqdci;     // dwdot[o]/d[oh]
  J[48] += 2 * dqdci; // dwdot[oh]/d[oh]
  J[49] -= dqdci;     // dwdot[h2o]/d[oh]
  // d()/d[h2o]
  dqdci = +k_f * sc[2];
  J[57] -= dqdci;     // dwdot[o]/d[h2o]
  J[59] += 2 * dqdci; // dwdot[oh]/d[h2o]
  J[60] -= dqdci;     // dwdot[h2o]/d[h2o]
  // d()/dT
  J[112] -= dqdT;     // dwdot[o]/dT
  J[114] += 2 * dqdT; // dwdot[oh]/dT
  J[115] -= dqdT;     // dwdot[h2o]/dT

  // reaction 9: h + ho2 <=> h2 + o2
  // a non-third-body and non-pressure-fall-off reaction
  // forward
  phi_f = sc[0] * sc[7];
  k_f = 16600000 * exp(-(414.147315957284) * invT);
  dlnkfdT = (414.147315957284) * invT2;
  // reverse
  phi_r = sc[1] * sc[3];
  Kc = exp(g_RT[0] - g_RT[1] - g_RT[3] + g_RT[7]);
  k_r = k_f / Kc;
  dlnKcdT = invT * (-(h_RT[0] + h_RT[7]) + (h_RT[1] + h_RT[3]));
  dkrdT = (dlnkfdT - dlnKcdT) * k_r;
  // rate of progress
  q = k_f * phi_f - k_r * phi_r;
  dqdT = (dlnkfdT * k_f * phi_f - dkrdT * phi_r);
  // update wdot
  wdot[0] -= q; // h
  wdot[1] += q; // h2
  wdot[3] += q; // o2
  wdot[7] -= q; // ho2
  // d()/d[h]
  dqdci = +k_f * sc[7];
  J[0] -= dqdci; // dwdot[h]/d[h]
  J[1] += dqdci; // dwdot[h2]/d[h]
  J[3] += dqdci; // dwdot[o2]/d[h]
  J[7] -= dqdci; // dwdot[ho2]/d[h]
  // d()/d[h2]
  dqdci = -k_r * sc[3];
  J[11] -= dqdci; // dwdot[h]/d[h2]
  J[12] += dqdci; // dwdot[h2]/d[h2]
  J[14] += dqdci; // dwdot[o2]/d[h2]
  J[18] -= dqdci; // dwdot[ho2]/d[h2]
  // d()/d[o2]
  dqdci = -k_r * sc[1];
  J[33] -= dqdci; // dwdot[h]/d[o2]
  J[34] += dqdci; // dwdot[h2]/d[o2]
  J[36] += dqdci; // dwdot[o2]/d[o2]
  J[40] -= dqdci; // dwdot[ho2]/d[o2]
  // d()/d[ho2]
  dqdci = +k_f * sc[0];
  J[77] -= dqdci; // dwdot[h]/d[ho2]
  J[78] += dqdci; // dwdot[h2]/d[ho2]
  J[80] += dqdci; // dwdot[o2]/d[ho2]
  J[84] -= dqdci; // dwdot[ho2]/d[ho2]
  // d()/dT
  J[110] -= dqdT; // dwdot[h]/dT
  J[111] += dqdT; // dwdot[h2]/dT
  J[113] += dqdT; // dwdot[o2]/dT
  J[117] -= dqdT; // dwdot[ho2]/dT

  // reaction 10: h + ho2 <=> 2 oh
  // a non-third-body and non-pressure-fall-off reaction
  // forward
  phi_f = sc[0] * sc[7];
  k_f = 70790000 * exp(-(148.448916412392) * invT);
  dlnkfdT = (148.448916412392) * invT2;
  // reverse
  phi_r = (sc[4] * sc[4]);
  Kc = exp(g_RT[0] - 2.000000 * g_RT[4] + g_RT[7]);
  k_r = k_f / Kc;
  dlnKcdT = invT * (-(h_RT[0] + h_RT[7]) + (2.000000 * h_RT[4]));
  dkrdT = (dlnkfdT - dlnKcdT) * k_r;
  // rate of progress
  q = k_f * phi_f - k_r * phi_r;
  dqdT = (dlnkfdT * k_f * phi_f - dkrdT * phi_r);
  // update wdot
  wdot[0] -= q;     // h
  wdot[4] += 2 * q; // oh
  wdot[7] -= q;     // ho2
  // d()/d[h]
  dqdci = +k_f * sc[7];
  J[0] -= dqdci;     // dwdot[h]/d[h]
  J[4] += 2 * dqdci; // dwdot[oh]/d[h]
  J[7] -= dqdci;     // dwdot[ho2]/d[h]
  // d()/d[oh]
  dqdci = -k_r * 2.000000 * sc[4];
  J[44] -= dqdci;     // dwdot[h]/d[oh]
  J[48] += 2 * dqdci; // dwdot[oh]/d[oh]
  J[51] -= dqdci;     // dwdot[ho2]/d[oh]
  // d()/d[ho2]
  dqdci = +k_f * sc[0];
  J[77] -= dqdci;     // dwdot[h]/d[ho2]
  J[81] += 2 * dqdci; // dwdot[oh]/d[ho2]
  J[84] -= dqdci;     // dwdot[ho2]/d[ho2]
  // d()/dT
  J[110] -= dqdT;     // dwdot[h]/dT
  J[114] += 2 * dqdT; // dwdot[oh]/dT
  J[117] -= dqdT;     // dwdot[ho2]/dT

  // reaction 11: ho2 + o <=> o2 + oh
  // a non-third-body and non-pressure-fall-off reaction
  // forward
  phi_f = sc[2] * sc[7];
  k_f = 32500000;
  dlnkfdT = 0.0;
  // reverse
  phi_r = sc[3] * sc[4];
  Kc = exp(g_RT[2] - g_RT[3] - g_RT[4] + g_RT[7]);
  k_r = k_f / Kc;
  dlnKcdT = invT * (-(h_RT[2] + h_RT[7]) + (h_RT[3] + h_RT[4]));
  dkrdT = (dlnkfdT - dlnKcdT) * k_r;
  // rate of progress
  q = k_f * phi_f - k_r * phi_r;
  dqdT = (dlnkfdT * k_f * phi_f - dkrdT * phi_r);
  // update wdot
  wdot[2] -= q; // o
  wdot[3] += q; // o2
  wdot[4] += q; // oh
  wdot[7] -= q; // ho2
  // d()/d[o]
  dqdci = +k_f * sc[7];
  J[24] -= dqdci; // dwdot[o]/d[o]
  J[25] += dqdci; // dwdot[o2]/d[o]
  J[26] += dqdci; // dwdot[oh]/d[o]
  J[29] -= dqdci; // dwdot[ho2]/d[o]
  // d()/d[o2]
  dqdci = -k_r * sc[4];
  J[35] -= dqdci; // dwdot[o]/d[o2]
  J[36] += dqdci; // dwdot[o2]/d[o2]
  J[37] += dqdci; // dwdot[oh]/d[o2]
  J[40] -= dqdci; // dwdot[ho2]/d[o2]
  // d()/d[oh]
  dqdci = -k_r * sc[3];
  J[46] -= dqdci; // dwdot[o]/d[oh]
  J[47] += dqdci; // dwdot[o2]/d[oh]
  J[48] += dqdci; // dwdot[oh]/d[oh]
  J[51] -= dqdci; // dwdot[ho2]/d[oh]
  // d()/d[ho2]
  dqdci = +k_f * sc[2];
  J[79] -= dqdci; // dwdot[o]/d[ho2]
  J[80] += dqdci; // dwdot[o2]/d[ho2]
  J[81] += dqdci; // dwdot[oh]/d[ho2]
  J[84] -= dqdci; // dwdot[ho2]/d[ho2]
  // d()/dT
  J[112] -= dqdT; // dwdot[o]/dT
  J[113] += dqdT; // dwdot[o2]/dT
  J[114] += dqdT; // dwdot[oh]/dT
  J[117] -= dqdT; // dwdot[ho2]/dT

  // reaction 12: ho2 + oh <=> h2o + o2
  // a non-third-body and non-pressure-fall-off reaction
  // forward
  phi_f = sc[4] * sc[7];
  k_f = 28900000 * exp(-(-250.098682904946) * invT);
  dlnkfdT = (-250.098682904946) * invT2;
  // reverse
  phi_r = sc[3] * sc[5];
  Kc = exp(-g_RT[3] + g_RT[4] - g_RT[5] + g_RT[7]);
  k_r = k_f / Kc;
  dlnKcdT = invT * (-(h_RT[4] + h_RT[7]) + (h_RT[3] + h_RT[5]));
  dkrdT = (dlnkfdT - dlnKcdT) * k_r;
  // rate of progress
  q = k_f * phi_f - k_r * phi_r;
  dqdT = (dlnkfdT * k_f * phi_f - dkrdT * phi_r);
  // update wdot
  wdot[3] += q; // o2
  wdot[4] -= q; // oh
  wdot[5] += q; // h2o
  wdot[7] -= q; // ho2
  // d()/d[o2]
  dqdci = -k_r * sc[5];
  J[36] += dqdci; // dwdot[o2]/d[o2]
  J[37] -= dqdci; // dwdot[oh]/d[o2]
  J[38] += dqdci; // dwdot[h2o]/d[o2]
  J[40] -= dqdci; // dwdot[ho2]/d[o2]
  // d()/d[oh]
  dqdci = +k_f * sc[7];
  J[47] += dqdci; // dwdot[o2]/d[oh]
  J[48] -= dqdci; // dwdot[oh]/d[oh]
  J[49] += dqdci; // dwdot[h2o]/d[oh]
  J[51] -= dqdci; // dwdot[ho2]/d[oh]
  // d()/d[h2o]
  dqdci = -k_r * sc[3];
  J[58] += dqdci; // dwdot[o2]/d[h2o]
  J[59] -= dqdci; // dwdot[oh]/d[h2o]
  J[60] += dqdci; // dwdot[h2o]/d[h2o]
  J[62] -= dqdci; // dwdot[ho2]/d[h2o]
  // d()/d[ho2]
  dqdci = +k_f * sc[4];
  J[80] += dqdci; // dwdot[o2]/d[ho2]
  J[81] -= dqdci; // dwdot[oh]/d[ho2]
  J[82] += dqdci; // dwdot[h2o]/d[ho2]
  J[84] -= dqdci; // dwdot[ho2]/d[ho2]
  // d()/dT
  J[113] += dqdT; // dwdot[o2]/dT
  J[114] -= dqdT; // dwdot[oh]/dT
  J[115] += dqdT; // dwdot[h2o]/dT
  J[117] -= dqdT; // dwdot[ho2]/dT

  // reaction 13: 2 ho2 <=> h2o2 + o2
  // a non-third-body and non-pressure-fall-off reaction
  // forward
  phi_f = (sc[7] * sc[7]);
  k_f = 420000000 * exp(-(6028.53565634054) * invT);
  dlnkfdT = (6028.53565634054) * invT2;
  // reverse
  phi_r = sc[3] * sc[8];
  Kc = exp(-g_RT[3] + 2.000000 * g_RT[7] - g_RT[8]);
  k_r = k_f / Kc;
  dlnKcdT = invT * (-(2.000000 * h_RT[7]) + (h_RT[3] + h_RT[8]));
  dkrdT = (dlnkfdT - dlnKcdT) * k_r;
  // rate of progress
  q = k_f * phi_f - k_r * phi_r;
  dqdT = (dlnkfdT * k_f * phi_f - dkrdT * phi_r);
  // update wdot
  wdot[3] += q;     // o2
  wdot[7] -= 2 * q; // ho2
  wdot[8] += q;     // h2o2
  // d()/d[o2]
  dqdci = -k_r * sc[8];
  J[36] += dqdci;      // dwdot[o2]/d[o2]
  J[40] += -2 * dqdci; // dwdot[ho2]/d[o2]
  J[41] += dqdci;      // dwdot[h2o2]/d[o2]
  // d()/d[ho2]
  dqdci = +k_f * 2.000000 * sc[7];
  J[80] += dqdci;      // dwdot[o2]/d[ho2]
  J[84] += -2 * dqdci; // dwdot[ho2]/d[ho2]
  J[85] += dqdci;      // dwdot[h2o2]/d[ho2]
  // d()/d[h2o2]
  dqdci = -k_r * sc[3];
  J[91] += dqdci;      // dwdot[o2]/d[h2o2]
  J[95] += -2 * dqdci; // dwdot[ho2]/d[h2o2]
  J[96] += dqdci;      // dwdot[h2o2]/d[h2o2]
  // d()/dT
  J[113] += dqdT;      // dwdot[o2]/dT
  J[117] += -2 * dqdT; // dwdot[ho2]/dT
  J[118] += dqdT;      // dwdot[h2o2]/dT

  // reaction 14: 2 ho2 <=> h2o2 + o2
  // a non-third-body and non-pressure-fall-off reaction
  // forward
  phi_f = (sc[7] * sc[7]);
  k_f = 130000 * exp(-(-819.739948595888) * invT);
  dlnkfdT = (-819.739948595888) * invT2;
  // reverse
  phi_r = sc[3] * sc[8];
  Kc = exp(-g_RT[3] + 2.000000 * g_RT[7] - g_RT[8]);
  k_r = k_f / Kc;
  dlnKcdT = invT * (-(2.000000 * h_RT[7]) + (h_RT[3] + h_RT[8]));
  dkrdT = (dlnkfdT - dlnKcdT) * k_r;
  // rate of progress
  q = k_f * phi_f - k_r * phi_r;
  dqdT = (dlnkfdT * k_f * phi_f - dkrdT * phi_r);
  // update wdot
  wdot[3] += q;     // o2
  wdot[7] -= 2 * q; // ho2
  wdot[8] += q;     // h2o2
  // d()/d[o2]
  dqdci = -k_r * sc[8];
  J[36] += dqdci;      // dwdot[o2]/d[o2]
  J[40] += -2 * dqdci; // dwdot[ho2]/d[o2]
  J[41] += dqdci;      // dwdot[h2o2]/d[o2]
  // d()/d[ho2]
  dqdci = +k_f * 2.000000 * sc[7];
  J[80] += dqdci;      // dwdot[o2]/d[ho2]
  J[84] += -2 * dqdci; // dwdot[ho2]/d[ho2]
  J[85] += dqdci;      // dwdot[h2o2]/d[ho2]
  // d()/d[h2o2]
  dqdci = -k_r * sc[3];
  J[91] += dqdci;      // dwdot[o2]/d[h2o2]
  J[95] += -2 * dqdci; // dwdot[ho2]/d[h2o2]
  J[96] += dqdci;      // dwdot[h2o2]/d[h2o2]
  // d()/dT
  J[113] += dqdT;      // dwdot[o2]/dT
  J[117] += -2 * dqdT; // dwdot[ho2]/dT
  J[118] += dqdT;      // dwdot[h2o2]/dT

  // reaction 16: h + h2o2 <=> h2o + oh
  // a non-third-body and non-pressure-fall-off reaction
  // forward
  phi_f = sc[0] * sc[8];
  k_f = 24100000 * exp(-(1997.77016324474) * invT);
  dlnkfdT = (1997.77016324474) * invT2;
  // reverse
  phi_r = sc[4] * sc[5];
  Kc = exp(g_RT[0] - g_RT[4] - g_RT[5] + g_RT[8]);
  k_r = k_f / Kc;
  dlnKcdT = invT * (-(h_RT[0] + h_RT[8]) + (h_RT[4] + h_RT[5]));
  dkrdT = (dlnkfdT - dlnKcdT) * k_r;
  // rate of progress
  q = k_f * phi_f - k_r * phi_r;
  dqdT = (dlnkfdT * k_f * phi_f - dkrdT * phi_r);
  // update wdot
  wdot[0] -= q; // h
  wdot[4] += q; // oh
  wdot[5] += q; // h2o
  wdot[8] -= q; // h2o2
  // d()/d[h]
  dqdci = +k_f * sc[8];
  J[0] -= dqdci; // dwdot[h]/d[h]
  J[4] += dqdci; // dwdot[oh]/d[h]
  J[5] += dqdci; // dwdot[h2o]/d[h]
  J[8] -= dqdci; // dwdot[h2o2]/d[h]
  // d()/d[oh]
  dqdci = -k_r * sc[5];
  J[44] -= dqdci; // dwdot[h]/d[oh]
  J[48] += dqdci; // dwdot[oh]/d[oh]
  J[49] += dqdci; // dwdot[h2o]/d[oh]
  J[52] -= dqdci; // dwdot[h2o2]/d[oh]
  // d()/d[h2o]
  dqdci = -k_r * sc[4];
  J[55] -= dqdci; // dwdot[h]/d[h2o]
  J[59] += dqdci; // dwdot[oh]/d[h2o]
  J[60] += dqdci; // dwdot[h2o]/d[h2o]
  J[63] -= dqdci; // dwdot[h2o2]/d[h2o]
  // d()/d[h2o2]
  dqdci = +k_f * sc[0];
  J[88] -= dqdci; // dwdot[h]/d[h2o2]
  J[92] += dqdci; // dwdot[oh]/d[h2o2]
  J[93] += dqdci; // dwdot[h2o]/d[h2o2]
  J[96] -= dqdci; // dwdot[h2o2]/d[h2o2]
  // d()/dT
  J[110] -= dqdT; // dwdot[h]/dT
  J[114] += dqdT; // dwdot[oh]/dT
  J[115] += dqdT; // dwdot[h2o]/dT
  J[118] -= dqdT; // dwdot[h2o2]/dT

  // reaction 17: h + h2o2 <=> h2 + ho2
  // a non-third-body and non-pressure-fall-off reaction
  // forward
  phi_f = sc[0] * sc[8];
  k_f = 60250000 * exp(-(4000.57249314752) * invT);
  dlnkfdT = (4000.57249314752) * invT2;
  // reverse
  phi_r = sc[1] * sc[7];
  Kc = exp(g_RT[0] - g_RT[1] - g_RT[7] + g_RT[8]);
  k_r = k_f / Kc;
  dlnKcdT = invT * (-(h_RT[0] + h_RT[8]) + (h_RT[1] + h_RT[7]));
  dkrdT = (dlnkfdT - dlnKcdT) * k_r;
  // rate of progress
  q = k_f * phi_f - k_r * phi_r;
  dqdT = (dlnkfdT * k_f * phi_f - dkrdT * phi_r);
  // update wdot
  wdot[0] -= q; // h
  wdot[1] += q; // h2
  wdot[7] += q; // ho2
  wdot[8] -= q; // h2o2
  // d()/d[h]
  dqdci = +k_f * sc[8];
  J[0] -= dqdci; // dwdot[h]/d[h]
  J[1] += dqdci; // dwdot[h2]/d[h]
  J[7] += dqdci; // dwdot[ho2]/d[h]
  J[8] -= dqdci; // dwdot[h2o2]/d[h]
  // d()/d[h2]
  dqdci = -k_r * sc[7];
  J[11] -= dqdci; // dwdot[h]/d[h2]
  J[12] += dqdci; // dwdot[h2]/d[h2]
  J[18] += dqdci; // dwdot[ho2]/d[h2]
  J[19] -= dqdci; // dwdot[h2o2]/d[h2]
  // d()/d[ho2]
  dqdci = -k_r * sc[1];
  J[77] -= dqdci; // dwdot[h]/d[ho2]
  J[78] += dqdci; // dwdot[h2]/d[ho2]
  J[84] += dqdci; // dwdot[ho2]/d[ho2]
  J[85] -= dqdci; // dwdot[h2o2]/d[ho2]
  // d()/d[h2o2]
  dqdci = +k_f * sc[0];
  J[88] -= dqdci; // dwdot[h]/d[h2o2]
  J[89] += dqdci; // dwdot[h2]/d[h2o2]
  J[95] += dqdci; // dwdot[ho2]/d[h2o2]
  J[96] -= dqdci; // dwdot[h2o2]/d[h2o2]
  // d()/dT
  J[110] -= dqdT; // dwdot[h]/dT
  J[111] += dqdT; // dwdot[h2]/dT
  J[117] += dqdT; // dwdot[ho2]/dT
  J[118] -= dqdT; // dwdot[h2o2]/dT

  // reaction 18: h2o2 + o <=> ho2 + oh
  // a non-third-body and non-pressure-fall-off reaction
  // forward
  phi_f = sc[2] * sc[8];
  k_f = 9.55 * exp(2 * logT - (1997.77016324474) * invT);
  dlnkfdT = 2 * invT + (1997.77016324474) * invT2;
  // reverse
  phi_r = sc[4] * sc[7];
  Kc = exp(g_RT[2] - g_RT[4] - g_RT[7] + g_RT[8]);
  k_r = k_f / Kc;
  dlnKcdT = invT * (-(h_RT[2] + h_RT[8]) + (h_RT[4] + h_RT[7]));
  dkrdT = (dlnkfdT - dlnKcdT) * k_r;
  // rate of progress
  q = k_f * phi_f - k_r * phi_r;
  dqdT = (dlnkfdT * k_f * phi_f - dkrdT * phi_r);
  // update wdot
  wdot[2] -= q; // o
  wdot[4] += q; // oh
  wdot[7] += q; // ho2
  wdot[8] -= q; // h2o2
  // d()/d[o]
  dqdci = +k_f * sc[8];
  J[24] -= dqdci; // dwdot[o]/d[o]
  J[26] += dqdci; // dwdot[oh]/d[o]
  J[29] += dqdci; // dwdot[ho2]/d[o]
  J[30] -= dqdci; // dwdot[h2o2]/d[o]
  // d()/d[oh]
  dqdci = -k_r * sc[7];
  J[46] -= dqdci; // dwdot[o]/d[oh]
  J[48] += dqdci; // dwdot[oh]/d[oh]
  J[51] += dqdci; // dwdot[ho2]/d[oh]
  J[52] -= dqdci; // dwdot[h2o2]/d[oh]
  // d()/d[ho2]
  dqdci = -k_r * sc[4];
  J[79] -= dqdci; // dwdot[o]/d[ho2]
  J[81] += dqdci; // dwdot[oh]/d[ho2]
  J[84] += dqdci; // dwdot[ho2]/d[ho2]
  J[85] -= dqdci; // dwdot[h2o2]/d[ho2]
  // d()/d[h2o2]
  dqdci = +k_f * sc[2];
  J[90] -= dqdci; // dwdot[o]/d[h2o2]
  J[92] += dqdci; // dwdot[oh]/d[h2o2]
  J[95] += dqdci; // dwdot[ho2]/d[h2o2]
  J[96] -= dqdci; // dwdot[h2o2]/d[h2o2]
  // d()/dT
  J[112] -= dqdT; // dwdot[o]/dT
  J[114] += dqdT; // dwdot[oh]/dT
  J[117] += dqdT; // dwdot[ho2]/dT
  J[118] -= dqdT; // dwdot[h2o2]/dT

  // reaction 19: h2o2 + oh <=> h2o + ho2
  // a non-third-body and non-pressure-fall-off reaction
  // forward
  phi_f = sc[4] * sc[8];
  k_f = 1000000;
  dlnkfdT = 0.0;
  // reverse
  phi_r = sc[5] * sc[7];
  Kc = exp(g_RT[4] - g_RT[5] - g_RT[7] + g_RT[8]);
  k_r = k_f / Kc;
  dlnKcdT = invT * (-(h_RT[4] + h_RT[8]) + (h_RT[5] + h_RT[7]));
  dkrdT = (dlnkfdT - dlnKcdT) * k_r;
  // rate of progress
  q = k_f * phi_f - k_r * phi_r;
  dqdT = (dlnkfdT * k_f * phi_f - dkrdT * phi_r);
  // update wdot
  wdot[4] -= q; // oh
  wdot[5] += q; // h2o
  wdot[7] += q; // ho2
  wdot[8] -= q; // h2o2
  // d()/d[oh]
  dqdci = +k_f * sc[8];
  J[48] -= dqdci; // dwdot[oh]/d[oh]
  J[49] += dqdci; // dwdot[h2o]/d[oh]
  J[51] += dqdci; // dwdot[ho2]/d[oh]
  J[52] -= dqdci; // dwdot[h2o2]/d[oh]
  // d()/d[h2o]
  dqdci = -k_r * sc[7];
  J[59] -= dqdci; // dwdot[oh]/d[h2o]
  J[60] += dqdci; // dwdot[h2o]/d[h2o]
  J[62] += dqdci; // dwdot[ho2]/d[h2o]
  J[63] -= dqdci; // dwdot[h2o2]/d[h2o]
  // d()/d[ho2]
  dqdci = -k_r * sc[5];
  J[81] -= dqdci; // dwdot[oh]/d[ho2]
  J[82] += dqdci; // dwdot[h2o]/d[ho2]
  J[84] += dqdci; // dwdot[ho2]/d[ho2]
  J[85] -= dqdci; // dwdot[h2o2]/d[ho2]
  // d()/d[h2o2]
  dqdci = +k_f * sc[4];
  J[92] -= dqdci; // dwdot[oh]/d[h2o2]
  J[93] += dqdci; // dwdot[h2o]/d[h2o2]
  J[95] += dqdci; // dwdot[ho2]/d[h2o2]
  J[96] -= dqdci; // dwdot[h2o2]/d[h2o2]
  // d()/dT
  J[114] -= dqdT; // dwdot[oh]/dT
  J[115] += dqdT; // dwdot[h2o]/dT
  J[117] += dqdT; // dwdot[ho2]/dT
  J[118] -= dqdT; // dwdot[h2o2]/dT

  // reaction 20: h2o2 + oh <=> h2o + ho2
  // a non-third-body and non-pressure-fall-off reaction
  // forward
  phi_f = sc[4] * sc[8];
  k_f = 580000000 * exp(-(4809.24167509571) * invT);
  dlnkfdT = (4809.24167509571) * invT2;
  // reverse
  phi_r = sc[5] * sc[7];
  Kc = exp(g_RT[4] - g_RT[5] - g_RT[7] + g_RT[8]);
  k_r = k_f / Kc;
  dlnKcdT = invT * (-(h_RT[4] + h_RT[8]) + (h_RT[5] + h_RT[7]));
  dkrdT = (dlnkfdT - dlnKcdT) * k_r;
  // rate of progress
  q = k_f * phi_f - k_r * phi_r;
  dqdT = (dlnkfdT * k_f * phi_f - dkrdT * phi_r);
  // update wdot
  wdot[4] -= q; // oh
  wdot[5] += q; // h2o
  wdot[7] += q; // ho2
  wdot[8] -= q; // h2o2
  // d()/d[oh]
  dqdci = +k_f * sc[8];
  J[48] -= dqdci; // dwdot[oh]/d[oh]
  J[49] += dqdci; // dwdot[h2o]/d[oh]
  J[51] += dqdci; // dwdot[ho2]/d[oh]
  J[52] -= dqdci; // dwdot[h2o2]/d[oh]
  // d()/d[h2o]
  dqdci = -k_r * sc[7];
  J[59] -= dqdci; // dwdot[oh]/d[h2o]
  J[60] += dqdci; // dwdot[h2o]/d[h2o]
  J[62] += dqdci; // dwdot[ho2]/d[h2o]
  J[63] -= dqdci; // dwdot[h2o2]/d[h2o]
  // d()/d[ho2]
  dqdci = -k_r * sc[5];
  J[81] -= dqdci; // dwdot[oh]/d[ho2]
  J[82] += dqdci; // dwdot[h2o]/d[ho2]
  J[84] += dqdci; // dwdot[ho2]/d[ho2]
  J[85] -= dqdci; // dwdot[h2o2]/d[ho2]
  // d()/d[h2o2]
  dqdci = +k_f * sc[4];
  J[92] -= dqdci; // dwdot[oh]/d[h2o2]
  J[93] += dqdci; // dwdot[h2o]/d[h2o2]
  J[95] += dqdci; // dwdot[ho2]/d[h2o2]
  J[96] -= dqdci; // dwdot[h2o2]/d[h2o2]
  // d()/dT
  J[114] -= dqdT; // dwdot[oh]/dT
  J[115] += dqdT; // dwdot[h2o]/dT
  J[117] += dqdT; // dwdot[ho2]/dT
  J[118] -= dqdT; // dwdot[h2o2]/dT

  amrex::Real c_R[10], dcRdT[10], e_RT[10];
  amrex::Real* eh_RT;
  if (HP == 1) {
    cp_R(c_R, T);
    dcvpRdT(dcRdT, T);
    eh_RT = &h_RT[0];
  } else {
    cv_R(c_R, T);
    dcvpRdT(dcRdT, T);
    speciesInternalEnergy(e_RT, T);
    eh_RT = &e_RT[0];
  }

  amrex::Real cmix = 0.0, ehmix = 0.0, dcmixdT = 0.0, dehmixdT = 0.0;
  for (int k = 0; k < 10; ++k) {
    cmix += c_R[k] * sc[k];
    dcmixdT += dcRdT[k] * sc[k];
    ehmix += eh_RT[k] * wdot[k];
    dehmixdT += invT * (c_R[k] - eh_RT[k]) * wdot[k] + eh_RT[k] * J[110 + k];
  }

  amrex::Real cmixinv = 1.0 / cmix;
  amrex::Real tmp1 = ehmix * cmixinv;
  amrex::Real tmp3 = cmixinv * T;
  amrex::Real tmp2 = tmp1 * tmp3;
  amrex::Real dehmixdc;
  // dTdot/d[X]
  for (int k = 0; k < 10; ++k) {
    dehmixdc = 0.0;
    for (int m = 0; m < 10; ++m) {
      dehmixdc += eh_RT[m] * J[k * 11 + m];
    }
    J[k * 11 + 10] = tmp2 * c_R[k] - tmp3 * dehmixdc;
  }
  // dTdot/dT
  J[120] = -tmp1 + tmp2 * dcmixdT - tmp3 * dehmixdT;
#else
  amrex::Abort();
#endif
}

// compute an approx to the reaction Jacobian (for preconditioning)
AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE void
DWDOT_SIMPLIFIED(
  amrex::Real* J, const amrex::Real* sc, const amrex::Real* Tp, const int* HP)
{
  amrex::Real c[10];

  for (int k = 0; k < 10; k++) {
    c[k] = 1.e6 * sc[k];
  }

  aJacobian_precond(J, c, *Tp, *HP);

  // dwdot[k]/dT
  // dTdot/d[X]
  for (int k = 0; k < 10; k++) {
    J[110 + k] *= 1.e-6;
    J[k * 11 + 10] *= 1.e6;
  }
}

// compute the reaction Jacobian
AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE void
aJacobian(
  amrex::Real* J, const amrex::Real* sc, const amrex::Real T, const int consP)
{

#if defined(PELE_COMPILE_AJACOBIAN) || !defined(AMREX_USE_HIP)
  for (int i = 0; i < 121; i++) {
    J[i] = 0.0;
  }

  amrex::Real wdot[10];
  for (auto& val : wdot) {
    val = 0.0;
  }

  const amrex::Real invT = 1.0 / T;
  const amrex::Real invT2 = invT * invT;
  const amrex::Real logT = log(T);

  // reference concentration: P_atm / (RT) in inverse mol/m^3
  amrex::Real refC = 101325 / 8.31446 / T;
  amrex::Real refCinv = 1.0 / refC;

  // compute the mixture concentration
  amrex::Real mixture = 0.0;
  for (int k = 0; k < 10; ++k) {
    mixture += sc[k];
  }

  // compute the Gibbs free energy
  amrex::Real g_RT[10];
  gibbs(g_RT, T);

  // compute the species enthalpy
  amrex::Real h_RT[10];
  speciesEnthalpy(h_RT, T);

  amrex::Real phi_f, k_f, k_r, phi_r, Kc, q, q_nocor, Corr, alpha;
  amrex::Real dlnkfdT, dlnk0dT, dlnKcdT, dkrdT, dqdT;
  amrex::Real dqdci, dcdc_fac, dqdc[10];
  amrex::Real Pr, fPr, F, k_0, logPr;
  amrex::Real logFcent, troe_c, troe_n, troePr_den, troePr, troe;
  amrex::Real Fcent1, Fcent2, Fcent3, Fcent;
  amrex::Real dlogFdc, dlogFdn, dlogFdcn_fac;
  amrex::Real dlogPrdT, dlogfPrdT, dlogFdT, dlogFcentdT, dlogFdlogPr, dlnCorrdT;
  const amrex::Real ln10 = log(10.0);
  const amrex::Real log10e = 1.0 / log(10.0);
  // reaction 8: h + o2 (+M) <=> ho2 (+M)
  // a third-body and pressure-fall-off reaction
  // 3-body correction factor
  alpha = mixture + (1.3 - 1) * sc[1] + (14 - 1) * sc[5] + (0.67 - 1) * sc[9];
  // forward
  phi_f = sc[0] * sc[3];
  k_f = 1475000 * exp(0.6 * logT);
  dlnkfdT = 0.6 * invT;
  // pressure-fall-off
  k_0 = 3.482e+16 * exp(-0.411 * logT - (-561.086582372262) * invT);
  Pr = 1e-12 * alpha / k_f * k_0;
  fPr = Pr / (1.0 + Pr);
  dlnk0dT = -0.411 * invT + (-561.086582372262) * invT2;
  dlogPrdT = log10e * (dlnk0dT - dlnkfdT);
  dlogfPrdT = dlogPrdT / (1.0 + Pr);
  // Troe form
  logPr = log10(Pr);
  Fcent1 = (1. - 0.5) * exp(-T / 1e-30);
  Fcent2 = 0.5 * exp(-T / 1e+30);
  Fcent3 = exp(-1e+100 * invT);
  Fcent = Fcent1 + Fcent2 + Fcent3;
  logFcent = log10(Fcent);
  troe_c = -.4 - .67 * logFcent;
  troe_n = .75 - 1.27 * logFcent;
  troePr_den = 1.0 / (troe_n - .14 * (troe_c + logPr));
  troePr = (troe_c + logPr) * troePr_den;
  troe = 1.0 / (1.0 + troePr * troePr);
  F = exp(M_LN10 * logFcent * troe);
  dlogFcentdT = log10e / Fcent *
                (-Fcent1 / 1e-30 - Fcent2 / 1e+30 + Fcent3 * 1e+100 * invT2);
  dlogFdcn_fac = 2.0 * logFcent * troe * troe * troePr * troePr_den;
  dlogFdc = -troe_n * dlogFdcn_fac * troePr_den;
  dlogFdn = dlogFdcn_fac * troePr;
  dlogFdlogPr = dlogFdc;
  dlogFdT = dlogFcentdT * (troe - 0.67 * dlogFdc - 1.27 * dlogFdn) +
            dlogFdlogPr * dlogPrdT;
  // reverse
  phi_r = sc[7];
  Kc = refCinv * exp(g_RT[0] + g_RT[3] - g_RT[7]);
  k_r = k_f / Kc;
  dlnKcdT = invT * (-(h_RT[0] + h_RT[3]) + (h_RT[7]) + 1.000000);
  dkrdT = (dlnkfdT - dlnKcdT) * k_r;
  // rate of progress
  q_nocor = k_f * phi_f - k_r * phi_r;
  Corr = fPr * F;
  q = Corr * q_nocor;
  dlnCorrdT = ln10 * (dlogfPrdT + dlogFdT);
  dqdT = Corr * (dlnkfdT * k_f * phi_f - dkrdT * phi_r) + dlnCorrdT * q;
  // update wdot
  wdot[0] -= q; // h
  wdot[3] -= q; // o2
  wdot[7] += q; // ho2
  // for convenience
  k_f *= Corr;
  k_r *= Corr;
  dcdc_fac = q / alpha * (1.0 / (Pr + 1.0) + dlogFdlogPr);
  if (consP == 1) {
    // d()/d[h]
    dqdci = +k_f * sc[3];
    J[0] -= dqdci; // dwdot[h]/d[h]
    J[3] -= dqdci; // dwdot[o2]/d[h]
    J[7] += dqdci; // dwdot[ho2]/d[h]
    // d()/d[h2]
    dqdci = (1.3 - 1) * dcdc_fac;
    J[11] -= dqdci; // dwdot[h]/d[h2]
    J[14] -= dqdci; // dwdot[o2]/d[h2]
    J[18] += dqdci; // dwdot[ho2]/d[h2]
    // d()/d[o2]
    dqdci = +k_f * sc[0];
    J[33] -= dqdci; // dwdot[h]/d[o2]
    J[36] -= dqdci; // dwdot[o2]/d[o2]
    J[40] += dqdci; // dwdot[ho2]/d[o2]
    // d()/d[h2o]
    dqdci = (14 - 1) * dcdc_fac;
    J[55] -= dqdci; // dwdot[h]/d[h2o]
    J[58] -= dqdci; // dwdot[o2]/d[h2o]
    J[62] += dqdci; // dwdot[ho2]/d[h2o]
    // d()/d[ho2]
    dqdci = -k_r;
    J[77] -= dqdci; // dwdot[h]/d[ho2]
    J[80] -= dqdci; // dwdot[o2]/d[ho2]
    J[84] += dqdci; // dwdot[ho2]/d[ho2]
    // d()/d[Xe]
    dqdci = (0.67 - 1) * dcdc_fac;
    J[99] -= dqdci;  // dwdot[h]/d[Xe]
    J[102] -= dqdci; // dwdot[o2]/d[Xe]
    J[106] += dqdci; // dwdot[ho2]/d[Xe]
  } else {
    dqdc[0] = dcdc_fac + k_f * sc[3];
    dqdc[1] = 1.3 * dcdc_fac;
    dqdc[2] = dcdc_fac;
    dqdc[3] = dcdc_fac + k_f * sc[0];
    dqdc[4] = dcdc_fac;
    dqdc[5] = 14 * dcdc_fac;
    dqdc[6] = dcdc_fac;
    dqdc[7] = dcdc_fac - k_r;
    dqdc[8] = dcdc_fac;
    dqdc[9] = 0.67 * dcdc_fac;
    for (int k = 0; k < 10; k++) {
      J[11 * k + 0] -= dqdc[k];
      J[11 * k + 3] -= dqdc[k];
      J[11 * k + 7] += dqdc[k];
    }
  }
  J[110] -= dqdT; // dwdot[h]/dT
  J[113] -= dqdT; // dwdot[o2]/dT
  J[117] += dqdT; // dwdot[ho2]/dT

  // reaction 15: h2o2 (+M) <=> 2 oh (+M)
  // a third-body and pressure-fall-off reaction
  // 3-body correction factor
  alpha = mixture + (2.5 - 1) * sc[1] + (12 - 1) * sc[5] + (0.64 - 1) * sc[9];
  // forward
  phi_f = sc[8];
  k_f = 295100000000000 * exp(-(24370.7831249226) * invT);
  dlnkfdT = (24370.7831249226) * invT2;
  // pressure-fall-off
  k_0 = 1.202e+17 * exp(-(22896.3582941147) * invT);
  Pr = 1e-6 * alpha / k_f * k_0;
  fPr = Pr / (1.0 + Pr);
  dlnk0dT = (22896.3582941147) * invT2;
  dlogPrdT = log10e * (dlnk0dT - dlnkfdT);
  dlogfPrdT = dlogPrdT / (1.0 + Pr);
  // Troe form
  logPr = log10(Pr);
  Fcent1 = (1. - 0.5) * exp(-T / 1e-30);
  Fcent2 = 0.5 * exp(-T / 1e+30);
  Fcent3 = exp(-1e+100 * invT);
  Fcent = Fcent1 + Fcent2 + Fcent3;
  logFcent = log10(Fcent);
  troe_c = -.4 - .67 * logFcent;
  troe_n = .75 - 1.27 * logFcent;
  troePr_den = 1.0 / (troe_n - .14 * (troe_c + logPr));
  troePr = (troe_c + logPr) * troePr_den;
  troe = 1.0 / (1.0 + troePr * troePr);
  F = exp(M_LN10 * logFcent * troe);
  dlogFcentdT = log10e / Fcent *
                (-Fcent1 / 1e-30 - Fcent2 / 1e+30 + Fcent3 * 1e+100 * invT2);
  dlogFdcn_fac = 2.0 * logFcent * troe * troe * troePr * troePr_den;
  dlogFdc = -troe_n * dlogFdcn_fac * troePr_den;
  dlogFdn = dlogFdcn_fac * troePr;
  dlogFdlogPr = dlogFdc;
  dlogFdT = dlogFcentdT * (troe - 0.67 * dlogFdc - 1.27 * dlogFdn) +
            dlogFdlogPr * dlogPrdT;
  // reverse
  phi_r = (sc[4] * sc[4]);
  Kc = refC * exp(-2.000000 * g_RT[4] + g_RT[8]);
  k_r = k_f / Kc;
  dlnKcdT = invT * (-(h_RT[8]) + (2.000000 * h_RT[4]) - 1.000000);
  dkrdT = (dlnkfdT - dlnKcdT) * k_r;
  // rate of progress
  q_nocor = k_f * phi_f - k_r * phi_r;
  Corr = fPr * F;
  q = Corr * q_nocor;
  dlnCorrdT = ln10 * (dlogfPrdT + dlogFdT);
  dqdT = Corr * (dlnkfdT * k_f * phi_f - dkrdT * phi_r) + dlnCorrdT * q;
  // update wdot
  wdot[4] += 2 * q; // oh
  wdot[8] -= q;     // h2o2
  // for convenience
  k_f *= Corr;
  k_r *= Corr;
  dcdc_fac = q / alpha * (1.0 / (Pr + 1.0) + dlogFdlogPr);
  if (consP == 1) {
    // d()/d[h2]
    dqdci = (2.5 - 1) * dcdc_fac;
    J[15] += 2 * dqdci; // dwdot[oh]/d[h2]
    J[19] -= dqdci;     // dwdot[h2o2]/d[h2]
    // d()/d[oh]
    dqdci = -k_r * 2.000000 * sc[4];
    J[48] += 2 * dqdci; // dwdot[oh]/d[oh]
    J[52] -= dqdci;     // dwdot[h2o2]/d[oh]
    // d()/d[h2o]
    dqdci = (12 - 1) * dcdc_fac;
    J[59] += 2 * dqdci; // dwdot[oh]/d[h2o]
    J[63] -= dqdci;     // dwdot[h2o2]/d[h2o]
    // d()/d[h2o2]
    dqdci = +k_f;
    J[92] += 2 * dqdci; // dwdot[oh]/d[h2o2]
    J[96] -= dqdci;     // dwdot[h2o2]/d[h2o2]
    // d()/d[Xe]
    dqdci = (0.64 - 1) * dcdc_fac;
    J[103] += 2 * dqdci; // dwdot[oh]/d[Xe]
    J[107] -= dqdci;     // dwdot[h2o2]/d[Xe]
  } else {
    dqdc[0] = dcdc_fac;
    dqdc[1] = 2.5 * dcdc_fac;
    dqdc[2] = dcdc_fac;
    dqdc[3] = dcdc_fac;
    dqdc[4] = dcdc_fac - k_r * 2.000000 * sc[4];
    dqdc[5] = 12 * dcdc_fac;
    dqdc[6] = dcdc_fac;
    dqdc[7] = dcdc_fac;
    dqdc[8] = dcdc_fac + k_f;
    dqdc[9] = 0.64 * dcdc_fac;
    for (int k = 0; k < 10; k++) {
      J[11 * k + 4] += 2 * dqdc[k];
      J[11 * k + 8] -= dqdc[k];
    }
  }
  J[114] += 2 * dqdT; // dwdot[oh]/dT
  J[118] -= dqdT;     // dwdot[h2o2]/dT

  // reaction 4: h2 + M <=> 2 h + M
  // a third-body and non-pressure-fall-off reaction
  // 3-body correction factor
  alpha = mixture + (2.5 - 1) * sc[1] + (12 - 1) * sc[5];
  // forward
  phi_f = sc[1];
  k_f = 45770000000000 * exp(-1.4 * logT - (52535.8199100127) * invT);
  dlnkfdT = -1.4 * invT + (52535.8199100127) * invT2;
  // reverse
  phi_r = (sc[0] * sc[0]);
  Kc = refC * exp(-2.000000 * g_RT[0] + g_RT[1]);
  k_r = k_f / Kc;
  dlnKcdT = invT * (-(h_RT[1]) + (2.000000 * h_RT[0]) - 1.000000);
  dkrdT = (dlnkfdT - dlnKcdT) * k_r;
  // rate of progress
  q_nocor = k_f * phi_f - k_r * phi_r;
  q = alpha * q_nocor;
  dqdT = alpha * (dlnkfdT * k_f * phi_f - dkrdT * phi_r);
  // update wdot
  wdot[0] += 2 * q; // h
  wdot[1] -= q;     // h2
  // for convenience
  k_f *= alpha;
  k_r *= alpha;
  if (consP == 1) {
    // d()/d[h]
    dqdci = -k_r * 2.000000 * sc[0];
    J[0] += 2 * dqdci; // dwdot[h]/d[h]
    J[1] -= dqdci;     // dwdot[h2]/d[h]
    // d()/d[h2]
    dqdci = (2.5 - 1) * q_nocor + k_f;
    J[11] += 2 * dqdci; // dwdot[h]/d[h2]
    J[12] -= dqdci;     // dwdot[h2]/d[h2]
    // d()/d[h2o]
    dqdci = (12 - 1) * q_nocor;
    J[55] += 2 * dqdci; // dwdot[h]/d[h2o]
    J[56] -= dqdci;     // dwdot[h2]/d[h2o]
  } else {
    dqdc[0] = q_nocor - k_r * 2.000000 * sc[0];
    dqdc[1] = 2.5 * q_nocor + k_f;
    dqdc[2] = q_nocor;
    dqdc[3] = q_nocor;
    dqdc[4] = q_nocor;
    dqdc[5] = 12 * q_nocor;
    dqdc[6] = q_nocor;
    dqdc[7] = q_nocor;
    dqdc[8] = q_nocor;
    dqdc[9] = q_nocor;
    for (int k = 0; k < 10; k++) {
      J[11 * k + 0] += 2 * dqdc[k];
      J[11 * k + 1] -= dqdc[k];
    }
  }
  J[110] += 2 * dqdT; // dwdot[h]/dT
  J[111] -= dqdT;     // dwdot[h2]/dT

  // reaction 5: 2 o + M <=> o2 + M
  // a third-body and non-pressure-fall-off reaction
  // 3-body correction factor
  alpha = mixture + (2.5 - 1) * sc[1] + (12 - 1) * sc[5] + (0.83 - 1) * sc[9];
  // forward
  phi_f = (sc[2] * sc[2]);
  k_f = 6165 * exp(-0.5 * logT);
  dlnkfdT = -0.5 * invT;
  // reverse
  phi_r = sc[3];
  Kc = refCinv * exp(2.000000 * g_RT[2] - g_RT[3]);
  k_r = k_f / Kc;
  dlnKcdT = invT * (-(2.000000 * h_RT[2]) + (h_RT[3]) + 1.000000);
  dkrdT = (dlnkfdT - dlnKcdT) * k_r;
  // rate of progress
  q_nocor = k_f * phi_f - k_r * phi_r;
  q = alpha * q_nocor;
  dqdT = alpha * (dlnkfdT * k_f * phi_f - dkrdT * phi_r);
  // update wdot
  wdot[2] -= 2 * q; // o
  wdot[3] += q;     // o2
  // for convenience
  k_f *= alpha;
  k_r *= alpha;
  if (consP == 1) {
    // d()/d[h2]
    dqdci = (2.5 - 1) * q_nocor;
    J[13] += -2 * dqdci; // dwdot[o]/d[h2]
    J[14] += dqdci;      // dwdot[o2]/d[h2]
    // d()/d[o]
    dqdci = +k_f * 2.000000 * sc[2];
    J[24] += -2 * dqdci; // dwdot[o]/d[o]
    J[25] += dqdci;      // dwdot[o2]/d[o]
    // d()/d[o2]
    dqdci = -k_r;
    J[35] += -2 * dqdci; // dwdot[o]/d[o2]
    J[36] += dqdci;      // dwdot[o2]/d[o2]
    // d()/d[h2o]
    dqdci = (12 - 1) * q_nocor;
    J[57] += -2 * dqdci; // dwdot[o]/d[h2o]
    J[58] += dqdci;      // dwdot[o2]/d[h2o]
    // d()/d[Xe]
    dqdci = (0.83 - 1) * q_nocor;
    J[101] += -2 * dqdci; // dwdot[o]/d[Xe]
    J[102] += dqdci;      // dwdot[o2]/d[Xe]
  } else {
    dqdc[0] = q_nocor;
    dqdc[1] = 2.5 * q_nocor;
    dqdc[2] = q_nocor + k_f * 2.000000 * sc[2];
    dqdc[3] = q_nocor - k_r;
    dqdc[4] = q_nocor;
    dqdc[5] = 12 * q_nocor;
    dqdc[6] = q_nocor;
    dqdc[7] = q_nocor;
    dqdc[8] = q_nocor;
    dqdc[9] = 0.83 * q_nocor;
    for (int k = 0; k < 10; k++) {
      J[11 * k + 2] += -2 * dqdc[k];
      J[11 * k + 3] += dqdc[k];
    }
  }
  J[112] += -2 * dqdT; // dwdot[o]/dT
  J[113] += dqdT;      // dwdot[o2]/dT

  // reaction 6: h + o + M <=> oh + M
  // a third-body and non-pressure-fall-off reaction
  // 3-body correction factor
  alpha = mixture + (2.5 - 1) * sc[1] + (12 - 1) * sc[5] + (0.75 - 1) * sc[9];
  // forward
  phi_f = sc[0] * sc[2];
  k_f = 4714000 * exp(-1 * logT);
  dlnkfdT = -1 * invT;
  // reverse
  phi_r = sc[4];
  Kc = refCinv * exp(g_RT[0] + g_RT[2] - g_RT[4]);
  k_r = k_f / Kc;
  dlnKcdT = invT * (-(h_RT[0] + h_RT[2]) + (h_RT[4]) + 1.000000);
  dkrdT = (dlnkfdT - dlnKcdT) * k_r;
  // rate of progress
  q_nocor = k_f * phi_f - k_r * phi_r;
  q = alpha * q_nocor;
  dqdT = alpha * (dlnkfdT * k_f * phi_f - dkrdT * phi_r);
  // update wdot
  wdot[0] -= q; // h
  wdot[2] -= q; // o
  wdot[4] += q; // oh
  // for convenience
  k_f *= alpha;
  k_r *= alpha;
  if (consP == 1) {
    // d()/d[h]
    dqdci = +k_f * sc[2];
    J[0] -= dqdci; // dwdot[h]/d[h]
    J[2] -= dqdci; // dwdot[o]/d[h]
    J[4] += dqdci; // dwdot[oh]/d[h]
    // d()/d[h2]
    dqdci = (2.5 - 1) * q_nocor;
    J[11] -= dqdci; // dwdot[h]/d[h2]
    J[13] -= dqdci; // dwdot[o]/d[h2]
    J[15] += dqdci; // dwdot[oh]/d[h2]
    // d()/d[o]
    dqdci = +k_f * sc[0];
    J[22] -= dqdci; // dwdot[h]/d[o]
    J[24] -= dqdci; // dwdot[o]/d[o]
    J[26] += dqdci; // dwdot[oh]/d[o]
    // d()/d[oh]
    dqdci = -k_r;
    J[44] -= dqdci; // dwdot[h]/d[oh]
    J[46] -= dqdci; // dwdot[o]/d[oh]
    J[48] += dqdci; // dwdot[oh]/d[oh]
    // d()/d[h2o]
    dqdci = (12 - 1) * q_nocor;
    J[55] -= dqdci; // dwdot[h]/d[h2o]
    J[57] -= dqdci; // dwdot[o]/d[h2o]
    J[59] += dqdci; // dwdot[oh]/d[h2o]
    // d()/d[Xe]
    dqdci = (0.75 - 1) * q_nocor;
    J[99] -= dqdci;  // dwdot[h]/d[Xe]
    J[101] -= dqdci; // dwdot[o]/d[Xe]
    J[103] += dqdci; // dwdot[oh]/d[Xe]
  } else {
    dqdc[0] = q_nocor + k_f * sc[2];
    dqdc[1] = 2.5 * q_nocor;
    dqdc[2] = q_nocor + k_f * sc[0];
    dqdc[3] = q_nocor;
    dqdc[4] = q_nocor - k_r;
    dqdc[5] = 12 * q_nocor;
    dqdc[6] = q_nocor;
    dqdc[7] = q_nocor;
    dqdc[8] = q_nocor;
    dqdc[9] = 0.75 * q_nocor;
    for (int k = 0; k < 10; k++) {
      J[11 * k + 0] -= dqdc[k];
      J[11 * k + 2] -= dqdc[k];
      J[11 * k + 4] += dqdc[k];
    }
  }
  J[110] -= dqdT; // dwdot[h]/dT
  J[112] -= dqdT; // dwdot[o]/dT
  J[114] += dqdT; // dwdot[oh]/dT

  // reaction 7: h + oh + M <=> h2o + M
  // a third-body and non-pressure-fall-off reaction
  // 3-body correction factor
  alpha = mixture + (0.73 - 1) * sc[1] + (12 - 1) * sc[5] + (0.38 - 1) * sc[9];
  // forward
  phi_f = sc[0] * sc[4];
  k_f = 45000000000 * exp(-2 * logT);
  dlnkfdT = -2 * invT;
  // reverse
  phi_r = sc[5];
  Kc = refCinv * exp(g_RT[0] + g_RT[4] - g_RT[5]);
  k_r = k_f / Kc;
  dlnKcdT = invT * (-(h_RT[0] + h_RT[4]) + (h_RT[5]) + 1.000000);
  dkrdT = (dlnkfdT - dlnKcdT) * k_r;
  // rate of progress
  q_nocor = k_f * phi_f - k_r * phi_r;
  q = alpha * q_nocor;
  dqdT = alpha * (dlnkfdT * k_f * phi_f - dkrdT * phi_r);
  // update wdot
  wdot[0] -= q; // h
  wdot[4] -= q; // oh
  wdot[5] += q; // h2o
  // for convenience
  k_f *= alpha;
  k_r *= alpha;
  if (consP == 1) {
    // d()/d[h]
    dqdci = +k_f * sc[4];
    J[0] -= dqdci; // dwdot[h]/d[h]
    J[4] -= dqdci; // dwdot[oh]/d[h]
    J[5] += dqdci; // dwdot[h2o]/d[h]
    // d()/d[h2]
    dqdci = (0.73 - 1) * q_nocor;
    J[11] -= dqdci; // dwdot[h]/d[h2]
    J[15] -= dqdci; // dwdot[oh]/d[h2]
    J[16] += dqdci; // dwdot[h2o]/d[h2]
    // d()/d[oh]
    dqdci = +k_f * sc[0];
    J[44] -= dqdci; // dwdot[h]/d[oh]
    J[48] -= dqdci; // dwdot[oh]/d[oh]
    J[49] += dqdci; // dwdot[h2o]/d[oh]
    // d()/d[h2o]
    dqdci = (12 - 1) * q_nocor - k_r;
    J[55] -= dqdci; // dwdot[h]/d[h2o]
    J[59] -= dqdci; // dwdot[oh]/d[h2o]
    J[60] += dqdci; // dwdot[h2o]/d[h2o]
    // d()/d[Xe]
    dqdci = (0.38 - 1) * q_nocor;
    J[99] -= dqdci;  // dwdot[h]/d[Xe]
    J[103] -= dqdci; // dwdot[oh]/d[Xe]
    J[104] += dqdci; // dwdot[h2o]/d[Xe]
  } else {
    dqdc[0] = q_nocor + k_f * sc[4];
    dqdc[1] = 0.73 * q_nocor;
    dqdc[2] = q_nocor;
    dqdc[3] = q_nocor;
    dqdc[4] = q_nocor + k_f * sc[0];
    dqdc[5] = 12 * q_nocor - k_r;
    dqdc[6] = q_nocor;
    dqdc[7] = q_nocor;
    dqdc[8] = q_nocor;
    dqdc[9] = 0.38 * q_nocor;
    for (int k = 0; k < 10; k++) {
      J[11 * k + 0] -= dqdc[k];
      J[11 * k + 4] -= dqdc[k];
      J[11 * k + 5] += dqdc[k];
    }
  }
  J[110] -= dqdT; // dwdot[h]/dT
  J[114] -= dqdT; // dwdot[oh]/dT
  J[115] += dqdT; // dwdot[h2o]/dT

  // reaction 0: h + o2 <=> o + oh
  // a non-third-body and non-pressure-fall-off reaction
  // forward
  phi_f = sc[0] * sc[3];
  k_f = 191500000 * exp(-(8272.88198582959) * invT);
  dlnkfdT = (8272.88198582959) * invT2;
  // reverse
  phi_r = sc[2] * sc[4];
  Kc = exp(g_RT[0] - g_RT[2] + g_RT[3] - g_RT[4]);
  k_r = k_f / Kc;
  dlnKcdT = invT * (-(h_RT[0] + h_RT[3]) + (h_RT[2] + h_RT[4]));
  dkrdT = (dlnkfdT - dlnKcdT) * k_r;
  // rate of progress
  q = k_f * phi_f - k_r * phi_r;
  dqdT = (dlnkfdT * k_f * phi_f - dkrdT * phi_r);
  // update wdot
  wdot[0] -= q; // h
  wdot[2] += q; // o
  wdot[3] -= q; // o2
  wdot[4] += q; // oh
  // d()/d[h]
  dqdci = +k_f * sc[3];
  J[0] -= dqdci; // dwdot[h]/d[h]
  J[2] += dqdci; // dwdot[o]/d[h]
  J[3] -= dqdci; // dwdot[o2]/d[h]
  J[4] += dqdci; // dwdot[oh]/d[h]
  // d()/d[o]
  dqdci = -k_r * sc[4];
  J[22] -= dqdci; // dwdot[h]/d[o]
  J[24] += dqdci; // dwdot[o]/d[o]
  J[25] -= dqdci; // dwdot[o2]/d[o]
  J[26] += dqdci; // dwdot[oh]/d[o]
  // d()/d[o2]
  dqdci = +k_f * sc[0];
  J[33] -= dqdci; // dwdot[h]/d[o2]
  J[35] += dqdci; // dwdot[o]/d[o2]
  J[36] -= dqdci; // dwdot[o2]/d[o2]
  J[37] += dqdci; // dwdot[oh]/d[o2]
  // d()/d[oh]
  dqdci = -k_r * sc[2];
  J[44] -= dqdci; // dwdot[h]/d[oh]
  J[46] += dqdci; // dwdot[o]/d[oh]
  J[47] -= dqdci; // dwdot[o2]/d[oh]
  J[48] += dqdci; // dwdot[oh]/d[oh]
  // d()/dT
  J[110] -= dqdT; // dwdot[h]/dT
  J[112] += dqdT; // dwdot[o]/dT
  J[113] -= dqdT; // dwdot[o2]/dT
  J[114] += dqdT; // dwdot[oh]/dT

  // reaction 1: h2 + o <=> h + oh
  // a non-third-body and non-pressure-fall-off reaction
  // forward
  phi_f = sc[1] * sc[2];
  k_f = 0.0508 * exp(2.67 * logT - (3166.2392612433) * invT);
  dlnkfdT = 2.67 * invT + (3166.2392612433) * invT2;
  // reverse
  phi_r = sc[0] * sc[4];
  Kc = exp(-g_RT[0] + g_RT[1] + g_RT[2] - g_RT[4]);
  k_r = k_f / Kc;
  dlnKcdT = invT * (-(h_RT[1] + h_RT[2]) + (h_RT[0] + h_RT[4]));
  dkrdT = (dlnkfdT - dlnKcdT) * k_r;
  // rate of progress
  q = k_f * phi_f - k_r * phi_r;
  dqdT = (dlnkfdT * k_f * phi_f - dkrdT * phi_r);
  // update wdot
  wdot[0] += q; // h
  wdot[1] -= q; // h2
  wdot[2] -= q; // o
  wdot[4] += q; // oh
  // d()/d[h]
  dqdci = -k_r * sc[4];
  J[0] += dqdci; // dwdot[h]/d[h]
  J[1] -= dqdci; // dwdot[h2]/d[h]
  J[2] -= dqdci; // dwdot[o]/d[h]
  J[4] += dqdci; // dwdot[oh]/d[h]
  // d()/d[h2]
  dqdci = +k_f * sc[2];
  J[11] += dqdci; // dwdot[h]/d[h2]
  J[12] -= dqdci; // dwdot[h2]/d[h2]
  J[13] -= dqdci; // dwdot[o]/d[h2]
  J[15] += dqdci; // dwdot[oh]/d[h2]
  // d()/d[o]
  dqdci = +k_f * sc[1];
  J[22] += dqdci; // dwdot[h]/d[o]
  J[23] -= dqdci; // dwdot[h2]/d[o]
  J[24] -= dqdci; // dwdot[o]/d[o]
  J[26] += dqdci; // dwdot[oh]/d[o]
  // d()/d[oh]
  dqdci = -k_r * sc[0];
  J[44] += dqdci; // dwdot[h]/d[oh]
  J[45] -= dqdci; // dwdot[h2]/d[oh]
  J[46] -= dqdci; // dwdot[o]/d[oh]
  J[48] += dqdci; // dwdot[oh]/d[oh]
  // d()/dT
  J[110] += dqdT; // dwdot[h]/dT
  J[111] -= dqdT; // dwdot[h2]/dT
  J[112] -= dqdT; // dwdot[o]/dT
  J[114] += dqdT; // dwdot[oh]/dT

  // reaction 2: h2 + oh <=> h + h2o
  // a non-third-body and non-pressure-fall-off reaction
  // forward
  phi_f = sc[1] * sc[4];
  k_f = 216 * exp(1.51 * logT - (1726.03316371019) * invT);
  dlnkfdT = 1.51 * invT + (1726.03316371019) * invT2;
  // reverse
  phi_r = sc[0] * sc[5];
  Kc = exp(-g_RT[0] + g_RT[1] + g_RT[4] - g_RT[5]);
  k_r = k_f / Kc;
  dlnKcdT = invT * (-(h_RT[1] + h_RT[4]) + (h_RT[0] + h_RT[5]));
  dkrdT = (dlnkfdT - dlnKcdT) * k_r;
  // rate of progress
  q = k_f * phi_f - k_r * phi_r;
  dqdT = (dlnkfdT * k_f * phi_f - dkrdT * phi_r);
  // update wdot
  wdot[0] += q; // h
  wdot[1] -= q; // h2
  wdot[4] -= q; // oh
  wdot[5] += q; // h2o
  // d()/d[h]
  dqdci = -k_r * sc[5];
  J[0] += dqdci; // dwdot[h]/d[h]
  J[1] -= dqdci; // dwdot[h2]/d[h]
  J[4] -= dqdci; // dwdot[oh]/d[h]
  J[5] += dqdci; // dwdot[h2o]/d[h]
  // d()/d[h2]
  dqdci = +k_f * sc[4];
  J[11] += dqdci; // dwdot[h]/d[h2]
  J[12] -= dqdci; // dwdot[h2]/d[h2]
  J[15] -= dqdci; // dwdot[oh]/d[h2]
  J[16] += dqdci; // dwdot[h2o]/d[h2]
  // d()/d[oh]
  dqdci = +k_f * sc[1];
  J[44] += dqdci; // dwdot[h]/d[oh]
  J[45] -= dqdci; // dwdot[h2]/d[oh]
  J[48] -= dqdci; // dwdot[oh]/d[oh]
  J[49] += dqdci; // dwdot[h2o]/d[oh]
  // d()/d[h2o]
  dqdci = -k_r * sc[0];
  J[55] += dqdci; // dwdot[h]/d[h2o]
  J[56] -= dqdci; // dwdot[h2]/d[h2o]
  J[59] -= dqdci; // dwdot[oh]/d[h2o]
  J[60] += dqdci; // dwdot[h2o]/d[h2o]
  // d()/dT
  J[110] += dqdT; // dwdot[h]/dT
  J[111] -= dqdT; // dwdot[h2]/dT
  J[114] -= dqdT; // dwdot[oh]/dT
  J[115] += dqdT; // dwdot[h2o]/dT

  // reaction 3: h2o + o <=> 2 oh
  // a non-third-body and non-pressure-fall-off reaction
  // forward
  phi_f = sc[2] * sc[5];
  k_f = 2.97 * exp(2.02 * logT - (6743.10332178324) * invT);
  dlnkfdT = 2.02 * invT + (6743.10332178324) * invT2;
  // reverse
  phi_r = (sc[4] * sc[4]);
  Kc = exp(g_RT[2] - 2.000000 * g_RT[4] + g_RT[5]);
  k_r = k_f / Kc;
  dlnKcdT = invT * (-(h_RT[2] + h_RT[5]) + (2.000000 * h_RT[4]));
  dkrdT = (dlnkfdT - dlnKcdT) * k_r;
  // rate of progress
  q = k_f * phi_f - k_r * phi_r;
  dqdT = (dlnkfdT * k_f * phi_f - dkrdT * phi_r);
  // update wdot
  wdot[2] -= q;     // o
  wdot[4] += 2 * q; // oh
  wdot[5] -= q;     // h2o
  // d()/d[o]
  dqdci = +k_f * sc[5];
  J[24] -= dqdci;     // dwdot[o]/d[o]
  J[26] += 2 * dqdci; // dwdot[oh]/d[o]
  J[27] -= dqdci;     // dwdot[h2o]/d[o]
  // d()/d[oh]
  dqdci = -k_r * 2.000000 * sc[4];
  J[46] -= dqdci;     // dwdot[o]/d[oh]
  J[48] += 2 * dqdci; // dwdot[oh]/d[oh]
  J[49] -= dqdci;     // dwdot[h2o]/d[oh]
  // d()/d[h2o]
  dqdci = +k_f * sc[2];
  J[57] -= dqdci;     // dwdot[o]/d[h2o]
  J[59] += 2 * dqdci; // dwdot[oh]/d[h2o]
  J[60] -= dqdci;     // dwdot[h2o]/d[h2o]
  // d()/dT
  J[112] -= dqdT;     // dwdot[o]/dT
  J[114] += 2 * dqdT; // dwdot[oh]/dT
  J[115] -= dqdT;     // dwdot[h2o]/dT

  // reaction 9: h + ho2 <=> h2 + o2
  // a non-third-body and non-pressure-fall-off reaction
  // forward
  phi_f = sc[0] * sc[7];
  k_f = 16600000 * exp(-(414.147315957284) * invT);
  dlnkfdT = (414.147315957284) * invT2;
  // reverse
  phi_r = sc[1] * sc[3];
  Kc = exp(g_RT[0] - g_RT[1] - g_RT[3] + g_RT[7]);
  k_r = k_f / Kc;
  dlnKcdT = invT * (-(h_RT[0] + h_RT[7]) + (h_RT[1] + h_RT[3]));
  dkrdT = (dlnkfdT - dlnKcdT) * k_r;
  // rate of progress
  q = k_f * phi_f - k_r * phi_r;
  dqdT = (dlnkfdT * k_f * phi_f - dkrdT * phi_r);
  // update wdot
  wdot[0] -= q; // h
  wdot[1] += q; // h2
  wdot[3] += q; // o2
  wdot[7] -= q; // ho2
  // d()/d[h]
  dqdci = +k_f * sc[7];
  J[0] -= dqdci; // dwdot[h]/d[h]
  J[1] += dqdci; // dwdot[h2]/d[h]
  J[3] += dqdci; // dwdot[o2]/d[h]
  J[7] -= dqdci; // dwdot[ho2]/d[h]
  // d()/d[h2]
  dqdci = -k_r * sc[3];
  J[11] -= dqdci; // dwdot[h]/d[h2]
  J[12] += dqdci; // dwdot[h2]/d[h2]
  J[14] += dqdci; // dwdot[o2]/d[h2]
  J[18] -= dqdci; // dwdot[ho2]/d[h2]
  // d()/d[o2]
  dqdci = -k_r * sc[1];
  J[33] -= dqdci; // dwdot[h]/d[o2]
  J[34] += dqdci; // dwdot[h2]/d[o2]
  J[36] += dqdci; // dwdot[o2]/d[o2]
  J[40] -= dqdci; // dwdot[ho2]/d[o2]
  // d()/d[ho2]
  dqdci = +k_f * sc[0];
  J[77] -= dqdci; // dwdot[h]/d[ho2]
  J[78] += dqdci; // dwdot[h2]/d[ho2]
  J[80] += dqdci; // dwdot[o2]/d[ho2]
  J[84] -= dqdci; // dwdot[ho2]/d[ho2]
  // d()/dT
  J[110] -= dqdT; // dwdot[h]/dT
  J[111] += dqdT; // dwdot[h2]/dT
  J[113] += dqdT; // dwdot[o2]/dT
  J[117] -= dqdT; // dwdot[ho2]/dT

  // reaction 10: h + ho2 <=> 2 oh
  // a non-third-body and non-pressure-fall-off reaction
  // forward
  phi_f = sc[0] * sc[7];
  k_f = 70790000 * exp(-(148.448916412392) * invT);
  dlnkfdT = (148.448916412392) * invT2;
  // reverse
  phi_r = (sc[4] * sc[4]);
  Kc = exp(g_RT[0] - 2.000000 * g_RT[4] + g_RT[7]);
  k_r = k_f / Kc;
  dlnKcdT = invT * (-(h_RT[0] + h_RT[7]) + (2.000000 * h_RT[4]));
  dkrdT = (dlnkfdT - dlnKcdT) * k_r;
  // rate of progress
  q = k_f * phi_f - k_r * phi_r;
  dqdT = (dlnkfdT * k_f * phi_f - dkrdT * phi_r);
  // update wdot
  wdot[0] -= q;     // h
  wdot[4] += 2 * q; // oh
  wdot[7] -= q;     // ho2
  // d()/d[h]
  dqdci = +k_f * sc[7];
  J[0] -= dqdci;     // dwdot[h]/d[h]
  J[4] += 2 * dqdci; // dwdot[oh]/d[h]
  J[7] -= dqdci;     // dwdot[ho2]/d[h]
  // d()/d[oh]
  dqdci = -k_r * 2.000000 * sc[4];
  J[44] -= dqdci;     // dwdot[h]/d[oh]
  J[48] += 2 * dqdci; // dwdot[oh]/d[oh]
  J[51] -= dqdci;     // dwdot[ho2]/d[oh]
  // d()/d[ho2]
  dqdci = +k_f * sc[0];
  J[77] -= dqdci;     // dwdot[h]/d[ho2]
  J[81] += 2 * dqdci; // dwdot[oh]/d[ho2]
  J[84] -= dqdci;     // dwdot[ho2]/d[ho2]
  // d()/dT
  J[110] -= dqdT;     // dwdot[h]/dT
  J[114] += 2 * dqdT; // dwdot[oh]/dT
  J[117] -= dqdT;     // dwdot[ho2]/dT

  // reaction 11: ho2 + o <=> o2 + oh
  // a non-third-body and non-pressure-fall-off reaction
  // forward
  phi_f = sc[2] * sc[7];
  k_f = 32500000;
  dlnkfdT = 0.0;
  // reverse
  phi_r = sc[3] * sc[4];
  Kc = exp(g_RT[2] - g_RT[3] - g_RT[4] + g_RT[7]);
  k_r = k_f / Kc;
  dlnKcdT = invT * (-(h_RT[2] + h_RT[7]) + (h_RT[3] + h_RT[4]));
  dkrdT = (dlnkfdT - dlnKcdT) * k_r;
  // rate of progress
  q = k_f * phi_f - k_r * phi_r;
  dqdT = (dlnkfdT * k_f * phi_f - dkrdT * phi_r);
  // update wdot
  wdot[2] -= q; // o
  wdot[3] += q; // o2
  wdot[4] += q; // oh
  wdot[7] -= q; // ho2
  // d()/d[o]
  dqdci = +k_f * sc[7];
  J[24] -= dqdci; // dwdot[o]/d[o]
  J[25] += dqdci; // dwdot[o2]/d[o]
  J[26] += dqdci; // dwdot[oh]/d[o]
  J[29] -= dqdci; // dwdot[ho2]/d[o]
  // d()/d[o2]
  dqdci = -k_r * sc[4];
  J[35] -= dqdci; // dwdot[o]/d[o2]
  J[36] += dqdci; // dwdot[o2]/d[o2]
  J[37] += dqdci; // dwdot[oh]/d[o2]
  J[40] -= dqdci; // dwdot[ho2]/d[o2]
  // d()/d[oh]
  dqdci = -k_r * sc[3];
  J[46] -= dqdci; // dwdot[o]/d[oh]
  J[47] += dqdci; // dwdot[o2]/d[oh]
  J[48] += dqdci; // dwdot[oh]/d[oh]
  J[51] -= dqdci; // dwdot[ho2]/d[oh]
  // d()/d[ho2]
  dqdci = +k_f * sc[2];
  J[79] -= dqdci; // dwdot[o]/d[ho2]
  J[80] += dqdci; // dwdot[o2]/d[ho2]
  J[81] += dqdci; // dwdot[oh]/d[ho2]
  J[84] -= dqdci; // dwdot[ho2]/d[ho2]
  // d()/dT
  J[112] -= dqdT; // dwdot[o]/dT
  J[113] += dqdT; // dwdot[o2]/dT
  J[114] += dqdT; // dwdot[oh]/dT
  J[117] -= dqdT; // dwdot[ho2]/dT

  // reaction 12: ho2 + oh <=> h2o + o2
  // a non-third-body and non-pressure-fall-off reaction
  // forward
  phi_f = sc[4] * sc[7];
  k_f = 28900000 * exp(-(-250.098682904946) * invT);
  dlnkfdT = (-250.098682904946) * invT2;
  // reverse
  phi_r = sc[3] * sc[5];
  Kc = exp(-g_RT[3] + g_RT[4] - g_RT[5] + g_RT[7]);
  k_r = k_f / Kc;
  dlnKcdT = invT * (-(h_RT[4] + h_RT[7]) + (h_RT[3] + h_RT[5]));
  dkrdT = (dlnkfdT - dlnKcdT) * k_r;
  // rate of progress
  q = k_f * phi_f - k_r * phi_r;
  dqdT = (dlnkfdT * k_f * phi_f - dkrdT * phi_r);
  // update wdot
  wdot[3] += q; // o2
  wdot[4] -= q; // oh
  wdot[5] += q; // h2o
  wdot[7] -= q; // ho2
  // d()/d[o2]
  dqdci = -k_r * sc[5];
  J[36] += dqdci; // dwdot[o2]/d[o2]
  J[37] -= dqdci; // dwdot[oh]/d[o2]
  J[38] += dqdci; // dwdot[h2o]/d[o2]
  J[40] -= dqdci; // dwdot[ho2]/d[o2]
  // d()/d[oh]
  dqdci = +k_f * sc[7];
  J[47] += dqdci; // dwdot[o2]/d[oh]
  J[48] -= dqdci; // dwdot[oh]/d[oh]
  J[49] += dqdci; // dwdot[h2o]/d[oh]
  J[51] -= dqdci; // dwdot[ho2]/d[oh]
  // d()/d[h2o]
  dqdci = -k_r * sc[3];
  J[58] += dqdci; // dwdot[o2]/d[h2o]
  J[59] -= dqdci; // dwdot[oh]/d[h2o]
  J[60] += dqdci; // dwdot[h2o]/d[h2o]
  J[62] -= dqdci; // dwdot[ho2]/d[h2o]
  // d()/d[ho2]
  dqdci = +k_f * sc[4];
  J[80] += dqdci; // dwdot[o2]/d[ho2]
  J[81] -= dqdci; // dwdot[oh]/d[ho2]
  J[82] += dqdci; // dwdot[h2o]/d[ho2]
  J[84] -= dqdci; // dwdot[ho2]/d[ho2]
  // d()/dT
  J[113] += dqdT; // dwdot[o2]/dT
  J[114] -= dqdT; // dwdot[oh]/dT
  J[115] += dqdT; // dwdot[h2o]/dT
  J[117] -= dqdT; // dwdot[ho2]/dT

  // reaction 13: 2 ho2 <=> h2o2 + o2
  // a non-third-body and non-pressure-fall-off reaction
  // forward
  phi_f = (sc[7] * sc[7]);
  k_f = 420000000 * exp(-(6028.53565634054) * invT);
  dlnkfdT = (6028.53565634054) * invT2;
  // reverse
  phi_r = sc[3] * sc[8];
  Kc = exp(-g_RT[3] + 2.000000 * g_RT[7] - g_RT[8]);
  k_r = k_f / Kc;
  dlnKcdT = invT * (-(2.000000 * h_RT[7]) + (h_RT[3] + h_RT[8]));
  dkrdT = (dlnkfdT - dlnKcdT) * k_r;
  // rate of progress
  q = k_f * phi_f - k_r * phi_r;
  dqdT = (dlnkfdT * k_f * phi_f - dkrdT * phi_r);
  // update wdot
  wdot[3] += q;     // o2
  wdot[7] -= 2 * q; // ho2
  wdot[8] += q;     // h2o2
  // d()/d[o2]
  dqdci = -k_r * sc[8];
  J[36] += dqdci;      // dwdot[o2]/d[o2]
  J[40] += -2 * dqdci; // dwdot[ho2]/d[o2]
  J[41] += dqdci;      // dwdot[h2o2]/d[o2]
  // d()/d[ho2]
  dqdci = +k_f * 2.000000 * sc[7];
  J[80] += dqdci;      // dwdot[o2]/d[ho2]
  J[84] += -2 * dqdci; // dwdot[ho2]/d[ho2]
  J[85] += dqdci;      // dwdot[h2o2]/d[ho2]
  // d()/d[h2o2]
  dqdci = -k_r * sc[3];
  J[91] += dqdci;      // dwdot[o2]/d[h2o2]
  J[95] += -2 * dqdci; // dwdot[ho2]/d[h2o2]
  J[96] += dqdci;      // dwdot[h2o2]/d[h2o2]
  // d()/dT
  J[113] += dqdT;      // dwdot[o2]/dT
  J[117] += -2 * dqdT; // dwdot[ho2]/dT
  J[118] += dqdT;      // dwdot[h2o2]/dT

  // reaction 14: 2 ho2 <=> h2o2 + o2
  // a non-third-body and non-pressure-fall-off reaction
  // forward
  phi_f = (sc[7] * sc[7]);
  k_f = 130000 * exp(-(-819.739948595888) * invT);
  dlnkfdT = (-819.739948595888) * invT2;
  // reverse
  phi_r = sc[3] * sc[8];
  Kc = exp(-g_RT[3] + 2.000000 * g_RT[7] - g_RT[8]);
  k_r = k_f / Kc;
  dlnKcdT = invT * (-(2.000000 * h_RT[7]) + (h_RT[3] + h_RT[8]));
  dkrdT = (dlnkfdT - dlnKcdT) * k_r;
  // rate of progress
  q = k_f * phi_f - k_r * phi_r;
  dqdT = (dlnkfdT * k_f * phi_f - dkrdT * phi_r);
  // update wdot
  wdot[3] += q;     // o2
  wdot[7] -= 2 * q; // ho2
  wdot[8] += q;     // h2o2
  // d()/d[o2]
  dqdci = -k_r * sc[8];
  J[36] += dqdci;      // dwdot[o2]/d[o2]
  J[40] += -2 * dqdci; // dwdot[ho2]/d[o2]
  J[41] += dqdci;      // dwdot[h2o2]/d[o2]
  // d()/d[ho2]
  dqdci = +k_f * 2.000000 * sc[7];
  J[80] += dqdci;      // dwdot[o2]/d[ho2]
  J[84] += -2 * dqdci; // dwdot[ho2]/d[ho2]
  J[85] += dqdci;      // dwdot[h2o2]/d[ho2]
  // d()/d[h2o2]
  dqdci = -k_r * sc[3];
  J[91] += dqdci;      // dwdot[o2]/d[h2o2]
  J[95] += -2 * dqdci; // dwdot[ho2]/d[h2o2]
  J[96] += dqdci;      // dwdot[h2o2]/d[h2o2]
  // d()/dT
  J[113] += dqdT;      // dwdot[o2]/dT
  J[117] += -2 * dqdT; // dwdot[ho2]/dT
  J[118] += dqdT;      // dwdot[h2o2]/dT

  // reaction 16: h + h2o2 <=> h2o + oh
  // a non-third-body and non-pressure-fall-off reaction
  // forward
  phi_f = sc[0] * sc[8];
  k_f = 24100000 * exp(-(1997.77016324474) * invT);
  dlnkfdT = (1997.77016324474) * invT2;
  // reverse
  phi_r = sc[4] * sc[5];
  Kc = exp(g_RT[0] - g_RT[4] - g_RT[5] + g_RT[8]);
  k_r = k_f / Kc;
  dlnKcdT = invT * (-(h_RT[0] + h_RT[8]) + (h_RT[4] + h_RT[5]));
  dkrdT = (dlnkfdT - dlnKcdT) * k_r;
  // rate of progress
  q = k_f * phi_f - k_r * phi_r;
  dqdT = (dlnkfdT * k_f * phi_f - dkrdT * phi_r);
  // update wdot
  wdot[0] -= q; // h
  wdot[4] += q; // oh
  wdot[5] += q; // h2o
  wdot[8] -= q; // h2o2
  // d()/d[h]
  dqdci = +k_f * sc[8];
  J[0] -= dqdci; // dwdot[h]/d[h]
  J[4] += dqdci; // dwdot[oh]/d[h]
  J[5] += dqdci; // dwdot[h2o]/d[h]
  J[8] -= dqdci; // dwdot[h2o2]/d[h]
  // d()/d[oh]
  dqdci = -k_r * sc[5];
  J[44] -= dqdci; // dwdot[h]/d[oh]
  J[48] += dqdci; // dwdot[oh]/d[oh]
  J[49] += dqdci; // dwdot[h2o]/d[oh]
  J[52] -= dqdci; // dwdot[h2o2]/d[oh]
  // d()/d[h2o]
  dqdci = -k_r * sc[4];
  J[55] -= dqdci; // dwdot[h]/d[h2o]
  J[59] += dqdci; // dwdot[oh]/d[h2o]
  J[60] += dqdci; // dwdot[h2o]/d[h2o]
  J[63] -= dqdci; // dwdot[h2o2]/d[h2o]
  // d()/d[h2o2]
  dqdci = +k_f * sc[0];
  J[88] -= dqdci; // dwdot[h]/d[h2o2]
  J[92] += dqdci; // dwdot[oh]/d[h2o2]
  J[93] += dqdci; // dwdot[h2o]/d[h2o2]
  J[96] -= dqdci; // dwdot[h2o2]/d[h2o2]
  // d()/dT
  J[110] -= dqdT; // dwdot[h]/dT
  J[114] += dqdT; // dwdot[oh]/dT
  J[115] += dqdT; // dwdot[h2o]/dT
  J[118] -= dqdT; // dwdot[h2o2]/dT

  // reaction 17: h + h2o2 <=> h2 + ho2
  // a non-third-body and non-pressure-fall-off reaction
  // forward
  phi_f = sc[0] * sc[8];
  k_f = 60250000 * exp(-(4000.57249314752) * invT);
  dlnkfdT = (4000.57249314752) * invT2;
  // reverse
  phi_r = sc[1] * sc[7];
  Kc = exp(g_RT[0] - g_RT[1] - g_RT[7] + g_RT[8]);
  k_r = k_f / Kc;
  dlnKcdT = invT * (-(h_RT[0] + h_RT[8]) + (h_RT[1] + h_RT[7]));
  dkrdT = (dlnkfdT - dlnKcdT) * k_r;
  // rate of progress
  q = k_f * phi_f - k_r * phi_r;
  dqdT = (dlnkfdT * k_f * phi_f - dkrdT * phi_r);
  // update wdot
  wdot[0] -= q; // h
  wdot[1] += q; // h2
  wdot[7] += q; // ho2
  wdot[8] -= q; // h2o2
  // d()/d[h]
  dqdci = +k_f * sc[8];
  J[0] -= dqdci; // dwdot[h]/d[h]
  J[1] += dqdci; // dwdot[h2]/d[h]
  J[7] += dqdci; // dwdot[ho2]/d[h]
  J[8] -= dqdci; // dwdot[h2o2]/d[h]
  // d()/d[h2]
  dqdci = -k_r * sc[7];
  J[11] -= dqdci; // dwdot[h]/d[h2]
  J[12] += dqdci; // dwdot[h2]/d[h2]
  J[18] += dqdci; // dwdot[ho2]/d[h2]
  J[19] -= dqdci; // dwdot[h2o2]/d[h2]
  // d()/d[ho2]
  dqdci = -k_r * sc[1];
  J[77] -= dqdci; // dwdot[h]/d[ho2]
  J[78] += dqdci; // dwdot[h2]/d[ho2]
  J[84] += dqdci; // dwdot[ho2]/d[ho2]
  J[85] -= dqdci; // dwdot[h2o2]/d[ho2]
  // d()/d[h2o2]
  dqdci = +k_f * sc[0];
  J[88] -= dqdci; // dwdot[h]/d[h2o2]
  J[89] += dqdci; // dwdot[h2]/d[h2o2]
  J[95] += dqdci; // dwdot[ho2]/d[h2o2]
  J[96] -= dqdci; // dwdot[h2o2]/d[h2o2]
  // d()/dT
  J[110] -= dqdT; // dwdot[h]/dT
  J[111] += dqdT; // dwdot[h2]/dT
  J[117] += dqdT; // dwdot[ho2]/dT
  J[118] -= dqdT; // dwdot[h2o2]/dT

  // reaction 18: h2o2 + o <=> ho2 + oh
  // a non-third-body and non-pressure-fall-off reaction
  // forward
  phi_f = sc[2] * sc[8];
  k_f = 9.55 * exp(2 * logT - (1997.77016324474) * invT);
  dlnkfdT = 2 * invT + (1997.77016324474) * invT2;
  // reverse
  phi_r = sc[4] * sc[7];
  Kc = exp(g_RT[2] - g_RT[4] - g_RT[7] + g_RT[8]);
  k_r = k_f / Kc;
  dlnKcdT = invT * (-(h_RT[2] + h_RT[8]) + (h_RT[4] + h_RT[7]));
  dkrdT = (dlnkfdT - dlnKcdT) * k_r;
  // rate of progress
  q = k_f * phi_f - k_r * phi_r;
  dqdT = (dlnkfdT * k_f * phi_f - dkrdT * phi_r);
  // update wdot
  wdot[2] -= q; // o
  wdot[4] += q; // oh
  wdot[7] += q; // ho2
  wdot[8] -= q; // h2o2
  // d()/d[o]
  dqdci = +k_f * sc[8];
  J[24] -= dqdci; // dwdot[o]/d[o]
  J[26] += dqdci; // dwdot[oh]/d[o]
  J[29] += dqdci; // dwdot[ho2]/d[o]
  J[30] -= dqdci; // dwdot[h2o2]/d[o]
  // d()/d[oh]
  dqdci = -k_r * sc[7];
  J[46] -= dqdci; // dwdot[o]/d[oh]
  J[48] += dqdci; // dwdot[oh]/d[oh]
  J[51] += dqdci; // dwdot[ho2]/d[oh]
  J[52] -= dqdci; // dwdot[h2o2]/d[oh]
  // d()/d[ho2]
  dqdci = -k_r * sc[4];
  J[79] -= dqdci; // dwdot[o]/d[ho2]
  J[81] += dqdci; // dwdot[oh]/d[ho2]
  J[84] += dqdci; // dwdot[ho2]/d[ho2]
  J[85] -= dqdci; // dwdot[h2o2]/d[ho2]
  // d()/d[h2o2]
  dqdci = +k_f * sc[2];
  J[90] -= dqdci; // dwdot[o]/d[h2o2]
  J[92] += dqdci; // dwdot[oh]/d[h2o2]
  J[95] += dqdci; // dwdot[ho2]/d[h2o2]
  J[96] -= dqdci; // dwdot[h2o2]/d[h2o2]
  // d()/dT
  J[112] -= dqdT; // dwdot[o]/dT
  J[114] += dqdT; // dwdot[oh]/dT
  J[117] += dqdT; // dwdot[ho2]/dT
  J[118] -= dqdT; // dwdot[h2o2]/dT

  // reaction 19: h2o2 + oh <=> h2o + ho2
  // a non-third-body and non-pressure-fall-off reaction
  // forward
  phi_f = sc[4] * sc[8];
  k_f = 1000000;
  dlnkfdT = 0.0;
  // reverse
  phi_r = sc[5] * sc[7];
  Kc = exp(g_RT[4] - g_RT[5] - g_RT[7] + g_RT[8]);
  k_r = k_f / Kc;
  dlnKcdT = invT * (-(h_RT[4] + h_RT[8]) + (h_RT[5] + h_RT[7]));
  dkrdT = (dlnkfdT - dlnKcdT) * k_r;
  // rate of progress
  q = k_f * phi_f - k_r * phi_r;
  dqdT = (dlnkfdT * k_f * phi_f - dkrdT * phi_r);
  // update wdot
  wdot[4] -= q; // oh
  wdot[5] += q; // h2o
  wdot[7] += q; // ho2
  wdot[8] -= q; // h2o2
  // d()/d[oh]
  dqdci = +k_f * sc[8];
  J[48] -= dqdci; // dwdot[oh]/d[oh]
  J[49] += dqdci; // dwdot[h2o]/d[oh]
  J[51] += dqdci; // dwdot[ho2]/d[oh]
  J[52] -= dqdci; // dwdot[h2o2]/d[oh]
  // d()/d[h2o]
  dqdci = -k_r * sc[7];
  J[59] -= dqdci; // dwdot[oh]/d[h2o]
  J[60] += dqdci; // dwdot[h2o]/d[h2o]
  J[62] += dqdci; // dwdot[ho2]/d[h2o]
  J[63] -= dqdci; // dwdot[h2o2]/d[h2o]
  // d()/d[ho2]
  dqdci = -k_r * sc[5];
  J[81] -= dqdci; // dwdot[oh]/d[ho2]
  J[82] += dqdci; // dwdot[h2o]/d[ho2]
  J[84] += dqdci; // dwdot[ho2]/d[ho2]
  J[85] -= dqdci; // dwdot[h2o2]/d[ho2]
  // d()/d[h2o2]
  dqdci = +k_f * sc[4];
  J[92] -= dqdci; // dwdot[oh]/d[h2o2]
  J[93] += dqdci; // dwdot[h2o]/d[h2o2]
  J[95] += dqdci; // dwdot[ho2]/d[h2o2]
  J[96] -= dqdci; // dwdot[h2o2]/d[h2o2]
  // d()/dT
  J[114] -= dqdT; // dwdot[oh]/dT
  J[115] += dqdT; // dwdot[h2o]/dT
  J[117] += dqdT; // dwdot[ho2]/dT
  J[118] -= dqdT; // dwdot[h2o2]/dT

  // reaction 20: h2o2 + oh <=> h2o + ho2
  // a non-third-body and non-pressure-fall-off reaction
  // forward
  phi_f = sc[4] * sc[8];
  k_f = 580000000 * exp(-(4809.24167509571) * invT);
  dlnkfdT = (4809.24167509571) * invT2;
  // reverse
  phi_r = sc[5] * sc[7];
  Kc = exp(g_RT[4] - g_RT[5] - g_RT[7] + g_RT[8]);
  k_r = k_f / Kc;
  dlnKcdT = invT * (-(h_RT[4] + h_RT[8]) + (h_RT[5] + h_RT[7]));
  dkrdT = (dlnkfdT - dlnKcdT) * k_r;
  // rate of progress
  q = k_f * phi_f - k_r * phi_r;
  dqdT = (dlnkfdT * k_f * phi_f - dkrdT * phi_r);
  // update wdot
  wdot[4] -= q; // oh
  wdot[5] += q; // h2o
  wdot[7] += q; // ho2
  wdot[8] -= q; // h2o2
  // d()/d[oh]
  dqdci = +k_f * sc[8];
  J[48] -= dqdci; // dwdot[oh]/d[oh]
  J[49] += dqdci; // dwdot[h2o]/d[oh]
  J[51] += dqdci; // dwdot[ho2]/d[oh]
  J[52] -= dqdci; // dwdot[h2o2]/d[oh]
  // d()/d[h2o]
  dqdci = -k_r * sc[7];
  J[59] -= dqdci; // dwdot[oh]/d[h2o]
  J[60] += dqdci; // dwdot[h2o]/d[h2o]
  J[62] += dqdci; // dwdot[ho2]/d[h2o]
  J[63] -= dqdci; // dwdot[h2o2]/d[h2o]
  // d()/d[ho2]
  dqdci = -k_r * sc[5];
  J[81] -= dqdci; // dwdot[oh]/d[ho2]
  J[82] += dqdci; // dwdot[h2o]/d[ho2]
  J[84] += dqdci; // dwdot[ho2]/d[ho2]
  J[85] -= dqdci; // dwdot[h2o2]/d[ho2]
  // d()/d[h2o2]
  dqdci = +k_f * sc[4];
  J[92] -= dqdci; // dwdot[oh]/d[h2o2]
  J[93] += dqdci; // dwdot[h2o]/d[h2o2]
  J[95] += dqdci; // dwdot[ho2]/d[h2o2]
  J[96] -= dqdci; // dwdot[h2o2]/d[h2o2]
  // d()/dT
  J[114] -= dqdT; // dwdot[oh]/dT
  J[115] += dqdT; // dwdot[h2o]/dT
  J[117] += dqdT; // dwdot[ho2]/dT
  J[118] -= dqdT; // dwdot[h2o2]/dT

  amrex::Real c_R[10], dcRdT[10], e_RT[10];
  amrex::Real* eh_RT;
  if (consP == 1) {
    cp_R(c_R, T);
    dcvpRdT(dcRdT, T);
    eh_RT = &h_RT[0];
  } else {
    cv_R(c_R, T);
    dcvpRdT(dcRdT, T);
    speciesInternalEnergy(e_RT, T);
    eh_RT = &e_RT[0];
  }

  amrex::Real cmix = 0.0, ehmix = 0.0, dcmixdT = 0.0, dehmixdT = 0.0;
  for (int k = 0; k < 10; ++k) {
    cmix += c_R[k] * sc[k];
    dcmixdT += dcRdT[k] * sc[k];
    ehmix += eh_RT[k] * wdot[k];
    dehmixdT += invT * (c_R[k] - eh_RT[k]) * wdot[k] + eh_RT[k] * J[110 + k];
  }

  amrex::Real cmixinv = 1.0 / cmix;
  amrex::Real tmp1 = ehmix * cmixinv;
  amrex::Real tmp3 = cmixinv * T;
  amrex::Real tmp2 = tmp1 * tmp3;
  amrex::Real dehmixdc;
  // dTdot/d[X]
  for (int k = 0; k < 10; ++k) {
    dehmixdc = 0.0;
    for (int m = 0; m < 10; ++m) {
      dehmixdc += eh_RT[m] * J[k * 11 + m];
    }
    J[k * 11 + 10] = tmp2 * c_R[k] - tmp3 * dehmixdc;
  }
  // dTdot/dT
  J[120] = -tmp1 + tmp2 * dcmixdT - tmp3 * dehmixdT;
#else
  amrex::Abort();
#endif
}

// compute the reaction Jacobian
AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE void
DWDOT(
  amrex::Real* J,
  const amrex::Real* sc,
  const amrex::Real* Tp,
  const int* consP)
{
  amrex::Real c[10];

  for (int k = 0; k < 10; k++) {
    c[k] = 1.e6 * sc[k];
  }

  aJacobian(J, c, *Tp, *consP);

  // dwdot[k]/dT
  // dTdot/d[X]
  for (int k = 0; k < 10; k++) {
    J[110 + k] *= 1.e-6;
    J[k * 11 + 10] *= 1.e6;
  }
}

// Transport function declarations

AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE void
egtransetLENIMC(int* LENIMC)
{
  *LENIMC = 42;
}

AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE void
egtransetLENRMC(int* LENRMC)
{
  *LENRMC = 2250;
}

AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE void
egtransetNO(int* NO)
{
  *NO = 4;
}

AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE void
egtransetKK(int* KK)
{
  *KK = 10;
}

AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE void
egtransetNLITE(int* NLITE)
{
  *NLITE = 2;
}

// Patm in ergs/cm3
AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE void
egtransetPATM(amrex::Real* PATM)
{
  *PATM = 0.1013250000000000E+07;
}

// the molecular weights in g/mol
AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE void
egtransetWT(amrex::Real* WT)
{
  WT[0] = 1.00800000E+00;
  WT[1] = 2.01600000E+00;
  WT[2] = 1.59990000E+01;
  WT[3] = 3.19980000E+01;
  WT[4] = 1.70070000E+01;
  WT[5] = 1.80150000E+01;
  WT[6] = 2.80140000E+01;
  WT[7] = 3.30060000E+01;
  WT[8] = 3.40140000E+01;
  WT[9] = 1.31293000E+02;
}

// the lennard-jones potential well depth eps/kb in K
AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE void
egtransetEPS(amrex::Real* EPS)
{
  EPS[0] = 1.45000000E+02;
  EPS[1] = 3.80000000E+01;
  EPS[2] = 8.00000000E+01;
  EPS[3] = 1.07400000E+02;
  EPS[4] = 8.00000000E+01;
  EPS[5] = 5.72400000E+02;
  EPS[6] = 9.75300000E+01;
  EPS[7] = 1.07400000E+02;
  EPS[8] = 1.07400000E+02;
  EPS[9] = 2.94000000E+02;
}

// the lennard-jones collision diameter in Angstroms
AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE void
egtransetSIG(amrex::Real* SIG)
{
  SIG[0] = 2.05000000E+00;
  SIG[1] = 2.92000000E+00;
  SIG[2] = 2.75000000E+00;
  SIG[3] = 3.45800000E+00;
  SIG[4] = 2.75000000E+00;
  SIG[5] = 2.60500000E+00;
  SIG[6] = 3.62100000E+00;
  SIG[7] = 3.45800000E+00;
  SIG[8] = 3.45800000E+00;
  SIG[9] = 4.10000000E+00;
}

// the dipole moment in Debye
AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE void
egtransetDIP(amrex::Real* DIP)
{
  DIP[0] = 0.00000000E+00;
  DIP[1] = 0.00000000E+00;
  DIP[2] = 0.00000000E+00;
  DIP[3] = 0.00000000E+00;
  DIP[4] = 0.00000000E+00;
  DIP[5] = 1.84400000E+00;
  DIP[6] = 0.00000000E+00;
  DIP[7] = 0.00000000E+00;
  DIP[8] = 0.00000000E+00;
  DIP[9] = 0.00000000E+00;
}

// the polarizability in cubic Angstroms
AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE void
egtransetPOL(amrex::Real* POL)
{
  POL[0] = 0.00000000E+00;
  POL[1] = 7.90000000E-01;
  POL[2] = 0.00000000E+00;
  POL[3] = 1.60000000E+00;
  POL[4] = 0.00000000E+00;
  POL[5] = 0.00000000E+00;
  POL[6] = 1.76000000E+00;
  POL[7] = 0.00000000E+00;
  POL[8] = 0.00000000E+00;
  POL[9] = 0.00000000E+00;
}

// the rotational relaxation collision number at 298 K
AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE void
egtransetZROT(amrex::Real* ZROT)
{
  ZROT[0] = 0.00000000E+00;
  ZROT[1] = 2.80000000E+02;
  ZROT[2] = 0.00000000E+00;
  ZROT[3] = 3.80000000E+00;
  ZROT[4] = 0.00000000E+00;
  ZROT[5] = 4.00000000E+00;
  ZROT[6] = 4.00000000E+00;
  ZROT[7] = 1.00000000E+00;
  ZROT[8] = 3.80000000E+00;
  ZROT[9] = 0.00000000E+00;
}

// 0: monoatomic, 1: linear, 2: nonlinear
AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE void
egtransetNLIN(int* NLIN)
{
  NLIN[0] = 0;
  NLIN[1] = 1;
  NLIN[2] = 0;
  NLIN[3] = 1;
  NLIN[4] = 1;
  NLIN[5] = 2;
  NLIN[6] = 1;
  NLIN[7] = 2;
  NLIN[8] = 2;
  NLIN[9] = 0;
}

// Poly fits for the viscosities, dim NO*KK
AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE void
egtransetCOFETA(amrex::Real* COFETA)
{
  COFETA[0] = -1.98744352E+01;
  COFETA[1] = 3.41660514E+00;
  COFETA[2] = -3.63206306E-01;
  COFETA[3] = 1.58671021E-02;
  COFETA[4] = -1.37549291E+01;
  COFETA[5] = 9.65530587E-01;
  COFETA[6] = -4.45720114E-02;
  COFETA[7] = 2.05871810E-03;
  COFETA[8] = -1.48001711E+01;
  COFETA[9] = 1.79491990E+00;
  COFETA[10] = -1.54008440E-01;
  COFETA[11] = 6.86719439E-03;
  COFETA[12] = -1.68118998E+01;
  COFETA[13] = 2.52362554E+00;
  COFETA[14] = -2.49309128E-01;
  COFETA[15] = 1.10211025E-02;
  COFETA[16] = -1.47696217E+01;
  COFETA[17] = 1.79491990E+00;
  COFETA[18] = -1.54008440E-01;
  COFETA[19] = 6.86719439E-03;
  COFETA[20] = -1.17770995E+01;
  COFETA[21] = -8.26744826E-01;
  COFETA[22] = 3.39009392E-01;
  COFETA[23] = -2.00674472E-02;
  COFETA[24] = -1.62526676E+01;
  COFETA[25] = 2.24839597E+00;
  COFETA[26] = -2.13428438E-01;
  COFETA[27] = 9.46192413E-03;
  COFETA[28] = -1.67963918E+01;
  COFETA[29] = 2.52362554E+00;
  COFETA[30] = -2.49309128E-01;
  COFETA[31] = 1.10211025E-02;
  COFETA[32] = -1.67813504E+01;
  COFETA[33] = 2.52362554E+00;
  COFETA[34] = -2.49309128E-01;
  COFETA[35] = 1.10211025E-02;
  COFETA[36] = -2.42036069E+01;
  COFETA[37] = 5.23221350E+00;
  COFETA[38] = -5.70348707E-01;
  COFETA[39] = 2.36892944E-02;
}

// Poly fits for the conductivities, dim NO*KK
AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE void
egtransetCOFLAM(amrex::Real* COFLAM)
{
  COFLAM[0] = -3.24554543E-01;
  COFLAM[1] = 3.41660514E+00;
  COFLAM[2] = -3.63206306E-01;
  COFLAM[3] = 1.58671021E-02;
  COFLAM[4] = 1.11035309E+01;
  COFLAM[5] = -1.31883723E+00;
  COFLAM[6] = 2.44042222E-01;
  COFLAM[7] = -8.99835317E-03;
  COFLAM[8] = 1.98515155E+00;
  COFLAM[9] = 1.79491990E+00;
  COFLAM[10] = -1.54008440E-01;
  COFLAM[11] = 6.86719439E-03;
  COFLAM[12] = -2.51295522E+00;
  COFLAM[13] = 3.15165699E+00;
  COFLAM[14] = -3.10007687E-01;
  COFLAM[15] = 1.34522334E-02;
  COFLAM[16] = 1.24593748E+01;
  COFLAM[17] = -2.63918077E+00;
  COFLAM[18] = 4.66620705E-01;
  COFLAM[19] = -2.10920232E-02;
  COFLAM[20] = 2.21730553E+01;
  COFLAM[21] = -8.46935470E+00;
  COFLAM[22] = 1.46153792E+00;
  COFLAM[23] = -7.29502313E-02;
  COFLAM[24] = 1.15507226E+01;
  COFLAM[25] = -2.91453572E+00;
  COFLAM[26] = 5.55045264E-01;
  COFLAM[27] = -2.75173246E-02;
  COFLAM[28] = 5.56033781E-01;
  COFLAM[29] = 1.59073629E+00;
  COFLAM[30] = -5.28054537E-02;
  COFLAM[31] = 4.07605156E-04;
  COFLAM[32] = 1.48802076E+00;
  COFLAM[33] = 1.06176261E+00;
  COFLAM[34] = 5.72195029E-02;
  COFLAM[35] = -6.38391114E-03;
  COFLAM[36] = -9.52318951E+00;
  COFLAM[37] = 5.23221350E+00;
  COFLAM[38] = -5.70348707E-01;
  COFLAM[39] = 2.36892944E-02;
}

// Poly fits for the diffusion coefficients, dim NO*KK*KK
AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE void
egtransetCOFD(amrex::Real* COFD)
{
  COFD[0] = -1.43693219E+01;
  COFD[1] = 4.03992999E+00;
  COFD[2] = -3.08044800E-01;
  COFD[3] = 1.32757775E-02;
  COFD[4] = -1.11808845E+01;
  COFD[5] = 2.66936727E+00;
  COFD[6] = -1.34411514E-01;
  COFD[7] = 5.92957488E-03;
  COFD[8] = -1.31860264E+01;
  COFD[9] = 3.38003453E+00;
  COFD[10] = -2.25783856E-01;
  COFD[11] = 9.85028660E-03;
  COFD[12] = -1.43713019E+01;
  COFD[13] = 3.70920439E+00;
  COFD[14] = -2.67274113E-01;
  COFD[15] = 1.15967481E-02;
  COFD[16] = -1.31877859E+01;
  COFD[17] = 3.38003453E+00;
  COFD[18] = -2.25783856E-01;
  COFD[19] = 9.85028660E-03;
  COFD[20] = -1.93611201E+01;
  COFD[21] = 5.51579726E+00;
  COFD[22] = -4.76061961E-01;
  COFD[23] = 1.96329391E-02;
  COFD[24] = -1.40298991E+01;
  COFD[25] = 3.55837688E+00;
  COFD[26] = -2.47785790E-01;
  COFD[27] = 1.07555332E-02;
  COFD[28] = -1.43717684E+01;
  COFD[29] = 3.70920439E+00;
  COFD[30] = -2.67274113E-01;
  COFD[31] = 1.15967481E-02;
  COFD[32] = -1.43722077E+01;
  COFD[33] = 3.70920439E+00;
  COFD[34] = -2.67274113E-01;
  COFD[35] = 1.15967481E-02;
  COFD[36] = -1.81087358E+01;
  COFD[37] = 4.95820816E+00;
  COFD[38] = -4.19498540E-01;
  COFD[39] = 1.78009875E-02;
  COFD[40] = -1.11808845E+01;
  COFD[41] = 2.66936727E+00;
  COFD[42] = -1.34411514E-01;
  COFD[43] = 5.92957488E-03;
  COFD[44] = -1.02395385E+01;
  COFD[45] = 2.15403244E+00;
  COFD[46] = -6.97480266E-02;
  COFD[47] = 3.23666871E-03;
  COFD[48] = -1.06250314E+01;
  COFD[49] = 2.15849701E+00;
  COFD[50] = -6.53886401E-02;
  COFD[51] = 2.81453370E-03;
  COFD[52] = -1.15797897E+01;
  COFD[53] = 2.43235504E+00;
  COFD[54] = -1.02890179E-01;
  COFD[55] = 4.52903603E-03;
  COFD[56] = -1.06283588E+01;
  COFD[57] = 2.15849701E+00;
  COFD[58] = -6.53886401E-02;
  COFD[59] = 2.81453370E-03;
  COFD[60] = -1.68759075E+01;
  COFD[61] = 4.49460341E+00;
  COFD[62] = -3.64766179E-01;
  COFD[63] = 1.56457173E-02;
  COFD[64] = -1.13253618E+01;
  COFD[65] = 2.31195095E+00;
  COFD[66] = -8.63988037E-02;
  COFD[67] = 3.77573452E-03;
  COFD[68] = -1.15806955E+01;
  COFD[69] = 2.43235504E+00;
  COFD[70] = -1.02890179E-01;
  COFD[71] = 4.52903603E-03;
  COFD[72] = -1.15815492E+01;
  COFD[73] = 2.43235504E+00;
  COFD[74] = -1.02890179E-01;
  COFD[75] = 4.52903603E-03;
  COFD[76] = -1.41947262E+01;
  COFD[77] = 3.33454102E+00;
  COFD[78] = -2.19933668E-01;
  COFD[79] = 9.59902063E-03;
  COFD[80] = -1.31860264E+01;
  COFD[81] = 3.38003453E+00;
  COFD[82] = -2.25783856E-01;
  COFD[83] = 9.85028660E-03;
  COFD[84] = -1.06250314E+01;
  COFD[85] = 2.15849701E+00;
  COFD[86] = -6.53886401E-02;
  COFD[87] = 2.81453370E-03;
  COFD[88] = -1.29877255E+01;
  COFD[89] = 2.80841511E+00;
  COFD[90] = -1.52629888E-01;
  COFD[91] = 6.72604927E-03;
  COFD[92] = -1.40864784E+01;
  COFD[93] = 3.07458927E+00;
  COFD[94] = -1.86899591E-01;
  COFD[95] = 8.19829781E-03;
  COFD[96] = -1.30027669E+01;
  COFD[97] = 2.80841511E+00;
  COFD[98] = -1.52629888E-01;
  COFD[99] = 6.72604927E-03;
  COFD[100] = -1.91096700E+01;
  COFD[101] = 5.02608697E+00;
  COFD[102] = -4.26959993E-01;
  COFD[103] = 1.80709910E-02;
  COFD[104] = -1.38756380E+01;
  COFD[105] = 2.98558426E+00;
  COFD[106] = -1.75507216E-01;
  COFD[107] = 7.71173691E-03;
  COFD[108] = -1.40915944E+01;
  COFD[109] = 3.07458927E+00;
  COFD[110] = -1.86899591E-01;
  COFD[111] = 8.19829781E-03;
  COFD[112] = -1.40964555E+01;
  COFD[113] = 3.07458927E+00;
  COFD[114] = -1.86899591E-01;
  COFD[115] = 8.19829781E-03;
  COFD[116] = -1.74450955E+01;
  COFD[117] = 4.17532821E+00;
  COFD[118] = -3.24644774E-01;
  COFD[119] = 1.39561994E-02;
  COFD[120] = -1.43713019E+01;
  COFD[121] = 3.70920439E+00;
  COFD[122] = -2.67274113E-01;
  COFD[123] = 1.15967481E-02;
  COFD[124] = -1.15797897E+01;
  COFD[125] = 2.43235504E+00;
  COFD[126] = -1.02890179E-01;
  COFD[127] = 4.52903603E-03;
  COFD[128] = -1.40864784E+01;
  COFD[129] = 3.07458927E+00;
  COFD[130] = -1.86899591E-01;
  COFD[131] = 8.19829781E-03;
  COFD[132] = -1.53110597E+01;
  COFD[133] = 3.37317428E+00;
  COFD[134] = -2.24900439E-01;
  COFD[135] = 9.81228151E-03;
  COFD[136] = -1.41066358E+01;
  COFD[137] = 3.07458927E+00;
  COFD[138] = -1.86899591E-01;
  COFD[139] = 8.19829781E-03;
  COFD[140] = -2.10639926E+01;
  COFD[141] = 5.50980703E+00;
  COFD[142] = -4.78335492E-01;
  COFD[143] = 1.98515434E-02;
  COFD[144] = -1.50096252E+01;
  COFD[145] = 3.25515933E+00;
  COFD[146] = -2.09710110E-01;
  COFD[147] = 9.15941830E-03;
  COFD[148] = -1.53187536E+01;
  COFD[149] = 3.37317428E+00;
  COFD[150] = -2.24900439E-01;
  COFD[151] = 9.81228151E-03;
  COFD[152] = -1.53261011E+01;
  COFD[153] = 3.37317428E+00;
  COFD[154] = -2.24900439E-01;
  COFD[155] = 9.81228151E-03;
  COFD[156] = -1.91005319E+01;
  COFD[157] = 4.59877085E+00;
  COFD[158] = -3.77297245E-01;
  COFD[159] = 1.61493276E-02;
  COFD[160] = -1.31877859E+01;
  COFD[161] = 3.38003453E+00;
  COFD[162] = -2.25783856E-01;
  COFD[163] = 9.85028660E-03;
  COFD[164] = -1.06283588E+01;
  COFD[165] = 2.15849701E+00;
  COFD[166] = -6.53886401E-02;
  COFD[167] = 2.81453370E-03;
  COFD[168] = -1.30027669E+01;
  COFD[169] = 2.80841511E+00;
  COFD[170] = -1.52629888E-01;
  COFD[171] = 6.72604927E-03;
  COFD[172] = -1.41066358E+01;
  COFD[173] = 3.07458927E+00;
  COFD[174] = -1.86899591E-01;
  COFD[175] = 8.19829781E-03;
  COFD[176] = -1.30182749E+01;
  COFD[177] = 2.80841511E+00;
  COFD[178] = -1.52629888E-01;
  COFD[179] = 6.72604927E-03;
  COFD[180] = -1.91256173E+01;
  COFD[181] = 5.02608697E+00;
  COFD[182] = -4.26959993E-01;
  COFD[183] = 1.80709910E-02;
  COFD[184] = -1.38948654E+01;
  COFD[185] = 2.98558426E+00;
  COFD[186] = -1.75507216E-01;
  COFD[187] = 7.71173691E-03;
  COFD[188] = -1.41119635E+01;
  COFD[189] = 3.07458927E+00;
  COFD[190] = -1.86899591E-01;
  COFD[191] = 8.19829781E-03;
  COFD[192] = -1.41170278E+01;
  COFD[193] = 3.07458927E+00;
  COFD[194] = -1.86899591E-01;
  COFD[195] = 8.19829781E-03;
  COFD[196] = -1.74722348E+01;
  COFD[197] = 4.17532821E+00;
  COFD[198] = -3.24644774E-01;
  COFD[199] = 1.39561994E-02;
  COFD[200] = -1.93611201E+01;
  COFD[201] = 5.51579726E+00;
  COFD[202] = -4.76061961E-01;
  COFD[203] = 1.96329391E-02;
  COFD[204] = -1.68759075E+01;
  COFD[205] = 4.49460341E+00;
  COFD[206] = -3.64766179E-01;
  COFD[207] = 1.56457173E-02;
  COFD[208] = -1.91096700E+01;
  COFD[209] = 5.02608697E+00;
  COFD[210] = -4.26959993E-01;
  COFD[211] = 1.80709910E-02;
  COFD[212] = -2.10639926E+01;
  COFD[213] = 5.50980703E+00;
  COFD[214] = -4.78335492E-01;
  COFD[215] = 1.98515434E-02;
  COFD[216] = -1.91256173E+01;
  COFD[217] = 5.02608697E+00;
  COFD[218] = -4.26959993E-01;
  COFD[219] = 1.80709910E-02;
  COFD[220] = -1.31492509E+01;
  COFD[221] = 1.48004045E+00;
  COFD[222] = 1.60499956E-01;
  COFD[223] = -1.19765871E-02;
  COFD[224] = -2.08123331E+01;
  COFD[225] = 5.42470175E+00;
  COFD[226] = -4.69700438E-01;
  COFD[227] = 1.95706912E-02;
  COFD[228] = -2.04177394E+01;
  COFD[229] = 5.31457079E+00;
  COFD[230] = -4.58216496E-01;
  COFD[231] = 1.91825910E-02;
  COFD[232] = -2.04229988E+01;
  COFD[233] = 5.31457079E+00;
  COFD[234] = -4.58216496E-01;
  COFD[235] = 1.91825910E-02;
  COFD[236] = -2.13840973E+01;
  COFD[237] = 5.20008362E+00;
  COFD[238] = -4.06087465E-01;
  COFD[239] = 1.54825257E-02;
  COFD[240] = -1.40298991E+01;
  COFD[241] = 3.55837688E+00;
  COFD[242] = -2.47785790E-01;
  COFD[243] = 1.07555332E-02;
  COFD[244] = -1.13253618E+01;
  COFD[245] = 2.31195095E+00;
  COFD[246] = -8.63988037E-02;
  COFD[247] = 3.77573452E-03;
  COFD[248] = -1.38756380E+01;
  COFD[249] = 2.98558426E+00;
  COFD[250] = -1.75507216E-01;
  COFD[251] = 7.71173691E-03;
  COFD[252] = -1.50096252E+01;
  COFD[253] = 3.25515933E+00;
  COFD[254] = -2.09710110E-01;
  COFD[255] = 9.15941830E-03;
  COFD[256] = -1.38948654E+01;
  COFD[257] = 2.98558426E+00;
  COFD[258] = -1.75507216E-01;
  COFD[259] = 7.71173691E-03;
  COFD[260] = -2.08123331E+01;
  COFD[261] = 5.42470175E+00;
  COFD[262] = -4.69700438E-01;
  COFD[263] = 1.95706912E-02;
  COFD[264] = -1.47639411E+01;
  COFD[265] = 3.15955654E+00;
  COFD[266] = -1.97590757E-01;
  COFD[267] = 8.64692156E-03;
  COFD[268] = -1.50168047E+01;
  COFD[269] = 3.25515933E+00;
  COFD[270] = -2.09710110E-01;
  COFD[271] = 9.15941830E-03;
  COFD[272] = -1.50236540E+01;
  COFD[273] = 3.25515933E+00;
  COFD[274] = -2.09710110E-01;
  COFD[275] = 9.15941830E-03;
  COFD[276] = -1.87415595E+01;
  COFD[277] = 4.47576226E+00;
  COFD[278] = -3.62421189E-01;
  COFD[279] = 1.55479328E-02;
  COFD[280] = -1.43717684E+01;
  COFD[281] = 3.70920439E+00;
  COFD[282] = -2.67274113E-01;
  COFD[283] = 1.15967481E-02;
  COFD[284] = -1.15806955E+01;
  COFD[285] = 2.43235504E+00;
  COFD[286] = -1.02890179E-01;
  COFD[287] = 4.52903603E-03;
  COFD[288] = -1.40915944E+01;
  COFD[289] = 3.07458927E+00;
  COFD[290] = -1.86899591E-01;
  COFD[291] = 8.19829781E-03;
  COFD[292] = -1.53187536E+01;
  COFD[293] = 3.37317428E+00;
  COFD[294] = -2.24900439E-01;
  COFD[295] = 9.81228151E-03;
  COFD[296] = -1.41119635E+01;
  COFD[297] = 3.07458927E+00;
  COFD[298] = -1.86899591E-01;
  COFD[299] = 8.19829781E-03;
  COFD[300] = -2.04177394E+01;
  COFD[301] = 5.31457079E+00;
  COFD[302] = -4.58216496E-01;
  COFD[303] = 1.91825910E-02;
  COFD[304] = -1.50168047E+01;
  COFD[305] = 3.25515933E+00;
  COFD[306] = -2.09710110E-01;
  COFD[307] = 9.15941830E-03;
  COFD[308] = -1.53265677E+01;
  COFD[309] = 3.37317428E+00;
  COFD[310] = -2.24900439E-01;
  COFD[311] = 9.81228151E-03;
  COFD[312] = -1.53340319E+01;
  COFD[313] = 3.37317428E+00;
  COFD[314] = -2.24900439E-01;
  COFD[315] = 9.81228151E-03;
  COFD[316] = -1.91129628E+01;
  COFD[317] = 4.59877085E+00;
  COFD[318] = -3.77297245E-01;
  COFD[319] = 1.61493276E-02;
  COFD[320] = -1.43722077E+01;
  COFD[321] = 3.70920439E+00;
  COFD[322] = -2.67274113E-01;
  COFD[323] = 1.15967481E-02;
  COFD[324] = -1.15815492E+01;
  COFD[325] = 2.43235504E+00;
  COFD[326] = -1.02890179E-01;
  COFD[327] = 4.52903603E-03;
  COFD[328] = -1.40964555E+01;
  COFD[329] = 3.07458927E+00;
  COFD[330] = -1.86899591E-01;
  COFD[331] = 8.19829781E-03;
  COFD[332] = -1.53261011E+01;
  COFD[333] = 3.37317428E+00;
  COFD[334] = -2.24900439E-01;
  COFD[335] = 9.81228151E-03;
  COFD[336] = -1.41170278E+01;
  COFD[337] = 3.07458927E+00;
  COFD[338] = -1.86899591E-01;
  COFD[339] = 8.19829781E-03;
  COFD[340] = -2.04229988E+01;
  COFD[341] = 5.31457079E+00;
  COFD[342] = -4.58216496E-01;
  COFD[343] = 1.91825910E-02;
  COFD[344] = -1.50236540E+01;
  COFD[345] = 3.25515933E+00;
  COFD[346] = -2.09710110E-01;
  COFD[347] = 9.15941830E-03;
  COFD[348] = -1.53340319E+01;
  COFD[349] = 3.37317428E+00;
  COFD[350] = -2.24900439E-01;
  COFD[351] = 9.81228151E-03;
  COFD[352] = -1.53416091E+01;
  COFD[353] = 3.37317428E+00;
  COFD[354] = -2.24900439E-01;
  COFD[355] = 9.81228151E-03;
  COFD[356] = -1.91249460E+01;
  COFD[357] = 4.59877085E+00;
  COFD[358] = -3.77297245E-01;
  COFD[359] = 1.61493276E-02;
  COFD[360] = -1.81087358E+01;
  COFD[361] = 4.95820816E+00;
  COFD[362] = -4.19498540E-01;
  COFD[363] = 1.78009875E-02;
  COFD[364] = -1.41947262E+01;
  COFD[365] = 3.33454102E+00;
  COFD[366] = -2.19933668E-01;
  COFD[367] = 9.59902063E-03;
  COFD[368] = -1.74450955E+01;
  COFD[369] = 4.17532821E+00;
  COFD[370] = -3.24644774E-01;
  COFD[371] = 1.39561994E-02;
  COFD[372] = -1.91005319E+01;
  COFD[373] = 4.59877085E+00;
  COFD[374] = -3.77297245E-01;
  COFD[375] = 1.61493276E-02;
  COFD[376] = -1.74722348E+01;
  COFD[377] = 4.17532821E+00;
  COFD[378] = -3.24644774E-01;
  COFD[379] = 1.39561994E-02;
  COFD[380] = -2.13840973E+01;
  COFD[381] = 5.20008362E+00;
  COFD[382] = -4.06087465E-01;
  COFD[383] = 1.54825257E-02;
  COFD[384] = -1.87415595E+01;
  COFD[385] = 4.47576226E+00;
  COFD[386] = -3.62421189E-01;
  COFD[387] = 1.55479328E-02;
  COFD[388] = -1.91129628E+01;
  COFD[389] = 4.59877085E+00;
  COFD[390] = -3.77297245E-01;
  COFD[391] = 1.61493276E-02;
  COFD[392] = -1.91249460E+01;
  COFD[393] = 4.59877085E+00;
  COFD[394] = -3.77297245E-01;
  COFD[395] = 1.61493276E-02;
  COFD[396] = -2.26405731E+01;
  COFD[397] = 5.51491566E+00;
  COFD[398] = -4.74542065E-01;
  COFD[399] = 1.95120022E-02;
}

// List of specs with small weight, dim NLITE
AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE void
egtransetKTDIF(int* KTDIF)
{
  KTDIF[0] = 0;
  KTDIF[1] = 1;
}

// Poly fits for thermal diff ratios, dim NO*NLITE*KK
AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE void
egtransetCOFTD(amrex::Real* COFTD)
{
  COFTD[0] = 0.00000000E+00;
  COFTD[1] = 0.00000000E+00;
  COFTD[2] = 0.00000000E+00;
  COFTD[3] = 0.00000000E+00;
  COFTD[4] = 1.52534742E-01;
  COFTD[5] = 5.46404022E-05;
  COFTD[6] = -2.93412470E-08;
  COFTD[7] = 4.87091914E-12;
  COFTD[8] = 2.70008280E-01;
  COFTD[9] = 3.61552588E-04;
  COFTD[10] = -1.80743499E-07;
  COFTD[11] = 2.75319341E-11;
  COFTD[12] = 2.20482081E-01;
  COFTD[13] = 4.80162630E-04;
  COFTD[14] = -2.32927140E-07;
  COFTD[15] = 3.46469239E-11;
  COFTD[16] = 2.72039997E-01;
  COFTD[17] = 3.64273144E-04;
  COFTD[18] = -1.82103531E-07;
  COFTD[19] = 2.77391022E-11;
  COFTD[20] = -1.41882969E-01;
  COFTD[21] = 7.66554625E-04;
  COFTD[22] = -3.06548330E-07;
  COFTD[23] = 4.02957302E-11;
  COFTD[24] = 2.40744276E-01;
  COFTD[25] = 4.45343183E-04;
  COFTD[26] = -2.18173742E-07;
  COFTD[27] = 3.26958310E-11;
  COFTD[28] = 2.20907136E-01;
  COFTD[29] = 4.81088308E-04;
  COFTD[30] = -2.33376187E-07;
  COFTD[31] = 3.47137178E-11;
  COFTD[32] = 2.21307723E-01;
  COFTD[33] = 4.81960701E-04;
  COFTD[34] = -2.33799385E-07;
  COFTD[35] = 3.47766667E-11;
  COFTD[36] = -1.95236803E-02;
  COFTD[37] = 7.78679260E-04;
  COFTD[38] = -3.39434193E-07;
  COFTD[39] = 4.70346869E-11;
  COFTD[40] = -1.52534742E-01;
  COFTD[41] = -5.46404022E-05;
  COFTD[42] = 2.93412470E-08;
  COFTD[43] = -4.87091914E-12;
  COFTD[44] = 0.00000000E+00;
  COFTD[45] = 0.00000000E+00;
  COFTD[46] = 0.00000000E+00;
  COFTD[47] = 0.00000000E+00;
  COFTD[48] = 4.15577509E-01;
  COFTD[49] = 1.09736860E-05;
  COFTD[50] = -3.96016410E-09;
  COFTD[51] = 1.14412838E-12;
  COFTD[52] = 4.42736017E-01;
  COFTD[53] = 7.11765887E-05;
  COFTD[54] = -3.84765396E-08;
  COFTD[55] = 6.86318682E-12;
  COFTD[56] = 4.21927217E-01;
  COFTD[57] = 1.11413555E-05;
  COFTD[58] = -4.02067239E-09;
  COFTD[59] = 1.16160979E-12;
  COFTD[60] = 6.02021050E-02;
  COFTD[61] = 5.61555733E-04;
  COFTD[62] = -2.55370065E-07;
  COFTD[63] = 3.63385926E-11;
  COFTD[64] = 4.45261429E-01;
  COFTD[65] = 4.94696577E-05;
  COFTD[66] = -2.63023124E-08;
  COFTD[67] = 4.90305626E-12;
  COFTD[68] = 4.44449675E-01;
  COFTD[69] = 7.14520855E-05;
  COFTD[70] = -3.86254673E-08;
  COFTD[71] = 6.88975155E-12;
  COFTD[72] = 4.46067449E-01;
  COFTD[73] = 7.17121674E-05;
  COFTD[74] = -3.87660620E-08;
  COFTD[75] = 6.91482989E-12;
  COFTD[76] = 3.05551565E-01;
  COFTD[77] = 3.85126233E-04;
  COFTD[78] = -1.93268064E-07;
  COFTD[79] = 2.95362433E-11;
}

// compute the critical parameters for each species
AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE void
GET_CRITPARAMS(
  amrex::Real* Tci, amrex::Real* ai, amrex::Real* bi, amrex::Real* acentric_i)
{

  amrex::Real EPS[10];
  amrex::Real SIG[10];
  amrex::Real wt[10];
  amrex::Real Rcst = 83.144598; // in bar [CGS] !
  amrex::Real avogadro = 6.02214199e23;
  amrex::Real boltzmann = 1.3806503e-16; // we work in CGS

  egtransetEPS(EPS);
  egtransetSIG(SIG);
  get_mw(wt);

  // species 0: h
  Tci[0] = 1.316 * EPS[0];
  ai[0] = (5.55 * avogadro * avogadro * EPS[0] * boltzmann * 1e-24 * SIG[0] *
           SIG[0] * SIG[0]) /
          (wt[0] * wt[0]);
  bi[0] = 0.855 * avogadro * 1e-24 * SIG[0] * SIG[0] * SIG[0] / (wt[0]);
  acentric_i[0] = 0.0;

  // species 1: h2
  Tci[1] = 1.316 * EPS[1];
  ai[1] = (5.55 * avogadro * avogadro * EPS[1] * boltzmann * 1e-24 * SIG[1] *
           SIG[1] * SIG[1]) /
          (wt[1] * wt[1]);
  bi[1] = 0.855 * avogadro * 1e-24 * SIG[1] * SIG[1] * SIG[1] / (wt[1]);
  acentric_i[1] = 0.0;

  // species 2: o
  Tci[2] = 1.316 * EPS[2];
  ai[2] = (5.55 * avogadro * avogadro * EPS[2] * boltzmann * 1e-24 * SIG[2] *
           SIG[2] * SIG[2]) /
          (wt[2] * wt[2]);
  bi[2] = 0.855 * avogadro * 1e-24 * SIG[2] * SIG[2] * SIG[2] / (wt[2]);
  acentric_i[2] = 0.0;

  // species 3: o2
  Tci[3] = 1.316 * EPS[3];
  ai[3] = (5.55 * avogadro * avogadro * EPS[3] * boltzmann * 1e-24 * SIG[3] *
           SIG[3] * SIG[3]) /
          (wt[3] * wt[3]);
  bi[3] = 0.855 * avogadro * 1e-24 * SIG[3] * SIG[3] * SIG[3] / (wt[3]);
  acentric_i[3] = 0.0;

  // species 4: oh
  Tci[4] = 1.316 * EPS[4];
  ai[4] = (5.55 * avogadro * avogadro * EPS[4] * boltzmann * 1e-24 * SIG[4] *
           SIG[4] * SIG[4]) /
          (wt[4] * wt[4]);
  bi[4] = 0.855 * avogadro * 1e-24 * SIG[4] * SIG[4] * SIG[4] / (wt[4]);
  acentric_i[4] = 0.0;

  // species 5: h2o
  Tci[5] = 1.316 * EPS[5];
  ai[5] = (5.55 * avogadro * avogadro * EPS[5] * boltzmann * 1e-24 * SIG[5] *
           SIG[5] * SIG[5]) /
          (wt[5] * wt[5]);
  bi[5] = 0.855 * avogadro * 1e-24 * SIG[5] * SIG[5] * SIG[5] / (wt[5]);
  acentric_i[5] = 0.0;

  // species 6: n2
  Tci[6] = 1.316 * EPS[6];
  ai[6] = (5.55 * avogadro * avogadro * EPS[6] * boltzmann * 1e-24 * SIG[6] *
           SIG[6] * SIG[6]) /
          (wt[6] * wt[6]);
  bi[6] = 0.855 * avogadro * 1e-24 * SIG[6] * SIG[6] * SIG[6] / (wt[6]);
  acentric_i[6] = 0.0;

  // species 7: ho2
  Tci[7] = 1.316 * EPS[7];
  ai[7] = (5.55 * avogadro * avogadro * EPS[7] * boltzmann * 1e-24 * SIG[7] *
           SIG[7] * SIG[7]) /
          (wt[7] * wt[7]);
  bi[7] = 0.855 * avogadro * 1e-24 * SIG[7] * SIG[7] * SIG[7] / (wt[7]);
  acentric_i[7] = 0.0;

  // species 8: h2o2
  Tci[8] = 1.316 * EPS[8];
  ai[8] = (5.55 * avogadro * avogadro * EPS[8] * boltzmann * 1e-24 * SIG[8] *
           SIG[8] * SIG[8]) /
          (wt[8] * wt[8]);
  bi[8] = 0.855 * avogadro * 1e-24 * SIG[8] * SIG[8] * SIG[8] / (wt[8]);
  acentric_i[8] = 0.0;

  // species 9: Xe
  Tci[9] = 1.316 * EPS[9];
  ai[9] = (5.55 * avogadro * avogadro * EPS[9] * boltzmann * 1e-24 * SIG[9] *
           SIG[9] * SIG[9]) /
          (wt[9] * wt[9]);
  bi[9] = 0.855 * avogadro * 1e-24 * SIG[9] * SIG[9] * SIG[9] / (wt[9]);
  acentric_i[9] = 0.0;
}

// compute the critical parameter quantities for each species for SRK
AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE void
GET_CRITPARAMS_SRK(
  amrex::Real* sqrtOneOverTc,
  amrex::Real* sqrtAsti,
  amrex::Real* Bi,
  amrex::Real* Fomega)
{

  // species 0: h
  // Computed from Lennard-Jones
  sqrtOneOverTc[0] = 7.2391579785311e-02;
  sqrtAsti[0] = 5.8451221185793e+05;
  Bi[0] = 4.4006634563410e+00;
  Fomega[0] = 4.8508000000000e-01;

  // species 1: h2
  // Computed from Lennard-Jones
  sqrtOneOverTc[1] = 1.4141004388627e-01;
  sqrtAsti[1] = 2.5434012251993e+05;
  Bi[1] = 6.3587995142790e+00;
  Fomega[1] = 4.8508000000000e-01;

  // species 2: o
  // Computed from Lennard-Jones
  sqrtOneOverTc[2] = 9.7460146948181e-02;
  sqrtAsti[2] = 4.2500156012386e+04;
  Bi[2] = 6.6930234851885e-01;
  Fomega[2] = 4.8508000000000e-01;

  // species 3: o2
  // Computed from Lennard-Jones
  sqrtOneOverTc[3] = 8.4114315417892e-02;
  sqrtAsti[3] = 3.4718113488862e+04;
  Bi[3] = 6.6537922761553e-01;
  Fomega[3] = 4.8508000000000e-01;

  // species 4: oh
  // Computed from Lennard-Jones
  sqrtOneOverTc[4] = 9.7460146948181e-02;
  sqrtAsti[4] = 3.9981183985545e+04;
  Bi[4] = 6.2963299076575e-01;
  Fomega[4] = 4.8508000000000e-01;

  // species 5: h2o
  // Computed from Lennard-Jones
  sqrtOneOverTc[5] = 3.6435291028173e-02;
  sqrtAsti[5] = 9.3082162855299e+04;
  Bi[5] = 5.0524965107539e-01;
  Fomega[5] = 4.8508000000000e-01;

  // species 6: n2
  // Computed from Lennard-Jones
  sqrtOneOverTc[6] = 8.8267930269877e-02;
  sqrtAsti[6] = 4.0492641031864e+04;
  Bi[6] = 8.7262480899350e-01;
  Fomega[6] = 4.8508000000000e-01;

  // species 7: ho2
  // Computed from Lennard-Jones
  sqrtOneOverTc[7] = 8.4114315417892e-02;
  sqrtAsti[7] = 3.3657825710980e+04;
  Bi[7] = 6.4505861132042e-01;
  Fomega[7] = 4.8508000000000e-01;

  // species 8: h2o2
  // Computed from Lennard-Jones
  sqrtOneOverTc[8] = 8.4114315417892e-02;
  sqrtAsti[8] = 3.2660380884830e+04;
  Bi[8] = 6.2594239211036e-01;
  Fomega[8] = 4.8508000000000e-01;

  // species 9: Xe
  // Computed from Lennard-Jones
  sqrtOneOverTc[9] = 5.0839162801226e-02;
  sqrtAsti[9] = 1.8073705723809e+04;
  Bi[9] = 2.7028821119126e-01;
  Fomega[9] = 4.8508000000000e-01;
}

// gauss-jordan solver external routine
// Replace this routine with the one generated by the Gauss Jordan solver of DW
AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE void
sgjsolve(amrex::Real* /*A*/, amrex::Real* /*x*/, amrex::Real* /*b*/)
{
  amrex::Abort("sgjsolve not implemented, choose a different solver ");
}

// Replace this routine with the one generated by the Gauss Jordan solver of DW
AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE void
sgjsolve_simplified(amrex::Real* /*A*/, amrex::Real* /*x*/, amrex::Real* /*b*/)
{
  amrex::Abort(
    "sgjsolve_simplified not implemented, choose a different solver ");
}

#endif
