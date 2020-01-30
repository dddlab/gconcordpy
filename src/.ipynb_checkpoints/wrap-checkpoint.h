#ifndef GCONCORD_H_
#define GCONCORD_H_
#include <Python.h>

extern "C" {
    void gconcord(double* s, int sCol, int method, 
                  double* lam1, double lam2, 
                  double epstol, int maxitr, int steptype, 
                  double* out, int* outi, int* outj);
}

#endif