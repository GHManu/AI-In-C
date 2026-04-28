#ifndef LOGISTIC_REGRESSION_H
#define LOGISTIC_REGRESSION_H

#include <float.h>

#define eps DBL_EPSILON

typedef struct {
    double *X;  
    double *y;  
    int n_samples;  
    int n_features; 
} Data;

void init_zero(double *vet, int n);
double randn();
void split_data(Data *full_data, Data *train, Data *test, double ratio);
double* sigmoid(double *x, int n);
double* predict_probs(double* X, double* weights, int n_samples, int n_features, double bias);
double loss(double* y_true, double* y_pred, int n_samples);
double* dloss_dw(double* X, double* y_true, double* y_pred, int n_samples, int n_features, double* y_diff);
void fit_logistic(Data *data, double* weights, double l_rate, double* bias, int n_epochs);
double* predict_logistic(double* X, double* weights, int n_samples, int n_features, double bias);
double calculate_accuracy(double* y_pred, double* y_true, int n_samples);

#endif