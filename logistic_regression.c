#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include "logistic_regression.h"

void init_zero(double *vet, int n) {
    for(int i = 0; i < n; i++) vet[i] = 0;
}

//normal distribution
double randn() {
    double u1 = (double)rand() / RAND_MAX;  //uniform distribution
    double u2 = (double)rand() / RAND_MAX;
    if (u1 <= 0.0) u1 = eps;
    return sqrt(-2.0 * log(u1)) * cos(2.0 * M_PI * u2);
}

//divides the dataset in 2 set: training and test set
void split_data(Data *full_data, Data *train, Data *test, double ratio) {
    int train_size = (int)(full_data->n_samples * ratio);
    train->n_samples = train_size;
    train->n_features = full_data->n_features;
    train->X = full_data->X; 
    train->y = full_data->y;

    test->n_samples = full_data->n_samples - train_size;
    test->n_features = full_data->n_features;
    test->X = &full_data->X[train_size * full_data->n_features];
    test->y = &full_data->y[train_size];
}

// --- TRAINING ---

//sigmoid function
/*
IN Python:
x è un np.array
sig = 1/ (1+np.exp(-x)) 
*/
double* sigmoid(double *x, int n) {
    double* result = malloc(n * sizeof(double));
    for(int i = 0; i < n; i++) result[i] = 1.0 / (1.0 + exp(-x[i]));
    return result;
}

//Calculate the probability that an example belongs to the class
/*

*/ 
double* predict_probs(double* X, double* weights, int n_samples, int n_features, double bias) {
    double* p = malloc(n_samples * sizeof(double));
    for (int i = 0; i < n_samples; i++) {
        double sum = bias;
        for (int j = 0; j < n_features; j++) sum += X[(i * n_features) + j] * weights[j];
        p[i] = sum;
    }
    double* probs = sigmoid(p, n_samples);
    free(p);
    return probs;
}

//loss function
/*
In Python:
    y_true e y_pred sono np.array
    return - np.mean(y_true * np.log(y_pred + eps) + (1-y_true) * np.log(1- y_pred + eps), axis = 0)#media sull'asse 0 poichè è l'asse degli esempi
*/
double loss(double* y_true, double* y_pred, int n_samples) {
    double sum = 0;
    for(int i = 0; i < n_samples; i++) {
        double pred = y_pred[i];
        if (pred < eps) pred = eps;
        if (pred > 1.0 - eps) pred = 1.0 - eps;
        sum += y_true[i] * log(pred) + (1.0 - y_true[i]) * log(1.0 - pred);
    }
    return -sum / n_samples;
}

//the derivative of loss function
/*
In Python:
N = X.shape[0]
der = np.dot(X.T, (y_pred - y_true)) / N    #np.dot prodotto riga per colonna, non metto la sommaotria poichè lo fa già con il prodotto riga per colonna
*/
double* dloss_dw(double* X, double* y_true, double* y_pred, int n_samples, int n_features, double* y_diff) {
    double* gradients = malloc(n_features * sizeof(double));
    init_zero(gradients, n_features);

    for(int i = 0; i < n_samples; i++) y_diff[i] = y_pred[i] - y_true[i];

    for(int i = 0; i < n_features; i++) {
        for(int j = 0; j < n_samples; j++) 
            gradients[i] += X[(j * n_features) + i] * y_diff[j];

        gradients[i] /= n_samples;
    }
    return gradients;
}

//TRAINING FUNCTION
/* In Python:
n_samples, n_features = X.shape
for e in range(n_epochs):

            #l'operatore @ è uguale al np.dot 
            z = X @ self.w + self.b       # (n,)  combinazione lineare                
            p = sigmoid(z) # (n,)  probabilità predette

            L = loss(Y, p)

            if verbose and e % 500 == 0:
                 print(f'Epoch {e:4d}: loss={L}')

            # gradiente ovvero calcolo quanto sono sbagliati i pesi e aggiornamento dei pesi
            dw = dloss_dw(Y, p, X)          # (p̂ − y)·x / n
            db = np.mean(p - Y)              # (p̂ − y) / n  per il bias

            self.w -= learning_rate * dw    # w(t+1) = w(t) − η·∇w
            self.b -= learning_rate * db
*/
void fit_logistic(Data *data, double* weights, double l_rate, double* bias, int n_epochs) {
    double* y_pred = NULL;
    double* gradients = NULL;
    double* y_diff = malloc(sizeof(double) * data->n_samples);
    for(int i = 0; i < n_epochs; i++) {
        if (y_pred) free(y_pred);
        y_pred = predict_probs(data->X, weights, data->n_samples, data->n_features, *bias);

        double L = loss(data->y, y_pred, data->n_samples);

        if (gradients) free(gradients);
        gradients = dloss_dw(data->X, data->y, y_pred, data->n_samples, data->n_features, y_diff);

        for(int j = 0; j < data->n_features; j++) weights[j] -= gradients[j] * l_rate;

        double b_sum = 0;
        for(int j = 0; j < data->n_samples; j++) 
            b_sum += y_diff[j];
        *bias -= (b_sum / data->n_samples) * l_rate;
        if(i % 100 == 0) printf("Epoch %d - Loss: %f\n", i, L);
    }
    free(y_pred); free(gradients); free(y_diff);
}

// --- TEST ---
/*
 z = X @ self.w + self.b   # a) combinazione lineare
p = sigmoid(z)              # b) probabilità in (0,1)
#return (p >= 0.5).astype(int) # c) discretizza -> {0,1} --> oppure potevo fare np.round
return np.round(p)
*/
double* predict_logistic(double* X, double* weights, int n_samples, int n_features, double bias) {
    double* probs = predict_probs(X, weights, n_samples, n_features, bias);
    double* classes = malloc(n_samples * sizeof(double));
    for (int i = 0; i < n_samples; i++) classes[i] = (probs[i] >= 0.5) ? 1.0 : 0.0;
    free(probs);
    return classes;
}

double calculate_accuracy(double* y_pred, double* y_true, int n_samples) {
    int correct = 0;
    for (int i = 0; i < n_samples; i++) if (y_pred[i] == y_true[i]) correct++;
    return (double)correct / n_samples;
}