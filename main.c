#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include "logistic_regression.h"

int main(int argc, char *argv[]) {
    if (argc < 2) {
        printf("Utilizzo: %s <modello>\nModelli disponibili: logistic\n", argv[0]);
        return 1;
    }

    srand(time(NULL));
    Data *full_data = malloc(sizeof(Data));
    full_data->n_samples = 10000;
    full_data->n_features = 5;
    full_data->X = malloc(sizeof(double) * full_data->n_samples * full_data->n_features);
    full_data->y = malloc(sizeof(double) * full_data->n_samples);

    double *true_weights = malloc(sizeof(double) * full_data->n_features);
    for(int j = 0; j < full_data->n_features; j++) true_weights[j] = randn();
    for(int i = 0; i < full_data->n_samples; i++) {
        double dot = 0;
        for(int j = 0; j < full_data->n_features; j++) {
            full_data->X[i * full_data->n_features + j] = randn();
            dot += full_data->X[i * full_data->n_features + j] * true_weights[j];
        }
        full_data->y[i] = (dot > 0.0) ? 1.0 : 0.0; 
    }

    Data train_set, test_set;
    split_data(full_data, &train_set, &test_set, 0.8);

    if (strcmp(argv[1], "logistic") == 0) {
        double *weights = malloc(sizeof(double) * full_data->n_features);
        for(int i = 0; i < full_data->n_features; i++) weights[i] = randn() * 0.001;
        double bias = 0.1;

        fit_logistic(&train_set, weights, 0.1, &bias, 10000);
        
        double* predictions = predict_logistic(test_set.X, weights, test_set.n_samples, test_set.n_features, bias);
        printf("Accuracy Logistic Regression: %.2f%%\n", calculate_accuracy(predictions, test_set.y, test_set.n_samples) * 100);
        
        free(predictions);
        free(weights);
    } else {
        printf("Modello '%s' non riconosciuto.\n", argv[1]);
    }

    free(true_weights);
    free(full_data->X);
    free(full_data->y);
    free(full_data);
    return 0;
}