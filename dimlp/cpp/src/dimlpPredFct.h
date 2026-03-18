#ifndef DIMLPPREDFCT_H
#define DIMLPPREDFCT_H

#include <string>

class Parameters;

/**
 * @brief Displays the parameters for dimlpPred.
 */
void showDimlpPredParams();

/**
 * @brief Sets default hyperparameters and checks the logic and validity of the parameters of dimlpPred.
 */
void checkDimlpPredParametersLogicValues(Parameters &p);

/**
 * @brief Executes the Dimlp prediction process on test set with specified parameters for a model trained with dimlpTrn.
 */
int dimlpPred(const std::string &command = "");

#endif // DIMLPPREDFCT_H
