#ifndef DIMLPTRNFCT_H
#define DIMLPTRNFCT_H

#include <string>

class Parameters;

/**
 * @brief Displays the parameters for dimlpTrn.
 */
void showDimlpTrnParams();

/**
 * @brief Sets default hyperparameters and checks the logic and validity of the parameters of dimlpTrn.
 */
void checkDimlpTrnParametersLogicValues(Parameters &p);

/**
 * @brief Executes the Dimlp training process with specified parameters and optionally performs rule extraction with the Dimlp algorithm.
 */
int dimlpTrn(const std::string &command = "");

#endif // DIMLPTRNFCT_H
