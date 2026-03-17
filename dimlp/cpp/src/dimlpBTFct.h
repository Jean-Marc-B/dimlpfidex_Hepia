#ifndef DIMLPBTFCT_H
#define DIMLPBTFCT_H

#include <string>

class Parameters;

/**
 * @brief Displays the parameters for dimlpBT.
 */
void showDimlpBTParams();

/**
 * @brief Sets default hyperparameters and checks the logic and validity of the parameters of dimlpBT.
 */
void checkDimlpBTParametersLogicValues(Parameters &p);

/**
 * @brief Executes the Dimlp Bagging Training (dimlpBT) process with specified parameters and optionally performs rule extraction with the Dimlp algorithm.
 */
int dimlpBT(const std::string &command = "");

#endif // DIMLPBTFCT_H
