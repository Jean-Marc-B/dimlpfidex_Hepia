#ifndef FIDEXFCT_H
#define FIDEXFCT_H

#include <string>

class Parameters;

/**
 * @brief Displays the parameters for fidex.
 */
void showFidexParams();

/**
 * @brief Sets default hyperparameters and checks the logic and validity of the parameters of fidex.
 */
void checkFidexParametersLogicValues(Parameters &p);

/**
 * @brief Executes the Fidex algorithm with specified parameters to extract an explanation rule for one or several given samples.
 */
int fidex(const std::string &command = "");

#endif // FIDEXFCT_H
