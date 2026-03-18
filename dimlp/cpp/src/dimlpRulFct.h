#ifndef DIMLPRULFCT_H
#define DIMLPRULFCT_H

#include <string>

class Parameters;

/**
 * @brief Displays the parameters for dimlpRul.
 */
void showDimlpRulParams();

/**
 * @brief Sets default hyperparameters and checks the logic and validity of the parameters of dimlpRul.
 */
void checkDimlpRulParametersLogicValues(Parameters &p);

/**
 * @brief Executes the Dimlp rule extraction process with specified parameters to obtain explaining rules and statistics for train, test and validation datasets for a model trained with dimlpTrn.
 */
int dimlpRul(const std::string &command = "");

#endif // DIMLPRULFCT_H
