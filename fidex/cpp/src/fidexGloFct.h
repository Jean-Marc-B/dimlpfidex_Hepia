#ifndef FIDEXGLOFCT_H
#define FIDEXGLOFCT_H

#include "../../../common/cpp/src/rule.h"
#include <string>
#include <vector>

class DataSetFid;
class Parameters;
class Hyperspace;

/**
 * @brief Displays the parameters for fidexGlo.
 */
void showFidexGloParams();

/**
 * @brief Sets default hyperparameters and checks the logic and validity of the parameters of fidexGlo.
 */
void checkParametersLogicValues(Parameters &p);

/**
 * @brief Executes the Fidex algorithm to extract an explanation rule for a given sample.
 */
void executeFidex(std::vector<std::string> &lines, std::vector<Rule> &generatedRules, DataSetFid &trainDataset, Parameters &p, Hyperspace &hyperspace, const std::vector<double> &mainSampleValues, int mainSamplePred, double mainSamplePredScore, const std::vector<std::string> &attributeNames, const std::vector<std::string> &classNames);

/**
 * @brief Executes the FidexGlo algorithm with specified parameters to extract explanation rules for each test sample.
 */
int fidexGlo(const std::string &command = "");

#endif // FIDEXGLOFCT_H
