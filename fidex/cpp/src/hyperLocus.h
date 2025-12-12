#ifndef HYPERLOCUSFCT_H
#define HYPERLOCUSFCT_H

#include "../../../common/cpp/src/checkFun.h"
#include "../../../common/cpp/src/dataSet.h"
#include "../../../common/cpp/src/errorHandler.h"
#include "../../../common/cpp/src/rule.h"
#include <fstream>
#include <iomanip>
#include <regex>
#include <set>
#include <sstream>
#include <string.h>

/**
 * @brief Calculates the hyperlocus matrix containing all possible hyperplanes in the feature
 * space that discriminate between different classes of samples, based on the weights training file.
 */
std::vector<std::vector<double>> calcHypLocus(DataSetFid &dataset, int nbQuantLevels, double hiKnot);

/**
 * @brief Calculates the hyperlocus matrix containing all possible hyperplanes in the feature
 * space that discriminate between different classes of samples, based on the rules training file.
 */
std::vector<std::vector<double>> calcHypLocus(const std::string &rulesFile, DataSetFid &dataset);

/**
 * @brief Optimizes a hyperlocus by removing barriers (knots/thresholds) that do not bound any data samples.
 * For each feature, a barrier is retained only if it forms, together with an adjacent barrier, an interval that contains at least one data sample.
 * If a barrier does not participate in enclosing any sample within its lower or upper interval, it is removed.
 *
 * @param originalHypLocus the hyperlocus to be optimized.
 * @param datas dataset used to filter the barriers.
 */
void optimizeHypLocus(std::vector<std::vector<double>> &originalHypLocus, DataSetFid &ds);

#endif // HYPERLOCUSFCT_H
