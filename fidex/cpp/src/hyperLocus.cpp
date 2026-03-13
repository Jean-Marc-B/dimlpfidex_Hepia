#include "hyperLocus.h"
#include "../../../common/cpp/src/checkFun.h"
#include "../../../common/cpp/src/dataSet.h"
#include "../../../common/cpp/src/errorHandler.h"
#include "../../../common/cpp/src/rule.h"
#include <algorithm>
#include <cmath>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <regex>
#include <set>

/**
 * @brief Calculates the hyperlocus matrix containing all possible hyperplanes in the feature space that discriminate between different classes of samples, based on the weights training file.
 *
 * This function calculates the positions of the hyperplanes using the number of quantization levels,
 * the size of the interval, the bias and the weights, and then stores these positions in the hyperlocus matrix.
 *
 * @param dataset Dataset object.
 * @param nbQuantLevels Number of quantization levels.
 * @param hiKnot Upper bound of the interval.
 * @return Matrix representing the hyperlocus.
 */
std::vector<std::vector<double>> calcHypLocus(DataSetFid &dataset, int nbQuantLevels, double hiKnot) {
  const double minAbsWeight = 1e-6;

  // =========================================================================
  // 1) Prepare quantization and lattice parameters
  // =========================================================================
  const double lowKnot = -hiKnot;
  const double dist = hiKnot - lowKnot;         // Size of the interval
  const double binWidth = dist / nbQuantLevels; // Width of a box between 2 separations
  const int nbKnots = nbQuantLevels + 1;        // Number of separations per dimension
  const int nbNets = dataset.getNbNets();
  const int nbFeatures = dataset.getNbAttributes();
  std::vector<std::vector<double>> hyperlocus(nbFeatures, std::vector<double>(nbKnots * nbNets));

  std::cout << "\nHyperlocus parameters:\n";
  std::cout << "- # stairs:\t" << nbQuantLevels << "\n";
  std::cout << "- Interval:\t[" << lowKnot << "," << hiKnot << "]\n";
  std::cout << "- # Nets:\t" << nbNets << "\n\n";

  // =========================================================================
  // 2) Compute raw barriers from network weights and biases
  // =========================================================================
  for (int n = 0; n < nbNets; n++) {
    const std::vector<double> &bias = dataset.getInputBias(n);
    const std::vector<double> &weights = dataset.getInputWeights(n);

    for (int i = 0; i < nbFeatures; i++) { // Loop on dimension
      const double safeWeight = (std::abs(weights[i]) < minAbsWeight) ? (weights[i] < 0.0 ? -minAbsWeight : minAbsWeight) : weights[i];
      for (int k = 0; k < nbKnots; ++k) {
        int j = n * nbKnots + k; // Index on the hyperlocus
        double knot = lowKnot + binWidth * k;
        double barrier = (knot - bias[i]) / safeWeight;
        hyperlocus[i][j] = barrier; // Placement of the hyperplan
      }
    }
  }

  // =========================================================================
  // 3) Sort barriers per feature to ease downstream filtering
  // =========================================================================
  // sorting barrers by value in order to ease filtering if enabled
  for (int i = 0; i < nbFeatures; i++) {
    sort(hyperlocus[i].begin(), hyperlocus[i].end());
  }

  std::cout << "HyperLocus computed.\n\n";

  return hyperlocus;
}

/**
 * @brief Finds which barrier bounds a given data and updates its scores.
 *
 * @param data Value to be placed somewhere between barriers.
 * @param barriers Vector of barriers.
 * @param scores Vector of barriers scores.
 * @return true If the data has found an enclosing barrier.
 * @return false If barriers/scores are invalid or if no enclosing interval is found (should never happen).
 */
bool searchBarriers(double data, const std::vector<double> &barriers, std::vector<int> &scores) {
  const size_t nbBarriers = barriers.size();
  if (nbBarriers == 0 || scores.size() != nbBarriers) {
    return false;
  }

  // if below lowest barrier
  if (data < barriers[0]) {
    scores[0]++;
    return true;
  }

  // if above or equal highest barrier
  if (data >= barriers[nbBarriers - 1]) {
    scores[nbBarriers - 1]++;
    return true;
  }

  // if between some barriers
  for (size_t bId = 1; bId < nbBarriers; ++bId) {
    if (data >= barriers[bId - 1] && data < barriers[bId]) {
      scores[bId - 1]++;
      scores[bId]++;
      return true;
    }
  }

  return false;
}

/**
 * @brief Revives a barrier between two non-zero barriers separated by a zero gap.
 *
 * When two non-zero barriers enclose a contiguous gap of zero-valued barriers,
 * the barrier at the center of the gap is assigned the sum of their scores.
 * The enclosing barriers are reset to zero. The operation modifies the input
 * vector in place and processes gaps from left to right.
 *
 * @param scores Vector of barrier scores (0 indicates a dead barrier).
 */
void reviveBarriersScore(std::vector<int> &scores) {
  int gap = 0;
  int low_id = 0;
  bool counting = false;

  // std::cout << "Before reviving process:\n";
  // for (int score : scores) {
  //   std::cout << score << ",";
  // }
  // std::cout << "\n";

  for (size_t i = 1; i < scores.size(); ++i) {
    if (!counting) {
      if (scores[i - 1] > 0 && scores[i] == 0) {
        // enable counting if current position is the beginning of a gap
        counting = true;
        low_id = static_cast<int>(i) - 1;
        gap = 1;
      }

    } else if (counting) {
      if (scores[i] > 0 && gap > 0) {
        // revive the centered barriers
        int middle_id = low_id + gap / 2 + 1;
        counting = false;

        scores[middle_id] = scores[low_id] + scores[i];
        scores[low_id] = 0;
        scores[i] = 0;

      } else {
        // continue counting if allowed
        gap++;
      }
    }
  }

  // std::cout << "After reviving process:\n";
  // for (int score : scores) {
  //   std::cout << score << ",";
  // }
  // std::cout << "\n\n";
}

/**
 * @brief Creates a new vector of barriers from another one that is filtered depending on how many datas each barrier bounds.
 *
 * @param barriers Original barriers to filter.
 * @param scores Vector with the number of contained datas per barrier.
 * @return std::vector<double> The filtered vector of barrier.
 */
std::vector<double> filterBarriers(const std::vector<double> &barriers, const std::vector<int> &scores) {
  std::vector<double> filteredBarriers;
  filteredBarriers.reserve(barriers.size());

  for (size_t i = 0; i < barriers.size(); ++i) {
    if (scores[i] > 0) {
      filteredBarriers.push_back(barriers[i]);
    }
  }
  return filteredBarriers;
}

/**
 * @brief Computes the total number of elements inside a 2D vector.
 *
 * @param vec 2D vector to be measured.
 * @return int The number of elements.
 */
template <typename T>
int sizeOf2DVector(const std::vector<std::vector<T>> &vec) {
  int sum = 0;

  for (const auto &line : vec) {
    sum += static_cast<int>(line.size());
  }

  return sum;
}

/**
 * @brief Optimizes a hyperlocus by removing barriers (knots/thresholds) that do not bound any data samples.
 * For each feature, a barrier is retained only if it forms, together with an adjacent barrier, an interval that contains at least one data sample.
 * If a barrier does not participate in enclosing any sample within its lower or upper interval, it is removed.
 *
 * @param originalHypLocus the hyperlocus to be optimized.
 * @param datas dataset used to filter the barriers.
 */
void optimizeHypLocus(std::vector<std::vector<double>> &originalHypLocus, DataSetFid &ds, bool enableRevive) {
  // =========================================================================
  // 1) Validate inputs and summarize optimization context
  // =========================================================================
  if (originalHypLocus.empty()) {
    std::cout << "Hyperlocus optimization skipped: no dimensions.\n";
    return;
  }

  const auto &datas = ds.getDatas();
  if (datas.empty()) {
    std::cout << "Hyperlocus optimization skipped: no samples.\n";
    return;
  }

  const int nbNets = ds.getNbNets();
  if (nbNets <= 0) {
    std::cout << "Hyperlocus optimization skipped: invalid number of networks.\n";
    return;
  }
  const int hyperlocusSize = static_cast<int>(originalHypLocus.size());
  const int nbSamples = static_cast<int>(datas.size());
  const int barriers_per_net = static_cast<int>(originalHypLocus[0].size()) / nbNets;
  const int totalNbHyperplans = sizeOf2DVector(originalHypLocus);
  std::cout << "original hyperlocus dimensions: " << nbNets << "*" << hyperlocusSize << "*" << barriers_per_net << "=" << totalNbHyperplans << "\n";

  // =========================================================================
  // 2) Score and filter barriers feature by feature
  // =========================================================================
  for (int hlId = 0; hlId < hyperlocusSize; hlId++) {
    std::vector<double> &currentBarriers = originalHypLocus[hlId];
    if (currentBarriers.empty()) {
      continue;
    }
    const int nbBarriers = static_cast<int>(currentBarriers.size());
    std::vector<int> currentBarriersScores(nbBarriers);

    for (int sId = 0; sId < nbSamples; sId++) {
      double currentData = datas[sId][hlId];
      searchBarriers(currentData, currentBarriers, currentBarriersScores);
    }

    if (enableRevive) {
      reviveBarriersScore(currentBarriersScores); // TODO test
    }

    originalHypLocus[hlId] = filterBarriers(currentBarriers, currentBarriersScores);
  }

  // =========================================================================
  // 3) Report optimization result
  // =========================================================================
  const int remainingNbHyperplans = sizeOf2DVector(originalHypLocus);
  const double remainingPercent = totalNbHyperplans > 0 ? remainingNbHyperplans * 100.0 / static_cast<double>(totalNbHyperplans) : 0.0;

  std::streamsize defaultPrecision = std::cout.precision();
  std::cout << "Optimization done. " << remainingNbHyperplans << "/" << totalNbHyperplans << " hyperplan(s) remaining (" << std::setprecision(3) << remainingPercent << std::setprecision(defaultPrecision) << "%).\n";
}

/**
 * @brief Calculates the hyperlocus matrix containing all possible hyperplanes in the feature space that discriminate between different classes of samples, based on the rules training file.
 *
 * This function reads a rules file to extract thresholds for each attribute from the
 * antecedents of the rules. These thresholds are then used to build the hyperlocus matrix,
 * which contains the positions of the hyperplanes for each attribute in the dataset.
 *
 * @param rulesFile Path to the file containing the rules.
 * @param dataset Reference to the dataset object.
 * @return Matrix representing the hyperlocus.
 */
std::vector<std::vector<double>> calcHypLocus(const std::string &rulesFile, DataSetFid &dataset) {
  std::string line;
  std::vector<std::vector<double>> matHypLocus(dataset.getNbAttributes());
  std::vector<std::set<double>> thresholds(dataset.getNbAttributes()); // Thresholds (=knots) for each attribute

  // =========================================================================
  // 1) Prepare parsing patterns depending on attribute format in rule file
  // =========================================================================
  // Get pattern for attributes
  std::regex attributePattern;
  bool attributeIdsInFile = getRulesPatternsFromRuleFile(rulesFile, dataset, false)[0];
  if (attributeIdsInFile) { // If rules use attribute IDs
    attributePattern = getAntStrPatternWithAttrIds(dataset.getNbAttributes());
  } else {
    attributePattern = getAntStrPatternWithAttrNames();
  }
  std::regex classPattern(getStrPatternWithClassIds(dataset.getNbClasses()));

  std::ifstream fileDta(rulesFile);

  if (!fileDta) {
    throw FileNotFoundError("Error : file " + rulesFile + " not found");
  }

  // =========================================================================
  // 2) Parse rules and collect threshold values per attribute
  // =========================================================================
  // Get thresholds values from rules file
  while (getline(fileDta, line)) {
    if (line.find("Rule") == 0) { // If line begins with "Rule"
      Rule rule;
      if (stringToRule(rule, line, attributePattern, classPattern, !attributeIdsInFile, false, dataset)) {
        for (const Antecedent &ant : rule.getAntecedents()) {
          thresholds[ant.getAttribute()].insert(ant.getValue());
        }
      }
    }
  }

  // =========================================================================
  // 3) Convert threshold sets to hyperlocus vectors
  // =========================================================================
  for (size_t i = 0; i < thresholds.size(); ++i) {
    matHypLocus[i].assign(thresholds[i].begin(), thresholds[i].end());
  }

  return matHypLocus;
}
