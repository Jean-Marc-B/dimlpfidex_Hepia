#include "hyperLocus.h"

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
  double lowKnot = -hiKnot;
  double dist = hiKnot - lowKnot;         // Size of the interval
  double binWidth = dist / nbQuantLevels; // Width of a box between 2 separations
  int nbKnots = nbQuantLevels + 1;        // Number of separations per dimension
  int nbNets = dataset.getNbNets();
  int nbFeatures = dataset.getNbAttributes();
  std::vector<std::vector<double>> hyperlocus(nbFeatures, std::vector<double>(nbKnots));

  std::cout << "\nHyperlocus parameters:\n";
  std::cout << "- # stairs:\t" << nbQuantLevels << "\n";
  std::cout << "- Interval:\t[" << lowKnot << "," << hiKnot << "]\n";
  std::cout << "- # Nets:\t" << nbNets << "\n\n";

  for (int n = 0; n < nbNets; n++) {
    std::vector<double> bias = dataset.getInputBias(n);
    std::vector<double> weights = dataset.getInputWeights(n);

    for (int i = 0; i < nbFeatures; i++) { // Loop on dimension
      for (int j = 0; j < nbKnots; j++) {
        double knot = lowKnot + binWidth * j;
        double barrier = (knot - bias[i]) / weights[i];
        hyperlocus[i][j] = barrier; // Placement of the hyperplan
      }
    }
  }

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
 * @return false If the data remains unenclosed (should never happen).
 */
bool searchBarriers(double data, std::vector<double> &barriers, std::vector<int> &scores) {
  int nbBarriers = barriers.size();

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
  for (int bId = 1; bId < nbBarriers; bId++) {
    if (data >= barriers[bId - 1] && data < barriers[bId]) {
      scores[bId - 1]++;
      scores[bId]++;
      return true;
    }
  }

  return false;
}

/**
 * @brief Creates a new vector of barriers from another one that is filtered depending on how many datas each barrier bounds.
 *
 * @param barriers Original barriers to filter.
 * @param scores Vector with the number of contained datas per barrier.
 * @return std::vector<double> The filtered vector of barrier.
 */
std::vector<double> filterBarriers(std::vector<double> &barriers, std::vector<int> &scores) {
  std::vector<double> filteredBarriers;

  for (int i = 0; i < barriers.size(); i++) {
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
int sizeOf2DVector(std::vector<std::vector<T>> vec) {
  int sum = 0;

  for (std::vector<T> line : vec) {
    sum += line.size();
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
void optimizeHypLocus(std::vector<std::vector<double>> &originalHypLocus, DataSetFid &ds) {
  std::vector<std::vector<double>> datas = ds.getDatas();

  if (sizeOf2DVector(datas) < 1) {
    throw InternalError("Connot optimize Hyperlocus. The given dataset does not contain any sample.");
  }

  int nbNets = ds.getNbNets();
  int hyperlocusSize = originalHypLocus.size();
  int nbSamples = ds.getDatas().size();
  int totalNbHyperplans = sizeOf2DVector(originalHypLocus);
  std::cout << "original hyperlocus dimensions: " << hyperlocusSize << "*" << originalHypLocus[0].size() << "=" << totalNbHyperplans << "\n";

  for (int hlId = 0; hlId < hyperlocusSize; hlId++) {
    std::vector<double> &currentBarriers = originalHypLocus[hlId];
    int nbBarriers = currentBarriers.size();
    std::vector<int> currentBarriersScores(nbBarriers);

    for (int sId = 0; sId < nbSamples; sId++) {
      double currentData = datas[sId][hlId];
      searchBarriers(currentData, currentBarriers, currentBarriersScores);
    }

    originalHypLocus[hlId] = filterBarriers(currentBarriers, currentBarriersScores);
  }

  int remainingNbHyperplans = sizeOf2DVector(originalHypLocus);
  double remainingPercent = remainingNbHyperplans * 100.0 / (double)totalNbHyperplans;

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

  // Get pattern for attributes
  std::regex attributePattern;
  bool attributeIdsInFile = getRulesPatternsFromRuleFile(rulesFile, dataset, false)[0];
  if (attributeIdsInFile) { // If we have attribute names
    attributePattern = getAntStrPatternWithAttrIds(dataset.getNbAttributes());
  } else {
    attributePattern = getAntStrPatternWithAttrNames();
  }
  std::regex classPattern(getStrPatternWithClassIds(dataset.getNbClasses()));

  std::ifstream fileDta(rulesFile);

  if (!fileDta) {
    throw FileNotFoundError("Error : file " + rulesFile + " not found");
  }
  // Get thresholds values from rules file
  while (getline(fileDta, line)) {
    if (line.find("Rule") == 0) { // If line begins with "Rule"
      Rule rule;
      if (stringToRule(rule, line, attributePattern, classPattern, !attributeIdsInFile, false, dataset)) {
        for (Antecedent ant : rule.getAntecedents()) {
          thresholds[ant.getAttribute()].insert(ant.getValue());
        }
      }
    }
  }

  for (size_t i = 0; i < thresholds.size(); ++i) {
    matHypLocus[i].assign(thresholds[i].begin(), thresholds[i].end());
  }

  fileDta.close(); // close data file
  return matHypLocus;
}
