#include "hyperLocus.h"

/**
 * @brief Calculates the hyperlocus matrix containing all possible hyperplanes in the feature space that discriminate between different classes of samples, based on the weights training file.
 *
 * This function calculates the positions of the hyperplanes using the number of quantization levels,
 * the size of the interval, the bias and the weights, and then stores these positions in the hyperlocus matrix.
 *
 * @param dataFileWeights Path to the file containing the weights.
 * @param nbQuantLevels Number of quantization levels.
 * @param hiKnot Upper bound of the interval.
 * @return Matrix representing the hyperlocus.
 */
std::vector<std::vector<double>> calcHypLocus(DataSetFid &dataset, int nbQuantLevels, double hiKnot) {
  double lowKnot = -hiKnot;
  double dist = hiKnot - lowKnot;         // Size of the interval
  double binWidth = dist / nbQuantLevels; // Width of a box between 2 separations
  int nbKnots = nbQuantLevels + 1;        // Number of separations per dimension

  std::cout << "\nHyperlocus parameters:\n";
  std::cout << "- # stairs:\t" << nbQuantLevels << "\n";
  std::cout << "- Interval:\t[" << lowKnot << "," << hiKnot << "]\n\n";

  int nbNets = dataset.getNbNets();
  std::vector<std::vector<double>> matHypLocus;

  for (int n = 0; n < nbNets; n++) {
    std::vector<double> bias = dataset.getInputBias(n);
    std::vector<double> weights = dataset.getInputWeights(n);
    size_t nbInputs = bias.size();                              // Number of neurons in the first hidden layer (May be the number of input variables)
    std::vector<std::vector<double>> matHypLocusTemp(nbInputs); // Matrix of hyperplans (dim x hyp)

    for (int i = 0; i < nbInputs; i++) { // Loop on dimension
      for (int j = 0; j < nbKnots; j++) {
        double knot = lowKnot + binWidth * j;
        double threshold = (knot - bias[i]) / weights[i];
        matHypLocusTemp[i].push_back(threshold); // Placement of the hyperplan
      }
    }
    matHypLocus.insert(matHypLocus.end(), matHypLocusTemp.begin(), matHypLocusTemp.end());
  }

  std::cout << "HyperLocus computed.\n\n";

  return matHypLocus;
}

/**
 * @brief Checks wether a data is contained between 2 barriers.
 *
 * @param data
 * @param lowerbarrier
 * @param upperBarrier
 * @return true
 * @return false
 */
bool isBetweenBarriers(double data, double lowerBarrier, double upperBarrier) {
  if (lowerBarrier > upperBarrier) {
    return data > upperBarrier && data <= lowerBarrier;
  } else {
    return data >= lowerBarrier && data < upperBarrier;
  }
}

bool isBelowLowestBarrier(double data, double lowestBarrier, double highestBarrier) {
  return (lowestBarrier < highestBarrier) ? data < lowestBarrier : data >= lowestBarrier;
}


bool isAboveHighestBarrier(double data, double lowestBarrier, double highestBarrier) {
  return (highestBarrier > lowestBarrier) ? data >= highestBarrier : data < highestBarrier;
}

/**
 * @brief Finds which barrier bounds a given data and updates its scores.
 * 
 * @param data to be placed somewhere between barriers
 * @param barriers vector of barriers
 * @param scores vector of barriers scores
 * @return true if the data has found an eclosing barrier
 * @return false if the data remains unenclosed (should never happen)
 */
bool searchBarriers(double data, std::vector<double> &barriers, std::vector<int> &scores) {
  int nbBarriers = barriers.size();

  if (isBelowLowestBarrier(data, barriers[0], barriers[nbBarriers - 1])) {
    scores[0]++;
    return true;
  }

  if (isAboveHighestBarrier(data, barriers[0], barriers[nbBarriers - 1])) {
    scores[nbBarriers - 1]++;
    return true;
  }

  for (int bId = 1; bId < nbBarriers; bId++) {
    if (isBetweenBarriers(data, barriers[bId - 1], barriers[bId])) {
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
 * @param barriers original barriers to filter
 * @param scores vector with the number of contained datas per barrier
 * @return std::vector<double> the filtered vector of barrier
 */
std::vector<double> filterBarriers(std::vector<double> &barriers, std::vector<int> &scores) {
  std::vector<double> filteredBarriers;

  for (int i = 0; i < scores.size(); i++) {
    if (scores[i] > 0) {
      filteredBarriers.push_back(barriers[i]);
    }
  }

  return filteredBarriers;
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
  int nbNets = ds.getNbNets();

  if (datas.size() < 1) {
    throw InternalError("Connot optimize Hyperlocus. The given dataset does not contain any sample.");
  }

  int hyperlocusSize = originalHypLocus.size();
  int nbFeatures = hyperlocusSize / nbNets;
  int nbSamples = ds.getDatas().size();

  for (int hlId = 0; hlId < hyperlocusSize; hlId++) {
    std::vector<double> &currentBarriers = originalHypLocus[hlId];
    int nbBarriers = currentBarriers.size();
    std::vector<int> currentBarriersScores(nbBarriers);

    for (int sId = 0; sId < nbSamples; sId++) {
      double currentData = datas[sId][hlId % nbFeatures];
      searchBarriers(currentData, currentBarriers, currentBarriersScores) ? 1 : 0;
    }

    originalHypLocus[hlId] = filterBarriers(currentBarriers, currentBarriersScores);
  }
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
