#include "hyperbox.h"

/**
 * @brief Constructs a Hyperbox object with specified discriminative hyperplanes.
 *
 * @param m_discriminativeHyperplans Vector of (attribute index, hyperplane value) pairs representing discriminative hyperplanes.
 */
Hyperbox::Hyperbox(const std::vector<std::pair<int, int>> &m_discriminativeHyperplans)
    : discriminativeHyperplans(m_discriminativeHyperplans) {
}

/**
 * @brief Default constructor for Hyperbox.
 */
Hyperbox::Hyperbox() = default;

/**
 * @brief Sets the covered samples of the hyperbox.
 *
 * @param m_coveredSamples Vector of integers representing the new covered samples.
 */
void Hyperbox::setCoveredSamples(const std::vector<int> &m_coveredSamples) {
  coveredSamples = m_coveredSamples;
}

/**
 * @brief Gets the discriminative hyperplanes of the hyperbox.
 *
 * @return Vector of (attribute index, hyperplane value) pairs representing the discriminative hyperplanes.
 */
std::vector<std::pair<int, int>> Hyperbox::getDiscriminativeHyperplans() const {
  return discriminativeHyperplans;
}

/**
 * @brief Computes the new covered samples with a new discriminative hyperplane.
 *
 * @param ancienCoveredSamples Vector of previously covered sample IDs.
 * @param attribut Index of the attribute (dimension) used for the hyperplane.
 * @param trainData Training data matrix.
 * @param mainSampleGreater Boolean indicating if the sample of interest is greater than the hyperplane value.
 * @param hypValue Value of the hyperplane.
 */
void Hyperbox::computeCoveredSamples(const std::vector<int> &ancienCoveredSamples, int attribut, std::vector<std::vector<double>> &trainData, bool mainSampleGreater, double hypValue) {
  std::vector<int> newCoveredSamples;
  for (int idCoveredSample : ancienCoveredSamples) { // We check all already covered samples
    double sampleValue = trainData[idCoveredSample][attribut];
    bool sampleGreater = hypValue <= sampleValue;
    if (mainSampleGreater == sampleGreater) {       // If both samples are on same side of hyperplane
      newCoveredSamples.push_back(idCoveredSample); // This sample is covered again
    }
  }
  coveredSamples = newCoveredSamples;
}

/**
 * @brief Gets the covered samples of the hyperbox.
 *
 * @return Vector of integers representing the covered samples.
 */
std::vector<int> Hyperbox::getCoveredSamples() const {
  return coveredSamples;
}

/**
 * @brief Gets the covering sizes for each new antecedant(discriminative hypaerplan) in the hyperbox.
 *
 * @return Vector of integers representing the covering sizes.
 */
std::vector<int> Hyperbox::getCoveringSizesWithNewAntecedent() const {
  return coveringSizesWithNewAntecedent;
}

/**
 * @brief Adds the new covering size of the rule with the new antecedant in the list.
 *
 * @param covSize Covering size of the rule with the new added antecedant.
 */
void Hyperbox::addCoveringSizesWithNewAntecedent(int covSize) {
  coveringSizesWithNewAntecedent.push_back(covSize);
}

/**
 * @brief Reset the covering sizes for each new antecedant(discriminative hypaerplan) in the hyperbox.
 */
void Hyperbox::resetCoveringSizesWithNewAntecedent() {
  std::vector<int> covSizeAnt;
  coveringSizesWithNewAntecedent = covSizeAnt;
}

/**
 * @brief Computes the fidelity of the samples covered by the hyperbox with respect to the model's prediction.
 *
 * @param mainsamplePred Prediction of the sample of interest.
 * @param trainPreds Vector of predictions of the training data.
 */
void Hyperbox::computeFidelity(const int mainsamplePred, std::vector<int> &trainPreds) {
  int coveredTrueClass = 0;                       // Number of samples covered by the hyperbox and of same class as the example
  size_t nbCovered = coveredSamples.size();       // Number of samples covered by the hyperbox
  for (int idSample : coveredSamples) {           // Loop on all covered samples
    if (mainsamplePred == trainPreds[idSample]) { // Check if sample is of right class (class predicted by dimlp network for our main sample)
      coveredTrueClass += 1;
    }
  }

  fidelity = (double)coveredTrueClass / (double)nbCovered;
}

/**
 * @brief Gets the fidelity of the samples covered by the hyperbox.
 *
 * @return Double representing the fidelity.
 */
double Hyperbox::getFidelity() const {
  return fidelity;
}

/**
 * @brief Sets the fidelity of the samples covered by the hyperbox.
 *
 * @param x New fidelity value.
 */
void Hyperbox::setFidelity(double x) {
  fidelity = x;
}

/**
 * @brief Gets the increased fidelity for each new antecedent.
 *
 * @return Vector of the increased fidelity for each new antecedent.
 */
std::vector<double> Hyperbox::getIncreasedFidelity() const {
  return increasedFidelity;
}

/**
 * @brief Adds in the list the increased fidelity with the new antecedent.
 *
 * @param newFidelity New fidelity with the new antecedent.
 */
void Hyperbox::addIncreasedFidelity(double newFidelity) {

  double incrFidelity = newFidelity;
  if (increasedFidelity.size() != 0) {
    incrFidelity -= std::accumulate(increasedFidelity.begin(), increasedFidelity.end(), 0.0); // We remove the sum of precedents increase, which is the last fidelity obtained
  }
  increasedFidelity.push_back(incrFidelity);
}

/**
 * @brief Reset the increased fidelity for each new antecedent.
 */
void Hyperbox::resetIncreasedFidelity() {
  std::vector<double> incrFid;
  increasedFidelity = incrFid;
}

/**
 * @brief Gets the accuracy changes for each new antecedent.
 *
 * @return Vector of the accuracy changes for each new antecedent.
 */
std::vector<double> Hyperbox::getAccuracyChanges() const {
  return accuracyChanges;
}

/**
 * @brief Adds in the list the accuracy changes with the new antecedent.
 *
 * @param newAccuracy New accuracy with the new antecedent.
 */
void Hyperbox::addAccuracyChanges(double newAccuracy) {

  double changeAcc = newAccuracy;
  if (accuracyChanges.size() != 0) {
    changeAcc -= std::accumulate(accuracyChanges.begin(), accuracyChanges.end(), 0.0); // We remove the sum of precedent changes, which is the last accuracy obtained
  }
  accuracyChanges.push_back(changeAcc);
}

/**
 * @brief Reset the accuracy changes for each new antecedent.
 */
void Hyperbox::resetAccuracyChanges() {
  std::vector<double> changeAcc;
  accuracyChanges = changeAcc;
}

/**
 * @brief Adds a new discriminative hyperplane to the hyperbox.
 *
 * @param dimVal Index of the attribute.
 * @param hypVal Hyperplane value.
 */
void Hyperbox::discriminateHyperplan(int dimVal, int hypVal) {
  discriminativeHyperplans.push_back(std::make_pair(dimVal, hypVal));
}

/**
 * @brief Resets the discriminative hyperplanes of the hyperbox.
 */
void Hyperbox::resetDiscriminativeHyperplans() {
  std::vector<std::pair<int, int>> disc;
  discriminativeHyperplans = disc;
}
