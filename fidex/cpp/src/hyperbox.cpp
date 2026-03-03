#include "hyperbox.h"

#include <numeric>

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
const std::vector<std::pair<int, int>> &Hyperbox::getDiscriminativeHyperplans() const {
  return discriminativeHyperplans;
}

/**
 * @brief Adds a new discriminative hyperplane to the hyperbox.
 *
 * @param dimVal Index of the attribute.
 * @param hypVal Hyperplane value.
 */
void Hyperbox::addDiscriminativeHyperplan(int dimVal, int hypValIndex) {
  discriminativeHyperplans.emplace_back(dimVal, hypValIndex);
}

/**
 * @brief Removes a discriminative hyperplane based on its array position.
 *
 * @param index item index to remove.
 */
void Hyperbox::removeDiscriminativehyperplan(int index) {
  if (index < 0 || index >= discriminativeHyperplans.size()) {
    return;
  }

  discriminativeHyperplans.erase(discriminativeHyperplans.begin() + index);
}

/**
 * @brief Resets the discriminative hyperplanes of the hyperbox.
 */
void Hyperbox::resetDiscriminativeHyperplans() {
  discriminativeHyperplans.clear();
}

/**
 * @brief Sets the discriminative hyperplanes of the hyperbox.
 *
 * @param newDiscriminativeHyperplan Vector of (attribute index, hyperplane value) pairs representing discriminative hyperplanes to be set.
 */
void Hyperbox::setDiscriminativeHyperplans(std::vector<std::pair<int, int>> newDiscriminativeHyperplan) {
  discriminativeHyperplans = std::move(newDiscriminativeHyperplan);
}

/**
 * @brief Computes the new covered samples with a new discriminative hyperplane.
 *
 * @param ancientCoveredSamples Vector of previously covered sample IDs.
 * @param attribute Index of the attribute (dimension) used for the hyperplane.
 * @param trainData Training data matrix.
 * @param mainSampleGreater Boolean indicating if the sample of interest is greater than the hyperplane value.
 * @param hypValue Value of the hyperplane.
 */
void Hyperbox::computeCoveredSamples(const std::vector<int> &ancientCoveredSamples, int attribute, const std::vector<std::vector<double>> &trainData, bool mainSampleGreater, double hypValue) {
  std::vector<int> newCoveredSamples;
  newCoveredSamples.reserve(ancientCoveredSamples.size());
  for (int idCoveredSample : ancientCoveredSamples) { // We check all already covered samples
    bool sampleGreater = hypValue <= trainData[idCoveredSample][attribute];
    if (mainSampleGreater == sampleGreater) {       // If both samples are on same side of hyperplane
      newCoveredSamples.push_back(idCoveredSample); // This sample is covered again
    }
  }
  coveredSamples = std::move(newCoveredSamples);
}

/**
 * @brief Computes the new covered samples and their fidelity in a single pass.
 *
 * @param ancientCoveredSamples Vector of previously covered sample IDs.
 * @param attribute Index of the attribute (dimension) used for the hyperplane.
 * @param trainData Training data matrix.
 * @param mainSampleGreater Boolean indicating if the sample of interest is greater than the hyperplane value.
 * @param hypValue Value of the hyperplane.
 * @param mainSamplePred Prediction of the sample of interest.
 * @param trainPreds Vector of predictions of the training data.
 */
void Hyperbox::computeCoveredSamplesAndFidelity(const std::vector<int> &ancientCoveredSamples, int attribute, const std::vector<std::vector<double>> &trainData, bool mainSampleGreater, double hypValue, const int mainSamplePred, const std::vector<int> &trainPreds) {
  std::vector<int> newCoveredSamples;
  newCoveredSamples.reserve(ancientCoveredSamples.size());

  size_t coveredTrueClass = 0;
  for (int idCoveredSample : ancientCoveredSamples) { // We check all already covered samples
    bool sampleGreater = hypValue <= trainData[idCoveredSample][attribute];
    if (mainSampleGreater == sampleGreater) { // If both samples are on same side of hyperplane
      newCoveredSamples.push_back(idCoveredSample);
      if (mainSamplePred == trainPreds[idCoveredSample]) {
        coveredTrueClass += 1;
      }
    }
  }

  const size_t nbCovered = newCoveredSamples.size();
  coveredSamples = std::move(newCoveredSamples);
  fidelity = (nbCovered == 0) ? 0.0 : static_cast<double>(coveredTrueClass) / static_cast<double>(nbCovered);
}

/**
 * @brief Gets the covered samples of the hyperbox.
 *
 * @return Vector of integers representing the covered samples.
 */
const std::vector<int> &Hyperbox::getCoveredSamples() const {
  return coveredSamples;
}

/**
 * @brief Gets the covering sizes for each new antecedent(discriminative hyperplan) in the hyperbox.
 *
 * @return Vector of integers representing the covering sizes.
 */
const std::vector<int> &Hyperbox::getCoveringSizesWithNewAntecedent() const {
  return coveringSizesWithNewAntecedent;
}

/**
 * @brief sets the new covering sizes.
 *
 * @param newCoveringSizesWithNewAntecedents Covering sizes to set.
 */
void Hyperbox::setCoveringSizesWithNewAntecedent(std::vector<int> newCoveringSizesWithNewAntecedents) {
  coveringSizesWithNewAntecedent = std::move(newCoveringSizesWithNewAntecedents);
}

/**
 * @brief Adds the new covering size of the rule with the new antecedent in the list.
 *
 * @param covSize Covering size of the rule with the new added antecedent.
 */
void Hyperbox::addCoveringSizesWithNewAntecedent(int covSize) {
  coveringSizesWithNewAntecedent.push_back(covSize);
}

/**
 * @brief remove a covering sizes with new antecedent stat based on its array position.
 *
 * @param index item index to remove.
 */
void Hyperbox::removeCoveringSizesWithNewAntecedent(int index) {
  if (index < 0 || index >= coveringSizesWithNewAntecedent.size()) {
    return;
  }

  coveringSizesWithNewAntecedent.erase(coveringSizesWithNewAntecedent.begin() + index);
}

/**
 * @brief Reset the covering sizes for each new antecedent(discriminative hyperplan) in the hyperbox.
 */
void Hyperbox::resetCoveringSizesWithNewAntecedent() {
  coveringSizesWithNewAntecedent.clear();
}

/**
 * @brief Computes the fidelity of the samples covered by the hyperbox with respect to the model's prediction.
 *
 * @param mainSamplePred Prediction of the sample of interest.
 * @param trainPreds Vector of predictions of the training data.
 */
void Hyperbox::computeFidelity(const int mainSamplePred, const std::vector<int> &trainPreds) {
  const size_t nbCovered = coveredSamples.size();
  if (nbCovered == 0) {
    fidelity = 0.0; // Invalid fidelity when no samples are covered
    return;
  }
  size_t coveredTrueClass = 0;                    // Number of samples covered by the hyperbox and of same class as the example
  for (int idSample : coveredSamples) {           // Loop on all covered samples
    if (mainSamplePred == trainPreds[idSample]) { // Check if sample is of right class (class predicted by dimlp network for our main sample)
      coveredTrueClass += 1;
    }
  }

  fidelity = static_cast<double>(coveredTrueClass) / static_cast<double>(nbCovered);
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
const std::vector<double> &Hyperbox::getIncreasedFidelity() const {
  return increasedFidelity;
}

/**
 * @brief Sets the increased fidelities list.
 *
 * @param newIncreasedFidelities New increased fidelities to set.
 */
void Hyperbox::setIncreasedFidelity(std::vector<double> newIncreasedFidelities) {
  increasedFidelity = std::move(newIncreasedFidelities);
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
 * @brief Removes an increased fidelity stat based on its array position.
 *
 * @param index item index to remove.
 */
void Hyperbox::removeIncreasedFidelity(int index) {
  if (index < 0 || index >= increasedFidelity.size()) {
    return;
  }

  increasedFidelity.erase(increasedFidelity.begin() + index);
}

/**
 * @brief Reset the increased fidelity for each new antecedent.
 */
void Hyperbox::resetIncreasedFidelity() {
  increasedFidelity.clear();
}

/**
 * @brief Gets the accuracy changes for each new antecedent.
 *
 * @return Vector of the accuracy changes for each new antecedent.
 */
const std::vector<double> &Hyperbox::getAccuracyChanges() const {
  return accuracyChanges;
}

/**
 * @brief Sets the accuracy changes list.
 *
 * @param accuracyChanges New accuracy changes to set.
 */
void Hyperbox::setAccuracyChanges(std::vector<double> newAccuracyChanges) {
  accuracyChanges = std::move(newAccuracyChanges);
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
 * @brief Removes an accuracy change stat based on its array position.
 *
 * @param index item index to remove.
 */
void Hyperbox::removeAccuracyChange(int index) {
  if (index < 0 || index >= accuracyChanges.size()) {
    return;
  }

  accuracyChanges.erase(accuracyChanges.begin() + index);
}

/**
 * @brief Reset the accuracy changes for each new antecedent.
 */
void Hyperbox::resetAccuracyChanges() {
  accuracyChanges.clear();
}

Hyperbox Hyperbox::deepCopy() {
  Hyperbox copy = Hyperbox();

  copy.setDiscriminativeHyperplans(discriminativeHyperplans);
  copy.setCoveringSizesWithNewAntecedent(coveringSizesWithNewAntecedent);
  copy.setCoveredSamples(coveredSamples);
  copy.setFidelity(fidelity);
  copy.setAccuracyChanges(accuracyChanges);
  copy.setIncreasedFidelity(increasedFidelity);

  return copy;
}

/**
 * @brief Computes the accuracy of the rule with respect to the rule prediction and true classes of the covered samples.
 *
 * @param mainSamplePred Prediction of the rule and main sample.
 * @param trainTrueClass True classes of the training data.
 * @return The accuracy of the rule.
 */
double Hyperbox::computeRuleAccuracy(const int mainSamplePred, const std::vector<int> &trainTrueClass) const { // Percentage of correct rule predictions on samples covered by the rule
  const size_t nbCovered = coveredSamples.size();
  if (nbCovered == 0) { // Invalid accuracy when no samples are covered
    return 0.0;
  }

  size_t nbCorrect = 0;
  for (int idSample : coveredSamples) {
    if (mainSamplePred == trainTrueClass[idSample]) {
      nbCorrect += 1;
    }
  }

  return static_cast<double>(nbCorrect) / static_cast<double>(nbCovered);
}
