#include "hyperspace.h"
#include "../../../common/cpp/src/errorHandler.h"

/**
 * @brief Constructs a Hyperspace object with a specified 2D vector of doubles of possible hyperplanes.
 *
 * @param matHypLocus 2D vector of doubles representing all the possible hyperplanes.
 */
Hyperspace::Hyperspace(const std::vector<std::vector<double>> &matHypLocus) : hyperLocus(matHypLocus) {
  hyperbox = std::make_shared<Hyperbox>();
}

/**
 * @brief Gets the 2D vector of doubles of possible hyperplanes in the hyperspace.
 *
 * @return 2D vector of doubles representing all the possible hyperplanes.
 */
const std::vector<std::vector<double>> &Hyperspace::getHyperLocus() const {
  return hyperLocus;
}

/**
 * @brief Resets the hyperbox to an empty state while keeping the same hyperlocus.
 */
void Hyperspace::resetHyperbox() {
  hyperbox = std::make_shared<Hyperbox>();
}

/**
 * @brief Gets the hyperbox associated with the hyperspace.
 *
 * @return Shared pointer to the Hyperbox object.
 */
const std::shared_ptr<Hyperbox> &Hyperspace::getHyperbox() const {
  return hyperbox;
}

/**
 * @brief Extracts a rule from the hyperspace based on the main sample data and its prediction.
 *
 * @param mainSampleData Data of the main sample.
 * @param mainSamplePred Prediction of the main sample.
 * @param ruleAccuracy Accuracy of the rule.
 * @param ruleConfidence Confidence of the rule.
 * @param mus Means for the denormalization of the rule values (optional).
 * @param sigmas Standard deviations for the denormalization of the rule values (optional).
 * @param normalizationIndices Indices for the denormalization of the rule values (optional).
 * @return A Rule object representing the extracted rule.
 */
Rule Hyperspace::ruleExtraction(const std::vector<double> &mainSampleData, const int mainSamplePred, double ruleAccuracy, double ruleConfidence, const std::vector<double> &mus, const std::vector<double> &sigmas, const std::vector<int> &normalizationIndices) {

  // =========================================================================
  // 1) Validate inputs and optional normalization metadata
  // =========================================================================
  const bool hasMus = !mus.empty();
  const bool hasSigmas = !sigmas.empty();
  const bool hasNormalizationIndices = !normalizationIndices.empty();
  const bool denormalizing = hasMus && hasSigmas && hasNormalizationIndices;

  if (denormalizing) {
    if (!(mus.size() == sigmas.size() && mus.size() == normalizationIndices.size())) {
      throw InternalError("Error during rule extraction : Means, standard deviations, and normalization indices must have the same number of elements.");
    }
  } else if (hasMus || hasSigmas || hasNormalizationIndices) {
    throw InternalError("Error during rule extraction : Means, standard deviations, and normalization indices must either all be specified or none at all.");
  }
  if (mainSampleData.empty()) {
    throw InternalError("Error during rule extraction : Main sample data cannot be empty.");
  }

  const int nbAttributes = static_cast<int>(mainSampleData.size());
  const auto &discrHyperplanes = hyperbox->getDiscriminativeHyperplans();
  std::vector<Antecedent> antecedents;
  antecedents.reserve(discrHyperplanes.size());

  // =========================================================================
  // 2) Build antecedents from discriminative hyperplanes
  // =========================================================================
  for (const auto &discrHyperplane : discrHyperplanes) {
    const int hyperplaneDimension = discrHyperplane.first;
    const int hyperplaneIndex = discrHyperplane.second;
    const int attribute = hyperplaneDimension % nbAttributes;
    double hypValue = hyperLocus[hyperplaneDimension][hyperplaneIndex];
    const double mainSampleValue = mainSampleData[attribute];
    const bool inequalityBool = (hypValue <= mainSampleValue);

    // Denormalization of values in case it was previously normalized
    if (denormalizing) {
      int index = -1;
      for (size_t i = 0; i < normalizationIndices.size(); ++i) {
        if (normalizationIndices[i] == attribute) {
          index = static_cast<int>(i);
          break;
        }
      }
      if (index != -1) {
        hypValue = hypValue * sigmas[index] + mus[index];
      }
    }

    antecedents.emplace_back(attribute, inequalityBool, hypValue);
  }
  return Rule(antecedents, hyperbox->getCoveredSamples(), hyperbox->getCoveringSizesWithNewAntecedent(), mainSamplePred, hyperbox->getFidelity(), hyperbox->getIncreasedFidelity(), ruleAccuracy, hyperbox->getAccuracyChanges(), ruleConfidence);
}

/**
 * @brief Computes the rule confidence as the mean model score on the rule class over covered samples.
 *
 * In Fidex, the rule is built to explain the main sample, so the rule prediction is the same
 * as the main sample prediction. Therefore, `mainSamplePred` is also the rule class used here.
 *
 * @param trainPredictionScores Model prediction scores for training samples.
 * @param mainSamplePred Predicted class of the main sample (also the rule prediction).
 * @param mainSamplePredScore Optional score of the main sample on `mainSamplePred`, included in the mean.
 * @return Mean model score on the rule class over covered samples (and optionally the main sample).
 */
double Hyperspace::computeRuleConfidence(const std::vector<std::vector<double>> &trainPredictionScores, const int mainSamplePred, double mainSamplePredScore) const {

  double total = 0.0; // Sum of prediction scores on the rule class for covered samples

  const auto &coveredSamples = hyperbox->getCoveredSamples();
  size_t nbCovered = coveredSamples.size();
  const bool hasMainSampleScore = (mainSamplePredScore != -1.0);

  // Sum prediction scores on the rule class (same as mainSamplePred in Fidex) over covered training samples
  for (int idSample : coveredSamples) {
    total += trainPredictionScores[idSample][mainSamplePred];
  }

  if (hasMainSampleScore) { // Include the main/test sample score when provided
    total += mainSamplePredScore;
    nbCovered += 1;
  }

  if (nbCovered == 0) {
    return 0.0;
  }

  return total / static_cast<double>(nbCovered);
}
