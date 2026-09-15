#include "fidexAlgo.h"

#include <cmath>
#include <limits>
#include <utility>

namespace {
// ============================================================================
// Local constants
// ============================================================================
constexpr double kDropoutActivationThreshold = 0.001;
constexpr double kFidelityRelaxationStep = 0.05;
constexpr double kThresholdPowerExponent = 4.0;
constexpr double kThresholdVeryPowerExponent = 8.0;
constexpr double kThresholdExpLambda = 6.0;

enum class ThresholdDecayFunction {
  Linear,
  FastPower,
  SlowPower,
  VeryFastPower,
  VerySlowPower,
  FastExponential,
  SlowExponential
};

constexpr ThresholdDecayFunction kThresholdDecayFunction = ThresholdDecayFunction::FastExponential;

const char *thresholdDecayFunctionName(ThresholdDecayFunction function) {
  switch (function) {
  case ThresholdDecayFunction::FastPower:
    return "FastPower";
  case ThresholdDecayFunction::SlowPower:
    return "SlowPower";
  case ThresholdDecayFunction::VeryFastPower:
    return "VeryFastPower";
  case ThresholdDecayFunction::VerySlowPower:
    return "VerySlowPower";
  case ThresholdDecayFunction::FastExponential:
    return "FastExponential";
  case ThresholdDecayFunction::SlowExponential:
    return "SlowExponential";
  case ThresholdDecayFunction::Linear:
  default:
    return "Linear";
  }
}

double computeThresholdFromProgress(double progress, ThresholdDecayFunction function) {
  if (progress <= 0.0) {
    return 1.0;
  }
  if (progress >= 1.0) {
    return 0.0;
  }

  switch (function) {
  case ThresholdDecayFunction::FastPower:
    return std::pow(1.0 - progress, kThresholdPowerExponent);
  case ThresholdDecayFunction::SlowPower:
    return 1.0 - std::pow(progress, kThresholdPowerExponent);
  case ThresholdDecayFunction::VeryFastPower:
    return std::pow(1.0 - progress, kThresholdVeryPowerExponent);
  case ThresholdDecayFunction::VerySlowPower:
    return 1.0 - std::pow(progress, kThresholdVeryPowerExponent);
  case ThresholdDecayFunction::FastExponential:
    return (std::exp(-kThresholdExpLambda * progress) - std::exp(-kThresholdExpLambda)) / (1.0 - std::exp(-kThresholdExpLambda));
  case ThresholdDecayFunction::SlowExponential:
    return 1.0 - (std::exp(kThresholdExpLambda * progress) - 1.0) / (std::exp(kThresholdExpLambda) - 1.0);
  case ThresholdDecayFunction::Linear:
  default:
    return 1.0 - progress;
  }
}

size_t computeThresholdZeroVisitCount(size_t nbHyperplans, double zeroFidelityRatio) {
  if (nbHyperplans == 0) {
    return 0;
  }

  double ratio = zeroFidelityRatio;
  if (ratio <= 0.0) {
    ratio = 1.0 / static_cast<double>(nbHyperplans); // Visit at least one hyperplane before threshold reaches 0
  } else if (ratio > 1.0) {
    ratio = 1.0; // Threshold reaches 0 only at the end of the visit of all hyperplanes
  }

  size_t thresholdZeroVisitCount = static_cast<size_t>(std::ceil(static_cast<double>(nbHyperplans) * ratio)); // Number of hyperplanes to visit before threshold reaches 0
  if (thresholdZeroVisitCount == 0) {
    thresholdZeroVisitCount = 1; // Visit at least one hyperplane before threshold reaches 0
  }
  if (thresholdZeroVisitCount > nbHyperplans) {
    thresholdZeroVisitCount = nbHyperplans; // Threshold reaches 0 only at the end of the visit of all hyperplanes
  }
  return thresholdZeroVisitCount;
}

/**
 * @brief Computes the gain in fidelity of a candidate rule compared to the current rule, adjusted to reflect the gain to reach a perfect fidelity of 1.0.
 * 
 * This function calculates the gain in fidelity of a candidate rule compared to the current rule, and adjusts it to reflect the gain needed to reach a perfect fidelity of 1.0. The gain is computed as the difference between the candidate's fidelity and the current rule's fidelity, divided by the remaining fidelity needed to reach 1.0. If the remaining fidelity is less than or equal to a small epsilon value, the function returns 1.0, indicating that any improvement would effectively lead to a perfect fidelity.
 * * @param candidateFidelity The fidelity of the candidate rule.
 * @param currentRuleFidelity The fidelity of the current rule.
 * @param scoreEpsilon A small epsilon value for numerical stability.
 * @return double The adjusted gain in fidelity of the candidate rule compared to the current rule, reflecting the gain to reach a perfect fidelity of 1.0.
 */
double computeFidelityGainToOne(double candidateFidelity, double currentRuleFidelity, double scoreEpsilon) {
  const double remainingFidelity = 1.0 - currentRuleFidelity;
  if (remainingFidelity <= scoreEpsilon) { // If the remaining fidelity is very small, we consider that we are effectively at 1.0, and any improvement would lead to a perfect fidelity.
    return 1.0;
  }
  return (candidateFidelity - currentRuleFidelity) / remainingFidelity;
}

/**
 * @brief Scalar metadata kept for early-stopping candidates.
 *
 * The covered samples are intentionally not stored here. Once the selected candidate is known,
 * its hyperbox is recomputed once from the current covered samples, dimension and hyperplane.
 */
struct EarlyStoppingCandidate {
  /**
   * @brief Builds the scalar metadata needed to compare and later recompute a candidate.
   */
  EarlyStoppingCandidate(int dimension,
                         int indexHyp,
                         int attribute,
                         bool mainSampleGreater,
                         double hypValue,
                         double fidelityGain,
                         double coveringDrop,
                         double score,
                         size_t coverSize)
      : dimension(dimension),
        indexHyp(indexHyp),
        attribute(attribute),
        mainSampleGreater(mainSampleGreater),
        hypValue(hypValue),
        fidelityGain(fidelityGain),
        coveringDrop(coveringDrop),
        score(score),
        coverSize(coverSize) {
  }

  int dimension;
  int indexHyp;
  int attribute;
  bool mainSampleGreater;
  double hypValue;
  double fidelityGain;
  double coveringDrop;
  double score;
  size_t coverSize;
};

/**
 * @brief Computes the mixed score used to compare candidates.
 *
 * A high score means a good fidelity gain with a limited covering drop. A fidelityImportance of
 * 1.0 ignores covering, while 0.0 keeps only the covering-drop penalty.
 *
 * @param fidelityGain Relative fidelity gain toward 1.0.
 * @param coveringDrop Relative covering loss compared to the current rule.
 * @param fidelityImportance Weight of fidelity gain in the score.
 * @return The mixed candidate score.
 */
double computeCandidateSelectionScore(double fidelityGain, double coveringDrop, double fidelityImportance) {
  return fidelityImportance * fidelityGain - (1.0 - fidelityImportance) * coveringDrop;
}

/**
 * @brief Checks whether a candidate can take part in the mixed-score selection.
 *
 * Early stopping is still triggered by the fidelity threshold. The mixed score is applied only
 * to candidates with a positive fidelity gain and with enough gain relative to the trigger.
 *
 * @param candidateFidelityGain Candidate relative fidelity gain toward 1.0.
 * @param requiredFidelityGain Minimum relative fidelity gain required for mixed-score selection.
 * @param scoreEpsilon Tolerance for floating-point score comparisons.
 * @return True if the candidate is eligible for mixed-score selection.
 */
bool hasEnoughFidelityGainForMixedSelection(double candidateFidelityGain, double requiredFidelityGain, double scoreEpsilon) {
  return candidateFidelityGain > scoreEpsilon && candidateFidelityGain + scoreEpsilon >= requiredFidelityGain;
}

/**
 * @brief Compares candidates with fidelity-first ordering.
 *
 * This is used by the threshold trigger and by fidelity-only early stopping. Covering is only a
 * tie-breaker when fidelity gain is numerically equal.
 *
 * @param candidate Candidate being tested.
 * @param bestCandidate Best candidate known so far.
 * @param scoreEpsilon Tolerance for floating-point score comparisons.
 * @return True if candidate is better than bestCandidate.
 */
bool isBetterByFidelityGain(const EarlyStoppingCandidate &candidate, const EarlyStoppingCandidate &bestCandidate, double scoreEpsilon) {
  if (candidate.fidelityGain > bestCandidate.fidelityGain + scoreEpsilon) {
    return true;
  }
  if (candidate.fidelityGain + scoreEpsilon < bestCandidate.fidelityGain) {
    return false;
  }
  return candidate.coverSize > bestCandidate.coverSize;
}

/**
 * @brief Compares candidates with mixed-score ordering.
 *
 * The mixed score is the primary criterion. Equal scores are resolved by higher fidelity gain,
 * then by higher covering, so covering never beats fidelity gain when the score is tied.
 *
 * @param candidate Candidate being tested.
 * @param bestCandidate Best candidate known so far.
 * @param scoreEpsilon Tolerance for floating-point score comparisons.
 * @return True if candidate is better than bestCandidate.
 */
bool isBetterByMixedScore(const EarlyStoppingCandidate &candidate, const EarlyStoppingCandidate &bestCandidate, double scoreEpsilon) {
  if (candidate.score > bestCandidate.score + scoreEpsilon) {
    return true;
  }
  if (candidate.score + scoreEpsilon < bestCandidate.score) {
    return false;
  }

  return isBetterByFidelityGain(candidate, bestCandidate, scoreEpsilon);
}

/**
 * @brief Selects the candidate applied by mixed-score early stopping.
 *
 * The fallback is the candidate that triggered the threshold. It is already stored in seenCandidates.
 * For positive fidelity gains it also satisfies requiredFidelityGain because
 * requiredFidelityGain = triggerThreshold * fidelityImportance and fidelityImportance <= 1.
 *
 * @param seenCandidates Valid candidates seen before the early-stopping trigger.
 * @param requiredFidelityGain Minimum fidelity gain required for mixed-score selection.
 * @param scoreEpsilon Tolerance for floating-point score comparisons.
 * @param fallbackCandidateIndex Index of the threshold-triggering candidate.
 * @return Index of the selected candidate in seenCandidates.
 */
int selectBestEarlyStoppingCandidate(const std::vector<EarlyStoppingCandidate> &seenCandidates, double requiredFidelityGain, double scoreEpsilon, int fallbackCandidateIndex) {
  int bestCandidateIndex = -1;

  for (size_t i = 0; i < seenCandidates.size(); ++i) {
    const EarlyStoppingCandidate &candidate = seenCandidates[i];
    if (!hasEnoughFidelityGainForMixedSelection(candidate.fidelityGain, requiredFidelityGain, scoreEpsilon)) {
      continue;
    }

    if (bestCandidateIndex == -1 || isBetterByMixedScore(candidate, seenCandidates[bestCandidateIndex], scoreEpsilon)) {
      bestCandidateIndex = static_cast<int>(i);
    }
  }

  return bestCandidateIndex != -1 ? bestCandidateIndex : fallbackCandidateIndex;
}
} // namespace

std::string getThresholdDecayFunctionName() {
  return thresholdDecayFunctionName(kThresholdDecayFunction);
}

/**
 * @brief Constructs a Fidex object with the given training dataset, parameters, and hyperspace and sets the random seed.
 *
 * @param trainDataset Reference to the training dataset.
 * @param parameters Reference to the parameters object.
 * @param hyperspace Reference to the hyperspace.
 * @param usingTestSamples Boolean indicating whether test samples are being used.
 */
Fidex::Fidex(DataSetFid &trainDataset, Parameters &parameters, Hyperspace &hyperspace, bool usingTestSamples) : _trainDataset(&trainDataset), _parameters(&parameters), _hyperspace(&hyperspace), _usingTestSamples(usingTestSamples) {
  int seed = parameters.getInt(SEED);

  if (seed == 0) {
    auto currentTime = std::chrono::steady_clock::now();
    auto seedValue = currentTime.time_since_epoch().count();
    _rnd.seed(seedValue);
  } else {
    _rnd.seed(seed);
  }
}

/**
 * @brief Executes the Fidex algorithm to compute an explaining rule for the given sample based on the training samples and hyperlocus and directed by the given parameters.
 *
 * Fidex builds a rule that meets the specified fidelity and covering criteria. It is driven by
 * a few other parameters, including dropout and the maximum number of iterations allowed.
 * It works by identifying hyperplanes in the feature space that discriminate between different classes of samples and constructing
 * a rule based on these hyperplanes. It updates the provided rule object with the computed rule even if the rule doesn't meet the
 * criteria (minimum covering and minimum fidelity). It returns True if we found a rule metting the criteria.
 *
 * @param rule Reference to the Rule object to be updated by the computation.
 * @param mainSampleValues A vector of double values representing the main sample values.
 * @param mainSamplePred An integer representing the predicted class of the main sample.
 * @param minFidelity A double representing the minimum fidelity threshold for rule creation.
 * @param minCoverSize An integer representing the minimum number of samples a rule must cover.
 * @return True if a rule meeting the criteria is found.
 * @return False if no rule meeting the criteria is found.
 */
bool Fidex::computeFull(Rule &rule, const std::vector<double> &mainSampleValues, int mainSamplePred, double minFidelity, int minCoverSize) {
  // =========================================================================
  // 1) Setup and context initialization
  // =========================================================================

  specs.nbIt = 0;

  // Execution context
  bool showInitialFidelity = getShowInitialFidelity();
  double mainSamplePredictionScore = getMainSamplePredScore(); // Prediction score of the main sample on the predicted class

  // Cached references used throughout the search
  Hyperspace *hyperspace = _hyperspace; // Hyperspace containing the hyperbox and the hyperlocus
  int nbAttributes = _trainDataset->getNbAttributes();
  const std::vector<int> &trainPreds = _trainDataset->getPredictions();
  const std::vector<int> &trainTrueClass = _trainDataset->getClasses();
  const std::vector<std::vector<double>> &trainData = _trainDataset->getDatas();
  const std::vector<std::vector<double>> &trainPredictionScores = _trainDataset->getPredictionScores();
  const auto &hyperLocus = hyperspace->getHyperLocus();    // Matrix of all possible hyperplanes for each dimension
  const auto &hyperbox = hyperspace->getHyperbox();        // Main hyperbox of the rule
  auto nbDimensions = static_cast<int>(hyperLocus.size()); // Number of dimensions in the hyperlocus (must be a multiple of the number of attributes)
  int maxIterations = _parameters->getInt(MAX_ITERATIONS); // Max number of antecedents in the rule
  double dropoutDim = _parameters->getFloat(DROPOUT_DIM);
  double dropoutHyp = _parameters->getFloat(DROPOUT_HYP);
  bool allowNoFidChange = _parameters->getBool(ALLOW_NO_FID_CHANGE); // Whether to allow that a new antecedent does not increase the fidelity of the rule
  bool hasdd = dropoutDim > kDropoutActivationThreshold;
  bool hasdh = dropoutHyp > kDropoutActivationThreshold;
  const double scoreEpsilon = 1e-12;                                             // Tolerance for floating-point score comparisons
  double coeffFidelityImportance = _parameters->getFloat(FIDELITY_IMPORTANCE);   // Coefficient to adjust the importance of fidelity with respect to the covering in the candidate selection objective function (1 = maximise fidelity only, 0 = minimise drop of covering only)
  double thresholdFidelityOnly = _parameters->getFloat(THRESHOLD_FIDELITY_ONLY); // Ratio of max iterations from which it switches to fidelity-only mode
  int thresholdScoreMode = static_cast<int>(thresholdFidelityOnly * maxIterations);
  bool fidelityOnlyMode = coeffFidelityImportance >= 1.0 - scoreEpsilon; // Keep behavior close to computeFull() when only fidelity matters

  // Optional denormalization metadata
  std::vector<int> normalizationIndices;
  std::vector<double> mus;
  std::vector<double> sigmas;

  if (_parameters->isIntVectorSet(NORMALIZATION_INDICES)) {
    normalizationIndices = _parameters->getIntVector(NORMALIZATION_INDICES);
  }
  if (_parameters->isDoubleVectorSet(MUS)) {
    mus = _parameters->getDoubleVector(MUS);
  }
  if (_parameters->isDoubleVectorSet(SIGMAS)) {
    sigmas = _parameters->getDoubleVector(SIGMAS);
  }
  if (_parameters->isDoubleVectorSet(MUS) && !(_parameters->isIntVectorSet(NORMALIZATION_INDICES) && _parameters->isDoubleVectorSet(SIGMAS))) {
    throw InternalError("Error during computation of Fidex: mus are specified but sigmas or normalization indices are not specified.");
  }

  // =========================================================================
  // 2) Input and consistency checks
  // =========================================================================

  // Input and consistency checks
  // Check that we have the prediction value of the main sample if we use test samples (used to compute rule confidence)
  if (mainSamplePredictionScore == -1.0 && _usingTestSamples) {
    throw InternalError("Error during computation of Fidex: Execution with a test sample but no sample prediction value has been given.");
  }

  // Check mainSampleValues size
  if (mainSampleValues.size() != nbAttributes) {
    throw InternalError("Error during computation of Fidex: main sample values size (" + std::to_string(mainSampleValues.size()) + ") is different from the number of attributes in the training dataset (" + std::to_string(nbAttributes) + ").");
  }
  if (nbAttributes <= 0) {
    throw InternalError("Error during computation of Fidex: number of attributes must be strictly positive.");
  }
  if (nbDimensions % nbAttributes != 0) {
    throw InternalError("Error during computation of Fidex: hyperlocus dimension count (" + std::to_string(nbDimensions) + ") must be a multiple of the number of attributes (" + std::to_string(nbAttributes) + ").");
  }

  if (minCoverSize < 1) {
    throw InternalError("Error during computation of Fidex: minimum covering must be >= 1.");
  }

  if (coeffFidelityImportance < 0 || coeffFidelityImportance > 1) {
    throw InternalError("Error during computation of Fidex: coefficient to adjust the importance of fidelity with respect to the covering in the candidate selection objective function must be between 0 and 1.");
  }

  std::uniform_real_distribution<double> dis(0.0, 1.0);

  // =========================================================================
  // 3) Hyperbox initialization
  // =========================================================================

  // Initialize the rule hyperbox with full covering
  std::vector<int> coveredSamples(trainData.size());   // Samples covered by the hyperbox
  iota(begin(coveredSamples), end(coveredSamples), 0); // The vector goes from 0 to len(coveredSamples)-1

  // Reset hyperbox state and compute initial fidelity
  hyperbox->setCoveredSamples(coveredSamples);
  hyperbox->computeFidelity(mainSamplePred, trainPreds); // Compute fidelity of initial hyperbox
  hyperbox->resetDiscriminativeHyperplans();             // We reset hyperbox discriminativeHyperplans
  hyperbox->resetIncreasedFidelity();                    // We reset the increased fidelities
  hyperbox->resetAccuracyChanges();                      // We reset the accuracy changes
  hyperbox->resetCoveringSizesWithNewAntecedent();       // We reset the covering sizes for each antecedent

  if (_usingTestSamples && showInitialFidelity) { // Test samples are not used with fidexGloRules, so we don't show the initial fidelity in this case
    std::cout << "Initial fidelity : " << hyperbox->getFidelity() << std::endl;
  }

  int nbIt = 0;
  std::vector<int> dimensions(nbDimensions);

  // =========================================================================
  // 4) Greedy antecedent search
  // =========================================================================

  // Main search loop: at each iteration, select and apply the best next antecedent if found.
  while (hyperbox->getFidelity() < minFidelity && nbIt < maxIterations) { // While fidelity of our hyperbox is not high enough, we try to add a new discriminative hyperplane (antecedent in the rule)

    if (nbIt >= thresholdScoreMode) {
      coeffFidelityImportance = 1.0;
      fidelityOnlyMode = true;
    }

    Hyperbox bestCandidateHyperbox; // best hyperbox to choose for next step
    Hyperbox candidateHyperbox;

    // Current rule covering used as the common baseline for all candidates in this iteration
    const auto &currentCoveredSamples = hyperbox->getCoveredSamples();
    const size_t ruleCoverSize = currentCoveredSamples.size();
    const double currentRuleFidelity = hyperbox->getFidelity();
    double mainSampleValue;
    int attribute;
    int dimension;
    int indexBestHyp = -1;
    int bestDimension = -1;
    double bestCandidateScore = -std::numeric_limits<double>::infinity();
    double bestCandidateGainedFidelity = -std::numeric_limits<double>::infinity();
    int bestCandidateType = -1; // -1: none, 0: candidateGainedFidelity==0 candidate, 1: candidateGainedFidelity>0 candidate

    // Explore dimensions in random order
    iota(begin(dimensions), end(dimensions), 0);
    shuffle(begin(dimensions), end(dimensions), _rnd);

    for (int d = 0; d < nbDimensions; d++) { // Loop on all dimensions
      if (indexBestHyp != -1 && bestCandidateHyperbox.getFidelity() >= minFidelity) {
        break;
      }

      dimension = dimensions[d];
      attribute = dimension % nbAttributes;
      mainSampleValue = mainSampleValues[attribute];

      // Test if we dropout this dimension
      if (hasdd && dis(_rnd) < dropoutDim) {
        continue; // Drop this dimension if below parameter ex: param=0.2 -> 20% are dropped
      }

      const size_t nbHyp = hyperLocus[dimension].size();
      if (nbHyp == 0) {
        continue; // No data on this dimension
      }

      // Evaluate every hyperplane candidate on this dimension
      for (size_t k = 0; k < nbHyp; ++k) { // for each possible hyperplane in this dimension (there is nbSteps+1 hyperplanes per dimension)
        // Test if we dropout this hyperplane
        if (hasdh && dis(_rnd) < dropoutHyp) {
          continue; // Drop this hyperplane if below parameter ex: param=0.2 -> 20% are dropped
        }

        double hypValue = hyperLocus[dimension][k];
        bool mainSampleGreater = hypValue <= mainSampleValue; // Check if the main sample is on the right side of the hyperplane

        candidateHyperbox.computeCoveredSamplesAndFidelity(currentCoveredSamples, attribute, trainData, mainSampleGreater, hypValue, mainSamplePred, trainPreds); // Compute new cover samples and fidelity

        const auto &candidateCoveredSamples = candidateHyperbox.getCoveredSamples();
        const size_t candidateCoverSize = candidateCoveredSamples.size();
        const double candidateFidelity = candidateHyperbox.getFidelity();

        // Skip candidates that already fail the minimum covering constraint
        if (candidateCoverSize < static_cast<size_t>(minCoverSize)) {
          continue;
        }

        const bool reducesCurrentRuleCover = candidateCoverSize < ruleCoverSize;
        if (!reducesCurrentRuleCover) {
          continue;
        }

        const double candidateGainedFidelity = (candidateFidelity - currentRuleFidelity) / (1 - currentRuleFidelity); // Percentage of gained fidelity with this antecedent out of the maximum possible gain to reach a perfect fidelity of 1.0.
        if (candidateGainedFidelity < -scoreEpsilon) {
          continue; // Worsens fidelity
        }

        const bool candidateIsImproving = candidateGainedFidelity > scoreEpsilon;
        const bool candidateIsFlat = !candidateIsImproving && candidateGainedFidelity >= -scoreEpsilon && candidateGainedFidelity <= scoreEpsilon;
        if (candidateIsFlat && !allowNoFidChange) {
          continue;
        }
        const int candidateType = candidateIsImproving ? 1 : 0;

        const double candidateCoveringDrop = (ruleCoverSize - candidateCoverSize) / static_cast<double>(ruleCoverSize);                          // Percentage of drop in covering with this antecedent
        const double candidateScore = computeCandidateSelectionScore(candidateGainedFidelity, candidateCoveringDrop, coeffFidelityImportance); // Combined score to select the best candidate in this iteration based on the parameters

        bool isBetterCandidate = false;
        if (candidateType > bestCandidateType) {
          // Strict priority: if any candidateGainedFidelity>0 candidate exists, it always beats candidateGainedFidelity==0 candidates.
          isBetterCandidate = true;
        } else if (candidateType == bestCandidateType) {
          const double scoreDeltaVsBestCandidate = candidateScore - bestCandidateScore;
          const size_t bestCandidateCoverSize = bestCandidateHyperbox.getCoveredSamples().size();
          const bool improvesBestCandidateScore = scoreDeltaVsBestCandidate > scoreEpsilon;
          const bool sameScore = scoreDeltaVsBestCandidate >= -scoreEpsilon && scoreDeltaVsBestCandidate <= scoreEpsilon;
          const bool sameScoreWithBetterFidelityGain = sameScore && candidateGainedFidelity > bestCandidateGainedFidelity;
          const bool sameScoreWithBetterCovering = sameScore && candidateCoverSize > bestCandidateCoverSize;
          // Tie-break policy:
          // - mixed objective (a < 1): equal score -> prefer higher fidelity gain
          // - fidelity-only mode (a = 1): equal score -> prefer larger covering (same behavior as computeFull())
          isBetterCandidate = improvesBestCandidateScore ||
                              (fidelityOnlyMode ? sameScoreWithBetterCovering : sameScoreWithBetterFidelityGain);
        }

        if (isBetterCandidate) {
          bestCandidateHyperbox.setFidelity(candidateHyperbox.getFidelity()); // Update best hyperbox
          bestCandidateHyperbox.setCoveredSamples(candidateCoveredSamples);
          bestCandidateType = candidateType;
          bestCandidateScore = candidateScore;
          bestCandidateGainedFidelity = candidateGainedFidelity;
          indexBestHyp = static_cast<int>(k);
          bestDimension = dimension;

          if (bestCandidateHyperbox.getFidelity() >= minFidelity) {
            break;
          }
        }
      }
    }

    // Apply the best candidate found during this outer iteration
    bool antecedentAdded = false;
    if (indexBestHyp != -1 && bestDimension != -1) { // If we found any good dimension with good hyperplane (with enough covering)
      const auto &bestCandidateCoveredSamples = bestCandidateHyperbox.getCoveredSamples();
      const size_t bestCandidateCoverSize = bestCandidateCoveredSamples.size();
      // Candidate acceptance (after selection above)
      // 1) accept if it improves the current rule fidelity
      // 2) otherwise, accept only if same fidelity is explicitly allowed (XOR workaround / progression path)
      const double bestCandidateFidelity = bestCandidateHyperbox.getFidelity();
      const double fidelityDeltaVsCurrentRule = bestCandidateFidelity - currentRuleFidelity;
      const bool improvesCurrentRuleFidelity = fidelityDeltaVsCurrentRule > scoreEpsilon;                                                                // Standard case: adding the antecedent strictly increases fidelity
      const bool sameFidelityAndAllowed = allowNoFidChange && fidelityDeltaVsCurrentRule >= -scoreEpsilon && fidelityDeltaVsCurrentRule <= scoreEpsilon; // Candidate keeps the same fidelity as the current rule (within epsilon) and this is allowed by the parameter

      if (improvesCurrentRuleFidelity || sameFidelityAndAllowed) {
        hyperbox->setFidelity(bestCandidateFidelity);
        hyperbox->addIncreasedFidelity(bestCandidateFidelity);
        hyperbox->setCoveredSamples(bestCandidateCoveredSamples);
        hyperbox->addCoveringSizesWithNewAntecedent(bestCandidateCoverSize);
        hyperbox->addDiscriminativeHyperplan(bestDimension, indexBestHyp);
        antecedentAdded = true;

        double ruleAccuracy = hyperbox->computeRuleAccuracy(mainSamplePred, trainTrueClass); // Percentage of covered samples whose true class matches the rule prediction
        hyperbox->addAccuracyChanges(ruleAccuracy);
      }
    }
    if (!antecedentAdded) {
      if (!(hasdd || hasdh)) {
        break;
      }
      nbIt += 1;
      continue;
    }
    nbIt += 1;
  }

  // =========================================================================
  // 5) Post-processing and final metrics
  // =========================================================================

  // Post-process: remove unnecessary antecedents
  while (optimizeRule(mainSampleValues, mainSamplePred)) {
  }

  // Final rule metrics
  double ruleAccuracy = hyperbox->computeRuleAccuracy(mainSamplePred, trainTrueClass);
  double ruleConfidence = hyperspace->computeRuleConfidence(trainPredictionScores, mainSamplePred, mainSamplePredictionScore); // Mean output value of prediction of class chosen by the rule for the covered samples

  // Extract the rule from the final hyperbox state
  if (_parameters->isDoubleVectorSet(MUS)) {
    rule = hyperspace->ruleExtraction(mainSampleValues, mainSamplePred, ruleAccuracy, ruleConfidence, mus, sigmas, normalizationIndices);
  } else {
    rule = hyperspace->ruleExtraction(mainSampleValues, mainSamplePred, ruleAccuracy, ruleConfidence);
  }

  // Persist execution stats
  specs.showInitialFidelity = false;
  setNbIt(nbIt);

  if (hyperbox->getFidelity() < minFidelity) {
    return false;
  }

  return true;
}

/**
 * @brief Executes the Fidex algorithm with a randomized threshold search for the next antecedent.
 *
 * Fidex builds a rule that explains the prediction of a model for a specific sample. It is based on
 * the training samples and the hyperlocus and directed by the given parameters, including the maximum
 * number of iterations allowed and whether an antecedent is allowed to keep the same fidelity.
 * It visits candidate hyperplanes in a shuffled order until the best fidelity gain seen so far reaches a
 * decreasing threshold. At that point, the threshold is used as a trigger only: the applied antecedent is
 * selected among the already seen candidates whose fidelity gain is at least
 * threshold * fidelity_importance, using the same mixed score as computeFull (fidelity gain versus covering
 * drop). This means fidelity_importance affects early-stopping selection only between candidates that have
 * improved fidelity enough with respect to the current threshold; it never lets covering compensate for a
 * fidelity decrease. When fidelity_importance is 1.0, or once threshold_fidelity_only is reached, the original
 * fidelity-only early-stopping path is kept and no candidate list is built. It updates the provided rule object
 * with the computed rule even if the rule doesn't meet the criteria (minimum covering and minimum fidelity).
 * It returns True if we found a rule meeting the criteria.
 *
 * @param rule Reference to the Rule object to be updated by the computation.
 * @param mainSampleValues A vector of double values representing the main sample values.
 * @param mainSamplePred An integer representing the predicted class of the main sample.
 * @param minFidelity A double representing the minimum fidelity threshold for rule creation.
 * @param minCoverSize An integer representing the minimum number of samples a rule must cover.
 * @return True if a rule meeting the criteria is found.
 * @return False if no rule meeting the criteria is found.
 */
bool Fidex::computeEarlyStopping(Rule &rule, const std::vector<double> &mainSampleValues, int mainSamplePred, double minFidelity, int minCoverSize) {
  // =========================================================================
  // 1) Setup and context initialization
  // =========================================================================
  specs.nbIt = 0;

  // Execution context
  bool showInitialFidelity = getShowInitialFidelity();
  double mainSamplePredictionScore = getMainSamplePredScore(); // Prediction score of the main sample on the predicted class

  // Cached references used throughout the search
  Hyperspace *hyperspace = _hyperspace; // Hyperspace containing the hyperbox and the hyperlocus
  int nbAttributes = _trainDataset->getNbAttributes();
  const std::vector<int> &trainPreds = _trainDataset->getPredictions();
  const std::vector<int> &trainTrueClass = _trainDataset->getClasses();
  const std::vector<std::vector<double>> &trainData = _trainDataset->getDatas();
  const std::vector<std::vector<double>> &trainPredictionScores = _trainDataset->getPredictionScores();
  const auto &hyperLocus = hyperspace->getHyperLocus();    // Matrix of all possible hyperplanes for each dimension
  const auto &hyperbox = hyperspace->getHyperbox();        // Main hyperbox of the rule
  auto nbDimensions = static_cast<int>(hyperLocus.size()); // Number of dimensions in the hyperlocus (must be a multiple of the number of attributes)
  int maxIterations = _parameters->getInt(MAX_ITERATIONS); // Max number of antecedents in the rule
  bool allowNoFidChange = _parameters->getBool(ALLOW_NO_FID_CHANGE); // Whether to allow that a new antecedent does not increase the fidelity of the rule
  const double scoreEpsilon = 1e-12;                                // Tolerance for floating-point score comparisons
  double fidelityImportance = _parameters->getFloat(FIDELITY_IMPORTANCE); // Weight of fidelity gain in the mixed candidate score
  double thresholdFidelityOnly = _parameters->getFloat(THRESHOLD_FIDELITY_ONLY); // Ratio of max iterations from which it switches to fidelity-only mode
  int thresholdScoreMode = static_cast<int>(thresholdFidelityOnly * maxIterations); // Iteration from which it switches to fidelity-only mode
  double zeroFidelityRatio = _parameters->getFloat(ZERO_FIDELITY_RATIO); // Ratio of hyperplanes to visit before the acceptance threshold reaches 0

  // Optional denormalization metadata
  std::vector<int> normalizationIndices;
  std::vector<double> mus;
  std::vector<double> sigmas;

  if (_parameters->isIntVectorSet(NORMALIZATION_INDICES)) {
    normalizationIndices = _parameters->getIntVector(NORMALIZATION_INDICES);
  }
  if (_parameters->isDoubleVectorSet(MUS)) {
    mus = _parameters->getDoubleVector(MUS);
  }
  if (_parameters->isDoubleVectorSet(SIGMAS)) {
    sigmas = _parameters->getDoubleVector(SIGMAS);
  }
  if (_parameters->isDoubleVectorSet(MUS) && !(_parameters->isIntVectorSet(NORMALIZATION_INDICES) && _parameters->isDoubleVectorSet(SIGMAS))) {
    throw InternalError("Error during computation of Fidex: mus are specified but sigmas or normalization indices are not specified.");
  }

  // =========================================================================
  // 2) Input and consistency checks
  // =========================================================================

  // Input and consistency checks
  // Check that we have the prediction value of the main sample if we use test samples (used to compute rule confidence)
  if (mainSamplePredictionScore == -1.0 && _usingTestSamples) {
    throw InternalError("Error during computation of Fidex: Execution with a test sample but no sample prediction value has been given.");
  }

  // Check mainSampleValues size
  if (mainSampleValues.size() != nbAttributes) {
    throw InternalError("Error during computation of Fidex: main sample values size (" + std::to_string(mainSampleValues.size()) + ") is different from the number of attributes in the training dataset (" + std::to_string(nbAttributes) + ").");
  }
  if (nbAttributes <= 0) {
    throw InternalError("Error during computation of Fidex: number of attributes must be strictly positive.");
  }
  if (nbDimensions % nbAttributes != 0) {
    throw InternalError("Error during computation of Fidex: hyperlocus dimension count (" + std::to_string(nbDimensions) + ") must be a multiple of the number of attributes (" + std::to_string(nbAttributes) + ").");
  }

  if (minCoverSize < 1) {
    throw InternalError("Error during computation of Fidex: minimum covering must be >= 1.");
  }

  if (fidelityImportance < 0 || fidelityImportance > 1) {
    throw InternalError("Error during computation of Fidex: coefficient to adjust the importance of fidelity with respect to the covering in the candidate selection objective function must be between 0 and 1.");
  }

  if (thresholdFidelityOnly < 0 || thresholdFidelityOnly > 1) {
    throw InternalError("Error during computation of Fidex: iteration ratio from which Fidex switches to fidelity-only mode must be between 0 and 1.");
  }

  // =========================================================================
  // 3) Hyperbox initialization
  // =========================================================================

  // Initialize the rule hyperbox with full covering
  std::vector<int> coveredSamples(trainData.size());   // Samples covered by the hyperbox
  iota(begin(coveredSamples), end(coveredSamples), 0); // The vector goes from 0 to len(coveredSamples)-1

  // Reset hyperbox state and compute initial fidelity
  hyperbox->setCoveredSamples(coveredSamples);
  hyperbox->computeFidelity(mainSamplePred, trainPreds); // Compute fidelity of initial hyperbox
  hyperbox->resetDiscriminativeHyperplans();             // We reset hyperbox discriminativeHyperplans
  hyperbox->resetIncreasedFidelity();                    // We reset the increased fidelities
  hyperbox->resetAccuracyChanges();                      // We reset the accuracy changes
  hyperbox->resetCoveringSizesWithNewAntecedent();       // We reset the covering sizes for each antecedent

  if (_usingTestSamples && showInitialFidelity) { // Test samples are not used with fidexGloRules, so we don't show the initial fidelity in this case
    std::cout << "Initial fidelity : " << hyperbox->getFidelity() << std::endl;
  }

  int nbIt = 0;
  std::vector<std::pair<int, int>> randomHyperplans;
  size_t nbHyperplans = 0;

  for (int dimension = 0; dimension < nbDimensions; ++dimension) {
    nbHyperplans += hyperLocus[dimension].size();
  }

  randomHyperplans.reserve(nbHyperplans);
  for (int dimension = 0; dimension < nbDimensions; ++dimension) {
    const size_t nbHyp = hyperLocus[dimension].size();
    for (size_t k = 0; k < nbHyp; ++k) {
      randomHyperplans.push_back(std::make_pair(dimension, static_cast<int>(k)));
    }
  }

  const size_t thresholdZeroVisitCount = computeThresholdZeroVisitCount(randomHyperplans.size(), zeroFidelityRatio);

  // =========================================================================
  // 4) Randomized threshold antecedent search
  // =========================================================================

  // Main search loop: stop when fidelity progress is good enough. In mixed mode, apply the best score candidate seen so far that still has enough fidelity gain.
  while (hyperbox->getFidelity() < minFidelity && nbIt < maxIterations) { // While fidelity of our hyperbox is not high enough, we try to add a new discriminative hyperplane (antecedent in the rule)

    const bool fidelityOnlySelection = fidelityImportance >= 1.0 - scoreEpsilon || nbIt >= thresholdScoreMode; // Keep the original early-stopping path when only fidelity matters or when the mixed-score phase is over

    Hyperbox bestCandidateHyperbox; // best hyperbox to choose for next step
    Hyperbox candidateHyperbox;

    // Current rule covering used as the common baseline for all candidates in this iteration
    const auto &currentCoveredSamples = hyperbox->getCoveredSamples();
    const size_t ruleCoverSize = currentCoveredSamples.size();
    const double currentRuleFidelity = hyperbox->getFidelity();
    int triggerCandidateIndex = -1;
    int selectedCandidateIndex = -1;
    int indexBestHyp = -1;
    int bestDimension = -1;
    double bestCandidateFidelityGain = -std::numeric_limits<double>::infinity();
    bool candidateAccepted = false;
    std::vector<EarlyStoppingCandidate> seenCandidates;
    if (!fidelityOnlySelection) {
      seenCandidates.reserve(std::min(randomHyperplans.size(), static_cast<size_t>(1024)));
    }

    // Explore all hyperplanes in random order without visiting the same pair twice in this iteration
    shuffle(begin(randomHyperplans), end(randomHyperplans), _rnd);

    for (size_t visitedHyperplans = 0; visitedHyperplans < randomHyperplans.size(); ++visitedHyperplans) {
      int dimension = randomHyperplans[visitedHyperplans].first;
      int indexHyp = randomHyperplans[visitedHyperplans].second;
      int attribute = dimension % nbAttributes;
      double mainSampleValue = mainSampleValues[attribute];
      double hypValue = hyperLocus[dimension][indexHyp];
      bool mainSampleGreater = hypValue <= mainSampleValue; // Check if the main sample is on the right side of the hyperplane

      candidateHyperbox.computeCoveredSamplesAndFidelity(currentCoveredSamples, attribute, trainData, mainSampleGreater, hypValue, mainSamplePred, trainPreds); // Compute new cover samples and fidelity

      const auto &candidateCoveredSamples = candidateHyperbox.getCoveredSamples();
      const size_t candidateCoverSize = candidateCoveredSamples.size();
      const double candidateFidelity = candidateHyperbox.getFidelity();

      // Skip candidates that already fail the minimum covering constraint
      if (candidateCoverSize >= static_cast<size_t>(minCoverSize)) {
        const bool reducesCurrentRuleCover = candidateCoverSize < ruleCoverSize;

        if (reducesCurrentRuleCover) { // Only consider candidates that reduce the covering of the current rule
          const double fidelityDeltaVsCurrentRule = candidateFidelity - currentRuleFidelity;
          const bool improvesCurrentRuleFidelity = fidelityDeltaVsCurrentRule > scoreEpsilon;                                                                // Standard case: adding the antecedent strictly increases fidelity
          const bool sameFidelityAndAllowed = allowNoFidChange && fidelityDeltaVsCurrentRule >= -scoreEpsilon && fidelityDeltaVsCurrentRule <= scoreEpsilon; // Candidate keeps the same fidelity as the current rule (within epsilon) and this is allowed by the parameter

          if (improvesCurrentRuleFidelity || sameFidelityAndAllowed) {
            const double candidateFidelityGain = computeFidelityGainToOne(candidateFidelity, currentRuleFidelity, scoreEpsilon); // Percentage of gained fidelity with this antecedent out of the maximum possible gain to reach fidelity 1

            if (fidelityOnlySelection) {
              bool isBetterCandidate = false;
              if (indexBestHyp == -1) {
                isBetterCandidate = true;
              } else if (candidateFidelityGain > bestCandidateFidelityGain + scoreEpsilon) {
                isBetterCandidate = true;
              } else if (candidateFidelityGain + scoreEpsilon >= bestCandidateFidelityGain && candidateCoverSize > bestCandidateHyperbox.getCoveredSamples().size()) {
                isBetterCandidate = true;
              }

              if (isBetterCandidate) {
                bestCandidateHyperbox.setFidelity(candidateHyperbox.getFidelity());
                bestCandidateHyperbox.setCoveredSamples(candidateCoveredSamples);
                bestCandidateFidelityGain = candidateFidelityGain;
                indexBestHyp = indexHyp;
                bestDimension = dimension;
              }
            } else { // In mixed mode, we keep track of all candidates seen until now to be able to select the best one above the threshold when we reach it, but we don't update the best candidate at each step based on the mixed score because we only want to apply the score-based selection once we reach the threshold, not at each step.
              const double candidateCoveringDrop = (ruleCoverSize - candidateCoverSize) / static_cast<double>(ruleCoverSize);
              const double candidateScore = computeCandidateSelectionScore(candidateFidelityGain, candidateCoveringDrop, fidelityImportance);

              // Store scalar metadata only. The covered samples are recomputed once for the selected candidate.
              seenCandidates.push_back(EarlyStoppingCandidate(dimension,
                                                              indexHyp,
                                                              attribute,
                                                              mainSampleGreater,
                                                              hypValue,
                                                              candidateFidelityGain,
                                                              candidateCoveringDrop,
                                                              candidateScore,
                                                              candidateCoverSize));

              const int currentCandidateIndex = static_cast<int>(seenCandidates.size()) - 1;
              // Update the trigger candidate, which is the candidate with the best fidelity gain among those seen so far
              if (triggerCandidateIndex == -1 || isBetterByFidelityGain(seenCandidates[currentCandidateIndex], seenCandidates[triggerCandidateIndex], scoreEpsilon)) {
                triggerCandidateIndex = currentCandidateIndex;
              }
            }
          }
        }
      }

      if (thresholdZeroVisitCount > 0) {
        const double progress = static_cast<double>(visitedHyperplans + 1) / static_cast<double>(thresholdZeroVisitCount);
        const double threshold = computeThresholdFromProgress(progress, kThresholdDecayFunction);

        if (fidelityOnlySelection && indexBestHyp != -1 && bestCandidateFidelityGain + scoreEpsilon >= threshold) {
          candidateAccepted = true;
          break;
        }

        // In mixed mode, stop scanning as soon as the best fidelity gain seen so far reaches the decreasing threshold.
        // The applied candidate is then selected by mixed score, but only among candidates whose fidelity gain is at
        // least threshold * fidelity_importance. If no such improving candidate exists (possible only for the
        // allow_no_fid_change fallback when the threshold reaches 0), use the threshold-triggering candidate.
        if (!fidelityOnlySelection && triggerCandidateIndex != -1 && seenCandidates[triggerCandidateIndex].fidelityGain + scoreEpsilon >= threshold) {
          const double requiredFidelityGain = threshold * fidelityImportance;
          selectedCandidateIndex = selectBestEarlyStoppingCandidate(seenCandidates, requiredFidelityGain, scoreEpsilon, triggerCandidateIndex);
          candidateAccepted = true;
          break;
        }
      }
    }

    // If we visited all hyperplanes without accepting a candidate, no valid antecedent can be found for the current rule
    if (!candidateAccepted) {
      break;
    }

    int selectedDimension = bestDimension;
    int selectedIndexHyp = indexBestHyp;
    if (!fidelityOnlySelection) {
      const EarlyStoppingCandidate &selectedCandidate = seenCandidates[selectedCandidateIndex];
      bestCandidateHyperbox.computeCoveredSamplesAndFidelity(currentCoveredSamples,
                                                             selectedCandidate.attribute,
                                                             trainData,
                                                             selectedCandidate.mainSampleGreater,
                                                             selectedCandidate.hypValue,
                                                             mainSamplePred,
                                                             trainPreds);
      selectedDimension = selectedCandidate.dimension;
      selectedIndexHyp = selectedCandidate.indexHyp;
    }
    const auto &bestCandidateCoveredSamples = bestCandidateHyperbox.getCoveredSamples();
    const size_t bestCandidateCoverSize = bestCandidateCoveredSamples.size();
    const double bestCandidateFidelity = bestCandidateHyperbox.getFidelity();

    hyperbox->setFidelity(bestCandidateFidelity);
    hyperbox->addIncreasedFidelity(bestCandidateFidelity);
    hyperbox->setCoveredSamples(bestCandidateCoveredSamples);
    hyperbox->addCoveringSizesWithNewAntecedent(bestCandidateCoverSize);
    hyperbox->addDiscriminativeHyperplan(selectedDimension, selectedIndexHyp);

    double ruleAccuracy = hyperbox->computeRuleAccuracy(mainSamplePred, trainTrueClass); // Percentage of covered samples whose true class matches the rule prediction
    hyperbox->addAccuracyChanges(ruleAccuracy);

    nbIt += 1;
  }

  // =========================================================================
  // 5) Post-processing and final metrics
  // =========================================================================

  // Post-process: remove unnecessary antecedents
  while (optimizeRule(mainSampleValues, mainSamplePred)) {
  }

  // Final rule metrics
  double ruleAccuracy = hyperbox->computeRuleAccuracy(mainSamplePred, trainTrueClass);
  double ruleConfidence = hyperspace->computeRuleConfidence(trainPredictionScores, mainSamplePred, mainSamplePredictionScore); // Mean output value of prediction of class chosen by the rule for the covered samples

  // Extract the rule from the final hyperbox state
  if (_parameters->isDoubleVectorSet(MUS)) {
    rule = hyperspace->ruleExtraction(mainSampleValues, mainSamplePred, ruleAccuracy, ruleConfidence, mus, sigmas, normalizationIndices);
  } else {
    rule = hyperspace->ruleExtraction(mainSampleValues, mainSamplePred, ruleAccuracy, ruleConfidence);
  }

  // Persist execution stats
  specs.showInitialFidelity = false;
  setNbIt(nbIt);

  if (hyperbox->getFidelity() < minFidelity) {
    return false;
  }

  return true;
}

/**
 * @brief Attempts to compute a rule with Fidex algorithm based on given parameters and updates the rule object if successful.
 *
 * @param rule Reference to the Rule object to be potentially updated by the computation.
 * @param mainSampleValues A vector of double values representing the main sample values.
 * @param mainSamplePred An integer representing the predicted class of the main sample.
 * @param minFidelity A double representing the minimum fidelity threshold for rule creation.
 * @param minCoverSize An integer representing the minimum number of samples a rule must cover.
 * @param verbose A boolean flag for detailed verbose output.
 * @param detailedVerbose A boolean flag for detailed verbose output. Default is false.
 * @param foundRule A boolean indicating whether a rule was found in a previous attempt. Default is false.
 * @return True if a rule meeting the criteria is successfully computed.
 * @return False if no rule meeting the criteria can be computed.
 */
bool Fidex::tryComputeFidex(Rule &rule, const std::vector<double> &mainSampleValues, int mainSamplePred, double minFidelity, int minCoverSize, bool verbose, bool detailedVerbose, bool foundRule) {
  // =========================================================================
  // One attempt wrapper (optional logs + compute Fidex)
  // =========================================================================

  if (detailedVerbose && verbose) {
    if (foundRule) {
      std::cout << "A rule has been found. ";
    } else {
      std::cout << "Fidelity is too low. ";
    }
    std::cout << "Restarting fidex with a minimum covering of " << minCoverSize << " and a minimum accepted fidelity of " << minFidelity << "." << std::endl;
  }

  const std::string fidexVersion = _parameters->getString(FIDEX_VERSION);
  const bool ruleCreated = fidexVersion == "fidexFull"
                               ? computeFull(rule, mainSampleValues, mainSamplePred, minFidelity, minCoverSize)
                               : computeEarlyStopping(rule, mainSampleValues, mainSamplePred, minFidelity, minCoverSize);
  if (verbose) {
    std::cout << "Final fidelity : " << rule.getFidelity() << std::endl;
  }
  return ruleCreated;
}

/**
 * @brief Performs a dichotomic (binary) search to find a rule with the best covering that meets the minimum fidelity criteria.
 *
 * It adjusts the search range based on the fidelity and covering size of the rules computed in each iteration to find rule with
 * the best covering size possible.
 *
 * @param bestRule Reference to the Rule object to store the best rule found during the search.
 * @param mainSampleValues A vector of double values representing the main sample values.
 * @param mainSamplePred An integer representing the predicted class of the main sample.
 * @param minFidelity A double representing the minimum fidelity threshold for rule creation.
 * @param left The starting point of the search range.
 * @param right The ending point of the search range.
 * @param verbose A boolean flag for detailed verbose output.
 * @return The best covering found that meets the minimum fidelity criteria. Returns -1 if no such covering is found.
 */
int Fidex::dichotomicSearch(Rule &bestRule, const std::vector<double> &mainSampleValues, int mainSamplePred, double minFidelity, int left, int right, bool verbose) {
  // =========================================================================
  // Dichotomic search on minimum covering
  // =========================================================================

  int bestCovering = -1;
  bool foundRule = false;
  while (left <= right) {
    const int currentMinCoverSize = left + (right - left + 1) / 2; // Upper mid to search the largest feasible covering
    Rule tempRule;
    if (tryComputeFidex(tempRule, mainSampleValues, mainSamplePred, minFidelity, currentMinCoverSize, verbose, true, foundRule)) {
      bestCovering = currentMinCoverSize;
      bestRule = std::move(tempRule);
      left = currentMinCoverSize + 1;
      foundRule = true;
    } else {
      right = currentMinCoverSize - 1;
      foundRule = false;
    }
  }
  return bestCovering;
}

/**
 * @brief Attempts to compute a rule multiple times up to a maximum number of failed attempts, adjusting fidelity if necessary.
 *
 * @param rule Reference to the Rule object to be potentially updated by the computation.
 * @param mainSampleValues A vector of double values representing the main sample values.
 * @param mainSamplePred An integer representing the predicted class of the main sample.
 * @param minFidelity A double representing the current minimum fidelity threshold for rule creation.
 * @param minCoverSize An integer representing the minimum number of samples a rule must cover.
 * @param verbose A boolean flag for detailed verbose output.
 * @return True if a rule meeting the criteria is successfully computed within the maximum number of attempts.
 * @return False if no rule meeting the criteria can be computed within the maximum number of attempts.
 */
bool Fidex::retryComputeFidex(Rule &rule, const std::vector<double> &mainSampleValues, int mainSamplePred, double minFidelity, int minCoverSize, bool verbose) {
  // =========================================================================
  // Retry loop at fixed fidelity/covering
  // =========================================================================

  int counterFailed = 0; // Number of times we failed to find a rule with maximal fidexlity when minCoverSize is 1
  int maxFailedAttempts = _parameters->getInt(MAX_FAILED_ATTEMPTS);
  bool allowNoFidChange = _parameters->getBool(ALLOW_NO_FID_CHANGE);
  bool covering_strategy = _parameters->getBool(COVERING_STRATEGY);
  bool ruleCreated = false;
  bool hasDropout = _parameters->getFloat(DROPOUT_DIM) > kDropoutActivationThreshold || _parameters->getFloat(DROPOUT_HYP) > kDropoutActivationThreshold;
  do {
    ruleCreated = tryComputeFidex(rule, mainSampleValues, mainSamplePred, minFidelity, minCoverSize, verbose, true);
    if (!ruleCreated) {
      counterFailed += 1;
    }
    if (counterFailed >= maxFailedAttempts && verbose) {
      std::cout << "\nWARNING Fidelity is too low after trying " << std::to_string(maxFailedAttempts) << " times with a minimum covering of " << minCoverSize << " and a minimum accepted fidelity of " << minFidelity << "! You may want to try again with a lower min_covering or a lower min_fidelity." << std::endl;
      if (hasDropout) {
        std::cout << "You can also try to not use dropout." << std::endl;
      }
      if (!covering_strategy) {
        std::cout << "You can also try to use the min cover strategy (--covering_strategy)." << std::endl;
      }
      if (!allowNoFidChange) {
        std::cout << "You could also try to allow to add a new antecedant without changing the fidelity by setting allow_no_fid_change to true." << std::endl;
      }
    }
  } while (!ruleCreated && counterFailed < maxFailedAttempts);

  return ruleCreated;
}

/**
 * @brief Launches the Fidex algorithm with specified parameters to attempt creating a rule for the given sample that meets given minimum covering and minimum fidelity criteria.
 *
 * Fidex is based on the training samples and hyperlocus and directed by the given parameters,
 * including dropout and the maximum number of iterations allowed.
 * It works by identifying hyperplanes in the feature space that discriminate between different
 * classes of samples and constructing a rule based on these hyperplanes.
 *
 * Computes Fidex until a rule is created or until the max failed attempts limit is reached.<br>
 *   - First attempt to generate a rule with a covering greater or equal to 'min_covering' and a fidelity greater or equal to 'min_fidelity'.<br>
 *   - If the attempt failed and the 'covering_strategy' is on, Fidex is computed to find a rule with the max possible minimal covering that can be lower than 'min_covering'.<br>
 *   - If all attempts failed, the targeted fidelity is gradually lowered until it succeed or 'lowest_min_fidelity' is reached.<br>
 *   - Each failed attempt on lowest minimal fidelity are counted.<br>
 *   - If the max failed attempts limit is reached, then the rule couldn't be computed for this sample.
 *
 * @param rule Reference to the Rule object to be potentially updated by the computation.
 * @param mainSampleValues A vector of double values representing the main sample values.
 * @param mainSamplePred An integer representing the predicted class of the main sample.
 * @param verbose A boolean flag for detailed verbose output. Default is false.
 * @return True if a rule meeting the criteria is successfully computed.
 * @return False if no rule meeting the criteria can be computed.
 */
bool Fidex::launchFidex(Rule &rule, const std::vector<double> &mainSampleValues, int mainSamplePred, bool verbose) {
  // =========================================================================
  // 1) Initial target setup
  // =========================================================================

  int minCoverSize = _parameters->getInt(MIN_COVERING);
  double minFidelity = _parameters->getFloat(MIN_FIDELITY);
  const bool coveringStrategy = _parameters->getBool(COVERING_STRATEGY);
  const double lowestMinFidelity = _parameters->getFloat(LOWEST_MIN_FIDELITY);

  if (verbose) {
    setShowInitialFidelity(true);
  }

  // =========================================================================
  // 2) First direct attempt with requested thresholds
  // =========================================================================

  // First attempt with requested fidelity/covering
  if (tryComputeFidex(rule, mainSampleValues, mainSamplePred, minFidelity, minCoverSize, verbose)) {
    return true;
  }

  // =========================================================================
  // 3) Optional covering strategy (dichotomic search)
  // =========================================================================

  // Without covering strategy, stop after the first failure
  if (!coveringStrategy) {
    if (verbose) {
      std::cout << "\nWARNING Fidelity is too low! You may want to try again." << std::endl;
      std::cout << "If you can't find a rule with the wanted fidelity, try a lowest minimal covering or a lower fidelity" << std::endl;
      std::cout << "You can also try to use the min cover strategy (--covering_strategy)" << std::endl;
      std::cout << "If this is not enough, put the min covering to 1 and do not use dropout." << std::endl;
      std::cout << "You may also want to allow to add a new antecedant without changing the fidelity by setting allow_no_fid_change to true.\n"
                << std::endl;
    }
    return false;
  }

  // Covering strategy: search for the best feasible covering below the requested minimum
  int right = minCoverSize - 1;
  int bestCovering = -1;
  Rule bestRule;

  if (right > 0) {
    bestCovering = dichotomicSearch(bestRule, mainSampleValues, mainSamplePred, minFidelity, 1, right, verbose);
  }

  if (bestCovering != -1) { // A valid rule was found during dichotomic search
    rule = std::move(bestRule);
    if (verbose) {
      std::cout << std::endl;
    }
    return true;
  }

  // =========================================================================
  // 4) Fidelity relaxation, then retry at lowest reached fidelity
  // =========================================================================

  // Couldn't find a rule with the current minimum fidelity: progressively lower the target.
  bool ruleCreated = false;
  double currentMinFidelity = minFidelity;
  while (!ruleCreated && currentMinFidelity > lowestMinFidelity) {
    currentMinFidelity = std::max(lowestMinFidelity, currentMinFidelity - kFidelityRelaxationStep);
    ruleCreated = tryComputeFidex(rule, mainSampleValues, mainSamplePred, currentMinFidelity, 1, verbose, true);
  }

  // Still no rule: retry several times at the current lowered fidelity.
  if (!ruleCreated) {
    ruleCreated = retryComputeFidex(rule, mainSampleValues, mainSamplePred, currentMinFidelity, 1, verbose);
  }

  if (verbose) {
    std::cout << std::endl;
  }
  return ruleCreated;
}

/**
 * @brief @brief Attepts to filter unnecessary attributes in a rule.
 *
 * @param mainSampleValues A vector of double values representing the main sample values.
 * @param mainSamplePred An integer representing the predicted class of the main sample.
 * @return wether an optimisation has been done or not.
 */
bool Fidex::optimizeRule(const std::vector<double> &mainSampleValues, int mainSamplePred) {
  // =========================================================================
  // 1) Prepare optimization candidates
  // =========================================================================

  const auto &originalHyperbox = _hyperspace->getHyperbox();
  const auto &originalDiscrHyperplans = originalHyperbox->getDiscriminativeHyperplans();
  const size_t nbAntecedents = originalDiscrHyperplans.size();
  if (nbAntecedents == 0) {
    return false;
  }

  const auto &trainData = _trainDataset->getDatas();
  const auto &trainPreds = _trainDataset->getPredictions();
  const auto &trainTrueClass = _trainDataset->getClasses();
  const auto &hyperLocus = _hyperspace->getHyperLocus();
  std::vector<int> coveredSamples(trainData.size()); // Samples covered by the hyperbox
  Hyperbox bestHyperbox = originalHyperbox->deepCopy();
  iota(begin(coveredSamples), end(coveredSamples), 0); // Vector from 0 to len(coveredSamples)-1
  int nbAttributes = _trainDataset->getNbAttributes();
  bool hasBeenOptimized = false;

  // =========================================================================
  // 2) Try removing each antecedent once
  // =========================================================================

  for (size_t i = 0; i < nbAntecedents; i++) {
    std::vector<std::pair<int, int>> copyDiscrHyperplans(originalDiscrHyperplans); // create original's copy
    Hyperbox copyHyperbox;
    copyDiscrHyperplans.erase(copyDiscrHyperplans.begin() + i); // hide an antecedant
    copyHyperbox.setDiscriminativeHyperplans(copyDiscrHyperplans);
    copyHyperbox.setCoveredSamples(coveredSamples);

    for (const auto &antecedant : copyDiscrHyperplans) {
      int dimension = antecedant.first;
      int hypIndex = antecedant.second;
      int feature = dimension % nbAttributes;

      double hypValue = hyperLocus[dimension][hypIndex];
      double mainSampleValue = mainSampleValues[feature];
      bool isMainSampleGreater = hypValue <= mainSampleValue;

      copyHyperbox.computeCoveredSamplesAndFidelity(copyHyperbox.getCoveredSamples(), feature, trainData, isMainSampleGreater, hypValue, mainSamplePred, trainPreds);
      copyHyperbox.addIncreasedFidelity(copyHyperbox.getFidelity());
      copyHyperbox.addCoveringSizesWithNewAntecedent(copyHyperbox.getCoveredSamples().size());
      copyHyperbox.addAccuracyChanges(copyHyperbox.computeRuleAccuracy(mainSamplePred, trainTrueClass));
    }

    // If the original rule had a single antecedent, removing it leaves an empty rule:
    // no inner iteration ran, so fidelity still needs to be computed once on full covering.
    if (nbAntecedents == 1) {
      copyHyperbox.computeFidelity(mainSamplePred, trainPreds);
    }

    if (copyHyperbox.getFidelity() >= bestHyperbox.getFidelity()) {
      hasBeenOptimized = true;
      bestHyperbox = std::move(copyHyperbox);
    }
  }

  // =========================================================================
  // 3) Commit the best simplified hyperbox (if any)
  // =========================================================================

  if (hasBeenOptimized) {
    originalHyperbox->setAccuracyChanges(bestHyperbox.getAccuracyChanges());
    originalHyperbox->setIncreasedFidelity(bestHyperbox.getIncreasedFidelity());
    originalHyperbox->setCoveringSizesWithNewAntecedent(bestHyperbox.getCoveringSizesWithNewAntecedent());
    originalHyperbox->setFidelity(bestHyperbox.getFidelity());
    originalHyperbox->setCoveredSamples(bestHyperbox.getCoveredSamples());
    originalHyperbox->setDiscriminativeHyperplans(bestHyperbox.getDiscriminativeHyperplans());
  }

  return hasBeenOptimized;
}
