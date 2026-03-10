#include "fidexAlgo.h"

namespace {
// ============================================================================
// Local constants
// ============================================================================
constexpr double kDropoutActivationThreshold = 0.001;
constexpr double kFidelityRelaxationStep = 0.05;
} // namespace

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
    auto currentTime = std::chrono::high_resolution_clock::now();
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
bool Fidex::compute(Rule &rule, const std::vector<double> &mainSampleValues, int mainSamplePred, double minFidelity, int minCoverSize) {

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
  std::vector<int> &trainPreds = _trainDataset->getPredictions();
  std::vector<int> &trainTrueClass = _trainDataset->getClasses();
  std::vector<std::vector<double>> &trainData = _trainDataset->getDatas();
  std::vector<std::vector<double>> &trainPredictionScores = _trainDataset->getPredictionScores();
  const auto &hyperLocus = hyperspace->getHyperLocus();    // Matrix of all possible hyperplanes for each dimension
  const auto &hyperbox = hyperspace->getHyperbox();        // Main hyperbox of the rule
  auto nbDimensions = static_cast<int>(hyperLocus.size()); // Number of dimensions in the hyperlocus (must be a multiple of the number of attributes)
  int maxIterations = _parameters->getInt(MAX_ITERATIONS); // Max number of antecedents in the rule
  double dropoutDim = _parameters->getFloat(DROPOUT_DIM);
  double dropoutHyp = _parameters->getFloat(DROPOUT_HYP);
  bool allowNoFidChange = _parameters->getBool(ALLOW_NO_FID_CHANGE); // Whether to allow that a new antecedent does not increase the fidelity of the rule
  bool hasdd = dropoutDim > kDropoutActivationThreshold;
  bool hasdh = dropoutHyp > kDropoutActivationThreshold;
  const double fidelityEpsilon = 1e-12; // Tolerance for floating-point fidelity comparisons

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

  // Main search loop: at each iteration, select and apply the best next antecedent if found
  while (hyperbox->getFidelity() < minFidelity && nbIt < maxIterations) { // While fidelity of our hyperbox is not high enough, we try to add a new discriminative hyperplane (antecedent in the rule)
    Hyperbox bestCandidateHyperbox;                                       // best hyperbox to choose for next step
    Hyperbox candidateHyperbox;

    // Current rule covering used as the common baseline for all candidates in this iteration
    const auto &ruleCoveredSamples = hyperbox->getCoveredSamples();
    const size_t ruleCoverSize = ruleCoveredSamples.size();
    double mainSampleValue;
    int attribute;
    int dimension;
    int indexBestHyp = -1;
    int bestDimension = -1;

    // Explore dimensions in random order
    iota(begin(dimensions), end(dimensions), 0);
    shuffle(begin(dimensions), end(dimensions), _rnd);

    for (int d = 0; d < nbDimensions; d++) { // Loop on all dimensions
      if (bestCandidateHyperbox.getFidelity() >= minFidelity) {
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

        candidateHyperbox.computeCoveredSamplesAndFidelity(ruleCoveredSamples, attribute, trainData, mainSampleGreater, hypValue, mainSamplePred, trainPreds); // Compute new cover samples and fidelity

        const auto &candidateCoveredSamples = candidateHyperbox.getCoveredSamples();
        const size_t candidateCoverSize = candidateCoveredSamples.size();

        // Skip candidates that already fail the minimum covering constraint
        if (candidateCoverSize < static_cast<size_t>(minCoverSize)) {
          continue;
        }

        const size_t bestCandidateCoverSize = bestCandidateHyperbox.getCoveredSamples().size();
        const double candidateFidelity = candidateHyperbox.getFidelity();
        const double bestCandidateFidelity = bestCandidateHyperbox.getFidelity();
        const double fidelityDeltaVsBestCandidate = candidateFidelity - bestCandidateFidelity;

        // Candidate ranking within this outer iteration
        // 1) The covering size has to decrease (otherwise the antecedant is useless)
        // 2) Among valid candidates, prefer higher fidelity; if fidelity is (approximately) equal, prefer larger covering
        const bool reducesCurrentRuleCover = candidateCoverSize < ruleCoverSize;                                                                                                                     // Candidate must reduce the covering size compared to current rule (otherwise the new antecedent is useless and we would never progress)
        const bool improvesBestCandidateFidelity = fidelityDeltaVsBestCandidate > fidelityEpsilon;                                                                                                   // Primary ranking criterion: higher fidelity is always better
        const bool sameFidelityWithBetterCover = fidelityDeltaVsBestCandidate >= -fidelityEpsilon && fidelityDeltaVsBestCandidate <= fidelityEpsilon && candidateCoverSize > bestCandidateCoverSize; // Tie-breaker: with equal fidelity (within epsilon), prefer the candidate covering more samples
        const bool isBetterCandidate = improvesBestCandidateFidelity || sameFidelityWithBetterCover;                                                                                                 // Combined ranking criterion used for this iteration

        if (reducesCurrentRuleCover && isBetterCandidate) {
          bestCandidateHyperbox.setFidelity(candidateHyperbox.getFidelity()); // Update best hyperbox
          bestCandidateHyperbox.setCoveredSamples(candidateCoveredSamples);
          indexBestHyp = static_cast<int>(k);
          bestDimension = dimension;

          if (bestCandidateHyperbox.getFidelity() >= minFidelity) {
            break;
          }
        }
      }
    }

    // Apply the best candidate found during this outer iteration
    if (indexBestHyp != -1 && bestDimension != -1) { // If we found any good dimension with good hyperplane (with enough covering)
      const auto &bestCandidateCoveredSamples = bestCandidateHyperbox.getCoveredSamples();
      const size_t bestCandidateCoverSize = bestCandidateCoveredSamples.size();
      // Candidate acceptance (after selection above)
      // 1) accept if it improves the current rule fidelity
      // 2) otherwise, accept only if same fidelity is explicitly allowed (XOR workaround / progression path)
      const double currentRuleFidelity = hyperbox->getFidelity();
      const double bestCandidateFidelity = bestCandidateHyperbox.getFidelity();
      const double fidelityDeltaVsCurrentRule = bestCandidateFidelity - currentRuleFidelity;
      const bool improvesCurrentRuleFidelity = fidelityDeltaVsCurrentRule > fidelityEpsilon;                                                                   // Standard case: adding the antecedent strictly increases fidelity
      const bool sameFidelityAndAllowed = allowNoFidChange && fidelityDeltaVsCurrentRule >= -fidelityEpsilon && fidelityDeltaVsCurrentRule <= fidelityEpsilon; // Candidate keeps the same fidelity as the current rule (within epsilon) and this is allowed by the parameter

      if (improvesCurrentRuleFidelity || sameFidelityAndAllowed) {

        hyperbox->setFidelity(bestCandidateFidelity);
        hyperbox->addIncreasedFidelity(bestCandidateFidelity);
        hyperbox->setCoveredSamples(bestCandidateCoveredSamples);
        hyperbox->addCoveringSizesWithNewAntecedent(bestCandidateCoverSize);
        hyperbox->addDiscriminativeHyperplan(bestDimension, indexBestHyp);

        double ruleAccuracy = hyperbox->computeRuleAccuracy(mainSamplePred, trainTrueClass); // Percentage of covered samples whose true class matches the rule prediction
        hyperbox->addAccuracyChanges(ruleAccuracy);
      }
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

  const bool ruleCreated = compute(rule, mainSampleValues, mainSamplePred, minFidelity, minCoverSize);
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
