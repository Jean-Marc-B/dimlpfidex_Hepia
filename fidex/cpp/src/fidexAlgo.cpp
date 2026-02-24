#include "fidexAlgo.h"

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
 * @param minNbCover An integer representing the minimum number of samples a rule must cover.
 * @return True if a rule meeting the criteria is found.
 * @return False if no rule meeting the criteria is found.
 */
bool Fidex::compute(Rule &rule, const std::vector<double> &mainSampleValues, int mainSamplePred, double minFidelity, int minNbCover) {

  specs.nbIt = 0;

  bool showInitialFidelity = getShowInitialFidelity();
  double mainSamplePredValue = getMainSamplePredValue(); // Output value of the main sample on the predicted class

  Hyperspace *hyperspace = _hyperspace;
  int nbAttributes = _trainDataset->getNbAttributes();
  std::vector<int> &trainPreds = _trainDataset->getPredictions();
  std::vector<int> &trainTrueClass = _trainDataset->getClasses();
  std::vector<std::vector<double>> &trainData = _trainDataset->getDatas();
  std::vector<std::vector<double>> &trainPredictionScores = _trainDataset->getPredictionScores();
  const auto &hyperLocus = hyperspace->getHyperLocus();
  auto hyperbox = hyperspace->getHyperbox();
  auto nbDimensions = static_cast<int>(hyperLocus.size()); // number of features
  int maxIterations = _parameters->getInt(MAX_ITERATIONS);
  double dropoutDim = _parameters->getFloat(DROPOUT_DIM);
  double dropoutHyp = _parameters->getFloat(DROPOUT_HYP);
  bool allowNoFidChange = _parameters->getBool(ALLOW_NO_FID_CHANGE);
  bool hasdd = dropoutDim > 0.001;
  bool hasdh = dropoutHyp > 0.001;

  // For denormalization :
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

  if (mainSamplePredValue == -1.0 && _usingTestSamples) {
    throw InternalError("Error during computation of Fidex: Execution with a test sample but no sample prediction value has been given.");
  }

  std::uniform_real_distribution<double> dis(0.0, 1.0);

  // Compute initial covering
  std::vector<int> coveredSamples(trainData.size());   // Samples covered by the hyperbox
  iota(begin(coveredSamples), end(coveredSamples), 0); // The vector goes from 0 to len(coveredSamples)-1

  // Store covering and compute initial fidelity
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
  std::vector<int> dimensions(nbDimensions); // Vector of all features

  while (hyperbox->getFidelity() < minFidelity && nbIt < maxIterations) { // While fidelity of our hyperbox is not high enough, we try to add a new discriminative hyperplane (antecedent in the rule)
    Hyperbox bestCandidateHyperbox;                                       // best hyperbox to choose for next step
    Hyperbox candidateHyperbox;
    const auto &ruleCoveredSamples = hyperbox->getCoveredSamples();
    const size_t ruleCoverSize = ruleCoveredSamples.size();
    double mainSampleValue;
    int attribute;
    int dimension;
    int indexBestHyp = -1;
    int bestDimension = -1;
    // Randomize dimensions

    iota(begin(dimensions), end(dimensions), 0); // Vector from 0 to nbIn-1
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

      size_t nbHyp = hyperLocus[dimension].size();
      if (nbHyp == 0) {
        continue; // No data on this dimension
      }

      for (int k = 0; k < nbHyp; k++) { // for each possible hyperplane in this dimension (there is nbSteps+1 hyperplanes per dimension)
        // Test if we dropout this hyperplane
        if (hasdh && dis(_rnd) < dropoutHyp) {
          continue; // Drop this hyperplane if below parameter ex: param=0.2 -> 20% are dropped
        }

        double hypValue = hyperLocus[dimension][k];
        bool mainSampleGreater = hypValue <= mainSampleValue;                                                           // Check if main sample value is on the right of the hyperplane
        candidateHyperbox.computeCoveredSamples(ruleCoveredSamples, attribute, trainData, mainSampleGreater, hypValue); // Compute new cover samples
        candidateHyperbox.computeFidelity(mainSamplePred, trainPreds);                                                  // Compute fidelity
        const auto &candidateCoveredSamples = candidateHyperbox.getCoveredSamples();
        const size_t candidateCoverSize = candidateCoveredSamples.size();
        const size_t bestCandidateCoverSize = bestCandidateHyperbox.getCoveredSamples().size();

        // Candidate selection for this iteration:
        // 1) It must satisfy the minimum covering constraint
        // 2) The covering size has to decrease (otherwise the antecedant is useless)
        // 2) Among valid candidates, prefer higher fidelity; if fidelity is equal, prefer larger covering
        const bool hasEnoughCover = candidateCoverSize >= static_cast<size_t>(minNbCover); // Candidate must satisfy the minimum covering constraint
        // const bool respectsNoFidChangeCoverConstraint = !allowNoFidChange || candidateCoverSize < ruleCoverSize; // When allowNoFidChange is enabled, same-fidelity path is only valid with strictly lower covering than current rule
        const bool reducesCurrentRuleCover = candidateCoverSize < ruleCoverSize;                                                                                        // Candidate must reduce the covering size compared to current rule (otherwise the new antecedent is useless and we would never progress)
        const bool improvesBestCandidateFidelity = candidateHyperbox.getFidelity() > bestCandidateHyperbox.getFidelity();                                               // Primary ranking criterion: higher fidelity is always better
        const bool sameFidelityWithBetterCover = candidateHyperbox.getFidelity() == bestCandidateHyperbox.getFidelity() && candidateCoverSize > bestCandidateCoverSize; // Tie-breaker: with equal fidelity, prefer the candidate covering more samples
        const bool isBetterCandidate = improvesBestCandidateFidelity || sameFidelityWithBetterCover;                                                                    // Combined ranking criterion used for this iteration

        if (hasEnoughCover && reducesCurrentRuleCover && isBetterCandidate) {
          bestCandidateHyperbox.setFidelity(candidateHyperbox.getFidelity()); // Update best hyperbox
          bestCandidateHyperbox.setCoveredSamples(candidateCoveredSamples);
          indexBestHyp = k;
          bestDimension = dimension;
        }
        if (bestCandidateHyperbox.getFidelity() >= minFidelity) {
          break;
        }
      }
    }

    // Modification of our hyperbox with the best at this iteration and modify discriminative hyperplanes
    if (indexBestHyp != -1 && bestDimension != -1) { // If we found any good dimension with good hyperplane (with enough covering)
      const auto &bestCandidateCoveredSamples = bestCandidateHyperbox.getCoveredSamples();
      const size_t bestCandidateCoverSize = bestCandidateCoveredSamples.size();
      // Candidate acceptance (after selection above):
      // 1) accept if it improves the current rule fidelity
      // 2) otherwise, accept only if same fidelity is explicitly allowed (XOR workaround / progression path)
      const bool improvesCurrentRuleFidelity = bestCandidateHyperbox.getFidelity() > hyperbox->getFidelity();                 // Standard case: adding the antecedent strictly increases fidelity
      const bool sameFidelityAndAllowed = allowNoFidChange && bestCandidateHyperbox.getFidelity() == hyperbox->getFidelity(); // Candidate keeps the same fidelity as the current rule and this is allowed by the parameter

      if (improvesCurrentRuleFidelity || sameFidelityAndAllowed) {

        hyperbox->setFidelity(bestCandidateHyperbox.getFidelity());
        hyperbox->addIncreasedFidelity(bestCandidateHyperbox.getFidelity());
        hyperbox->setCoveredSamples(bestCandidateCoveredSamples);
        hyperbox->addCoveringSizesWithNewAntecedent(bestCandidateCoverSize);
        hyperbox->addDiscriminativeHyperplan(bestDimension, indexBestHyp);

        double ruleAccuracy;
        ruleAccuracy = hyperbox->computeRuleAccuracy(mainSamplePred, trainTrueClass); // Percentage of covered samples whose true class matches the rule prediction
        hyperbox->addAccuracyChanges(ruleAccuracy);
      }
    }
    nbIt += 1;
  }

  // trying to remove unecessary antecedants
  while (optimizeRule(mainSampleValues, mainSamplePred))
    ;

  // Compute rule accuracy and confidence
  double ruleAccuracy = hyperbox->computeRuleAccuracy(mainSamplePred, trainTrueClass);
  double ruleConfidence = hyperspace->computeRuleConfidence(trainPredictionScores, mainSamplePred, mainSamplePredValue); // Mean output value of prediction of class chosen by the rule for the covered samples

  // Extract rules
  if (_parameters->isDoubleVectorSet(MUS)) {
    rule = hyperspace->ruleExtraction(mainSampleValues, mainSamplePred, ruleAccuracy, ruleConfidence, mus, sigmas, normalizationIndices);
  } else {
    rule = hyperspace->ruleExtraction(mainSampleValues, mainSamplePred, ruleAccuracy, ruleConfidence);
  }

  // Reset execution specs
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
 * @param minFidelity A float representing the minimum fidelity threshold for rule creation.
 * @param minNbCover An integer representing the minimum number of samples a rule must cover.
 * @param verbose A boolean flag for detailed verbose output.
 * @param detailedVerbose A boolean flag for detailed verbose output. Default is false.
 * @param foundRule A boolean indicating whether a rule was found in a previous attempt. Default is false.
 * @return True if a rule meeting the criteria is successfully computed.
 * @return False if no rule meeting the criteria can be computed.
 */
bool Fidex::tryComputeFidex(Rule &rule, const std::vector<double> &mainSampleValues, int mainSamplePred, float minFidelity, int minNbCover, bool verbose, bool detailedVerbose, bool foundRule) {
  if (detailedVerbose && verbose) {
    if (foundRule) {
      std::cout << "A rule has been found. ";
    } else {
      std::cout << "Fidelity is too low. ";
    }
    std::cout << "Restarting fidex with a minimum covering of " << minNbCover << " and a minimum accepted fidelity of " << minFidelity << "." << std::endl;
  }

  bool ruleCreated = compute(rule, mainSampleValues, mainSamplePred, minFidelity, minNbCover);
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
 * @param minFidelity A float representing the minimum fidelity threshold for rule creation.
 * @param left The starting point of the search range.
 * @param right The ending point of the search range.
 * @param verbose A boolean flag for detailed verbose output.
 * @return The best covering found that meets the minimum fidelity criteria. Returns -1 if no such covering is found.
 */
int Fidex::dichotomicSearch(Rule &bestRule, const std::vector<double> &mainSampleValues, int mainSamplePred, float minFidelity, int left, int right, bool verbose) {
  int bestCovering = -1;
  int currentMinNbCover = right + 1;
  bool foundRule = false;
  while (currentMinNbCover != ceil((right + left) / 2.0) && left <= right) {
    currentMinNbCover = static_cast<int>(ceil((right + left) / 2.0));
    Rule tempRule;
    if (tryComputeFidex(tempRule, mainSampleValues, mainSamplePred, minFidelity, currentMinNbCover, verbose, true, foundRule)) {
      bestCovering = currentMinNbCover;
      bestRule = tempRule;
      left = currentMinNbCover + 1;
      foundRule = true;
    } else {
      right = currentMinNbCover - 1;
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
 * @param minFidelity A reference to a float representing the current minimum fidelity threshold for rule creation. May be adjusted.
 * @param minNbCover An integer representing the minimum number of samples a rule must cover.
 * @param verbose A boolean flag for detailed verbose output.
 * @return True if a rule meeting the criteria is successfully computed within the maximum number of attempts.
 * @return False if no rule meeting the criteria can be computed within the maximum number of attempts.
 */
bool Fidex::retryComputeFidex(Rule &rule, const std::vector<double> &mainSampleValues, int mainSamplePred, float minFidelity, int minNbCover, bool verbose) {
  int counterFailed = 0; // Number of times we failed to find a rule with maximal fidexlity when minNbCover is 1
  int maxFailedAttempts = _parameters->getInt(MAX_FAILED_ATTEMPTS);
  bool allowNoFidChange = _parameters->getBool(ALLOW_NO_FID_CHANGE);
  bool covering_strategy = _parameters->getBool(COVERING_STRATEGY);
  bool ruleCreated = false;
  bool hasDropout = _parameters->getFloat(DROPOUT_DIM) > 0.001 || _parameters->getFloat(DROPOUT_HYP) > 0.001;
  do {
    ruleCreated = tryComputeFidex(rule, mainSampleValues, mainSamplePred, minFidelity, minNbCover, verbose, true);
    if (!ruleCreated) {
      counterFailed += 1;
    }
    if (counterFailed >= maxFailedAttempts && verbose) {
      std::cout << "\nWARNING Fidelity is too low after trying " << std::to_string(maxFailedAttempts) << " times with a minimum covering of " << minNbCover << " and a minimum accepted fidelity of " << minFidelity << "! You may want to try again with a lower min_covering or a lower min_fidelity." << std::endl;
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

  int minNbCover = _parameters->getInt(MIN_COVERING);
  float minFidelity = _parameters->getFloat(MIN_FIDELITY);
  bool ruleCreated = false;
  bool foundRule = true;

  if (verbose) {
    setShowInitialFidelity(true);
  }

  // Try to find a rule
  if (!tryComputeFidex(rule, mainSampleValues, mainSamplePred, minFidelity, minNbCover, verbose)) {
    // If no rule is found

    // With covering strategy
    if (_parameters->getBool(COVERING_STRATEGY)) {
      int right = minNbCover - 1;
      int bestCovering = -1;
      Rule bestRule;

      // Dichotomic search to find a rule with best covering
      if (right > 0) {
        bestCovering = dichotomicSearch(bestRule, mainSampleValues, mainSamplePred, minFidelity, 1, right, verbose);
      }

      // Coundn't find a rule with minimal fidelity 1, we search for a lower minimal fidelity
      if (bestCovering == -1) {
        float currentMinFidelity = minFidelity;
        while (!ruleCreated && currentMinFidelity > _parameters->getFloat(LOWEST_MIN_FIDELITY)) {
          currentMinFidelity -= 0.05f;
          ruleCreated = tryComputeFidex(rule, mainSampleValues, mainSamplePred, currentMinFidelity, 1, verbose, true);
        }
        // Coundn't find a rule, we retry maxFailedAttempts times
        if (!ruleCreated) {
          foundRule = retryComputeFidex(rule, mainSampleValues, mainSamplePred, currentMinFidelity, 1, verbose);
        }

      } else { // If we found a correct rule during dichotomic search
        rule = bestRule;
      }
      if (verbose) {
        std::cout << std::endl;
      }
      // Without covering strategy
    } else if (verbose) {
      std::cout << "\nWARNING Fidelity is too low! You may want to try again." << std::endl;
      std::cout << "If you can't find a rule with the wanted fidelity, try a lowest minimal covering or a lower fidelity" << std::endl;
      std::cout << "You can also try to use the min cover strategy (--covering_strategy)" << std::endl;
      std::cout << "If this is not enough, put the min covering to 1 and do not use dropout." << std::endl;
      std::cout << "You may also want to allow to add a new antecedant without changing the fidelity by setting allow_no_fid_change to true.\n"
                << std::endl;
      foundRule = false;
    }
  }
  if (!foundRule) {
    return false;
  }
  return true;
}

/**
 * @brief @brief Attepts to filter unnecessary attributes in a rule.
 *
 * @param mainSampleValues A vector of double values representing the main sample values.
 * @param mainSamplePred An integer representing the predicted class of the main sample.
 * @return wether an optimisation has been done or not.
 */
bool Fidex::optimizeRule(const std::vector<double> &mainSampleValues, int mainSamplePred) {
  std::shared_ptr<Hyperbox> originalHyperbox = _hyperspace->getHyperbox();
  std::vector<std::pair<int, int>> originalDiscrHyperplans = originalHyperbox->getDiscriminativeHyperplans();
  std::vector<std::vector<double>> &trainData = _trainDataset->getDatas();
  std::vector<int> &trainPreds = _trainDataset->getPredictions();
  std::vector<int> &trainTrueClass = _trainDataset->getClasses();
  const auto &hyperLocus = _hyperspace->getHyperLocus();
  std::vector<int> coveredSamples(trainData.size()); // Samples covered by the hyperbox
  Hyperbox bestHyperbox = originalHyperbox->deepCopy();
  iota(begin(coveredSamples), end(coveredSamples), 0); // Vector from 0 to len(coveredSamples)-1
  int nbAttributes = _trainDataset->getNbAttributes();
  int antededantIndex = -1;
  bool hasBeenOptimized = false;

  for (int i = 0; i < originalHyperbox->getDiscriminativeHyperplans().size(); i++) {
    std::vector<std::pair<int, int>> copyDiscrHyperplans(originalDiscrHyperplans); // create original's copy
    Hyperbox copyHyperbox = Hyperbox();
    copyDiscrHyperplans.erase(copyDiscrHyperplans.begin() + i); // hide an antecedant
    copyHyperbox.setDiscriminativeHyperplans(copyDiscrHyperplans);
    copyHyperbox.setCoveredSamples(coveredSamples);

    for (auto antecedant : copyDiscrHyperplans) {
      int dimension = antecedant.first;
      int hypIndex = antecedant.second;
      int feature = dimension % nbAttributes;

      double hypValue = hyperLocus[dimension][hypIndex];
      double mainSampleValue = mainSampleValues[feature];
      bool isMainSampleGreater = hypValue <= mainSampleValue;

      copyHyperbox.computeCoveredSamples(copyHyperbox.getCoveredSamples(), feature, trainData, isMainSampleGreater, hypValue);
      copyHyperbox.computeFidelity(mainSamplePred, trainPreds);
      copyHyperbox.addIncreasedFidelity(copyHyperbox.getFidelity());
      copyHyperbox.addCoveringSizesWithNewAntecedent(copyHyperbox.getCoveredSamples().size());
      copyHyperbox.addAccuracyChanges(copyHyperbox.computeRuleAccuracy(mainSamplePred, trainTrueClass));
    }

    copyHyperbox.computeFidelity(mainSamplePred, trainPreds);

    if (copyHyperbox.getFidelity() >= bestHyperbox.getFidelity()) {
      hasBeenOptimized = true;
      antededantIndex = i;
      bestHyperbox = copyHyperbox;
    }
  }

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
