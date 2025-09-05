#ifndef HYPERBOX_H
#define HYPERBOX_H

#include <cstring>
#include <fstream>
#include <iostream>
#include <numeric>
#include <sstream>
#include <string>
#include <vector>

/**
 * @brief Represents a hyperbox composed of discriminative hyperplanes in the feature space used to construct an explaining rule for a sample of interest, discriminating between different classes of samples.
 *
 * This class encapsulates the attributes and methods needed to handle hyperboxes
 * which include discriminative hyperplanes and the samples they cover.
 */
class Hyperbox {
  std::vector<std::pair<int, int>> discriminativeHyperplans; ///< Vector of discriminative hyperplanes represented by a dimension (attribute index) and a index leading to the hyperplane value in this dimension.
  std::vector<int> coveredSamples;                           ///< Vector of sample IDs covered by the hyperbox.
  std::vector<int> coveringSizesWithNewAntecedent;           ///< Vector of the number of samples covered by the rule for each new antecedent in the rule (with 1, 2, 3, ... antecedents).
  std::vector<double> increasedFidelity;                     ///< Vector of the increased fidelity for each new antecedent.
  std::vector<double> accuracyChanges;                       ///< Vector of the accuracy changes for each new antecedent.
  double fidelity = -1;                                      ///< Fidelity of the samples covered by the hyperbox with respect to the model's prediction.

public:
  /**
   * @brief Default constructor for Hyperbox.
   */
  Hyperbox();

  /**
   * @brief Constructs a Hyperbox object with specified discriminative hyperplanes.
   */
  explicit Hyperbox(const std::vector<std::pair<int, int>> &m_discriminativeHyperplans);

  /**
   * @brief Sets the covered samples of the hyperbox.
   */
  void setCoveredSamples(const std::vector<int> &m_coveredSamples);

  /**
   * @brief Gets the discriminative hyperplanes of the hyperbox.
   */
  std::vector<std::pair<int, int>> getDiscriminativeHyperplans() const;

  /**
   * @brief Sets the discriminative hyperplanes of the hyperbox.
   */
  void setDiscriminativeHyperplans(std::vector<std::pair<int, int>> newDiscriminativehyperplan);

  /**
   * @brief remove a discriminative hyperplan based on its array position.
   */
  void removeDiscriminativehyperplan(int index);

  /**
   * @brief Resets the discriminative hyperplanes of the hyperbox.
   */
  void resetDiscriminativeHyperplans();

  /**
   * @brief Computes the new covered samples with a new discriminative hyperplane.
   */
  void computeCoveredSamples(const std::vector<int> &ancienCoveredSamples, int attribut, std::vector<std::vector<double>> &trainData, bool mainSampleGreater, double hypValue);

  /**
   * @brief Computes the fidelity of the samples covered by the hyperbox with respect to the model's prediction.
   */
  void computeFidelity(const int mainsamplePred, std::vector<int> &trainPreds);

  /**
   * @brief Gets the fidelity of the samples covered by the hyperbox.
   */
  double getFidelity() const;

  /**
   * @brief Sets the fidelity of the samples covered by the hyperbox.
   */
  void setFidelity(double x);

  /**
   * @brief Gets the increased fidelity for each new antecedent.
   */
  std::vector<double> getIncreasedFidelity() const;

  /**
   * @brief Adds in the list the increased fidelity with the new antecedent.
   */
  void addIncreasedFidelity(double incrFidelity);

  /**
   * @brief remove a increased fidelity stat based on its array position.
   */
  void removeIncreasedFidelity(int index);

  /**
   * @brief Sets the increased fidelities list.
   */
  void setIncreasedFidelity(std::vector<double> newIncreasedFidelities);

  /**
   * @brief Reset the increased fidelity for each new antecedent.
   */
  void resetIncreasedFidelity();

  /**
   * @brief Gets the accuracy changes for each new antecedent.
   */
  std::vector<double> getAccuracyChanges() const;

  /**
   * @brief Adds in the list the accuracy changes with the new antecedent.
   */
  void addAccuracyChanges(double incrFidelity);

  /**
   * @brief remove an accuracy change stat based on its array position.
   */
  void removeAccuracyChange(int index);
  
  /**
   * @brief Sets the accuracy changes list.
   */
  void setAccuracyChanges(std::vector<double> newAccuracyChanges);

  /**
   * @brief Reset the accuracy changes for each new antecedent.
   */
  void resetAccuracyChanges();

  /**
   * @brief Gets the covered samples of the hyperbox.
   */
  std::vector<int> getCoveredSamples() const;

  /**
   * @brief Gets the covering sizes for each new antecedant(discriminative hypaerplan) in the hyperbox.
   */
  std::vector<int> getCoveringSizesWithNewAntecedent() const;

  /**
   * @brief Adds the new covering size of the rule with the new antecedant in the list.
   */
  void addCoveringSizesWithNewAntecedent(int covSize);

  /**
   * @brief remove a covering sizes with new antecedent stat based on its array position.
   */
  void removeCoveringSizesWithNewAntecedent(int index);

  /**
   * @brief sets the new covering sizes.
   */
  void setCoveringSizesWithNewAntecedent(std::vector<int> coveringSizesWithNewAntecedents);

  /**
   * @brief Reset the covering sizes for each new antecedant(discriminative hypaerplan) in the hyperbox.
   */
  void resetCoveringSizesWithNewAntecedent();

  /**
   * @brief Adds a new discriminative hyperplane to the hyperbox.
   */
  void addDiscriminativeHyperplan(int dimVal, int hypValIndex);

  Hyperbox deepCopy();
};

inline std::ostream &operator<<(std::ostream &stream, const Hyperbox &hyperbox) {
  stream << "Discriminative hyperplans:        ";
  for (auto discHyp : hyperbox.getDiscriminativeHyperplans()) {
    stream << "[" << discHyp.first << "," << discHyp.second << "], ";
  }
  stream << std::endl;

  stream << "Increased fidelity by antededant: ";
  for (auto increasedFidelity : hyperbox.getIncreasedFidelity()) {
    stream << increasedFidelity << ",";
  }
  stream << std::endl;

  stream << "Covering size w/ new antecedant:  ";
  for (auto coveringSize : hyperbox.getCoveringSizesWithNewAntecedent()) {
    stream << coveringSize << ",";
  }
  stream << std::endl;

  stream << "Covered samples:                  ";
  for (int i = 0; i < 5; i++) {
    stream << hyperbox.getCoveredSamples()[i] << ",";
  }
  stream << "... (size=" << hyperbox.getCoveredSamples().size() << ")" << std::endl;

  stream << "Accuracy Changes:                 ";
  for (auto accuracyChange : hyperbox.getAccuracyChanges()) {
    stream << accuracyChange << ",";
  }
  stream << std::endl;
  stream << "Fidelity:                         " << hyperbox.getFidelity() << std::endl;

  return stream;
}

#endif // HYPERBOX_H
