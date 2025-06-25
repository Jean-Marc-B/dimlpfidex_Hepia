class Rule:
    """
    Represents a classification rule consisting of multiple antecedents and a target class.

    Attributes:
    - antecedents: A list of Antecedent objects representing conditions that make up the rule.
    - target_class: The target class associated with the rule when the antecedents are satisfied.
    - covering_size: (Optional) The number of samples covered by the rule.
    - fidelity: (Optional) The fidelity of the rule when evaluated on a train dataset.
    - accuracy: (Optional) The accuracy of the rule.
    - confidence: (Optional) The confidence of the rule.
    - covered_samples: (Optional) The list of samples covered by the rule.
    - include_X: Boolean indicating whether to include the prefix 'X' when displaying antecedents.

    Methods:
    - __str__: Returns a string representation of the rule.
    - __repr__: Returns a string representation (same as __str__).
    """

    def __init__(self, antecedents, target_class, covering_size=None, coveringSizesWithNewAntecedent=None, fidelity=None, increasedFidelity=None, accuracy=None, accuracyChanges=None, confidence=None, covered_samples=None, include_X=True):

        self.antecedents = antecedents
        self.target_class = target_class
        self.covering_size = covering_size
        self.coveringSizesWithNewAntecedent = coveringSizesWithNewAntecedent
        self.covered_samples = covered_samples
        self.fidelity = fidelity
        self.increasedFidelity = increasedFidelity
        self.accuracy = accuracy
        self.accuracyChanges = accuracyChanges
        self.confidence = confidence
        self.include_X = include_X
        if not self.include_X:
            for antecedent in self.antecedents:
                antecedent.include_X = False

    @property
    def include_X(self):
        """Gets or sets the include_X property, updating all antecedents when modified."""
        return self._include_X

    @include_X.setter
    def include_X(self, value):
        self._include_X = value
        for antecedent in self.antecedents:
            antecedent.include_X = self._include_X

    def __str__(self):
        """
        Returns a string representation of the rule, including:
        - Antecedents and their conditions.
        - Target class.
        - Optional metrics such as covering size, fidelity, accuracy, and confidence if available.
        """
        antecedents_str = ' '.join(str(antecedent) for antecedent in self.antecedents)
        metrics = []
        if self.covering_size is not None:
            metrics.append(f"Train Covering_size : {self.covering_size}")
        if self.fidelity is not None:
            metrics.append(f"Train Fidelity : {self.fidelity:.2f}")
        if self.accuracy is not None:
            metrics.append(f"Train Accuracy : {self.accuracy:.2f}")
        if self.confidence is not None:
            metrics.append(f"Train Confidence : {self.confidence:.2f}")
        if self.coveringSizesWithNewAntecedent is not None:
            formatted_covering = ' '.join(str(v) for v in self.coveringSizesWithNewAntecedent)
            metrics.append(f"Train Covering size evolution with antecedents : {formatted_covering}")
        if self.increasedFidelity is not None:
            formatted_fidelity = ' '.join(f"{v:.2f}" for v in self.increasedFidelity)
            metrics.append(f"Train Fidelity increase with antecedents : {formatted_fidelity}")
        if self.accuracyChanges is not None:
            formatted_accuracy = ' '.join(f"{v:.2f}" for v in self.accuracyChanges)
            metrics.append(f"Train Accuracy variation with antecedents : {formatted_accuracy}")

        rule_str = f"{antecedents_str} -> class {self.target_class}"
        if metrics:
            rule_str += "\n\t" + '\n\t'.join(metrics) + "\n"
        return rule_str

    def __repr__(self):
        """Returns the string representation of the rule (same as __str__)."""
        return self.__str__()
