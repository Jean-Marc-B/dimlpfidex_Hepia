class Rule:

    def __init__(self, antecedents, target_class, covering_size=None, fidelity=None, accuracy=None, confidence=None, covered_samples=None, include_X=True):

        self.antecedents = antecedents
        self.target_class = target_class
        self.covering_size = covering_size
        self.covered_samples = covered_samples
        self.fidelity = fidelity
        self.accuracy = accuracy
        self.confidence = confidence
        self.include_X = include_X
        if not self.include_X:
            for antecedent in self.antecedents:
                antecedent.include_X = False

    @property
    def include_X(self):
        return self._include_X

    @include_X.setter
    def include_X(self, value):
        self._include_X = value
        for antecedent in self.antecedents:
            antecedent.include_X = self._include_X

    def __str__(self):
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
        rule_str = f"{antecedents_str} -> class {self.target_class}"
        if metrics:
            rule_str += "\n\t" + '\n\t'.join(metrics) + "\n"
        return rule_str

    def __repr__(self):
        return self.__str__()
