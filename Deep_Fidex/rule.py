class Rule:

    def __init__(self, antecedents, target_class):
        self.antecedents = antecedents
        self.target_class = target_class

    def __str__(self):
        antecedents_str = ' '.join(str(antecedent) for antecedent in self.antecedents)
        return f"{antecedents_str} -> class {self.target_class}"

    def __repr__(self):
        return self.__str__()
