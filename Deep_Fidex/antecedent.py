class Antecedent:

    def __init__(self, attribute, inequality, value, include_X=True):
        self.attribute = attribute
        self.inequality = inequality    # True if the condition is '>=', False if '<'
        self.value = value
        self.include_X = include_X

    def __str__(self):
        ineq="<"
        if self.inequality:
            ineq=">="
        if self.include_X:
            return f"X{self.attribute}{ineq}{self.value}"
        else:
            return f"{self.attribute}{ineq}{self.value}"

    def __repr__(self):
        return self.__str__()
