class Antecedent:

    def __init__(self, attribute, inequality, value):
        self.attribute = attribute
        self.inequality = inequality    # True if the condition is '>=', False if '<'
        self.value = value

    def __str__(self):
        ineq="<"
        if self.inequality:
            ineq=">="
        return f"X{self.attribute}{ineq}{self.value}"

    def __repr__(self):
        return self.__str__()
