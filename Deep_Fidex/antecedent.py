class Antecedent:
    """
    Represents an antecedent (condition) in a rule, with an attribute, inequality, and value.

    Attributes:
    - attribute: The attribute (or feature) used in the antecedent.
    - inequality: Boolean indicating the type of inequality ('>=' if True, '<' if False).
    - value: The threshold value for the inequality.
    - include_X: Boolean indicating whether to include the prefix 'X' when displaying the antecedent.

    Methods:
    - __str__: Returns a string representation of the antecedent.
    - __repr__: Returns a string representation (same as __str__).
    """

    def __init__(self, attribute, inequality, value, include_X=True):
        self.attribute = attribute
        self.inequality = inequality    # True if the condition is '>=', False if '<'
        self.value = value
        self.include_X = include_X

    def __str__(self):
        """
        Returns a string representation of the antecedent.
        - The prefix 'X' is included if include_X is True.
        - The inequality symbol ('>=' or '<') is shown based on the inequality attribute.
        """
        ineq="<"
        if self.inequality:
            ineq=">="
        if self.include_X:
            return f"X{self.attribute}{ineq}{self.value}"
        else:
            return f"{self.attribute}{ineq}{self.value}"

    def __repr__(self):
        """Returns the string representation of the antecedent (same as __str__)."""
        return self.__str__()
