"""What the constraint-handling techniques do, for the pages and tutorials that explain them."""

# What each value of the parameter that handles the constraints does, as Evolver's tutorial
# "Problems with constraints" describes them.
CONSTRAINT_HANDLING = {
    "cdp": "Deb's constrained-domination principle. A feasible solution is better than an "
    "infeasible one, two infeasible ones are compared by their violation, and two feasible ones "
    "by dominance.",
    "violatedConstraints": "As `cdp`, but two infeasible solutions are compared by how many "
    "constraints they violate.",
    "epsilonConstrained": "As `cdp`, but a violation below a tolerance counts as feasible; the "
    "tolerance decreases to 0 during the run.",
    "stochasticRanking": "When one of two solutions is infeasible, they are compared by their "
    "objectives with a given probability, and by their violation otherwise.",
    "pushPull": "First the constraints are ignored (push) until the population settles, and then "
    "it is pulled to the feasible region.",
    "selfAdaptivePenalty": "The objectives are penalized with the violation, with a penalty that "
    "adapts to the fraction of feasible solutions.",
    "feasibilityRules": "Deb's feasibility rules decide whether a new solution replaces the one "
    "of a subproblem.",
    "improvedEpsilon": "The improved epsilon method of Fan et al. (2019): a tolerance on the "
    "violation that adapts to the fraction of feasible solutions.",
    "violationThreshold": "A violation below an adaptive threshold does not count.",
}
