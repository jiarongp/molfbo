import torch
from platypus import NSGAII, Problem, Real, Solution
from platypus.core import _convert_constraint
from platypus.config import default_variator


# platypus problem
def get_platypus_problem(problem):
    class MyProblem(Problem):
        def __init__(self):
            super().__init__(
                nvars=problem.dim,
                nobjs=problem.num_objectives,
                nconstrs=len(problem.ref_point),
                function=problem,
            )
            self.types[:] = [Real(*bound) for bound in problem.bounds.numpy().T]
            self.directions[:] = Problem.MAXIMIZE
            self.constraints = _convert_constraint([f">={pt}" for pt in problem.ref_point])

        @property
        def bounds(self):
            return self.function.bounds
        
        def evaluate(self, solution):
            x = solution.variables
            solution.objectives = self.function(torch.tensor(x)).tolist()
            solution.constraints = solution.objectives

    return MyProblem()


class NSGAII(NSGAII):

    def __init__(
        self,
        problem,
        population_size=100,
        archive=None,
        **kwargs
    ):
        super().__init__(
            problem=problem,
            population_size=population_size,
            archive=archive,
            **kwargs
        )

    def initialize(self, X_init=None, y_init=None):
        if X_init is None:
            self.population = [self.generator.generate(self.problem) for _ in range(self.population_size)]
            self.evaluate_all(self.population)
        else:
            self.population = []
            for x, y in zip(X_init, y_init):
                solution = Solution(self.problem)
                solution.variables = x.tolist()
                solution.objectives = y.tolist()
                self.population.append(solution)

        if self.archive is not None:
            self.archive += self.population

        if self.variator is None:
            self.variator = default_variator(self.problem)

        self.nfe = len(X_init)


class NSGA2:

    def __init__(self, problem, batch_size=5, device="cpu:0", dtype=torch.double):
        # get problem for platypus 
        self.problem = get_platypus_problem(problem)
        self.tkwargs = {"dtype": dtype, "device": device}
        self.standard_bounds = torch.zeros(2, problem.dim, **self.tkwargs)
        self.standard_bounds[1] = 1

        self.nsga2 = NSGAII(
            self.problem,
            population_size=batch_size,
            archive=[]
        )

    def observe_and_suggest(self, X_obs, y_obs, X_pen=None, batch_size=1, *args, **kwargs):
        # initialize nsga2
        if self.nsga2.nfe == 0:
            self.nsga2.initialize(X_obs, y_obs)

        self.nsga2.step()

        new_x = [pop.variables for pop in self.nsga2.population]
        return torch.tensor(new_x, **self.tkwargs)
