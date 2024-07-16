from botorch.utils.sampling import draw_sobol_samples


class Sobol:

    def __init__(self, problem, *args, **kwargs):
        self.problem = problem

    def observe_and_suggest(self, X_obs, y_obs, X_pen=None, batch_size=1, *args, **kwargs):
        """
        Samples a set of random weights for each candidate in the batch, performs sequential greedy optimization of the qNParEGO
        acquisition function, and returns a new candidate and obervation.
        """
        return draw_sobol_samples(bounds=self.problem.bounds, n=1, q=batch_size).squeeze(1)
