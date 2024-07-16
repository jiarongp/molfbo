# https://github.com/georgedeath/mbore/blob/main/mbore/rankers.py

import numpy as np
import pygmo as pg
from typing import Tuple, Union


class BaseScalarizer:
    """
    Functionality we want:
        instantiation(F):
            - F: n by n_obj numpy ndarray of objective values

        get_ranks():
            - return the ranking of the objective values in indices form
              ideally this should cache the result unless the result is
              stochastic

    """
    def __init__(self, reference_point: np.ndarray):
        self.ref_point = np.ravel(reference_point)

    def _scalarize(self, y: np.ndarray) -> np.ndarray:
        """returns the scalarised version of y"""
        raise NotImplementedError

    def scalarize(self, y):
        return self._scalarize(y)

    def get_ranks(
        self, y: np.ndarray, return_scalers: bool = False
    ) -> Union[np.ndarray, Tuple[np.ndarray, np.ndarray]]:

        scalars = self._scalarize(y)

        # rank the values, lower rank = larger value
        ranks = np.argsort(scalars)[::-1]

        if return_scalers:
            return ranks, scalars

        return ranks


class HypI(BaseScalarizer):

    def _scalarize(self, y):
        # get the list of shell indices
        # return the non dominated fronts
        ndf, _, _, _ = pg.fast_non_dominated_sorting(y)

        hypi = np.zeros((y.shape[0],))
        zero_arr = np.array([0], dtype=ndf[0].dtype)

        for shell, next_shell in zip(ndf[:-1], ndf[1:]):
            # preallocate the indices of the locations
            combined_front_inds = np.concatenate([next_shell, zero_arr])

            for location_idx in shell:
                # add each location from shell n to shell n+1
                combined_front_inds[-1] = location_idx

                # calculate the hypervolume of the combined front
                hypi[location_idx] = pg.core.hypervolume(
                    y[combined_front_inds]
                ).compute(self.ref_point)

        # last shell: hypervolume of each individual shell memeber
        last_shell = ndf[-1]

        # calculate the volume of the hyper-rectangle, with edges spanning
        # from each element of the shell to the reference points, by simply
        # taking the product of its edge lengths.
        hypi[last_shell] = np.prod(self.ref_point[np.newaxis, :] - y[last_shell], axis=1)

        return hypi


class DomRank(BaseScalarizer):

    def _scalarize(self, y):
        # get non-donmination ranks of y
        _, _, dom_count, _ = pg.fast_non_dominated_sorting(y)
        domrank = 1 - (dom_count / (y.shape[0] - 1))
        return domrank


class HypervoumeContribution(BaseScalarizer):

    def _scalarize(self, y):
        ndf, _, _, _ = pg.fast_non_dominated_sorting(y)
        hvc = np.zeros((y.shape[0],))

        for shell_idx in range(len(ndf) - 1, -1, -1):
            shell = ndf[shell_idx]

            hv_class = pg.core.hypervolume(y[shell])

            # first calculate the exclusive hypervolume for each shell
            # (in reverse order) and add it to the current hypervolume value
            hvc[shell] += hv_class.contributions(self.ref_point)

            # then, if not dealing with the last shell, add the maximum
            # of this shell's contributions (plus previous added contributions)
            if shell_idx > 0:
                hvc[ndf[shell_idx - 1]] += np.max(hvc[shell])
        
        return hvc
