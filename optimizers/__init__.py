from .sobol.sobol import Sobol
from .mbore.mbore import MBORE
from .mbore.mbore_mlp import MBORE_MLP
from .mbore.mbore_composite import MBORE_COMPOSITE
from .mbore.mbore_rff import MBORE_RFF
from .mbore.mbore_mdre_ei import MBORE_MDRE_EI
from .mbore.mbore_mdre_ei_aux import MBORE_MDRE_EI_AUX
from .angle_decomposition import AngleDecomp
from .parego import ParEGO
from .qparego.qparego import qParEGO
from .qehvi.qehvi import qEHVI
from .pfns import PFNs
from .qsucb import qSUCB
from .linear_scalarization import LinearScalarization
from .ts_tch.ts_tch import TS_TCH
from .set_rank import SetRank
from .nsga2.nsga2 import NSGA2
from .morbo.morbo import MORBO


OPTIMIZERS = {
    "Sobol": Sobol,
    "MBORE": MBORE,
    "MBORE_MLP": MBORE_MLP,
    "MBORE_RFF": MBORE_RFF,
    "MBORE_COMPOSITE": MBORE_COMPOSITE,
    "MBORE_MDRE_EI": MBORE_MDRE_EI,
    "MBORE_MDRE_EI_AUX": MBORE_MDRE_EI_AUX,
    "AngleDecomp":AngleDecomp,
    "TS_TCH":TS_TCH,
    "ParEGO": ParEGO,
    "qParEGO": qParEGO,
    "qSUCB": qSUCB,
    "qEHVI": qEHVI,
    "PFNs": PFNs,
    "LinearScalarization": LinearScalarization,
    "SetRank": SetRank,
    "NSGA2": NSGA2,
    "MORBO": MORBO,
}