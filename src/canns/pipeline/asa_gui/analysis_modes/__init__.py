"""Analysis modes for ASA GUI."""

from .base import AbstractAnalysisMode
from .batch_mode import BatchMode
from .cohomap_mode import CohoMapMode
from .cohospace_mode import CohoSpaceMode
from .decode_mode import DecodeMode
from .fr_mode import FRMode
from .frm_mode import FRMMode
from .gridscore_mode import GridScoreInspectMode, GridScoreMode
from .pathcompare_mode import PathCompareMode
from .tda_mode import TDAMode

__all__ = [
    "AbstractAnalysisMode",
    "BatchMode",
    "CohoMapMode",
    "CohoSpaceMode",
    "DecodeMode",
    "FRMode",
    "FRMMode",
    "GridScoreMode",
    "GridScoreInspectMode",
    "PathCompareMode",
    "TDAMode",
    "get_analysis_modes",
]


def get_analysis_modes() -> list[AbstractAnalysisMode]:
    return [
        TDAMode(),
        DecodeMode(),
        CohoMapMode(),
        PathCompareMode(),
        CohoSpaceMode(),
        FRMode(),
        FRMMode(),
        GridScoreMode(),
        GridScoreInspectMode(),
        BatchMode(),
    ]
