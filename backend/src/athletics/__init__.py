"""Athletics reference profile and technique analysis package (Chunk 8D).

Provides event-specific technique modeling and injury screening for:
- Long Jump
- Shot Put
- Javelin Throw
"""

from src.athletics.analyzer import AthleticsAnalyzer, AttemptResult
from src.athletics.long_jump import LongJumpAnalyzer
from src.athletics.shot_put import ShotPutAnalyzer
from src.athletics.javelin import JavelinAnalyzer

__all__ = [
    "AthleticsAnalyzer",
    "AttemptResult",
    "LongJumpAnalyzer",
    "ShotPutAnalyzer",
    "JavelinAnalyzer",
]
