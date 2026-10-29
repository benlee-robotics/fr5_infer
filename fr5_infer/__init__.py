"""Deploy a LeRobot policy checkpoint on the Fairino FR5.

The robot client, safety limits and observation conversion come from the
sibling ``fr5_teleop`` checkout (package ``teleop``), so the deployed policy
sees the same vectors and obeys the same limits as the recorded teleop.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path


__version__ = "0.1.2"

TELEOP_ROOT = Path(
    os.environ.get(
        "FR5_TELEOP_ROOT", Path(__file__).resolve().parents[2] / "fr5_teleop"
    )
)

try:
    import teleop  # noqa: F401
except ImportError:
    # Also runs in spawned robot/camera processes, which re-import this package.
    sys.path.insert(0, str(TELEOP_ROOT))
    import teleop  # noqa: F401
