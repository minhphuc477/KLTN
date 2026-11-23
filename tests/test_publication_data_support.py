"""Raw source targets must support the declared full-game endpoint."""

import numpy as np
import pytest

from src.core.definitions import SEMANTIC_PALETTE
from src.publication.data_support import require_raw_validity_target_support


@pytest.mark.parametrize("ids", [[], [1, 2, 10, 42], [1, 21], [1, 22]])
def test_missing_endpoint_classes_block_raw_validity_claim(ids):
    with pytest.raises(ValueError, match="unsupported by the training targets"):
        require_raw_validity_target_support([np.asarray(ids, dtype=np.int32).reshape(1, -1)])


def test_endpoint_support_is_checked_over_all_training_targets():
    require_raw_validity_target_support([
        np.asarray([[SEMANTIC_PALETTE["START"]]]),
        np.asarray([[SEMANTIC_PALETTE["TRIFORCE"]]]),
    ])
