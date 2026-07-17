from __future__ import annotations

import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

from feature_extraction_utils import (  # noqa: E402
    extract_resumable,
    finalize_features,
    prepare_outputs,
    save_array_safely,
)


class FeatureExtractionUtilsTests(unittest.TestCase):
    def test_resumes_partial_array_and_preserves_completed_rows(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            output = root / "features.npy"
            mask_output = root / "features_valid_mask.npy"
            temp = prepare_outputs(
                output, mask_output, overwrite=False, resume=True
            )
            valid = np.array([True, False, True, True, False])
            partial = np.lib.format.open_memmap(
                temp, mode="w+", dtype=np.float32, shape=(5, 3)
            )
            partial[:] = 0
            partial[0] = [1, 2, 3]
            partial.flush()
            del partial

            seen: list[int] = []

            def infer(indices: np.ndarray) -> np.ndarray:
                seen.extend(indices.tolist())
                return np.repeat(indices[:, None], 3, axis=1) + 1

            shape, dtype, extracted = extract_resumable(
                valid_mask=valid,
                temp_file=temp,
                infer_batch=infer,
                batch_size=2,
                log_every=1,
            )
            finalize_features(temp, output)
            save_array_safely(mask_output, valid)

            result = np.load(output)
            self.assertEqual(shape, (5, 3))
            self.assertEqual(dtype, np.dtype("float32"))
            self.assertEqual(extracted, 2)
            self.assertEqual(seen, [2, 3])
            np.testing.assert_array_equal(result[0], [1, 2, 3])
            np.testing.assert_array_equal(result[1], [0, 0, 0])
            np.testing.assert_array_equal(np.load(mask_output), valid)


if __name__ == "__main__":
    unittest.main()
