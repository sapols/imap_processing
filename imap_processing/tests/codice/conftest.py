from unittest.mock import patch

import cdflib
import numpy as np
import pytest
from imap_data_access import ScienceFilePath

from imap_processing import imap_module_directory
from imap_processing.cdf.utils import load_cdf

TEST_DATA_PATH = imap_module_directory / "tests" / "codice" / "data"

VALIDATION_FILE_DATE = "20260204"
VALIDATION_FILE_VERSION = "v029"

IALIRT_VALIDATION_FILE_DATE = "20250814"
IALIRT_VALIDATION_FILE_VERSION = "v015"

# epoch_delta = num_spins * spin_period / 2, with spin_period VALIDMAX = 16 s
# and num_spins max = 16 in the current CoDICE timing model. That yields a
# worst-case delta of 128 s = 128000000000 ns.
EXPECTED_EPOCH_DELTA_VALIDMAX = 128000000000

# The test LUT returned for each ancillary descriptor the processing code requests
LUT_FILES = {
    "l1a-sci-lut": "l1a_lut/imap_codice_l1a-sci-lut_20260129_v002.json",
    "l2-hi-omni-efficiency": (
        "l2_lut/imap_codice_l2-hi-omni-efficiency_20251212_v004.csv"
    ),
    "l2-hi-sectored-efficiency": (
        "l2_lut/imap_codice_l2-hi-sectored-efficiency_20251212_v004.csv"
    ),
    "l2-lo-efficiency": "l2_lut/imap_codice_l2-lo-efficiency_20251008_v003.csv",
    "l2-lo-gfactor": "l2_lut/imap_codice_l2-lo-gfactor_20251212_v003.csv",
    "l2-lo-onboard-mpq-cal": (
        "l2_lut/imap_codice_l2-lo-onboard-mpq-cal_20250101_v001.csv"
    ),
    "l2-lo-onboard-energy-bins": (
        "l2_lut/imap_codice_l2-lo-onboard-energy-bins_20250101_v001.csv"
    ),
    "l2-lo-onboard-energy-table": (
        "l2_lut/imap_codice_l2-lo-onboard-energy-table_20250101_v001.csv"
    ),
    "l2-hi-energy-table": "l2_lut/imap_codice_l2-hi-energy-table_20250101_v002.csv",
    "l2-hi-tof-table": "l2_lut/imap_codice_l2-hi-tof-table_20250101_v001.csv",
}


def l0_file(descriptor):
    """Return the L0 test file the validation data was produced from."""
    return (
        TEST_DATA_PATH
        / "l1a_input"
        / f"imap_codice_l0_{descriptor}_{VALIDATION_FILE_DATE}_v001.pkts"
    )


def validation_file(level, descriptor):
    """Return the CoDICE team's validation CDF for a product at a data level."""
    return (
        TEST_DATA_PATH
        / f"{level}_validation"
        / (
            f"imap_codice_{level}_{descriptor}_{VALIDATION_FILE_DATE}"
            f"_{VALIDATION_FILE_VERSION}.cdf"
        )
    )


def lut_file_paths(descriptor):
    """Return the test LUT for an ancillary descriptor, like get_file_paths."""
    return [TEST_DATA_PATH / LUT_FILES[descriptor]]


def file_paths_for(science_file):
    """Return a get_file_paths side effect for a science file plus the test LUTs."""
    science_descriptor = ScienceFilePath(science_file).descriptor

    def get_file_paths(descriptor=None, data_type=None):
        if data_type == "l0" or descriptor == science_descriptor:
            return [science_file]
        return lut_file_paths(descriptor)

    return get_file_paths


@pytest.fixture
def mock_get_file_paths():
    """Patch get_file_paths to serve the CoDICE test LUTs."""
    with patch(
        "imap_data_access.processing_input.ProcessingInputCollection.get_file_paths"
    ) as mock_get_file_paths:
        mock_get_file_paths.side_effect = lut_file_paths
        yield mock_get_file_paths


def assert_allclose_fillaware(actual, expected_da, rtol=1e-5, err_msg=""):
    """Compare processed data against a validation DataArray, treating the
    validation's CDF FILLVAL sentinel as equivalent to NaN.

    Some of our in-memory processed data represents "no data" as float NaN
    (e.g. after masking science counts), but integer-typed CDF variables (e.g.
    uint32 raw counts) can't hold NaN, so the validation CDF (loaded straight
    from disk) stores the CDF FILLVAL sentinel instead. Comparing those
    directly always reports a mismatch even though both sides mean the same
    thing, so normalize the validation side to NaN wherever it equals FILLVAL
    before comparing -- but only when our own data is actually float (and so
    could contain real NaN). Variables that stay integer end-to-end on both
    sides (e.g. half_spin_per_esa_step, which uses its own FILLVAL literally
    rather than NaN) should be compared as-is.
    """
    actual = np.asarray(actual)
    expected = expected_da.values
    fillval = expected_da.attrs.get("FILLVAL")
    if (
        fillval is not None
        and np.issubdtype(expected.dtype, np.integer)
        and np.issubdtype(actual.dtype, np.floating)
    ):
        expected = expected.astype(np.float64)
        expected[expected == fillval] = np.nan
    np.testing.assert_allclose(actual, expected, rtol=rtol, err_msg=err_msg)


def assert_matches_validation(processed, level, descriptor, rtol=1e-5, skip=()):
    """Assert every variable in a validation CDF matches the processed data.

    Variables the validation file leaves empty, such as the non-active Lo
    counters, have nothing to compare against and are skipped.
    """
    validation = load_cdf(validation_file(level, descriptor))
    for variable in validation.variables:
        expected = validation[variable]
        if variable in skip or expected.size == 0:
            continue
        if expected.dtype.kind == "U":
            np.testing.assert_array_equal(
                processed[variable], expected, err_msg=f"Mismatch in '{variable}'"
            )
        else:
            assert_allclose_fillaware(
                processed[variable],
                expected,
                rtol=rtol,
                err_msg=f"Mismatch in '{variable}'",
            )


def assert_epoch_delta_cdf_metadata(cdf_file):
    """Assert the written epoch delta variables use integer duration metadata."""
    with cdflib.CDF(cdf_file) as cdf:
        for variable in ("epoch_delta_minus", "epoch_delta_plus"):
            info = cdf.varinq(variable)
            attrs = cdf.varattsget(variable)
            assert info.Data_Type_Description == "CDF_INT8"
            assert attrs["FILLVAL"] == -9223372036854775808
            assert attrs["FORMAT"] == "I19"
            assert attrs["VALIDMIN"] == 0
            assert attrs["VALIDMAX"] == EXPECTED_EPOCH_DELTA_VALIDMAX


def assert_l2_epoch_delta_cdf_metadata(cdf_file):
    """Assert L2 epoch delta vars and epoch links are written correctly."""
    with cdflib.CDF(cdf_file) as cdf:
        epoch_attrs = cdf.varattsget("epoch")
        assert epoch_attrs["DELTA_MINUS_VAR"] == "epoch_delta_minus"
        assert epoch_attrs["DELTA_PLUS_VAR"] == "epoch_delta_plus"
    assert_epoch_delta_cdf_metadata(cdf_file)
