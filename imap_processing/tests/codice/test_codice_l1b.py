"""Tests the L1b processing for CoDICE L1a data"""

import pytest
from imap_data_access import ProcessingInputCollection

from imap_processing.cdf.utils import write_cdf
from imap_processing.codice.codice_l1a import process_l1a
from imap_processing.codice.codice_l1b import process_codice_l1b
from imap_processing.tests.codice.conftest import (
    VALIDATION_FILE_DATE,
    assert_matches_validation,
    file_paths_for,
    l0_file,
)
from imap_processing.utils import filter_day_boundary_data

pytestmark = pytest.mark.external_test_data

# TODO: write these with terminate_on_warning=True once their 1-D
# energy_*_minus/plus variables stop carrying both DEPEND_0 and DEPEND_1
ISTP_WARNING_PRODUCTS = ["hi-omni", "hi-sectored"]

# (descriptor, rtol, variables not compared)
L1B_PRODUCTS = [
    # TODO: compare spin_sector_pairs once our values and the validation
    # files' "0-30 deg" style labels agree
    ("lo-counters-aggregated", 1e-5, ["spin_sector_pairs"]),
    ("lo-counters-singles", 1e-5, ["spin_sector_pairs"]),
    ("lo-sw-priority", 1e-5, []),
    ("lo-nsw-priority", 1e-5, []),
    ("lo-sw-species", 1e-5, []),
    # Hi rates differ by up to 1.12e-5
    ("hi-counters-aggregated", 1.2e-5, []),
    ("hi-counters-singles", 1.2e-5, []),
    # The epoch deltas are unchanged from L1A, where they are compared
    ("hi-omni", 1.2e-5, ["epoch_delta_minus", "epoch_delta_plus"]),
    ("hi-sectored", 1.2e-5, []),
    ("hi-priority", 1.2e-5, []),
]


@pytest.mark.parametrize(
    ("descriptor", "rtol", "skip"),
    L1B_PRODUCTS,
    ids=[descriptor for descriptor, _, _ in L1B_PRODUCTS],
)
def test_l1b_validation(mock_get_file_paths, descriptor, rtol, skip):
    """Tests each L1b product against the CoDICE team's validation data."""
    mock_get_file_paths.side_effect = file_paths_for(l0_file(descriptor))
    l1a_data = filter_day_boundary_data(
        process_l1a(dependency=ProcessingInputCollection())[0], VALIDATION_FILE_DATE
    )
    processed_data = process_codice_l1b(write_cdf(l1a_data))
    assert_matches_validation(processed_data, "l1b", descriptor, rtol, skip)

    cdf_file = write_cdf(
        processed_data,
        terminate_on_warning=descriptor not in ISTP_WARNING_PRODUCTS,
    )
    assert (
        cdf_file.name
        == f"imap_codice_l1b_{descriptor}_{VALIDATION_FILE_DATE}_v001.0001.cdf"
    )
