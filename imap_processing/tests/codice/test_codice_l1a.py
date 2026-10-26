"""Tests the L1a processing for decommutated CoDICE data"""

import logging

import cdflib
import numpy as np
import pytest
from imap_data_access import ProcessingInputCollection

from imap_processing import imap_module_directory
from imap_processing.cdf.utils import write_cdf
from imap_processing.codice.codice_l1a import process_l1a
from imap_processing.codice.codice_l1a_de import l1a_direct_event
from imap_processing.codice.utils import CODICEAPID
from imap_processing.tests.codice.conftest import (
    TEST_DATA_PATH,
    VALIDATION_FILE_DATE,
    assert_epoch_delta_cdf_metadata,
    assert_matches_validation,
    file_paths_for,
    l0_file,
)
from imap_processing.utils import filter_day_boundary_data, packet_file_to_datasets

pytestmark = pytest.mark.external_test_data

LO_MODE_SUPPORT_VARIABLES = [
    "rgfo_half_spin",
    "rgfo_spin_sector",
    "rgfo_esa_step",
    "nso_half_spin",
    "nso_spin_sector",
    "nso_esa_step",
]

# (descriptor, rtol, variables not compared)
L1A_PRODUCTS = [
    # TODO: compare spin_sector_pairs once our values and the validation
    # files' "0-30 deg" style labels agree
    ("lo-counters-aggregated", 1e-5, ["spin_sector_pairs"]),
    ("lo-counters-singles", 1e-5, ["spin_sector_pairs"]),
    ("lo-sw-priority", 1e-5, []),
    ("lo-nsw-priority", 1e-5, []),
    ("lo-sw-species", 1e-5, []),
    ("hi-counters-aggregated", 1e-5, []),
    ("hi-counters-singles", 1e-5, []),
    # epoch_delta_minus/plus differ by up to 4.3e-5
    ("hi-omni", 5e-5, []),
    ("hi-sectored", 1e-5, []),
    ("hi-priority", 1e-5, []),
    ("lo-direct-events", 1e-5, []),
    # TODO: 2 of 2172 num_events values mismatch (epoch 257 & 312, priority 3).
    # combine_segmented_packets (imap_processing/utils.py) drops the
    # entire first packet of a segmented group whenever that group's
    # sequence flags look corrupted (see "Incorrect/incomplete sequence
    # flags" warnings), even though num_events is a header field that
    # lives entirely in that first packet and is unaffected by
    # corruption in later continuation packets. Downstream, that
    # priority is then treated as fully missing and zero-padded
    # (codice_l1a_de.py process_de_data), so we report num_events=0
    # while the validation file retains the real, recoverable count.
    # The per-event science arrays are unaffected (correctly fillval
    # on both sides) since the event byte payload really is unusable.
    # The CoDICE team is aware and will decide if we need to recover
    # these values before combine_segmented_packets.
    ("hi-direct-events", 1e-5, ["num_events"]),
]


def test_updated_packet_version(mock_get_file_paths, caplog):
    """Tests the new FSW changes (jan 2026)."""
    mock_get_file_paths.side_effect = file_paths_for(
        TEST_DATA_PATH / "l1a_input" / "imap_codice_l0_raw_20260130_v001.pkts"
    )
    datasets = process_l1a(dependency=ProcessingInputCollection())
    # Assert that we have all of the expected datasets
    assert len(datasets) == 14
    for ds in datasets:
        # Only check lo products. Skip direct-events
        if (
            "lo" not in ds.attrs["Data_type"]
            or "direct-events" in ds.attrs["Data_type"]
        ):
            continue
        # Check that the lo datasets contain the new unpacked variables
        expected_vars = [
            "rgfo_spin_sector",
            "nso_spin_sector",
            "rgfo_esa_step",
            "nso_esa_step",
        ]
        for var in expected_vars:
            assert var in ds.data_vars, (
                f"Expected variable '{var}' not found in dataset"
            )

        # check that a warning is logged for the missing "cnoplus" species
        assert (
            "Desired species cnoplus not found in actual species names from LUT"
            in caplog.text
        )


def test_hskp(mock_get_file_paths):
    """Tests the housekeeping."""
    mock_get_file_paths.side_effect = file_paths_for(l0_file("hskp"))
    processed_datasets = process_l1a(dependency=ProcessingInputCollection())

    assert len(processed_datasets) == 2
    processed_l1a = processed_datasets[0]
    processed_l1b = processed_datasets[1]

    # spot check the l1a value is an integer and the l1b is a float after conversion
    np.testing.assert_almost_equal(processed_l1a["fee_ssd_eb_temp_1_t"].values[0], 1367)
    np.testing.assert_almost_equal(
        processed_l1b["fee_ssd_eb_temp_1_t"].values[0], -0.317, decimal=2
    )


@pytest.mark.parametrize(
    ("descriptor", "rtol", "skip"),
    L1A_PRODUCTS,
    ids=[descriptor for descriptor, _, _ in L1A_PRODUCTS],
)
def test_l1a_validation(mock_get_file_paths, descriptor, rtol, skip):
    """Tests each L1a product against the CoDICE team's validation data."""
    mock_get_file_paths.side_effect = file_paths_for(l0_file(descriptor))
    processed_data = filter_day_boundary_data(
        process_l1a(dependency=ProcessingInputCollection())[0], VALIDATION_FILE_DATE
    )
    assert_matches_validation(processed_data, "l1a", descriptor, rtol, skip)

    cdf_file = write_cdf(processed_data, terminate_on_warning=True)
    assert (
        cdf_file.name
        == f"imap_codice_l1a_{descriptor}_{VALIDATION_FILE_DATE}_v001.0001.cdf"
    )
    assert_epoch_delta_cdf_metadata(cdf_file)


def test_lo_direct_events_mode_variables_are_support_data(mock_get_file_paths):
    mock_get_file_paths.side_effect = file_paths_for(l0_file("lo-direct-events"))
    cdf_file = write_cdf(process_l1a(dependency=ProcessingInputCollection())[0])
    with cdflib.CDF(cdf_file) as cdf:
        for variable in LO_MODE_SUPPORT_VARIABLES:
            attrs = cdf.varattsget(variable)
            assert attrs["VAR_TYPE"] == "support_data"


def test_direct_events_incomplete_groups(caplog):
    """Tests lo-direct-events with a packet containing incomplete groups."""

    xtce_file = (
        imap_module_directory
        / "codice/packet_definitions/imap_codice_packet-definition_20250101_v001.xml"
    )
    # Decom packet
    datasets_by_apid = packet_file_to_datasets(
        l0_file("lo-direct-events"),
        xtce_file,
    )
    apid = CODICEAPID.COD_LO_PHA
    de_dataset = datasets_by_apid[apid]

    # NOTE: This L0 test file already has some naturally incomplete priority
    # groups -- packets genuinely missing from the raw telemetry (confirmed via
    # gaps in the CCSDS source sequence counter), independent of the packet we
    # drop below. So dropping one packet below does NOT create the only
    # incomplete group in the file -- it makes an already-incomplete group even
    # more incomplete. These are the current baseline values for that first
    # group (acq_start_seconds, packet count) before our drop, taken from the
    # untouched data: if this L0 fixture ever changes, these will need updating.
    first_group_time = 507858646
    baseline_count = 4

    # Drop the first packet to test incomplete group handling
    # This mocks the case when one priority group is incomplete
    # in this example, the first group is missing the first priority
    len_epoch = de_dataset.sizes["epoch"]
    de_dataset = de_dataset.isel(epoch=slice(1, len_epoch))
    with caplog.at_level(logging.WARNING):
        dataset = l1a_direct_event(de_dataset, apid)
    # Check that fillvals are used for the first missing priority for the first epoch
    assert np.all(dataset.tof[0, 0, :].values == 65535)
    # Check that there is data for the remaining priorities
    assert np.any(dataset.tof[0, 1:, :].values != 65535)
    # Check logs report the first group with one fewer packet than baseline
    assert f"acq_start_seconds [{first_group_time}" in caplog.text
    assert f"counts [{baseline_count - 1}" in caplog.text
