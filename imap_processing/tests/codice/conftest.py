from pathlib import Path

import pytest

from imap_processing import imap_module_directory

TEST_DATA_PATH = imap_module_directory / "tests" / "codice" / "data"
TEST_DATA_L0_PATH = TEST_DATA_PATH / "l0_data"
TEST_L0_FILE = TEST_DATA_L0_PATH / "imap_codice_l0_raw_20241110_v001.pkts"

VALIDATION_FILE_DATE = "20260204"
VALIDATION_FILE_VERSION = "v016"

# IALiRT validation data is decoupled from the science products and pinned to
# the original epoch. IALiRT no longer derives from the same kind of CoDICE L0
# packets (it is sourced from a LASP API), so no L0 packets exist at the new
# science epoch, and the issue #3242 spin-angle fix does not affect any IALiRT
# output (HI_IALIRT_REF_SPIN_ANGLE feeds only the unused HI_IALIRT_SPIN_ANGLE).
# Reconciling the IALiRT fixtures with the LASP-API data source is tracked in #3304.
IALIRT_VALIDATION_FILE_DATE = "20250814"
IALIRT_VALIDATION_FILE_VERSION = "v015"

# Science L0 inputs under ``data/l1a_input``, keyed by test descriptor with the
# value being the product descriptor as it appears in the filename. These match
# the test descriptor except where the two differ (e.g. "hi-priorities" is
# stored as "hi-priority"). The date/version come from VALIDATION_FILE_DATE.
L0_FILE_DESCRIPTORS = {
    "lo-sw-species": "lo-sw-species",
    "hi-sectored": "hi-sectored",
    "hi-omni": "hi-omni",
    "lo-direct-events": "lo-direct-events",
    "hi-direct-events": "hi-direct-events",
    "lo-nsw-priority": "lo-nsw-priority",
    "lo-sw-priority": "lo-sw-priority",
    "hi-priorities": "hi-priority",
    "hi-counters-singles": "hi-counters-singles",
    "hi-counters-aggregated": "hi-counters-aggregated",
    "lo-counters-singles": "lo-counters-singles",
    "lo-counters-aggregated": "lo-counters-aggregated",
    "hskp": "hskp",
}

# Descriptors that have an L1B validation CDF under ``data/l1b_validation`` which
# the L2 tests feed back in as L1B input (requested with ``data_type == "l1b"``).
L1B_VALIDATION_DESCRIPTORS = ("lo-sw-species", "hi-sectored", "hi-omni")

# Ancillary inputs (LUTs and fixed-date packets), keyed by descriptor. Paths are
# relative to ``TEST_DATA_PATH`` and carry their own fixed dates, independent of
# the science validation epoch.
ANCILLARY_FILES = {
    "l1a-sci-lut": "l1a_lut/imap_codice_l1a-sci-lut_20251007_v005.json",
    "l1a-sci-lut-jan": "l1a_lut/imap_codice_l1a-sci-lut_20260129_v002.json",
    "l2-hi-omni-efficiency": (
        "l2_lut/imap_codice_l2-hi-omni-efficiency_20251212_v003.csv"
    ),
    "l2-hi-sectored-efficiency": (
        "l2_lut/imap_codice_l2-hi-sectored-efficiency_20251212_v003.csv"
    ),
    "l2-lo-efficiency": "l2_lut/imap_codice_l2-lo-efficiency_20251212_v003.csv",
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
    "l2-hi-energy-table": "l2_lut/imap_codice_l2-hi-energy-table_20250101_v001.csv",
    "l2-hi-tof-table": "l2_lut/imap_codice_l2-hi-tof-table_20250101_v001.csv",
    "fsw-changes": "l1a_input/imap_codice_l0_raw_20260130_v001.pkts",
}


@pytest.fixture(scope="session")
def codice_lut_path():
    """Return a callable side-effect that returns file paths based on descriptor.

    This fixture is intended to be used as the ``side_effect`` for
    ``ProcessingInputCollection.get_file_paths`` in tests, e.g.::

        mock_get_file_paths.side_effect = codice_lut_path

    The returned function accepts a ``descriptor`` and an optional ``data_type``
    and returns a single-element list of Paths, resolved from the module-level
    ``L0_FILE_DESCRIPTORS``, ``L1B_VALIDATION_DESCRIPTORS``, and
    ``ANCILLARY_FILES`` tables.
    """

    def _side_effect(
        descriptor: str | None = None, data_type: str | None = None
    ) -> list[Path]:
        # Science products have both an L0 input and an L1B validation CDF for
        # the same descriptor, so L0 vs L1B is disambiguated by ``data_type``.
        if data_type == "l0" and descriptor in L0_FILE_DESCRIPTORS:
            product = L0_FILE_DESCRIPTORS[descriptor]
            return [
                TEST_DATA_PATH
                / "l1a_input"
                / f"imap_codice_l0_{product}_{VALIDATION_FILE_DATE}_v001.pkts"
            ]
        if data_type == "l1b" and descriptor in L1B_VALIDATION_DESCRIPTORS:
            return [
                TEST_DATA_PATH
                / "l1b_validation"
                / (
                    f"imap_codice_l1b_{descriptor}_{VALIDATION_FILE_DATE}"
                    f"_{VALIDATION_FILE_VERSION}.cdf"
                )
            ]
        if descriptor in ANCILLARY_FILES:
            return [TEST_DATA_PATH / ANCILLARY_FILES[descriptor]]
        raise ValueError(f"Unknown descriptor: {descriptor}")

    return _side_effect
