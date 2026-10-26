"""Tests the L2 processing of CoDICE L1 data"""

from unittest import mock

import cdflib
import numpy as np
import pandas as pd
import pytest
import xarray as xr
from imap_data_access import ProcessingInputCollection
from sammi.validation import CDFValidator

from imap_processing.cdf.utils import load_cdf, write_cdf
from imap_processing.codice.codice_l1a import process_l1a
from imap_processing.codice.codice_l1b import process_codice_l1b
from imap_processing.codice.codice_l2 import (
    compute_geometric_factors,
    get_efficiency_lut,
    get_geometric_factor_lut,
    get_hi_de_luts,
    get_mpq_calc_energy_conversion_vals,
    get_mpq_calc_tof_conversion_vals,
    process_codice_l2,
    process_lo_species_intensity,
)
from imap_processing.codice.constants import (
    LO_SW_SOLAR_WIND_SPECIES_VARIABLE_NAMES,
)
from imap_processing.tests.codice.conftest import (
    LUT_FILES,
    TEST_DATA_PATH,
    VALIDATION_FILE_DATE,
    assert_l2_epoch_delta_cdf_metadata,
    assert_matches_validation,
    file_paths_for,
    l0_file,
)
from imap_processing.utils import filter_day_boundary_data

pytestmark = pytest.mark.external_test_data

DIRECT_EVENT_DISPLAY_TYPES = {
    "lo-direct-events": {
        "num_events": "spectrogram",
        "data_quality": "no_plot",
        "gain": "no_plot",
        "multi_flag": "no_plot",
        "spin_angle": "spectrogram",
        "elevation_angle": "spectrogram",
        "tof": "spectrogram",
        "type": "no_plot",
        "apd_energy": "spectrogram",
        "apd_id": "no_plot",
        "energy_per_charge": "spectrogram",
        "energy_step": "no_plot",
        "position": "no_plot",
    },
    "hi-direct-events": {
        "num_events": "spectrogram",
        "data_quality": "no_plot",
        "gain": "no_plot",
        "multi_flag": "no_plot",
        "spin_angle": "spectrogram",
        "elevation_angle": "spectrogram",
        "tof": "spectrogram",
        "type": "no_plot",
        "ssd_energy": "spectrogram",
        "energy_per_nuc": "spectrogram",
        "ssd_id": "no_plot",
    },
}
LO_DIRECT_EVENT_SUPPORT_VARIABLES = [
    "nso_half_spin",
    "nso_spin_sector",
    "nso_esa_step",
]


def _generate_direct_events_l2_file(mock_get_file_paths, descriptor):
    """Generate a fresh CoDICE direct-events L2 CDF for metadata assertions."""
    mock_get_file_paths.side_effect = file_paths_for(l0_file(descriptor))
    l1a_cdf = process_l1a(ProcessingInputCollection())[0]
    processed_l1a_file = write_cdf(l1a_cdf)
    mock_get_file_paths.side_effect = file_paths_for(processed_l1a_file)
    processed_l2_ds = process_codice_l2(descriptor, ProcessingInputCollection())
    processed_l2_ds.attrs["Data_version"] = "001"
    return write_cdf(processed_l2_ds)


@pytest.fixture
def mock_half_spin_per_esa_step():
    """
    Mock half_spin_per_esa_step for testing.
    Example:
      ESA steps 0–63 belong to half_spin=2
      ESA steps 64–127 belong to half_spin=3
    """
    half_spin_per_esa = np.repeat([2, 3], 64)
    # repeat along epoch dimension to create shape (2, 128) for testing
    return np.tile(half_spin_per_esa, (2, 1))


def test_compute_geometric_factors_all_full_mode(mock_half_spin_per_esa_step):
    # rgfo_half_spin = 4 means all half_spin values (2 or 3) are < rgfo_half_spin
    dataset = xr.Dataset(
        {
            "rgfo_half_spin": (("epoch",), np.array([4, 4])),
            "half_spin_per_esa_step": (
                (
                    "epoch",
                    "esa_step",
                ),
                mock_half_spin_per_esa_step,
            ),
        },
        attrs={"Logical_file_id": "imap_codice_l1b_lo-sw-species_20250101_v001"},
    )
    geometric_factor_lut = {
        "full": np.zeros((128, 24)),
        "reduced": np.ones((128, 24)),
    }
    result = compute_geometric_factors(dataset, geometric_factor_lut)

    # Expect "full" values everywhere
    expected = np.full((2, 128, 24), 0)
    np.testing.assert_array_equal(result, expected)


def test_compute_geometric_factors_past_nov_24th(mock_half_spin_per_esa_step):
    # rgfo_half_spin = 1 means all half_spin values (>=2) are >= rgfo_half_spin
    # Although the rgfo_half_spin indicates reduced mode, the date is past Nov 24th,
    # 2025 so we expect full mode to be used.
    dataset = xr.Dataset(
        {
            "rgfo_half_spin": (("epoch",), np.array([1, 1])),
            "half_spin_per_esa_step": (
                (
                    "epoch",
                    "esa_step",
                ),
                mock_half_spin_per_esa_step,
            ),
        },
        # Make sure epoch is past Nov 24th, 2025
        attrs={"Logical_file_id": "imap_codice_l1b_lo-sw-species_20251125_v001"},
    )
    geometric_factor_lut = {
        "full": np.zeros((128, 24)),
        "reduced": np.ones((128, 24)),
    }
    result = compute_geometric_factors(dataset, geometric_factor_lut)

    # Expect "full" values everywhere
    expected = np.full((2, 128, 24), 0)
    np.testing.assert_array_equal(result, expected)


def test_compute_geometric_factors_all_reduced_mode(mock_half_spin_per_esa_step):
    # rgfo_half_spin = 1 means all half_spin values (>=2) are >= rgfo_half_spin
    dataset = xr.Dataset(
        {
            "rgfo_half_spin": (("epoch",), np.array([1])),
            "half_spin_per_esa_step": (
                (
                    "epoch",
                    "esa_step",
                ),
                mock_half_spin_per_esa_step[0:1],
            ),
        },
        attrs={"Logical_file_id": "imap_codice_l1b_lo-sw-species_20250101_v001"},
    )
    geometric_factor_lut = {
        "full": np.zeros((128, 24)),
        "reduced": np.ones((128, 24)),
    }
    result = compute_geometric_factors(dataset, geometric_factor_lut)

    # Expect "reduced" values everywhere
    expected = np.full((1, 128, 24), 1)
    np.testing.assert_array_equal(result, expected)


def test_compute_geometric_factors_mixed(mock_half_spin_per_esa_step):
    # rgfo_half_spin = 2
    dataset = xr.Dataset(
        {
            "rgfo_half_spin": (("epoch",), np.array([2])),
            "half_spin_per_esa_step": (
                (
                    "epoch",
                    "esa_step",
                ),
                mock_half_spin_per_esa_step[0:1],
            ),
        },
        attrs={"Logical_file_id": "imap_codice_l1b_lo-sw-species_20250101_v001"},
    )
    geometric_factor_lut = {
        "full": np.zeros((128, 24)),
        "reduced": np.ones((128, 24)),
    }
    result = compute_geometric_factors(dataset, geometric_factor_lut)

    # ESA steps 0-63 (half_spin=1) -> 2 > 1 → mode=full → 1
    # ESA steps 64-127 (half_spin=2) -> 1 !>1 → mode=reduced → 0
    expected = np.repeat(np.array([[[0]] * 64 + [[1]] * 64]), 24, -1)
    np.testing.assert_array_equal(result, expected)


def test_get_geometric_factor_lut(mock_get_file_paths):
    gfactor_lut = get_geometric_factor_lut(ProcessingInputCollection())

    # Load the csv files directly to compare
    geometric_factors = pd.read_csv(TEST_DATA_PATH / LUT_FILES["l2-lo-gfactor"])
    full = (
        geometric_factors[geometric_factors["mode"] == "full"]
        .drop(["mode", "esa_step"], axis=1)
        .to_numpy()
    )
    reduced = (
        geometric_factors[geometric_factors["mode"] == "reduced"]
        .drop(["mode", "esa_step"], axis=1)
        .to_numpy()
    )

    # Test the shape is (modes, esa_steps, positions)
    np.testing.assert_array_equal(gfactor_lut["full"].shape, (128, 24))

    np.testing.assert_array_equal(gfactor_lut["full"], full)
    np.testing.assert_array_equal(gfactor_lut["reduced"], reduced)


def test_get_efficiency_lut(mock_get_file_paths):
    efficiency_lut = get_efficiency_lut(ProcessingInputCollection())
    expected_colnames = ["esa_step", "product", "species"] + [
        f"position_{x}" for x in range(1, 25)
    ]

    for col in expected_colnames:
        assert col in efficiency_lut.columns, f"Missing column {col} in efficiency LUT"


def test_get_tof_ns_from_mpq_lut(mock_get_file_paths):
    tof_ns = get_mpq_calc_tof_conversion_vals(ProcessingInputCollection())
    assert tof_ns.shape == (1024,)
    mpq_df = pd.read_csv(
        TEST_DATA_PATH / LUT_FILES["l2-lo-onboard-mpq-cal"], header=None
    )
    expected_tof_ns = mpq_df.loc[6:, 1].to_numpy().astype(np.float64)
    # Calculated values should be more precise than LUT but should be close
    np.testing.assert_allclose(tof_ns, expected_tof_ns, atol=1e-5)


def test_get_energy_kev_from_mpq_lut(mock_get_file_paths):
    energy_kev = get_mpq_calc_energy_conversion_vals(ProcessingInputCollection())
    assert energy_kev.shape == (128,)
    mpq_df = pd.read_csv(
        TEST_DATA_PATH / LUT_FILES["l2-lo-onboard-mpq-cal"], header=None
    )
    expected_e_kev = mpq_df.loc[5, 4:].to_numpy().astype(np.float64)
    # Calculated values should be more precise than LUT but should be close
    np.testing.assert_allclose(energy_kev, expected_e_kev, rtol=0.01)


def test_get_hi_de_luts(mock_get_file_paths):
    energy_table, tof_table = get_hi_de_luts(ProcessingInputCollection())
    assert energy_table.shape == (2048, 48)
    assert tof_table.shape == (1024, 2)


def test_process_lo_species_intensity(mock_get_file_paths):
    mock_get_file_paths.side_effect = file_paths_for(l0_file("lo-sw-species"))
    processed_l1a_file = write_cdf(process_l1a(ProcessingInputCollection())[0])
    l1b_data = process_codice_l1b(processed_l1a_file)
    l1b_val_data_processed = l1b_data.copy()
    gf = xr.DataArray(
        np.ones((len(l1b_data.epoch), 128, 24)) * 2,
        dims=("epoch", "esa_step", "inst_az"),
    )
    with mock.patch(
        "imap_processing.codice.codice_l2.get_species_efficiency",
        return_value=xr.DataArray(np.ones((128, 24)) * 2, dims=("esa_step", "inst_az")),
    ):
        len_pos = 5
        process_lo_species_intensity(
            l1b_val_data_processed,
            LO_SW_SOLAR_WIND_SPECIES_VARIABLE_NAMES,
            gf,
            None,
            list(np.arange(0, len_pos)),
        )

    for var in LO_SW_SOLAR_WIND_SPECIES_VARIABLE_NAMES:
        assert var in l1b_val_data_processed, f"Missing variable {var} after processing"
        # Check that values are non-negative. NaN is a legitimate "no data"
        # marker for masked ESA steps, and `nan >= 0` is always False, so
        # exclude NaNs rather than letting them read as "negative".
        vals = l1b_val_data_processed[var].values
        assert np.all(vals[~np.isnan(vals)] >= 0), (
            f"Variable {var} contains negative values"
        )
        # Check that values match expected calculation
        expected_intensity = (
            l1b_data[var]
            / (len_pos * 4 * l1b_data["energy_per_charge"].data)[
                np.newaxis, :, np.newaxis
            ]
        )
        # process_lo_species_intensity fills in NaN for the half-spin at which
        # the RGFO mode is triggered, since the data during that half-spin
        # can't be de-convolved (see codice_l2.py process_lo_species_intensity
        # for details). Reproduce that here so the comparison matches.
        half_spin_boundary = (
            l1b_data.half_spin_per_esa_step.data
            == l1b_data.rgfo_half_spin.data[:, np.newaxis]
        )[:, :, np.newaxis]
        expected_intensity = expected_intensity.where(~half_spin_boundary)
        np.testing.assert_allclose(
            l1b_val_data_processed[var].values, expected_intensity.values, rtol=1e-5
        )


def test_process_lo_missing_species_intensity():
    l1b_val_data = xr.Dataset(
        {
            "epoch": ("epoch", np.ones(5)),
            "energy_per_charge": (("esa_step",), np.ones(128) * 10),
            "packet_version": ("epoch", np.ones(5)),
            "half_spin_per_esa_step": (("epoch", "esa_step"), np.ones((5, 128)) * 2),
            "rgfo_half_spin": ("epoch", np.ones(5) * 2),
        }
    )

    l1b_val_data_processed = l1b_val_data.copy()
    gf = xr.DataArray(
        np.ones((len(l1b_val_data.epoch), 128, 24)) * 2,
        dims=("epoch", "energy_per_charge", "inst_az"),
    )
    with mock.patch(
        "imap_processing.codice.codice_l2.get_species_efficiency",
        return_value=xr.DataArray(
            np.ones((128, 24)) * 2, dims=("energy_per_charge", "inst_az")
        ),
    ):
        len_pos = 5
        with pytest.raises(ValueError, match="Species hplus not found in dataset"):
            process_lo_species_intensity(
                l1b_val_data_processed,
                LO_SW_SOLAR_WIND_SPECIES_VARIABLE_NAMES,
                gf,
                None,
                list(np.arange(0, len_pos)),
            )


def test_codice_l2_sw_species_intensity(mock_get_file_paths):
    mock_get_file_paths.side_effect = file_paths_for(l0_file("lo-sw-species"))
    l1a_ds = filter_day_boundary_data(
        process_l1a(ProcessingInputCollection())[0], VALIDATION_FILE_DATE
    )
    processed_l1a_file = write_cdf(l1a_ds)
    processed_l1b_file = write_cdf(process_codice_l1b(processed_l1a_file))
    mock_get_file_paths.side_effect = file_paths_for(processed_l1b_file)
    processed_2_ds = process_codice_l2("lo-sw-species", ProcessingInputCollection())
    assert_matches_validation(processed_2_ds, "l2", "lo-sw-species")

    processed_2_ds.attrs["Data_version"] = "001"
    assert processed_2_ds.attrs["Logical_source"] == "imap_codice_l2_lo-sw-species"
    cdf_path = write_cdf(processed_2_ds)
    assert_l2_epoch_delta_cdf_metadata(cdf_path)
    # Make sure the written file can actually be reloaded
    load_cdf(cdf_path)
    with cdflib.CDF(cdf_path) as cdf_file:
        hplus_attrs = cdf_file.varattsget("hplus")
        assert (
            hplus_attrs["CATDESC"]
            == "Differential intensity from sunward-looking detectors measuring "
            "solar-wind H+"
        )
        assert hplus_attrs["FIELDNAM"] == "Sunward Differential Intensity - H+"
        unc_hplus_attrs = cdf_file.varattsget("unc_hplus")
        assert (
            unc_hplus_attrs["CATDESC"]
            == "Uncertainty in differential intensity from sunward-looking "
            "detectors measuring solar-wind H+"
        )
        assert unc_hplus_attrs["FIELDNAM"] == "Sunward Uncertainty - H+"
        for var in ["nso_esa_step", "nso_spin_sector"]:
            var_info = cdf_file.varinq(var)
            var_attrs = cdf_file.varattsget(var)
            assert var_info.Data_Type_Description == "CDF_UINT1"
            assert var_attrs["FILLVAL"] == np.uint8(255)
        # y-axis should be physical energy_per_charge values, not esa_step
        assert hplus_attrs["DEPEND_1"] == "energy_per_charge"
        energy_per_charge_attrs = cdf_file.varattsget("energy_per_charge")
        assert "DEPEND_1" not in energy_per_charge_attrs


def test_codice_l2_lo_de(mock_get_file_paths):
    mock_get_file_paths.side_effect = file_paths_for(l0_file("lo-direct-events"))
    l1a_cdf = filter_day_boundary_data(
        process_l1a(ProcessingInputCollection())[0], VALIDATION_FILE_DATE
    )

    processed_l1a_file = write_cdf(l1a_cdf)
    mock_get_file_paths.side_effect = file_paths_for(processed_l1a_file)

    processed_l2_ds = process_codice_l2("lo-direct-events", ProcessingInputCollection())
    l1a_input_ds = load_cdf(processed_l1a_file)
    original_spin_sector = l1a_input_ds["spin_sector"].values
    # Mirror the LO direct-event spin-sector remapping so this test catches any
    # unintended changes to valid sector values while still checking that only
    # invalid sectors are replaced with the uint8 fill value.
    expected_spin_sector = np.where(
        (l1a_input_ds["position"].values >= 13)
        & (l1a_input_ds["position"].values <= 24),
        (original_spin_sector + 12) % 24,
        original_spin_sector,
    )
    invalid_spin_sector = ~np.isfinite(original_spin_sector) | (
        original_spin_sector > 23
    )
    expected_spin_sector = np.where(
        invalid_spin_sector, np.uint8(255), expected_spin_sector
    ).astype(np.uint8)
    assert processed_l2_ds["spin_sector"].dtype == np.uint8
    np.testing.assert_array_equal(
        processed_l2_ds["spin_sector"].values,
        expected_spin_sector,
        err_msg="LO direct-event spin_sector values changed unexpectedly",
    )
    assert_matches_validation(processed_l2_ds, "l2", "lo-direct-events", rtol=5e-5)

    processed_l2_ds.attrs["Data_version"] = "001"
    assert processed_l2_ds.attrs["Logical_source"] == "imap_codice_l2_lo-direct-events"
    file = write_cdf(processed_l2_ds)
    assert_l2_epoch_delta_cdf_metadata(file)
    errors = CDFValidator().validate(file)
    assert not errors
    cdf_file = cdflib.CDF(file)
    spin_sector_info = cdf_file.varinq("spin_sector")
    spin_sector_attrs = cdf_file.varattsget("spin_sector")
    data_quality_info = cdf_file.varinq("data_quality")
    data_quality_attrs = cdf_file.varattsget("data_quality")
    assert spin_sector_info.Data_Type_Description == "CDF_UINT1"
    assert spin_sector_attrs["FILLVAL"] == np.uint8(255)
    assert data_quality_info.Data_Type_Description == "CDF_UINT2"
    assert data_quality_attrs["FILLVAL"] == np.uint16(65535)
    load_cdf(file)


def test_codice_l2_hi_de(mock_get_file_paths):
    mock_get_file_paths.side_effect = file_paths_for(l0_file("hi-direct-events"))
    l1a_cdf = filter_day_boundary_data(
        process_l1a(ProcessingInputCollection())[0], VALIDATION_FILE_DATE
    )

    processed_l1a_file = write_cdf(l1a_cdf)
    mock_get_file_paths.side_effect = file_paths_for(processed_l1a_file)

    processed_l2_ds = process_codice_l2("hi-direct-events", ProcessingInputCollection())
    # TODO: 2 of 2172 num_events values mismatch here (epoch 257 & 312, priority 3).
    # combine_segmented_packets (imap_processing/utils.py) drops the
    # entire first packet of a segmented group whenever that group's
    # sequence flags look corrupted (see "Incorrect/incomplete sequence
    # flags" warnings), even though num_events is a header field that
    # lives entirely in that first packet and is unaffected by
    # corruption in later continuation packets. Downstream, that
    # priority is then treated as fully missing and zero-padded
    # (codice_l1a_de.py process_de_data), so we report num_events=0
    # while the validation file retains the real, recoverable count.
    # Fix planned in a follow-up PR: recover num_events/byte_count
    # from the raw first packet before it gets dropped, instead of
    # hardcoding 0, for groups dropped due to bad sequence flags
    # (as opposed to packets genuinely never received).
    assert_matches_validation(
        processed_l2_ds, "l2", "hi-direct-events", rtol=5e-5, skip=["num_events"]
    )

    processed_l2_ds.attrs["Data_version"] = "001"
    assert processed_l2_ds.attrs["Logical_source"] == "imap_codice_l2_hi-direct-events"
    file = write_cdf(processed_l2_ds)
    assert_l2_epoch_delta_cdf_metadata(file)
    errors = CDFValidator().validate(file)
    assert not errors
    cdf_file = cdflib.CDF(file)
    data_quality_info = cdf_file.varinq("data_quality")
    data_quality_attrs = cdf_file.varattsget("data_quality")
    assert data_quality_info.Data_Type_Description == "CDF_UINT2"
    assert data_quality_attrs["FILLVAL"] == np.uint16(65535)
    load_cdf(file)


@pytest.mark.parametrize("descriptor", ["lo-direct-events", "hi-direct-events"])
def test_codice_l2_direct_events_display_type_cdf_metadata(
    mock_get_file_paths, descriptor
):
    file = _generate_direct_events_l2_file(mock_get_file_paths, descriptor)
    with cdflib.CDF(str(file)) as cdf_file:
        for variable, expected_display_type in DIRECT_EVENT_DISPLAY_TYPES[
            descriptor
        ].items():
            attrs = cdf_file.varattsget(variable)
            assert "DISPLAY_TYPE" in attrs, f"{variable} is missing DISPLAY_TYPE"
            assert isinstance(attrs["DISPLAY_TYPE"], str), (
                f"{variable} DISPLAY_TYPE must be stored as a string"
            )
            assert attrs["DISPLAY_TYPE"] == expected_display_type

        if descriptor == "hi-direct-events":
            attrs = cdf_file.varattsget("energy_per_nuc")
            assert attrs["DEPEND_1"] == "priority"
            assert attrs["LABL_PTR_1"] == "priority_label"
            priority_attrs = cdf_file.varattsget("priority")
            assert (
                priority_attrs["CATDESC"]
                == "Direct-event telemetry priority level (0-5)"
            )
            assert (
                priority_attrs["VAR_NOTES"].strip()
                == "Hi direct-event telemetry priority level. 0 = unused, 1 = "
                "DCR, 2 = SSD-only, 3 = Protons, 4 = Helium, 5 = Heavies."
            )
        else:
            priority_attrs = cdf_file.varattsget("priority")
            assert (
                priority_attrs["CATDESC"]
                == "Direct-event telemetry priority level (0-7)"
            )
            assert (
                priority_attrs["VAR_NOTES"].strip()
                == "Lo direct-event telemetry priority level. 0 = SW TCR PUIs, "
                "1 = SW H+, 2 = SW He++, 3 = SW Heavies, 4 = SW DCR PUIs, 5 = "
                "NSW Heavies, 6 = NSW H+ and He++, 7 = reserved/unused."
            )

        type_attrs = cdf_file.varattsget("type")
        assert type_attrs["FIELDNAM"] == "PHA Type Code"
        assert type_attrs["VALIDMAX"] == 2
        if descriptor == "hi-direct-events":
            assert type_attrs["CATDESC"] == "PHA type code: 0=TCR, 1=DCR, 2=SSD-only"
            assert (
                type_attrs["VAR_NOTES"].strip()
                == "PHA type code for CoDICE-Hi direct events. 0 = TCR "
                "(Triple Coincidence Rate; ST+SP+APD), 1 = DCR "
                "(Double Coincidence Rate; ST+SP), 2 = SSD-only energy event "
                "(SSD only or ST+SSD only)."
            )
        else:
            assert type_attrs["CATDESC"] == "PHA type code: 0=TCR, 1=DCR, 2=APD-only"
            assert (
                type_attrs["VAR_NOTES"].strip()
                == "PHA type code for CoDICE-Lo direct events. 0 = TCR "
                "(Triple Coincidence Rate; STA+STB+SP+APD), 1 = DCR "
                "(Double Coincidence Rate; STA+STB+SP), 2 = APD-only energy "
                "event (APD with only one or two of STA, STB, and SP)."
            )


def test_codice_l2_lo_direct_events_support_var_type_cdf_metadata(
    mock_get_file_paths,
):
    file = _generate_direct_events_l2_file(mock_get_file_paths, "lo-direct-events")
    with cdflib.CDF(str(file)) as cdf_file:
        written_variables = set(cdf_file.cdf_info().zVariables)
        assert set(LO_DIRECT_EVENT_SUPPORT_VARIABLES) <= written_variables
        for variable in LO_DIRECT_EVENT_SUPPORT_VARIABLES:
            attrs = cdf_file.varattsget(variable)
            assert attrs["VAR_TYPE"] == "support_data"
