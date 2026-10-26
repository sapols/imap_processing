import cdflib
import numpy as np
import pytest
from imap_data_access.processing_input import ProcessingInputCollection

from imap_processing.cdf.utils import load_cdf, write_cdf
from imap_processing.codice.codice_l2 import (
    HI_SPECIES_DISPLAY_NAMES,
    process_codice_l2,
)
from imap_processing.tests.codice.conftest import (
    VALIDATION_FILE_DATE,
    assert_l2_epoch_delta_cdf_metadata,
    assert_matches_validation,
    file_paths_for,
    validation_file,
)

pytestmark = pytest.mark.external_test_data


def _expected_hi_energy_labels(species: str, energies: np.ndarray) -> np.ndarray:
    species_display = HI_SPECIES_DISPLAY_NAMES[species]
    return np.array(
        [f"{species_display} int @{energy:.3f} MeV/nuc" for energy in energies]
    )


def test_l2_hi_omni(mock_get_file_paths):
    mock_get_file_paths.side_effect = file_paths_for(validation_file("l1b", "hi-omni"))
    processed_l2 = process_codice_l2("hi-omni", ProcessingInputCollection())
    assert_matches_validation(processed_l2, "l2", "hi-omni")

    processed_l2.attrs["Data_version"] = "001"
    omni_cdf_file = write_cdf(processed_l2)
    assert (
        omni_cdf_file.name == f"imap_codice_l2_hi-omni_{VALIDATION_FILE_DATE}_v001.cdf"
    )
    assert_l2_epoch_delta_cdf_metadata(omni_cdf_file)
    with cdflib.CDF(omni_cdf_file) as cdf_file:
        data_quality_attrs = cdf_file.varattsget("data_quality")
        assert data_quality_attrs["VAR_TYPE"] == "data"
        assert cdf_file.varattsget("energy_h")["FORMAT"] == "F12.6"
        assert cdf_file.varattsget("energy_h_minus")["FORMAT"] == "F12.6"
        assert cdf_file.varattsget("energy_h_plus")["FORMAT"] == "F12.6"
        energy_h_label_attrs = cdf_file.varattsget("energy_h_label")
        assert energy_h_label_attrs["FORMAT"] == "A32"
        assert (
            energy_h_label_attrs["CATDESC"]
            == "Energy-channel labels for H differential intensity"
        )
        assert energy_h_label_attrs["FIELDNAM"] == "H Energy Channel Labels"
        energy_h_attrs = cdf_file.varattsget("energy_h")
        assert energy_h_attrs["CATDESC"] == "Geometric mean energy per nucleon for H"
        assert energy_h_attrs["FIELDNAM"] == "H Energy"
        energy_junk_label_attrs = cdf_file.varattsget("energy_junk_label")
        assert (
            energy_junk_label_attrs["CATDESC"]
            == "Energy-channel labels for Junk differential intensity"
        )
        assert energy_junk_label_attrs["FIELDNAM"] == "Junk Energy Channel Labels"
        h_attrs = cdf_file.varattsget("h")
        assert h_attrs["CATDESC"] == (
            "Differential intensity for H at root-2-spaced energy-per-nucleon channels"
        )
        assert h_attrs["DELTA_MINUS_VAR"] == "unc_h"
        assert h_attrs["DELTA_PLUS_VAR"] == "unc_h"
        assert h_attrs["FIELDNAM"] == "Differential Intensity - H"
        unc_h_attrs = cdf_file.varattsget("unc_h")
        assert unc_h_attrs["CATDESC"] == (
            "Uncertainty in differential intensity for H at root-2-spaced "
            "energy-per-nucleon channels"
        )
        assert unc_h_attrs["FIELDNAM"] == "Uncertainty - H"
        junk_attrs = cdf_file.varattsget("junk")
        assert junk_attrs["CATDESC"] == (
            "Differential intensity for Junk (unclassified counts) at root-2-spaced "
            "energy-per-nucleon channels"
        )
        assert junk_attrs["FIELDNAM"] == "Differential Intensity - Junk"
        assert junk_attrs["VAR_NOTES"].strip() == (
            "Catch-all bin for counts that do not fall into any CoDICE Hi "
            "species classification bin."
        )
        unc_junk_attrs = cdf_file.varattsget("unc_junk")
        assert unc_junk_attrs["CATDESC"] == (
            "Uncertainty in differential intensity for Junk (unclassified counts) "
            "at root-2-spaced energy-per-nucleon channels"
        )
        assert unc_junk_attrs["VAR_NOTES"].strip() == (
            "Catch-all uncertainty bin for counts that do not fall into any "
            "CoDICE Hi species classification bin."
        )
        uh_attrs = cdf_file.varattsget("uh")
        assert uh_attrs["CATDESC"] == (
            "Differential intensity for Ultra-Heavy ions at root-2-spaced "
            "energy-per-nucleon channels"
        )
        assert uh_attrs["FIELDNAM"] == "Differential Intensity - Ultra-Heavy"
        np.testing.assert_array_equal(
            cdf_file.varget("energy_h_label"),
            _expected_hi_energy_labels("h", processed_l2["energy_h"].values),
        )


def test_l2_hi_sectored(mock_get_file_paths):
    mock_get_file_paths.side_effect = file_paths_for(
        validation_file("l1b", "hi-sectored")
    )
    processed_l2 = process_codice_l2("hi-sectored", ProcessingInputCollection())
    assert_matches_validation(processed_l2, "l2", "hi-sectored")

    val_data = load_cdf(validation_file("l2", "hi-sectored"))
    for variable in val_data.variables:
        # The validation file puts the epoch deltas on a placeholder dimension
        if variable in ["epoch_delta_plus", "epoch_delta_minus"]:
            continue
        assert processed_l2[variable].dims == val_data[variable].dims, (
            f"Dimension mismatch in '{variable}'"
        )

    processed_l2.attrs["Data_version"] = "001"
    sectored_cdf_file = write_cdf(processed_l2)
    assert (
        sectored_cdf_file.name
        == f"imap_codice_l2_hi-sectored_{VALIDATION_FILE_DATE}_v001.cdf"
    )
    assert_l2_epoch_delta_cdf_metadata(sectored_cdf_file)
    with cdflib.CDF(sectored_cdf_file) as cdf_file:
        data_quality_attrs = cdf_file.varattsget("data_quality")
        assert data_quality_attrs["VAR_TYPE"] == "data"
        assert data_quality_attrs["FORMAT"] == "I3"
        spin_sector_attrs = cdf_file.varattsget("spin_sector")
        assert spin_sector_attrs["FORMAT"] == "I3"
        spin_angle_attrs = cdf_file.varattsget("spin_angle")
        assert spin_angle_attrs["VAR_TYPE"] == "support_data"
        assert cdf_file.varattsget("energy_h")["FORMAT"] == "F12.6"
        assert cdf_file.varattsget("energy_h_minus")["FORMAT"] == "F12.6"
        assert cdf_file.varattsget("energy_h_plus")["FORMAT"] == "F12.6"
        energy_h_label_attrs = cdf_file.varattsget("energy_h_label")
        assert energy_h_label_attrs["FORMAT"] == "A32"
        assert (
            energy_h_label_attrs["CATDESC"]
            == "Energy-channel labels for H differential intensity"
        )
        assert energy_h_label_attrs["FIELDNAM"] == "H Energy Channel Labels"
        energy_h_attrs = cdf_file.varattsget("energy_h")
        assert energy_h_attrs["CATDESC"] == "Geometric mean energy per nucleon for H"
        assert energy_h_attrs["FIELDNAM"] == "H Energy"
        h_attrs = cdf_file.varattsget("h")
        assert h_attrs["CATDESC"] == (
            "Differential intensity for H by energy, spin sector, and "
            "elevation at x2-spaced energy-per-nucleon channels"
        )
        assert h_attrs["FIELDNAM"] == "Differential Intensity - H"
        unc_h_attrs = cdf_file.varattsget("unc_h")
        assert unc_h_attrs["CATDESC"] == (
            "Uncertainty in differential intensity for H by energy, spin "
            "sector, and elevation at x2-spaced energy-per-nucleon channels"
        )
        assert unc_h_attrs["FIELDNAM"] == "Uncertainty - H"
        np.testing.assert_array_equal(
            cdf_file.varget("energy_h_label"),
            _expected_hi_energy_labels("h", processed_l2["energy_h"].values),
        )
