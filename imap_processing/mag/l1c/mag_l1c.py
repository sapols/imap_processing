"""MAG L1C processing module."""

import logging
from dataclasses import dataclass

import numpy as np
import xarray as xr

from imap_processing.cdf.imap_cdf_manager import ImapCdfAttributes
from imap_processing.mag import imap_mag_sdc_configuration_v001 as configuration
from imap_processing.mag.constants import ModeFlags, VecSec
from imap_processing.mag.l1c.interpolation_methods import InterpolationFunction
from imap_processing.spice.time import et_to_ttj2000ns, str_to_et

logger = logging.getLogger(__name__)

GAP_TOLERANCE = 0.075


def _to_int64_ns(values: np.ndarray | list | tuple) -> np.ndarray:
    """Convert timestamp-like values to nanosecond int64 without float round-trip loss."""
    array = np.asarray(values)
    if np.issubdtype(array.dtype, np.integer):
        return array.astype(np.int64, copy=False)

    return np.rint(array).astype(np.int64, copy=False)


def _to_int64_ns_scalar(value: int | float | np.integer | np.floating) -> int:
    """Scalar form of _to_int64_ns."""
    if isinstance(value, (int, np.integer)):
        return int(value)

    return int(np.rint(value))


@dataclass(frozen=True)
class GapFillPlan:
    """Synthetic NM timestamps and BM context for filling a single gap."""

    t_a: int
    t_b: int
    norm_rate: VecSec
    burst_rate: VecSec
    synthetic_epochs: np.ndarray
    source_indices: np.ndarray
    burst_window_start: int
    burst_window_end: int


def mag_l1c(
    first_input_dataset: xr.Dataset,
    day_to_process: np.datetime64,
    second_input_dataset: xr.Dataset = None,
) -> xr.Dataset:
    """
    Will process MAG L1C data from L1A data.

    This requires both the norm and burst data to be passed in.

    Parameters
    ----------
    first_input_dataset : xr.Dataset
        The first input dataset to process. This can be either burst or norm data, for
        mago or magi.
    day_to_process : np.datetime64
        The day to process, in np.datetime64[D] format. This is used to fill gaps at
        the beginning or end of the day if needed.
    second_input_dataset : xr.Dataset, optional
        The second input dataset to process. This should be burst if first_input_dataset
        was norm, or norm if first_input_dataset was burst. It should match the
        instrument - both inputs should be mago or magi.

    Returns
    -------
    output_dataset : xr.Dataset
        L1C data set.
    """
    # TODO:
    # find missing sequences and output them
    # Fix gaps at the beginning of the day by going to previous day's file
    # Fix gaps at the end of the day
    # Allow for one input to be missing
    # Missing burst file - just pass through norm file
    # Missing norm file - go back to previous L1C file to find timestamps, then
    # interpolate the entire day from burst

    input_logical_source_1 = first_input_dataset.attrs["Logical_source"]
    if isinstance(first_input_dataset.attrs["Logical_source"], list):
        input_logical_source_1 = first_input_dataset.attrs["Logical_source"][0]

    sensor = input_logical_source_1[-1:]
    output_logical_source = f"imap_mag_l1c_norm-mag{sensor}"

    normal_mode_dataset, burst_mode_dataset = select_datasets(
        first_input_dataset, second_input_dataset
    )

    interp_function = InterpolationFunction[configuration.L1C_INTERPOLATION_METHOD]
    if burst_mode_dataset is not None:
        # Only use day_to_process if there is no norm data
        day_to_process_arg = day_to_process if normal_mode_dataset is None else None
        full_interpolated_timeline: np.ndarray = process_mag_l1c(
            normal_mode_dataset, burst_mode_dataset, interp_function, day_to_process_arg
        )
    elif normal_mode_dataset is not None:
        full_interpolated_timeline = fill_normal_data(normal_mode_dataset)
    else:
        raise ValueError("At least one of norm or burst dataset must be provided.")

    completed_timeline = remove_missing_data(full_interpolated_timeline)

    attribute_manager = ImapCdfAttributes()
    attribute_manager.add_instrument_global_attrs("mag")
    attribute_manager.add_instrument_variable_attrs("mag", "l1c")
    compression = xr.DataArray(
        np.arange(2),
        name="compression",
        dims=["compression"],
        attrs=attribute_manager.get_variable_attributes(
            "compression_attrs", check_schema=False
        ),
    )

    direction = xr.DataArray(
        np.arange(4),
        name="direction",
        dims=["direction"],
        attrs=attribute_manager.get_variable_attributes(
            "direction_attrs", check_schema=False
        ),
    )

    epoch_time = xr.DataArray(
        completed_timeline[:, 0],
        name="epoch",
        dims=["epoch"],
        attrs=attribute_manager.get_variable_attributes("epoch"),
    )

    direction_label = xr.DataArray(
        direction.values.astype(str),
        name="direction_label",
        dims=["direction_label"],
        attrs=attribute_manager.get_variable_attributes(
            "direction_label", check_schema=False
        ),
    )

    compression_label = xr.DataArray(
        compression.values.astype(str),
        name="compression_label",
        dims=["compression_label"],
        attrs=attribute_manager.get_variable_attributes(
            "compression_label", check_schema=False
        ),
    )
    global_attributes = attribute_manager.get_global_attributes(output_logical_source)
    # TODO merge missing sequences? replace?
    global_attributes["missing_sequences"] = ""

    try:
        active_dataset = normal_mode_dataset or burst_mode_dataset

        global_attributes["is_mago"] = active_dataset.attrs["is_mago"]
        global_attributes["is_active"] = active_dataset.attrs["is_active"]

        # Check if all vectors are primary in both normal and burst datasets
        is_mago = active_dataset.attrs.get("is_mago", "False") == "True"
        normal_all_primary = active_dataset.attrs.get("all_vectors_primary", False)

        # Default for missing burst dataset: 1 if MAGO (expected primary), 0 if MAGI
        burst_all_primary = is_mago
        if burst_mode_dataset is not None:
            burst_all_primary = burst_mode_dataset.attrs.get(
                "all_vectors_primary", False
            )

        # Both datasets must have all vectors primary for the combined result to be True
        global_attributes["all_vectors_primary"] = (
            normal_all_primary and burst_all_primary
        )

        global_attributes["missing_sequences"] = active_dataset.attrs[
            "missing_sequences"
        ]
    except KeyError as e:
        logger.info(
            f"Key error when assigning global attributes, attribute not found in "
            f"L1B file with logical source "
            f"{active_dataset.attrs['Logical_source']}: {e}"
        )

    global_attributes["interpolation_method"] = interp_function.name

    output_dataset = xr.Dataset(
        coords={
            "epoch": epoch_time,
            "direction": direction,
            "direction_label": direction_label,
            "compression": compression,
            "compression_label": compression_label,
        },
        attrs=global_attributes,
    )

    output_dataset["vectors"] = xr.DataArray(
        completed_timeline[:, 1:5],
        name="vectors",
        dims=["epoch", "direction"],
        attrs=attribute_manager.get_variable_attributes("vector_attrs"),
    )

    if len(output_dataset["vectors"]) > 0:
        output_dataset["vector_magnitude"] = xr.apply_ufunc(
            lambda x: np.linalg.norm(x[:4]),
            output_dataset["vectors"],
            input_core_dims=[["direction"]],
            output_core_dims=[[]],
            vectorize=True,
        )
        output_dataset[
            "vector_magnitude"
        ].attrs = attribute_manager.get_variable_attributes("vector_magnitude_attrs")
    else:
        output_dataset["vector_magnitude"] = xr.DataArray(
            np.empty((0, 1)),
            name="vector_magnitude",
            dims=["epoch", "vector_magnitude"],
            attrs=attribute_manager.get_variable_attributes("vector_magnitude_attrs"),
        )

    output_dataset["compression_flags"] = xr.DataArray(
        completed_timeline[:, 6:8],
        name="compression_flags",
        dims=["epoch", "compression"],
        attrs=attribute_manager.get_variable_attributes("compression_flags_attrs"),
    )

    output_dataset["generated_flag"] = xr.DataArray(
        completed_timeline[:, 5],
        name="generated_flag",
        dims=["epoch"],
        attrs=attribute_manager.get_variable_attributes("generated_flag_attrs"),
    )

    return output_dataset


def select_datasets(
    first_input_dataset: xr.Dataset, second_input_dataset: xr.Dataset | None = None
) -> tuple[xr.Dataset, xr.Dataset]:
    """
    Given one or two datasets, assign one to norm and one to burst.

    If only one dataset is provided, the other will be marked as None. If two are
    provided, they will be validated to ensure one is norm and one is burst.

    Parameters
    ----------
    first_input_dataset : xr.Dataset
        The first input dataset.
    second_input_dataset : xr.Dataset, optional
        The second input dataset.

    Returns
    -------
    tuple
        Tuple containing norm_mode_dataset, burst_mode_dataset.
    """
    normal_mode_dataset = None
    burst_mode_dataset = None

    input_logical_source_1 = first_input_dataset.attrs["Logical_source"]

    if isinstance(first_input_dataset.attrs["Logical_source"], list):
        input_logical_source_1 = first_input_dataset.attrs["Logical_source"][0]

    if "norm" in input_logical_source_1:
        normal_mode_dataset = first_input_dataset

    if "burst" in input_logical_source_1:
        burst_mode_dataset = first_input_dataset

    if second_input_dataset is None:
        logger.info(
            f"Only one input dataset provided with logical source "
            f"{input_logical_source_1}"
        )
    else:
        input_logical_source_2 = second_input_dataset.attrs["Logical_source"]
        if isinstance(second_input_dataset.attrs["Logical_source"], list):
            input_logical_source_2 = second_input_dataset.attrs["Logical_source"][0]

        if "burst" in input_logical_source_2:
            burst_mode_dataset = second_input_dataset

        elif "norm" in input_logical_source_2:
            normal_mode_dataset = second_input_dataset

        # If there are two inputs, one should be norm and one should be burst
        if normal_mode_dataset is None or burst_mode_dataset is None:
            raise RuntimeError(
                "L1C requires one normal mode and one burst mode input file."
            )

    return normal_mode_dataset, burst_mode_dataset


def process_mag_l1c(
    normal_mode_dataset: xr.Dataset | None,
    burst_mode_dataset: xr.Dataset,
    interpolation_function: InterpolationFunction,
    day_to_process: np.datetime64 | None = None,
) -> np.ndarray:
    """
    Create MAG L1C data from L1B datasets.

    This function starts from the normal mode dataset and completes the following steps:
    1. find all the gaps in the dataset
    2. generate a new timeline with the gaps filled, including new timestamps to fill
    out the rest of the day to +/- 15 minutes on either side
    3. fill the timeline with normal mode data (so, all the non-gap timestamps)
    4. interpolate the gaps using the burst mode data and the method specified in
        interpolation_function.

    It returns an (n, 8) shaped array:
    0 - epoch (timestamp)
    1-4 - vector x, y, z, and range
    5 - generated flag (0 for normal data, 1 for interpolated data, -1 for missing data)
    6-7 - compression flags (is_compressed, compression_width)

    Parameters
    ----------
    normal_mode_dataset : xarray.Dataset
        The normal mode dataset, which acts as a base for the output.
    burst_mode_dataset : xarray.Dataset
        The burst mode dataset, which is used to fill in the gaps in the normal mode.
    interpolation_function : InterpolationFunction
        The interpolation function to use to fill in the gaps.
    day_to_process : np.datetime64, optional
        The day to process, in np.datetime64[D] format. This is used to fill
        gaps at the beginning or end of the day if needed. If not included, these
        gaps will not be filled.

    Returns
    -------
    np.ndarray
        An (n, 8) shaped array containing the completed timeline.
    """
    day_start_ns = None
    day_end_ns = None

    if day_to_process is not None:
        day_start = day_to_process.astype("datetime64[s]") - np.timedelta64(30, "m")

        # get the end of the day plus 30 minutes
        day_end = (
            day_to_process.astype("datetime64[s]")
            + np.timedelta64(1, "D")
            + np.timedelta64(30, "m")
        )

        day_start_ns = et_to_ttj2000ns(str_to_et(str(day_start)))
        day_end_ns = et_to_ttj2000ns(str_to_et(str(day_end)))

    if normal_mode_dataset:
        norm_epoch = _to_int64_ns(normal_mode_dataset["epoch"].data)
        if "vectors_per_second" in normal_mode_dataset.attrs:
            normal_vecsec_dict = vectors_per_second_from_string(
                normal_mode_dataset.attrs["vectors_per_second"]
            )
        else:
            normal_vecsec_dict = None

        gaps = find_all_gaps(norm_epoch, normal_vecsec_dict, day_start_ns, day_end_ns)
        gap_fill_plans = build_gap_fill_plans(burst_mode_dataset, gaps)
        new_timeline = build_timeline_from_gap_plans(norm_epoch, gap_fill_plans)
        norm_filled: np.ndarray = fill_normal_data(normal_mode_dataset, new_timeline)
        interpolated = interpolate_gaps(
            burst_mode_dataset, gap_fill_plans, norm_filled, interpolation_function
        )
    else:
        norm_epoch = [day_start_ns, day_end_ns]
        gaps = np.array(
            [
                [
                    day_start_ns,
                    day_end_ns,
                    VecSec.TWO_VECS_PER_S.value,
                ]
            ]
        )
        new_timeline = generate_timeline(norm_epoch, gaps)
        norm_filled = generate_empty_norm_array(new_timeline)
        interpolated = interpolate_gaps(
            burst_mode_dataset, gaps, norm_filled, interpolation_function
        )

    return interpolated


def get_vecsec_dict(
    dataset: xr.Dataset | None, default_rate: int = VecSec.TWO_VECS_PER_S.value
) -> dict[int, int]:
    """Return a vectors-per-second mapping with a sensible default."""
    if dataset is None or "vectors_per_second" not in dataset.attrs:
        return {0: default_rate}

    return vectors_per_second_from_string(dataset.attrs["vectors_per_second"])


def get_rate_segment(
    epoch_data: np.ndarray, vecsec_dict: dict[int, int], target_time: int
) -> tuple[VecSec, int, int]:
    """Return the active observed-cadence segment containing target_time."""
    epoch_data = _to_int64_ns(epoch_data)
    if epoch_data.shape[0] == 0:
        raise ValueError("Cannot resolve a rate segment for an empty epoch array.")

    rate_segments = _find_rate_segments(epoch_data, vecsec_dict)
    if len(rate_segments) == 0:
        default_rate = VecSec(int(next(iter(vecsec_dict.values()))))
        return default_rate, 0, epoch_data.shape[0]

    target_index = int(np.searchsorted(epoch_data, target_time, side="left"))
    target_index = min(max(target_index, 0), epoch_data.shape[0] - 1)

    segment_index = 0
    for index, (segment_start, _) in enumerate(rate_segments):
        if segment_start > target_index:
            break
        segment_index = index

    segment_start, segment_rate = rate_segments[segment_index]
    segment_end = epoch_data.shape[0]
    if segment_index + 1 < len(rate_segments):
        segment_end = rate_segments[segment_index + 1][0]

    return VecSec(int(segment_rate)), int(segment_start), int(segment_end)


def find_nearest_epoch_index(epoch_data: np.ndarray, target_time: int) -> int:
    """Return the index of the timestamp nearest target_time, preferring earlier ties."""
    if epoch_data.shape[0] == 0:
        raise ValueError("Cannot find a nearest timestamp in an empty array.")

    index = int(np.searchsorted(epoch_data, target_time, side="left"))
    if index == epoch_data.shape[0]:
        return epoch_data.shape[0] - 1

    if index > 0 and abs(epoch_data[index - 1] - target_time) <= abs(
        epoch_data[index] - target_time
    ):
        return index - 1

    return index


def build_decimated_indices(
    anchor_index: int, step: int, start_index: int, end_index: int
) -> np.ndarray:
    """Build a decimated index sequence around anchor_index, including the anchor."""
    forward = np.arange(anchor_index, end_index, step, dtype=np.int64)
    backward = np.arange(anchor_index - step, start_index - 1, -step, dtype=np.int64)
    if backward.size == 0:
        return forward

    return np.concatenate((backward[::-1], forward))


def build_regular_gap_epochs(
    t_a: int,
    source_indices: np.ndarray,
    anchor_index: int,
    burst_rate: VecSec,
) -> np.ndarray:
    """Convert decimated BM indices into a zero-jitter synthetic NM sequence."""
    sample_spacing_ns = int(1e9 / burst_rate.value)
    relative_samples = source_indices - anchor_index
    return (t_a + relative_samples * sample_spacing_ns).astype(np.int64, copy=False)


def build_gap_fill_plan(
    burst_epochs: np.ndarray, burst_vecsec_dict: dict[int, int], gap: np.ndarray
) -> GapFillPlan:
    """
    Build a BM-driven plan for a single NM gap per 7.3.4 step 3.

    Select the BM time closest to tA, decimate around that anchor to the target NM
    cadence, shift the decimated BM timestamps by (tA - tC), and trim to the open
    interval (tA, tB).
    """
    burst_epochs = _to_int64_ns(burst_epochs)
    t_a = _to_int64_ns_scalar(gap[0])
    t_b = _to_int64_ns_scalar(gap[1])
    norm_rate = VecSec(int(gap[2]))
    burst_rate, segment_start, segment_end = get_rate_segment(
        burst_epochs, burst_vecsec_dict, t_a
    )
    if burst_rate.value % norm_rate.value != 0:
        raise ValueError(
            f"Burst rate {burst_rate.value} must be divisible by normal rate "
            f"{norm_rate.value}."
        )

    t_c_index = segment_start + find_nearest_epoch_index(
        burst_epochs[segment_start:segment_end], t_a
    )
    decimation_factor = int(burst_rate.value / norm_rate.value)
    source_indices = build_decimated_indices(
        t_c_index, decimation_factor, segment_start, segment_end
    )
    synthetic_epochs = build_regular_gap_epochs(
        t_a, source_indices, t_c_index, burst_rate
    )

    keep = (
        (synthetic_epochs > t_a)
        & (synthetic_epochs < t_b)
        & (synthetic_epochs >= burst_epochs[segment_start])
        & (synthetic_epochs <= burst_epochs[segment_end - 1])
    )
    synthetic_epochs = synthetic_epochs[keep]
    source_indices = source_indices[keep]

    required_seconds = (1 / norm_rate.value) * 2
    burst_buffer = int(required_seconds * burst_rate.value)
    if source_indices.size == 0:
        burst_window_start = max(segment_start, t_c_index - burst_buffer)
        burst_window_end = min(segment_end, t_c_index + burst_buffer + 1)
    else:
        burst_window_start = max(segment_start, int(source_indices[0]) - burst_buffer)
        burst_window_end = min(segment_end, int(source_indices[-1]) + burst_buffer + 1)

    return GapFillPlan(
        t_a=t_a,
        t_b=t_b,
        norm_rate=norm_rate,
        burst_rate=burst_rate,
        synthetic_epochs=synthetic_epochs,
        source_indices=source_indices.astype(np.int64, copy=False),
        burst_window_start=burst_window_start,
        burst_window_end=burst_window_end,
    )


def build_gap_fill_plans(
    burst_dataset: xr.Dataset, gaps: np.ndarray
) -> list[GapFillPlan]:
    """Build BM-driven fill plans for each detected NM gap."""
    if gaps.shape[0] == 0:
        return []

    burst_epochs = _to_int64_ns(burst_dataset["epoch"].data)
    burst_vecsec_dict = get_vecsec_dict(burst_dataset)
    return [build_gap_fill_plan(burst_epochs, burst_vecsec_dict, gap) for gap in gaps]


def build_timeline_from_gap_plans(
    epoch_data: np.ndarray, gap_fill_plans: list[GapFillPlan]
) -> np.ndarray:
    """Insert spec-derived synthetic epochs into the NM timeline."""
    epoch_data = _to_int64_ns(epoch_data)
    if len(gap_fill_plans) == 0:
        return epoch_data.copy()

    timeline_parts: list[np.ndarray] = []
    last_index = 0
    for gap_fill_plan in gap_fill_plans:
        epoch_start_index = np.searchsorted(epoch_data, gap_fill_plan.t_a, side="left")
        gap_start_in_epoch = (
            epoch_start_index < epoch_data.shape[0]
            and epoch_data[epoch_start_index] == gap_fill_plan.t_a
        )
        epoch_copy_end = epoch_start_index + int(gap_start_in_epoch)
        timeline_parts.append(epoch_data[last_index:epoch_copy_end])
        if gap_fill_plan.synthetic_epochs.size != 0:
            timeline_parts.append(gap_fill_plan.synthetic_epochs)
        last_index = int(np.searchsorted(epoch_data, gap_fill_plan.t_b, side="left"))

    timeline_parts.append(epoch_data[last_index:])
    return np.concatenate(timeline_parts)


def build_default_gap_fill_plans(
    burst_dataset: xr.Dataset,
    gaps: np.ndarray,
    filled_norm_timeline: np.ndarray,
) -> list[GapFillPlan]:
    """
    Build simple plans from the existing NM scaffold timeline.

    This preserves the current no-NM fallback behavior used by T024 while the
    normal+burst path uses the spec-compliant BM shift/trim algorithm.
    """
    burst_epochs = _to_int64_ns(burst_dataset["epoch"].data)
    norm_epochs = _to_int64_ns(filled_norm_timeline[:, 0])
    burst_vecsec_dict = get_vecsec_dict(burst_dataset)
    gap_fill_plans = []

    for gap in gaps:
        t_a = _to_int64_ns_scalar(gap[0])
        t_b = _to_int64_ns_scalar(gap[1])
        norm_rate = VecSec(int(gap[2]))
        burst_rate, segment_start, segment_end = get_rate_segment(
            burst_epochs, burst_vecsec_dict, t_a
        )
        synthetic_epochs = norm_epochs[(norm_epochs > gap[0]) & (norm_epochs < gap[1])]
        synthetic_epochs = synthetic_epochs.astype(np.int64, copy=False)
        output_cadence_ns = int(1e9 / norm_rate.value)
        synthetic_epochs = synthetic_epochs[
            (synthetic_epochs > burst_epochs[segment_start])
            & (synthetic_epochs <= burst_epochs[segment_end - 1] - output_cadence_ns)
        ]

        required_seconds = (1 / norm_rate.value) * 2
        burst_buffer = int(required_seconds * burst_rate.value)
        if synthetic_epochs.size == 0:
            burst_gap_start = find_nearest_epoch_index(burst_epochs, t_a)
            burst_gap_end = find_nearest_epoch_index(burst_epochs, t_b)
            source_indices = np.empty((0,), dtype=np.int64)
            burst_window_start = max(segment_start, burst_gap_start - burst_buffer)
            burst_window_end = min(segment_end, burst_gap_end + burst_buffer + 1)
        else:
            source_indices = np.array(
                [
                    segment_start
                    + find_nearest_epoch_index(
                        burst_epochs[segment_start:segment_end], int(timestamp)
                    )
                    for timestamp in synthetic_epochs
                ],
                dtype=np.int64,
            )
            burst_window_start = max(segment_start, int(source_indices[0]) - burst_buffer)
            burst_window_end = min(
                segment_end, int(source_indices[-1]) + burst_buffer + 1
            )

        gap_fill_plans.append(
            GapFillPlan(
                t_a=t_a,
                t_b=t_b,
                norm_rate=norm_rate,
                burst_rate=burst_rate,
                synthetic_epochs=synthetic_epochs,
                source_indices=source_indices,
                burst_window_start=burst_window_start,
                burst_window_end=burst_window_end,
            )
        )

    return gap_fill_plans


def generate_empty_norm_array(new_timeline: np.ndarray) -> np.ndarray:
    """
    Generate an empty Normal mode array with the new timeline.

    Parameters
    ----------
    new_timeline : np.ndarray
        A 1D array of timestamps to fill.

    Returns
    -------
    np.ndarray
        An (n, 8) shaped array containing the timeline filled with `FILLVAL` data.
    """
    # TODO: fill with FILLVAL
    norm_filled: np.ndarray = np.zeros((len(new_timeline), 8))
    norm_filled[:, 0] = new_timeline
    # Flags, will also indicate any missed timestamps
    norm_filled[:, 5] = ModeFlags.MISSING.value

    return norm_filled


def fill_normal_data(
    normal_dataset: xr.Dataset,
    new_timeline: np.ndarray | None = None,
) -> np.ndarray:
    """
    Fill the new timeline with the normal mode data.

    If the timestamp exists in the normal mode data, it will be filled in the output.

    Parameters
    ----------
    normal_dataset : xr.Dataset
        The normal mode dataset.
    new_timeline : np.ndarray, optional
        A 1D array of timestamps to fill. If not provided, the normal mode timestamps
        will be used.

    Returns
    -------
    filled_timeline : np.ndarray
        An (n, 8) shaped array containing the timeline filled with normal mode data.
        Gaps are marked as -1 in the generated flag column at index 5.
        Indices: 0 - epoch, 1-4 - vector x, y, z, and range, 5 - generated flag,
        6-7 - compression flags.
    """
    if new_timeline is None:
        new_timeline = _to_int64_ns(normal_dataset["epoch"].data)
    else:
        new_timeline = _to_int64_ns(new_timeline)

    filled_timeline = generate_empty_norm_array(new_timeline)
    normal_epochs = _to_int64_ns(normal_dataset["epoch"].data)

    for index, timestamp in enumerate(normal_epochs):
        timeline_index = np.searchsorted(new_timeline, timestamp)
        filled_timeline[timeline_index, 1:5] = normal_dataset["vectors"].data[index]
        filled_timeline[timeline_index, 5] = ModeFlags.NORM.value
        filled_timeline[timeline_index, 6:8] = normal_dataset["compression_flags"].data[
            index
        ]

    return filled_timeline


def interpolate_gaps(
    burst_dataset: xr.Dataset,
    gap_data: list[GapFillPlan] | np.ndarray,
    filled_norm_timeline: np.ndarray,
    interpolation_function: InterpolationFunction,
) -> np.ndarray:
    """
    Interpolate the gaps in the filled timeline using the burst mode data.

    Returns an array that matches the format of filled_norm_timeline, with gaps filled
    using interpolated burst data.

    Parameters
    ----------
    burst_dataset : xarray.Dataset
        The L1B burst mode dataset.
    gap_data : list[GapFillPlan] | numpy.ndarray
        A list of spec-derived gap plans, or legacy gap tuples for the no-NM fallback.
    filled_norm_timeline : numpy.ndarray
        Timeline filled with normal mode data in the shape (n, 8).
    interpolation_function : InterpolationFunction
        The interpolation function to use to fill in the gaps.

    Returns
    -------
    numpy.ndarray
        An array of shape (n, 8) containing the fully filled timeline.
        Indices: 0 - epoch, 1-4 - vector x, y, z, and range, 5 - generated flag,
        6-7 - compression flags.
    """
    burst_epochs = _to_int64_ns(burst_dataset["epoch"].data)
    norm_epochs = _to_int64_ns(filled_norm_timeline[:, 0])
    burst_vectors = burst_dataset["vectors"].data
    if isinstance(gap_data, np.ndarray):
        gap_fill_plans = build_default_gap_fill_plans(
            burst_dataset, gap_data, filled_norm_timeline
        )
    else:
        gap_fill_plans = gap_data

    for gap_fill_plan in gap_fill_plans:
        gap_timeline = gap_fill_plan.synthetic_epochs
        if gap_timeline.size == 0:
            continue

        short = (gap_timeline >= burst_epochs[gap_fill_plan.burst_window_start]) & (
            gap_timeline <= burst_epochs[gap_fill_plan.burst_window_end - 1]
        )
        if not np.any(short):
            continue

        gap_timeline = gap_timeline[short]
        gap_source_indices = gap_fill_plan.source_indices[short]
        adjusted_gap_timeline, gap_fill = interpolation_function(
            burst_vectors[
                gap_fill_plan.burst_window_start : gap_fill_plan.burst_window_end, :3
            ],
            burst_epochs[
                gap_fill_plan.burst_window_start : gap_fill_plan.burst_window_end
            ],
            gap_timeline,
            input_rate=gap_fill_plan.burst_rate,
            output_rate=gap_fill_plan.norm_rate,
        )
        adjusted_gap_timeline = adjusted_gap_timeline.astype(np.int64, copy=False)
        source_lookup = {
            int(timestamp): int(source_index)
            for timestamp, source_index in zip(gap_timeline, gap_source_indices)
        }
        missing_timestamps = {int(timestamp) for timestamp in gap_timeline}

        for index, timestamp in enumerate(adjusted_gap_timeline):
            source_index = source_lookup[int(timestamp)]
            timeline_index = np.searchsorted(norm_epochs, timestamp)
            if filled_norm_timeline[timeline_index, 5] == ModeFlags.NORM.value:
                raise RuntimeError(
                    "Self-inconsistent data. Interpolated timestamps should not "
                    "overwrite normal-mode samples."
                )

            filled_norm_timeline[timeline_index, 1:4] = gap_fill[index]
            filled_norm_timeline[timeline_index, 4] = burst_vectors[source_index, 3]
            filled_norm_timeline[timeline_index, 5] = ModeFlags.BURST.value
            filled_norm_timeline[timeline_index, 6:8] = burst_dataset[
                "compression_flags"
            ].data[source_index]
            missing_timestamps.discard(int(timestamp))

        for timestamp in missing_timestamps:
            timeline_index = np.searchsorted(norm_epochs, timestamp)
            if filled_norm_timeline[timeline_index, 5] != ModeFlags.MISSING.value:
                raise RuntimeError(
                    "Self-inconsistent data. "
                    "Gaps not included in final timeline should be missing."
                )

    return filled_norm_timeline


def generate_timeline(epoch_data: np.ndarray, gaps: np.ndarray) -> np.ndarray:
    """
    Generate a new timeline from existing, gap-filled timeline and gaps.

    The gaps are generated at the cadence implied by the gap rate. If no rate is
    provided, a default cadence of 0.5 seconds is used.

    Parameters
    ----------
    epoch_data : numpy.ndarray
        The existing timeline data, in the shape (n,).
    gaps : numpy.ndarray
        An array of gaps to fill, with shape (n, 2) where n is the number of gaps.
        The gap is specified as (start, end).

    Returns
    -------
    numpy.ndarray
        The new timeline, filled with the existing data and the generated gaps.
    """
    epoch_data = _to_int64_ns(epoch_data)
    full_timeline: np.ndarray = np.array([], dtype=epoch_data.dtype)
    last_index = 0
    for gap in gaps:
        epoch_start_index = np.searchsorted(epoch_data, gap[0], side="left")
        gap_start_in_epoch = (
            epoch_start_index < epoch_data.shape[0]
            and epoch_data[epoch_start_index] == _to_int64_ns_scalar(gap[0])
        )
        epoch_copy_end = epoch_start_index + int(gap_start_in_epoch)
        full_timeline = np.concatenate(
            (full_timeline, epoch_data[last_index:epoch_copy_end])
        )
        generated_timestamps = generate_missing_timestamps(gap)
        if gap_start_in_epoch and generated_timestamps.size != 0:
            generated_timestamps = generated_timestamps[1:]
        if generated_timestamps.size == 0:
            last_index = int(np.searchsorted(epoch_data, gap[1], side="left"))
            continue

        # Remove any generated timestamps that are already in the timeline
        # Use np.isin to check for exact matches
        mask = ~np.isin(generated_timestamps, full_timeline)
        generated_timestamps = generated_timestamps[mask]

        if generated_timestamps.size == 0:
            print("All generated timestamps already exist in timeline")
            continue

        full_timeline = np.concatenate((full_timeline, generated_timestamps))
        last_index = int(np.searchsorted(epoch_data, gap[1], side="left"))

    full_timeline = np.concatenate((full_timeline, epoch_data[last_index:]))

    return full_timeline


def find_all_gaps(
    epoch_data: np.ndarray,
    vecsec_dict: dict | None = None,
    start_of_day_ns: float | None = None,
    end_of_day_ns: float | None = None,
) -> np.ndarray:
    """
    Find all the gaps in the epoch data.

    If vectors_per_second_attr is provided, it will be used to find the gaps. Otherwise,
    it will assume a nominal 1/2 second gap. A gap is defined as missing data from the
    expected sequence as defined by vectors_per_second_attr.

    If start_of_day_ns and end_of_day_ns are provided, gaps at the beginning and end of
    the day will be added if the epoch_data does not cover the full day.

    Parameters
    ----------
    epoch_data : numpy.ndarray
        The epoch data to find gaps in.
    vecsec_dict : dict, optional
        A dictionary of the form {start: vecsec, start: vecsec} where start is the time
        in nanoseconds and vecsec is the number of vectors per second. This will be
        used to find the gaps. If not provided, a 1/2 second gap is assumed.
    start_of_day_ns : float, optional
        The start of the day in nanoseconds since TTJ2000. If provided, a gap will be
        added from this time to the first epoch if they don't match.
    end_of_day_ns : float, optional
        The end of the day in nanoseconds since TTJ2000. If provided, a gap will be
        added from the last epoch to this time if they don't match.

    Returns
    -------
    numpy.ndarray
        An array of gaps with shape (n, 3) where n is the number of gaps. The gaps are
        specified as (start, end, vector_rate) where start and end both exist in the
        timeline.
    """
    epoch_data = _to_int64_ns(epoch_data)
    gaps: np.ndarray = np.empty((0, 3), dtype=np.int64)

    # TODO: when we go back to the previous file, also retrieve expected
    #  vectors per second

    vecsec_dict = {0: VecSec.TWO_VECS_PER_S.value} | (vecsec_dict or {})

    rate_segments = _find_rate_segments(epoch_data, vecsec_dict)
    if rate_segments:
        first_rate = rate_segments[0][1]
        last_rate = rate_segments[-1][1]
    else:
        default_rate = next(iter(vecsec_dict.values()))
        first_rate = default_rate
        last_rate = default_rate

    if start_of_day_ns is not None and epoch_data[0] > start_of_day_ns:
        # Add a gap from the start of the day to the first timestamp
        gaps = np.concatenate(
            (
                gaps,
                np.array([[start_of_day_ns, epoch_data[0], first_rate]], dtype=np.int64),
            )
        )

    for index, (start_index, vectors_per_second) in enumerate(rate_segments):
        next_start_index = (
            rate_segments[index + 1][0]
            if index + 1 < len(rate_segments)
            else epoch_data.shape[0] - 1
        )
        epoch_slice = epoch_data[start_index : next_start_index + 1]
        gaps = np.concatenate((gaps, find_gaps(epoch_slice, vectors_per_second)))

    if end_of_day_ns is not None and epoch_data[-1] < end_of_day_ns:
        gaps = np.concatenate(
            (
                gaps,
                np.array([[epoch_data[-1], end_of_day_ns, last_rate]], dtype=np.int64),
            )
        )

    return gaps


def find_gaps(timeline_data: np.ndarray, vectors_per_second: int) -> np.ndarray:
    """
    Find gaps in timeline_data that are larger than 1/vectors_per_second.

    Returns timestamps (start_gap, end_gap, vectors_per_second) where startgap and
    endgap both exist in timeline data.

    Parameters
    ----------
    timeline_data : numpy.ndarray
        Array of timestamps.
    vectors_per_second : int
        Number of vectors expected per second.

    Returns
    -------
    numpy.ndarray
        Array of timestamps of shape (n, 3) containing n gaps with start_gap and
        end_gap, as well as vectors_per_second. Start_gap and end_gap both correspond
        to points in timeline_data.
    """
    if timeline_data.shape[0] < 2:
        return np.empty((0, 3), dtype=np.int64)

    # Expected difference between timestamps in nanoseconds.
    expected_gap = 1 / vectors_per_second * 1e9

    diffs = abs(np.diff(timeline_data))

    # Gap can be up to 7.5% larger than expected vectors per second due to clock drift
    gap_index = np.asarray(diffs - expected_gap > expected_gap * GAP_TOLERANCE).nonzero()[
        0
    ]
    output: np.ndarray = np.zeros((len(gap_index), 3), dtype=np.int64)

    for index, gap in enumerate(gap_index):
        output[index, :] = [
            timeline_data[gap],
            timeline_data[gap + 1],
            vectors_per_second,
        ]

    return output


def generate_missing_timestamps(gap: np.ndarray) -> np.ndarray:
    """
    Generate a new timeline from input gaps.

    Any gaps specified in gaps will be filled with timestamps at the gap rate. If the
    gap rate is not included, the default cadence is 0.5 seconds.

    Parameters
    ----------
    gap : numpy.ndarray
        Array of timestamps of shape (2,) containing n gaps with start_gap and
        end_gap. Start_gap and end_gap both correspond to points in timeline_data and
        are included in the output timespan.

    Returns
    -------
    full_timeline: numpy.ndarray
        Completed timeline.
    """
    difference_ns = int(0.5 * 1e9)
    if gap.shape[0] > 2:
        difference_ns = int(1e9 / int(gap[2]))

    output: np.ndarray = np.arange(
        _to_int64_ns_scalar(gap[0]),
        _to_int64_ns_scalar(gap[1]),
        difference_ns,
        dtype=np.int64,
    )
    return output


def _is_expected_rate(timestamp_difference: float, vectors_per_second: int) -> bool:
    """Return True when a timestamp spacing matches a rate within tolerance."""
    expected_gap = 1 / vectors_per_second * 1e9
    return abs(timestamp_difference - expected_gap) <= expected_gap * GAP_TOLERANCE


def _find_rate_segments(
    epoch_data: np.ndarray, vecsec_dict: dict[int, int]
) -> list[tuple[int, int]]:
    """
    Build contiguous rate segments using observed cadence near each transition.

    The validation data can switch cadence before the corresponding configuration time
    appears in the metadata. Walk each configured transition backward while the observed
    cadence already matches the new rate so gaps stay attached to the correct segment.
    """
    if epoch_data.shape[0] == 0:
        return []

    segments: list[tuple[int, int]] = []
    for start_time, vectors_per_second in sorted(vecsec_dict.items()):
        start_index = int(np.searchsorted(epoch_data, start_time, side="left"))
        start_index = min(start_index, epoch_data.shape[0] - 1)
        lower_bound = segments[-1][0] if segments else 0

        while start_index > lower_bound and _is_expected_rate(
            epoch_data[start_index] - epoch_data[start_index - 1], vectors_per_second
        ):
            start_index -= 1

        if segments and start_index == segments[-1][0]:
            segments[-1] = (start_index, vectors_per_second)
        else:
            segments.append((start_index, vectors_per_second))

    return segments


def vectors_per_second_from_string(vecsec_string: str) -> dict:
    """
    Extract the vectors per second from a string into a dictionary.

    Dictionary format: {start_time: vecsec, start_time: vecsec}.

    Parameters
    ----------
    vecsec_string : str
        A string of the form "start:vecsec,start:vecsec" where start is the time in
        nanoseconds and vecsec is the number of vectors per second.

    Returns
    -------
    dict
        A dictionary of the form {start_time: vecsec, start_time: vecsec}.
    """
    vecsec_dict = {}
    vecsec_segments = vecsec_string.split(",")
    for vecsec_segment in vecsec_segments:
        if vecsec_segment:
            start_time, vecsec = vecsec_segment.split(":")
            vecsec_dict[int(start_time)] = int(vecsec)

    return vecsec_dict


def remove_missing_data(filled_timeline: np.ndarray) -> np.ndarray:
    """
    Remove timestamps with no data from the filled timeline.

    Anywhere that the generated flag is equal to -1, the data will be removed.

    Parameters
    ----------
    filled_timeline : numpy.ndarray
        An (n, 8) shaped array containing the filled timeline.
        Indices: 0 - epoch, 1-4 - vector x, y, z, and range, 5 - generated flag,
        6-7 - compression flags.

    Returns
    -------
    cleaned_array : numpy.ndarray
        The filled timeline with missing data removed.
    """
    cleaned_array: np.ndarray = filled_timeline[filled_timeline[:, 5] != -1]
    return cleaned_array
