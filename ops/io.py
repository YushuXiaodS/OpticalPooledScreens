import numpy as np
import os
import pandas as pd
import six
import struct
import warnings
import fire

from . import utils
import ops.constants
import tifffile
from .paper.cell_idr import QuitError

imagej_description = "".join(
    [
        "ImageJ=1.49v\nimages=%d\nchannels=%d\nslices=%d",
        "\nframes=%d\nhyperstack=true\nmode=composite",
        "\nunit=\\u00B5m\nspacing=8.0\nloop=false\n",
        "min=764.0\nmax=38220.0\n",
    ]
)

ramp = list(range(256))
ZERO = [0] * 256
RED = ramp + ZERO + ZERO
GREEN = ZERO + ramp + ZERO
BLUE = ZERO + ZERO + ramp
MAGENTA = ramp + ZERO + ramp
GRAY = ramp + ramp + ramp
CYAN = ZERO + ramp + ramp

DEFAULT_LUTS = GRAY, GREEN, RED, MAGENTA, CYAN, GRAY, GRAY


def read_lut(lut_string):
    return pd.read_csv(
        six.StringIO(lut_string), sep="\s+", header=None
    ).values.T.flatten()


GLASBEY = read_lut(ops.constants.GLASBEY_INVERTED)


def grid_view(files, bounds, padding=40, with_mask=False):
    """Bounds are given as [(i0, j0, i1, j1)].
    Mask is 1-indexed, zero indicates background.
    """
    padding = int(padding)
    bounds = np.round(bounds).astype(int)

    arr = []
    Is = {}
    for filename, bounds_ in zip(files, bounds):
        try:
            I = Is[filename]
        except KeyError:
            I = read_stack(filename, copy=False)
            Is[filename] = I
        I_cell = utils.subimage(I, bounds_, pad=padding)
        arr.append(I_cell.copy())

    if with_mask:
        arr_m = []
        for i, (i0, j0, i1, j1) in enumerate(bounds):
            shape = i1 - i0 + padding, j1 - j0 + padding
            img = np.zeros(shape, dtype=np.uint16) + i + 1
            arr_m += [img]
        return utils.pile(arr), utils.pile(arr_m)

    return utils.pile(arr)


@utils.memoize(active=False)
def read_stack(filename, copy=True):
    """Read a .tif file into a numpy array, with optional memory mapping."""
    with tifffile.TiffFile(filename, is_mmstack=False) as tif:
        data = tif.asarray()

    # preserve inner singleton dimensions
    while data.ndim > 2 and data.shape[0] == 1:
        data = np.squeeze(data, axis=0)

    if copy:
        data = data.copy()
    return data


def normalize_dtype(data):
    """Convert array data to an ImageJ-compatible dtype.

    Supported output dtypes: uint8, uint16, float32.

    Conversions:
        bool    -> uint8 (0, 255)
        int32   -> uint16 (if in range) or raises ValueError
        int64   -> uint16 (if in range) or float32
        float64 -> float32
    """
    if isinstance(data, list):
        data = np.array(data)

    if data.dtype == np.bool_:
        return data.astype(np.uint8) * 255

    if data.dtype == np.int64:
        if (data >= 0).all() and (data < 2**16).all():
            return data.astype(np.uint16)
        data = data.astype(np.float32)
        print("Cast int64 to float32")
        return data

    if data.dtype == np.float64:
        return data.astype(np.float32)

    if data.dtype == np.int32:
        if data.min() >= 0 and data.max() < 2**16:
            return data.astype(np.uint16)
        raise ValueError(
            f"Cannot cast int32 to uint16: data range [{data.min()}, {data.max()}] "
            f"exceeds [0, {2**16 - 1}]"
        )

    if data.dtype not in (np.uint8, np.uint16, np.float32):
        raise ValueError(f"Cannot save data of type {data.dtype}")

    return data


def save_stack(
    name,
    data,
    luts=None,
    display_ranges=None,
    resolution=1.0,
    compress=0,
    dimensions=None,
    display_mode="composite",
):
    """Save an array as an ImageJ-compatible TIFF hyperstack.

    Parameters
    ----------
    name : str
        Output filename. '.tif' is appended if not present.
    data : array_like
        Array with 2–5 dimensions, ordered [T, Z, C, Y, X].
    luts : sequence of array, optional
        ImageJ lookup tables per channel (e.g. GREEN, RED, BLUE).
    display_ranges : sequence of (min, max), optional
        Display range per channel.
    resolution : float
        Microns per pixel. Stored as its reciprocal (pixels per micron).
    compress : int
        Compression level. 1 saves significant space for integer masks.
    dimensions : str, optional
        Dimension labels for leading axes, e.g. "TZC". Inferred if None.
    display_mode : str
        ImageJ display mode ("composite", "color", "grayscale").

    Supported input dtypes
    ----------------------
    bool (-> uint8 0/255), uint8, uint16, int32 (-> uint16),
    int64 (-> uint16 or float32), float32, float64 (-> float32).

    Examples
    --------
    >>> random_data = np.random.randint(0, 2**16, size=(3, 100, 100), dtype=np.uint16)
    >>> luts = ops.io.GREEN, ops.io.RED, ops.io.BLUE
    >>> display_ranges = (0, 60000), (0, 40000), (0, 20000)
    >>> save_stack('random_image', random_data, luts=luts, display_ranges=display_ranges)
    """
    if not name.endswith(".tif"):
        name += ".tif"
    name = os.path.abspath(name)

    if isinstance(data, list):
        data = np.array(data)

    if not (2 <= data.ndim <= 5):
        raise ValueError(f"Input has shape {data.shape}, but ndim must be in [2, 5]")

    data = normalize_dtype(data)

    resolution_tag = (1 / resolution, 1 / resolution)

    out_dir = os.path.dirname(name)
    if out_dir and not os.path.isdir(out_dir):
        os.makedirs(out_dir, exist_ok=True)

    compression = "zlib" if compress else None

    if data.ndim == 2:
        lo, hi = single_contrast(data, display_ranges)
        metadata = {"min": lo, "max": hi}
        tifffile.imwrite(
            name,
            data,
            imagej=True,
            photometric="minisblack",
            resolution=resolution_tag,
            compression=compression,
            metadata=metadata,
        )
        return

    # --- hyperstack (ndim 3–5) ---
    luts, display_ranges = infer_luts_display_ranges(data, luts, display_ranges)

    leading_shape = data.shape[:-2]
    if dimensions is None:
        dimensions = "TZC"[::-1][: len(leading_shape)][::-1]

    # Build the metadata dict consumed by modern tifffile
    metadata = {"mode": display_mode}
    ij_metadata = {}

    if display_ranges is not None:
        # tifffile expects a flat sequence: [min0, max0, min1, max1, ...]
        ij_metadata["Ranges"] = [[float(v) for pair in display_ranges for v in pair]]

    if luts is not None:
        # Each LUT should be a (3, 256) uint8 array
        ij_metadata["LUTs"] = [
            np.asarray(lut, dtype=np.uint8).reshape(3, 256) for lut in luts
        ]

    # Map dimension letters to the metadata keys tifffile recognizes
    dim_key_map = {"T": "frames", "Z": "slices", "C": "channels"}
    hyperstack_shape = {}
    for dim_char, size in zip(dimensions, leading_shape):
        key = dim_key_map.get(dim_char)
        if key:
            hyperstack_shape[key] = size

    metadata.update(hyperstack_shape)
    metadata["hyperstack"] = True
    ij_tags = tifffile.imagej_metadata_tag(ij_metadata, "<")
    tifffile.imwrite(
        name,
        data,
        imagej=True,
        photometric="minisblack",
        resolution=resolution_tag,
        resolutionunit=1,
        compression=compression,
        metadata=metadata,
        extratags=ij_tags,
    )


def format_input(input_table, n_jobs=1, channel_order=None, **kwargs):
    """Formats filenames and concatenates images from the same field of view
    and cycle of imaging if necessary (e.g., when individual channels are split
    between tiff file outputs of microscope control software). See example
    input_file.xlsx table. Also saves `input/well_tile_list.csv` to use as input
    for a snakemake workflow.

    Parameters
    ----------
    input_table : str, path object, or file-like object
        Path to table defining how files should be formatted. Can be any valid
        input to `pd.read_excel` or `pd.read_csv`.

    channel_order : dict of channel name -> channel order, optional
        If not provided, default is for standard ordering of SBS channels. Can
        be used to provide custom ordering of channels.

    n_jobs : int, default 1
        Number of parallelized processes to use for formatting files.

    Other Parameters
    ----------------
    **kwargs
        Keyword arguments passed to `joblib.Parallel` for parallelized
        processing.

    """
    if input_table.endswith("xlsx"):
        df = pd.read_excel(input_table, engine="openpyxl").drop_duplicates()
    elif input_table.endswith("xls"):
        df = pd.read_excel(input_table, engine="xlrd").drop_duplicates()
    else:
        df = pd.read_csv(input_table).drop_duplicates()

    if channel_order is None:
        channel_order = {"ALL": 0, "DAPI": 1, "G": 2, "T": 3, "A": 4, "C": 5}

    df["channel_order"] = df["channel"].map(channel_order)

    # check for unknwon channel names
    unknown_channels = df["channel"][df["channel_order"].isna()].pipe(set)

    if len(unknown_channels) > 0:
        raise ValueError(
            f'Channel(s) {", ".join(unknown_channels)} '
            "not recognized. Custom channels and channel orders can be"
            ' defined using the "channel_order" argument'
        )

    # check for missing channel files when concatenating channels
    if df.query('channel!="ALL"').pipe(len) > 0:
        # same number of channels for every image within a cycle
        # (may be different across cycles)
        missing = (
            df.groupby("cycle")["channel"]
            .value_counts()
            .groupby("cycle")
            .nunique()
            .loc[lambda x: x != 1]
            .index.astype(str)
        )
        if len(missing) > 0:
            raise ValueError(
                "Varying number of channels per site in "
                f'cycle(s) {", ".join(missing)}, likely missing entry '
                "to input table."
            )

    def process_site(output_file, df_input):
        stacked = np.array(
            [
                read_stack(input_file)
                for input_file in df_input.sort_values("channel_order")[
                    "original filename"
                ]
            ]
        )
        save_stack(output_file, stacked)

    if n_jobs != 1:
        from joblib import Parallel, delayed

        Parallel(n_jobs=n_jobs, **kwargs)(
            delayed(process_site)(output_file, df_input)
            for output_file, df_input in df.groupby("snakemake filename")
        )
    else:
        for output_file, df_input in df.groupby("snakemake filename"):
            process_site(output_file, df_input)

    df[["well", "tile"]].drop_duplicates().to_csv(
        "input/well_tile_list.csv", index=False
    )


def infer_luts_display_ranges(data, luts, display_ranges):
    """Deal with user input."""
    nchannels = data.shape[-3]
    if luts is None:
        luts = DEFAULT_LUTS + (GRAY,) * (nchannels - len(DEFAULT_LUTS))

    if display_ranges is None:
        display_ranges = [None] * nchannels

    for i, dr in enumerate(display_ranges):
        if dr is None:
            x = data[..., i, :, :]
            display_ranges[i] = x.min(), x.max()
    if len(luts) < nchannels or len(display_ranges) < nchannels:
        error = "Must provide at least {} luts and display ranges"
        raise IndexError(error.format(nchannels))
    else:
        luts = luts[:nchannels]
        display_ranges = display_ranges[:nchannels]

    return luts, display_ranges


def single_contrast(data, display_ranges):
    try:
        min, max = np.array(display_ranges).flat[:2]
    except ValueError:
        min, max = data.min(), data.max()
    return min, max


def imagej_description_2D(min, max):
    return "ImageJ=1.49v\nimages=1\nmin={min}\nmax={max}".format(min=min, max=max)


def imagej_description(
    leading_shape, leading_axes, contrast=None, display_mode="composite"
):
    if len(leading_shape) != len(leading_axes):
        error = "mismatched axes, shape is {} but axis labels are {}"
        raise ValueError(error.format(leading_shape, leading_axes))

    prefix = "ImageJ=1.49v\n"
    suffix = "hyperstack=true\nmode={mode}\n".format(mode=display_mode)
    images = np.prod(leading_shape)
    sizes = {k: v for k, v in zip(leading_axes, leading_shape)}
    description = prefix + "images={}\n".format(images)
    if "C" in sizes:
        description += "channels={}\n".format(sizes["C"])
    if "Z" in sizes:
        description += "slices={}\n".format(sizes["Z"])
    if "T" in sizes:
        description += "frames={}\n".format(sizes["T"])
    if contrast is not None:
        min, max = contrast
        description += "min={0}\nmax={1}\n".format(min, max)
    description += suffix

    return description


def ij_tag_50838(nchannels):
    """ImageJ uses tag 50838 to indicate size of metadata elements (e.g., 768 bytes per ROI)
    Parameter `nchannels` must accurately describe the length of the display ranges and LUTs,
    or the values imported into LUTs will be shifted in memory.
    :param nchannels:
    :return:
    """
    info_block = (20,)  # summary of metadata fields
    display_block = (16 * nchannels,)  # display range block
    luts_block = (256 * 3,) * nchannels  #
    return info_block + display_block + luts_block


def ij_tag_50839(luts, display_ranges):
    """ImageJ uses tag 50839 to store metadata. Only range and luts are implemented here.
    :param tuple luts: tuple of 256*3=768 8-bit ints specifying RGB, e.g., constants io.RED etc.
    :param tuple display_ranges: tuple of (min, max) pairs for each channel
    :return:
    """
    d = struct.pack(
        "<" + "d" * len(display_ranges) * 2, *[y for x in display_ranges for y in x]
    )
    # insert display ranges
    tag = (
        "".join(
            [
                "JIJI",
                "gnar\x01\x00\x00\x00",
                "stul%s\x00\x00\x00" % chr(len(luts)),
            ]
        ).encode("ascii")
        + d
    )
    tag = struct.unpack("<" + "B" * len(tag), tag)
    return tag + tuple(sum([list(x) for x in luts], []))


def load_stitching_offsets(filename):
    """Load i,j coordinates from the text file saved by the Fiji
    Grid/Collection stitching plugin.
    """
    from ast import literal_eval

    with open(filename, "r") as fh:
        txt = fh.read()
    txt = txt.split("# Define the image coordinates")[1]
    lines = txt.split("\n")
    coordinates = []
    for line in lines:
        parts = line.split(";")
        if len(parts) == 3:
            coordinates += [parts[-1].strip()]

    return [(i, j) for j, i in map(literal_eval, coordinates)]


if __name__ == "__main__":
    commands = {"format_input": format_input}
    try:
        fire.Fire(commands)
    except QuitError:
        sys.exit(1)
