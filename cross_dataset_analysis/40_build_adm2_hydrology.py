"""
01_build_adm2_hydrology.py

Purpose
-------
Build an ADM2-level daily hydrology dataset for South Sudan using
ERA5 rainfall and runoff data.

The script:

1. Reads South Sudan ADM2 administrative boundaries.
2. Reads all ERA5 rainfall/runoff NetCDF files.
3. Converts ERA5 rainfall and runoff from metres to millimetres.
4. Aggregates hourly ERA5 values to daily totals.
5. Assigns ERA5 grid-cell centres to ADM2 counties.
6. Calculates area-weighted ADM2 rainfall and runoff.
7. Produces daily, monthly, annual and climatological datasets.
8. Produces ADM2 rainfall/runoff maps and seasonal figures.

Expected project structure
--------------------------

DBL-Group-22/
│
├── raw_data/
│   ├── administrative_boundaries/
│   │   └── ssd_admin2.geojson
│   │
│   └── rainfall_and_runoff/
│       ├── ERA5_2014.nc
│       ├── ERA5_2015.nc
│       ├── ...
│       └── ERA5_2025.nc
│
├── processed_data/
│
├── outputs/
│   └── figures/
│
└── scripts/
    └── cross_dataset_analysis/
        └── 01_build_adm2_hydrology.py

Outputs
-------

Processed data:
processed_data/cross_dataset_analysis/adm2_hydrology/

Figures:
outputs/figures/cross_dataset_analysis/adm2_hydrology/
"""


# ============================================================
# IMPORTS
# ============================================================

from pathlib import Path
import warnings

import numpy as np
import pandas as pd
import geopandas as gpd
import xarray as xr
import matplotlib.pyplot as plt

warnings.filterwarnings("ignore")


# ============================================================
# SETTINGS
# ============================================================

CLIP_NEGATIVE_VALUES = True

RAINFALL_VARIABLE_CANDIDATES = [
    "tp",
    "total_precipitation",
    "precipitation",
]

RUNOFF_VARIABLE_CANDIDATES = [
    "ro",
    "runoff",
    "total_runoff",
    "sro",
    "surface_runoff",
]

TIME_CANDIDATES = [
    "valid_time",
    "time",
]

LATITUDE_CANDIDATES = [
    "latitude",
    "lat",
]

LONGITUDE_CANDIDATES = [
    "longitude",
    "lon",
]


# ============================================================
# PROJECT PATHS
# ============================================================

SCRIPT_DIR = Path(__file__).resolve().parent

# Current file:
# DBL-Group-22/scripts/cross_dataset_analysis/
#
# parents[0] -> scripts
# parents[1] -> DBL-Group-22
PROJECT_ROOT = SCRIPT_DIR.parents[1]

RAW_DATA_DIR = PROJECT_ROOT / "raw_data"
PROCESSED_DATA_DIR = PROJECT_ROOT / "processed_data"
OUTPUTS_DIR = PROJECT_ROOT / "outputs"


# ============================================================
# INPUT PATHS
# ============================================================

ERA5_RAW_DIR = (
    RAW_DATA_DIR
    / "rainfall_and_runoff"
)

ADMIN_BOUNDARY_DIR = (
    RAW_DATA_DIR
    / "administrative_boundaries"
)

ADM2_FILE = (
    ADMIN_BOUNDARY_DIR
    / "ssd_admin2.geojson"
)


# ============================================================
# PROCESSED DATA OUTPUT PATH
# ============================================================

PROCESSED_OUTPUT_DIR = (
    PROCESSED_DATA_DIR
    / "cross_dataset_analysis"
    / "adm2_hydrology"
)

PROCESSED_OUTPUT_DIR.mkdir(
    parents=True,
    exist_ok=True,
)


# ============================================================
# FIGURE OUTPUT PATH
# ============================================================

FIGURE_OUTPUT_DIR = (
    OUTPUTS_DIR
    / "figures"
    / "cross_dataset_analysis"
    / "adm2_hydrology"
)

FIGURE_OUTPUT_DIR.mkdir(
    parents=True,
    exist_ok=True,
)


# ============================================================
# CSV OUTPUT FILES
# ============================================================

DAILY_OUTPUT = (
    PROCESSED_OUTPUT_DIR
    / "ADM2_daily_rainfall_runoff.csv"
)

MONTHLY_OUTPUT = (
    PROCESSED_OUTPUT_DIR
    / "ADM2_monthly_rainfall_runoff.csv"
)

ANNUAL_OUTPUT = (
    PROCESSED_OUTPUT_DIR
    / "ADM2_annual_rainfall_runoff.csv"
)

MONTHLY_CLIMATOLOGY_OUTPUT = (
    PROCESSED_OUTPUT_DIR
    / "ADM2_monthly_climatology.csv"
)

GRID_ASSIGNMENT_OUTPUT = (
    PROCESSED_OUTPUT_DIR
    / "ERA5_grid_ADM2_assignment.csv"
)

ADM2_GRID_COVERAGE_OUTPUT = (
    PROCESSED_OUTPUT_DIR
    / "ADM2_ERA5_grid_coverage.csv"
)


# ============================================================
# FIGURE FILES
# ============================================================

FIGURE_01 = (
    FIGURE_OUTPUT_DIR
    / "01_adm2_mean_annual_rainfall_map.png"
)

FIGURE_02 = (
    FIGURE_OUTPUT_DIR
    / "02_adm2_mean_annual_runoff_map.png"
)

FIGURE_03 = (
    FIGURE_OUTPUT_DIR
    / "03_adm2_monthly_rainfall_heatmap.png"
)

FIGURE_04 = (
    FIGURE_OUTPUT_DIR
    / "04_adm2_monthly_runoff_heatmap.png"
)

FIGURE_05 = (
    FIGURE_OUTPUT_DIR
    / "05_adm2_rainfall_runoff_scatter.png"
)

FIGURE_06 = (
    FIGURE_OUTPUT_DIR
    / "06_national_monthly_climatology.png"
)

FIGURE_07 = (
    FIGURE_OUTPUT_DIR
    / "07_adm2_peak_rainfall_month_map.png"
)

FIGURE_08 = (
    FIGURE_OUTPUT_DIR
    / "08_adm2_peak_runoff_month_map.png"
)

FIGURE_09 = (
    FIGURE_OUTPUT_DIR
    / "09_adm2_grid_coverage_map.png"
)


# ============================================================
# GENERAL HELPER FUNCTIONS
# ============================================================

def print_header(title):
    """
    Print a formatted section header.
    """

    print("\n" + "=" * 80)
    print(title)
    print("=" * 80)


def find_existing_name(
    available_names,
    candidates,
    description,
):
    """
    Find the first candidate name that exists in a dataset.
    """

    for candidate in candidates:

        if candidate in available_names:

            return candidate

    raise KeyError(
        f"Could not detect {description}.\n"
        f"Candidates tried: {candidates}\n"
        f"Available names: {list(available_names)}"
    )


def find_column(
    dataframe,
    candidates,
    required=False,
):
    """
    Detect a likely administrative boundary column.
    """

    columns = list(
        dataframe.columns
    )

    for candidate in candidates:

        if candidate in columns:

            return candidate

    lower_mapping = {
        str(column).lower(): column
        for column in columns
    }

    for candidate in candidates:

        if candidate.lower() in lower_mapping:

            return lower_mapping[
                candidate.lower()
            ]

    if required:

        raise KeyError(
            "Could not detect required administrative "
            "boundary column.\n"
            f"Tried: {candidates}\n"
            f"Available columns: {columns}"
        )

    return None


# ============================================================
# ERA5 STRUCTURE DETECTION
# ============================================================

def detect_era5_structure(dataset):
    """
    Automatically detect rainfall, runoff, time,
    latitude and longitude variables.
    """

    available_names = (
        set(dataset.variables)
        | set(dataset.coords)
    )

    rainfall_variable = find_existing_name(
        available_names,
        RAINFALL_VARIABLE_CANDIDATES,
        "rainfall variable",
    )

    runoff_variable = find_existing_name(
        available_names,
        RUNOFF_VARIABLE_CANDIDATES,
        "runoff variable",
    )

    time_name = find_existing_name(
        available_names,
        TIME_CANDIDATES,
        "time coordinate",
    )

    latitude_name = find_existing_name(
        available_names,
        LATITUDE_CANDIDATES,
        "latitude coordinate",
    )

    longitude_name = find_existing_name(
        available_names,
        LONGITUDE_CANDIDATES,
        "longitude coordinate",
    )

    return (
        rainfall_variable,
        runoff_variable,
        time_name,
        latitude_name,
        longitude_name,
    )


# ============================================================
# EXTRA DIMENSION HANDLING
# ============================================================

def remove_extra_dimensions(
    data_array,
    time_name,
    latitude_name,
    longitude_name,
):
    """
    Remove dimensions other than time, latitude and longitude.

    ERA5 files may sometimes contain an 'expver' dimension.
    In this case the valid streams are combined using a mean
    that ignores missing values.
    """

    expected_dimensions = {
        time_name,
        latitude_name,
        longitude_name,
    }

    extra_dimensions = [
        dimension
        for dimension in data_array.dims
        if dimension not in expected_dimensions
    ]

    for dimension in extra_dimensions:

        if dimension.lower() == "expver":

            print(
                f"  Combining '{dimension}' "
                "using mean(skipna=True)"
            )

            data_array = data_array.mean(
                dim=dimension,
                skipna=True,
            )

        else:

            print(
                f"  Selecting first value of "
                f"extra dimension '{dimension}'"
            )

            data_array = data_array.isel(
                {
                    dimension: 0
                }
            )

    return data_array


# ============================================================
# LOAD ADM2 BOUNDARIES
# ============================================================

def load_adm2():
    """
    Load South Sudan ADM2 boundaries and standardise
    administrative identifiers.
    """

    print_header(
        "LOADING ADM2 ADMINISTRATIVE BOUNDARIES"
    )

    if not ADM2_FILE.exists():

        raise FileNotFoundError(
            "ADM2 boundary file not found:\n"
            f"{ADM2_FILE}"
        )

    adm2 = gpd.read_file(
        ADM2_FILE
    )

    print(
        f"Boundary file:\n{ADM2_FILE}"
    )

    print(
        f"\nNumber of ADM2 polygons: "
        f"{len(adm2)}"
    )

    print(
        f"Original CRS: {adm2.crs}"
    )

    print(
        "\nAvailable boundary columns:"
    )

    for column in adm2.columns:

        print(
            f" - {column}"
        )

    # --------------------------------------------------------
    # CRS
    # --------------------------------------------------------

    if adm2.crs is None:

        print(
            "\nWARNING: No CRS found. "
            "Assuming EPSG:4326."
        )

        adm2 = adm2.set_crs(
            "EPSG:4326"
        )

    else:

        adm2 = adm2.to_crs(
            "EPSG:4326"
        )

    # --------------------------------------------------------
    # Detect ADM2 name
    # --------------------------------------------------------

    adm2_name_column = find_column(
        adm2,
        [
            "ADM2_EN",
            "ADM2_NAME",
            "ADM2",
            "NAME_2",
            "admin2Name",
            "admin2_name",
            "county",
            "County",
        ],
        required=True,
    )

    # --------------------------------------------------------
    # Detect ADM2 code
    # --------------------------------------------------------

    adm2_code_column = find_column(
        adm2,
        [
            "ADM2_PCODE",
            "ADM2_CODE",
            "PCODE",
            "GID_2",
            "admin2Pcode",
            "admin2_code",
        ],
        required=False,
    )

    # --------------------------------------------------------
    # Detect ADM1 name
    # --------------------------------------------------------

    adm1_name_column = find_column(
        adm2,
        [
            "ADM1_EN",
            "ADM1_NAME",
            "ADM1",
            "NAME_1",
            "admin1Name",
            "admin1_name",
            "state",
            "State",
        ],
        required=False,
    )

    # --------------------------------------------------------
    # Detect ADM1 code
    # --------------------------------------------------------

    adm1_code_column = find_column(
        adm2,
        [
            "ADM1_PCODE",
            "ADM1_CODE",
            "GID_1",
            "admin1Pcode",
            "admin1_code",
        ],
        required=False,
    )

    print(
        f"\nDetected ADM2 name column: "
        f"{adm2_name_column}"
    )

    print(
        f"Detected ADM2 code column: "
        f"{adm2_code_column}"
    )

    print(
        f"Detected ADM1 name column: "
        f"{adm1_name_column}"
    )

    print(
        f"Detected ADM1 code column: "
        f"{adm1_code_column}"
    )

    # --------------------------------------------------------
    # Standardise fields
    # --------------------------------------------------------

    adm2 = adm2.copy()

    adm2["adm2_name"] = (
        adm2[adm2_name_column]
        .astype(str)
        .str.strip()
    )

    if adm2_code_column is not None:

        adm2["adm2_code"] = (
            adm2[adm2_code_column]
            .astype(str)
            .str.strip()
        )

    else:

        adm2["adm2_code"] = (
            np.arange(
                1,
                len(adm2) + 1
            )
            .astype(str)
        )

    if adm1_name_column is not None:

        adm2["adm1_name"] = (
            adm2[adm1_name_column]
            .astype(str)
            .str.strip()
        )

    else:

        adm2["adm1_name"] = None

    if adm1_code_column is not None:

        adm2["adm1_code"] = (
            adm2[adm1_code_column]
            .astype(str)
            .str.strip()
        )

    else:

        adm2["adm1_code"] = None

    # --------------------------------------------------------
    # Repair invalid geometries
    # --------------------------------------------------------

    invalid_geometry_count = (
        ~adm2.geometry.is_valid
    ).sum()

    if invalid_geometry_count > 0:

        print(
            f"\nRepairing "
            f"{invalid_geometry_count} invalid geometries..."
        )

        adm2["geometry"] = (
            adm2.geometry.buffer(0)
        )

    adm2 = adm2.reset_index(
        drop=True
    )

    adm2["adm2_id"] = np.arange(
        len(adm2)
    )

    adm2 = adm2[
        [
            "adm2_id",
            "adm2_code",
            "adm2_name",
            "adm1_code",
            "adm1_name",
            "geometry",
        ]
    ]

    print(
        "\nADM2 boundaries successfully loaded."
    )

    return adm2


# ============================================================
# BUILD ERA5 GRID TO ADM2 ASSIGNMENT
# ============================================================

def build_grid_assignment(
    first_era5_file,
    adm2,
):
    """
    Create ERA5 grid-centre points and assign each point
    to an ADM2 polygon.
    """

    print_header(
        "BUILDING ERA5 GRID TO ADM2 ASSIGNMENT"
    )

    print(
        f"Reference ERA5 file:\n"
        f"{first_era5_file}"
    )

    with xr.open_dataset(
        first_era5_file
    ) as dataset:

        (
            rainfall_variable,
            runoff_variable,
            time_name,
            latitude_name,
            longitude_name,
        ) = detect_era5_structure(
            dataset
        )

        latitudes = np.asarray(
            dataset[
                latitude_name
            ].values,
            dtype=float,
        )

        longitudes = np.asarray(
            dataset[
                longitude_name
            ].values,
            dtype=float,
        )

    print(
        f"\nRainfall variable: "
        f"{rainfall_variable}"
    )

    print(
        f"Runoff variable: "
        f"{runoff_variable}"
    )

    print(
        f"Time coordinate: "
        f"{time_name}"
    )

    print(
        f"Latitude coordinate: "
        f"{latitude_name}"
    )

    print(
        f"Longitude coordinate: "
        f"{longitude_name}"
    )

    print(
        f"\nERA5 grid size: "
        f"{len(latitudes)} × "
        f"{len(longitudes)}"
    )

    # --------------------------------------------------------
    # Construct grid
    # --------------------------------------------------------

    longitude_grid, latitude_grid = np.meshgrid(
        longitudes,
        latitudes,
    )

    grid_dataframe = pd.DataFrame(
        {
            "grid_id": np.arange(
                latitude_grid.size
            ),
            "latitude": latitude_grid.ravel(),
            "longitude": longitude_grid.ravel(),
        }
    )

    grid_gdf = gpd.GeoDataFrame(
        grid_dataframe,
        geometry=gpd.points_from_xy(
            grid_dataframe["longitude"],
            grid_dataframe["latitude"],
        ),
        crs="EPSG:4326",
    )

    # --------------------------------------------------------
    # Restrict grid search to South Sudan area
    # --------------------------------------------------------

    min_x, min_y, max_x, max_y = (
        adm2.total_bounds
    )

    buffer_degrees = 0.3

    grid_subset = grid_gdf[
        (
            grid_gdf["longitude"]
            >= min_x - buffer_degrees
        )
        &
        (
            grid_gdf["longitude"]
            <= max_x + buffer_degrees
        )
        &
        (
            grid_gdf["latitude"]
            >= min_y - buffer_degrees
        )
        &
        (
            grid_gdf["latitude"]
            <= max_y + buffer_degrees
        )
    ].copy()

    print(
        f"\nTotal ERA5 grid cells: "
        f"{len(grid_gdf):,}"
    )

    print(
        f"Grid cells near South Sudan: "
        f"{len(grid_subset):,}"
    )

    # --------------------------------------------------------
    # Spatial join
    # --------------------------------------------------------

    joined = gpd.sjoin(
        grid_subset,
        adm2[
            [
                "adm2_id",
                "adm2_code",
                "adm2_name",
                "adm1_code",
                "adm1_name",
                "geometry",
            ]
        ],
        how="inner",
        predicate="intersects",
    )

    joined = (
        joined
        .sort_values(
            [
                "grid_id",
                "adm2_id",
            ]
        )
        .drop_duplicates(
            subset="grid_id",
            keep="first",
        )
        .copy()
    )

    # --------------------------------------------------------
    # Approximate cell-area correction
    # --------------------------------------------------------

    joined["area_weight"] = np.cos(
        np.deg2rad(
            joined["latitude"]
        )
    )

    print(
        f"Grid cells assigned to ADM2: "
        f"{len(joined):,}"
    )

    # --------------------------------------------------------
    # Coverage QA
    # --------------------------------------------------------

    coverage = (
        joined
        .groupby(
            [
                "adm2_id",
                "adm2_code",
                "adm2_name",
                "adm1_code",
                "adm1_name",
            ],
            dropna=False,
        )
        .agg(
            grid_cell_count=(
                "grid_id",
                "count",
            ),
            mean_grid_latitude=(
                "latitude",
                "mean",
            ),
            mean_grid_longitude=(
                "longitude",
                "mean",
            ),
        )
        .reset_index()
    )

    all_adm2 = adm2[
        [
            "adm2_id",
            "adm2_code",
            "adm2_name",
            "adm1_code",
            "adm1_name",
        ]
    ].copy()

    coverage = all_adm2.merge(
        coverage,
        on=[
            "adm2_id",
            "adm2_code",
            "adm2_name",
            "adm1_code",
            "adm1_name",
        ],
        how="left",
    )

    coverage["grid_cell_count"] = (
        coverage[
            "grid_cell_count"
        ]
        .fillna(0)
        .astype(int)
    )

    coverage.to_csv(
        ADM2_GRID_COVERAGE_OUTPUT,
        index=False,
    )

    # --------------------------------------------------------
    # Save grid assignment
    # --------------------------------------------------------

    joined[
        [
            "grid_id",
            "latitude",
            "longitude",
            "area_weight",
            "adm2_id",
            "adm2_code",
            "adm2_name",
            "adm1_code",
            "adm1_name",
        ]
    ].to_csv(
        GRID_ASSIGNMENT_OUTPUT,
        index=False,
    )

    missing_adm2 = coverage[
        coverage[
            "grid_cell_count"
        ] == 0
    ]

    print(
        f"\nADM2 units with ERA5 grid cells: "
        f"{(coverage['grid_cell_count'] > 0).sum()}"
    )

    print(
        f"ADM2 units without ERA5 grid cells: "
        f"{len(missing_adm2)}"
    )

    if len(missing_adm2) > 0:

        print(
            "\nWARNING: No ERA5 grid-cell centre "
            "was located inside:"
        )

        for county in missing_adm2[
            "adm2_name"
        ]:

            print(
                f" - {county}"
            )

    return (
        joined,
        coverage,
        latitudes,
        longitudes,
    )


# ============================================================
# WEIGHT MATRIX
# ============================================================

def build_weight_matrix(
    grid_assignment,
    number_of_grid_cells,
    number_of_adm2,
):
    """
    Build an ADM2 × ERA5-grid weight matrix.
    """

    weight_matrix = np.zeros(
        (
            number_of_adm2,
            number_of_grid_cells,
        ),
        dtype=np.float64,
    )

    for row in grid_assignment.itertuples():

        adm2_id = int(
            row.adm2_id
        )

        grid_id = int(
            row.grid_id
        )

        weight_matrix[
            adm2_id,
            grid_id,
        ] = float(
            row.area_weight
        )

    return weight_matrix


# ============================================================
# PREPARE DAILY ERA5 VARIABLE
# ============================================================

def prepare_daily_variable(
    data_array,
    time_name,
    latitude_name,
    longitude_name,
):
    """
    Convert an hourly ERA5 variable to daily totals
    expressed in millimetres per day.
    """

    data_array = remove_extra_dimensions(
        data_array,
        time_name,
        latitude_name,
        longitude_name,
    )

    data_array = data_array.transpose(
        time_name,
        latitude_name,
        longitude_name,
    )

    # ERA5 rainfall/runoff values:
    # metre -> millimetre
    data_array = (
        data_array.astype(
            "float64"
        )
        * 1000.0
    )

    if CLIP_NEGATIVE_VALUES:

        data_array = data_array.clip(
            min=0
        )

    # Hourly accumulation -> daily total
    daily_array = (
        data_array
        .resample(
            {
                time_name: "1D"
            }
        )
        .sum(
            skipna=True
        )
    )

    return daily_array


# ============================================================
# ADM2 WEIGHTED SPATIAL MEAN
# ============================================================

def calculate_weighted_adm2_mean(
    daily_values,
    weight_matrix,
):
    """
    Calculate the area-weighted spatial mean for each
    ADM2 and each day.
    """

    values = np.asarray(
        daily_values,
        dtype=np.float64,
    )

    finite_mask = np.isfinite(
        values
    )

    values_without_nan = np.where(
        finite_mask,
        values,
        0.0,
    )

    available_weight = (
        finite_mask.astype(
            np.float64
        )
        @ weight_matrix.T
    )

    weighted_sum = (
        values_without_nan
        @ weight_matrix.T
    )

    with np.errstate(
        divide="ignore",
        invalid="ignore",
    ):

        weighted_mean = (
            weighted_sum
            / available_weight
        )

    weighted_mean[
        available_weight == 0
    ] = np.nan

    return weighted_mean


# ============================================================
# PROCESS ONE ERA5 FILE
# ============================================================

def process_one_era5_file(
    era5_file,
    adm2,
    weight_matrix,
    reference_latitudes,
    reference_longitudes,
):
    """
    Process one ERA5 NetCDF file and return daily
    ADM2 rainfall and runoff.
    """

    print_header(
        f"PROCESSING {era5_file.name}"
    )

    with xr.open_dataset(
        era5_file
    ) as dataset:

        (
            rainfall_variable,
            runoff_variable,
            time_name,
            latitude_name,
            longitude_name,
        ) = detect_era5_structure(
            dataset
        )

        current_latitudes = np.asarray(
            dataset[
                latitude_name
            ].values,
            dtype=float,
        )

        current_longitudes = np.asarray(
            dataset[
                longitude_name
            ].values,
            dtype=float,
        )

        # ----------------------------------------------------
        # Grid consistency check
        # ----------------------------------------------------

        if not np.allclose(
            current_latitudes,
            reference_latitudes,
        ):

            raise ValueError(
                f"Latitude grid in {era5_file.name} "
                "does not match the reference grid."
            )

        if not np.allclose(
            current_longitudes,
            reference_longitudes,
        ):

            raise ValueError(
                f"Longitude grid in {era5_file.name} "
                "does not match the reference grid."
            )

        print(
            f"Rainfall variable: "
            f"{rainfall_variable}"
        )

        print(
            f"Runoff variable: "
            f"{runoff_variable}"
        )

        start_time = pd.to_datetime(
            dataset[
                time_name
            ].values[0]
        )

        end_time = pd.to_datetime(
            dataset[
                time_name
            ].values[-1]
        )

        print(
            f"Time range: "
            f"{start_time} -> {end_time}"
        )

        # ----------------------------------------------------
        # Daily rainfall
        # ----------------------------------------------------

        print(
            "\nAggregating rainfall to daily totals..."
        )

        rainfall_daily = prepare_daily_variable(
            dataset[
                rainfall_variable
            ],
            time_name,
            latitude_name,
            longitude_name,
        )

        # ----------------------------------------------------
        # Daily runoff
        # ----------------------------------------------------

        print(
            "Aggregating runoff to daily totals..."
        )

        runoff_daily = prepare_daily_variable(
            dataset[
                runoff_variable
            ],
            time_name,
            latitude_name,
            longitude_name,
        )

        dates = pd.to_datetime(
            rainfall_daily[
                time_name
            ].values
        )

        # ----------------------------------------------------
        # Flatten the spatial grid
        # ----------------------------------------------------

        rainfall_flat = (
            rainfall_daily
            .values
            .reshape(
                len(dates),
                -1,
            )
        )

        runoff_flat = (
            runoff_daily
            .values
            .reshape(
                len(dates),
                -1,
            )
        )

    # --------------------------------------------------------
    # Calculate ADM2 spatial means
    # --------------------------------------------------------

    print(
        "Calculating ADM2 rainfall..."
    )

    rainfall_adm2 = (
        calculate_weighted_adm2_mean(
            rainfall_flat,
            weight_matrix,
        )
    )

    print(
        "Calculating ADM2 runoff..."
    )

    runoff_adm2 = (
        calculate_weighted_adm2_mean(
            runoff_flat,
            weight_matrix,
        )
    )

    # --------------------------------------------------------
    # Convert to long-format dataframe
    # --------------------------------------------------------

    records = []

    adm2_sorted = (
        adm2
        .sort_values(
            "adm2_id"
        )
        .reset_index(
            drop=True
        )
    )

    for row in adm2_sorted.itertuples():

        adm2_id = int(
            row.adm2_id
        )

        county_dataframe = pd.DataFrame(
            {
                "date": dates,
                "adm2_id": adm2_id,
                "adm2_code": row.adm2_code,
                "adm2_name": row.adm2_name,
                "adm1_code": row.adm1_code,
                "adm1_name": row.adm1_name,
                "rainfall_mm": (
                    rainfall_adm2[
                        :,
                        adm2_id
                    ]
                ),
                "runoff_mm": (
                    runoff_adm2[
                        :,
                        adm2_id
                    ]
                ),
            }
        )

        records.append(
            county_dataframe
        )

    result = pd.concat(
        records,
        ignore_index=True,
    )

    print(
        f"\nGenerated "
        f"{len(result):,} ADM2-day observations."
    )

    return result


# ============================================================
# MONTHLY SUMMARY
# ============================================================

def build_monthly_summary(
    daily,
):
    """
    Create year-month summaries for each ADM2.
    """

    dataframe = daily.copy()

    dataframe["year"] = (
        dataframe["date"].dt.year
    )

    dataframe["month"] = (
        dataframe["date"].dt.month
    )

    dataframe["month_name"] = (
        dataframe[
            "date"
        ].dt.strftime(
            "%b"
        )
    )

    monthly = (
        dataframe
        .groupby(
            [
                "adm2_id",
                "adm2_code",
                "adm2_name",
                "adm1_code",
                "adm1_name",
                "year",
                "month",
                "month_name",
            ],
            dropna=False,
        )
        .agg(
            rainfall_mm=(
                "rainfall_mm",
                "sum",
            ),
            runoff_mm=(
                "runoff_mm",
                "sum",
            ),
            mean_daily_rainfall_mm=(
                "rainfall_mm",
                "mean",
            ),
            mean_daily_runoff_mm=(
                "runoff_mm",
                "mean",
            ),
            max_daily_rainfall_mm=(
                "rainfall_mm",
                "max",
            ),
            max_daily_runoff_mm=(
                "runoff_mm",
                "max",
            ),
            days_available=(
                "date",
                "count",
            ),
        )
        .reset_index()
    )

    return monthly


# ============================================================
# ANNUAL SUMMARY
# ============================================================

def build_annual_summary(
    daily,
):
    """
    Create annual hydrological statistics for each ADM2.
    """

    dataframe = daily.copy()

    dataframe["year"] = (
        dataframe["date"].dt.year
    )

    annual = (
        dataframe
        .groupby(
            [
                "adm2_id",
                "adm2_code",
                "adm2_name",
                "adm1_code",
                "adm1_name",
                "year",
            ],
            dropna=False,
        )
        .agg(
            annual_rainfall_mm=(
                "rainfall_mm",
                "sum",
            ),
            annual_runoff_mm=(
                "runoff_mm",
                "sum",
            ),
            mean_daily_rainfall_mm=(
                "rainfall_mm",
                "mean",
            ),
            mean_daily_runoff_mm=(
                "runoff_mm",
                "mean",
            ),
            max_daily_rainfall_mm=(
                "rainfall_mm",
                "max",
            ),
            max_daily_runoff_mm=(
                "runoff_mm",
                "max",
            ),
            days_available=(
                "date",
                "count",
            ),
        )
        .reset_index()
    )

    return annual


# ============================================================
# MONTHLY CLIMATOLOGY
# ============================================================

def build_monthly_climatology(
    monthly,
):
    """
    Calculate average monthly rainfall/runoff for each ADM2
    across all available years.
    """

    climatology = (
        monthly
        .groupby(
            [
                "adm2_id",
                "adm2_code",
                "adm2_name",
                "adm1_code",
                "adm1_name",
                "month",
                "month_name",
            ],
            dropna=False,
        )
        .agg(
            mean_monthly_rainfall_mm=(
                "rainfall_mm",
                "mean",
            ),
            mean_monthly_runoff_mm=(
                "runoff_mm",
                "mean",
            ),
            mean_daily_rainfall_mm=(
                "mean_daily_rainfall_mm",
                "mean",
            ),
            mean_daily_runoff_mm=(
                "mean_daily_runoff_mm",
                "mean",
            ),
        )
        .reset_index()
    )

    return climatology


# ============================================================
# FIGURE 1
# ADM2 MEAN ANNUAL RAINFALL MAP
# ============================================================

def plot_mean_annual_rainfall(
    adm2,
    annual,
):
    """
    Map long-term mean annual rainfall by ADM2.
    """

    mean_values = (
        annual
        .groupby(
            "adm2_id"
        )[
            "annual_rainfall_mm"
        ]
        .mean()
        .reset_index()
        .rename(
            columns={
                "annual_rainfall_mm":
                "mean_annual_rainfall_mm"
            }
        )
    )

    map_data = adm2.merge(
        mean_values,
        on="adm2_id",
        how="left",
    )

    fig, ax = plt.subplots(
        figsize=(10, 11)
    )

    map_data.plot(
        column="mean_annual_rainfall_mm",
        legend=True,
        ax=ax,
        edgecolor="black",
        linewidth=0.4,
        missing_kwds={
            "label": "No data",
        },
    )

    ax.set_title(
        "Mean Annual Rainfall by ADM2",
        fontsize=15,
    )

    ax.set_axis_off()

    plt.tight_layout()

    plt.savefig(
        FIGURE_01,
        dpi=300,
        bbox_inches="tight",
    )

    plt.close()

    print(
        f"Saved figure:\n{FIGURE_01}"
    )


# ============================================================
# FIGURE 2
# ADM2 MEAN ANNUAL RUNOFF MAP
# ============================================================

def plot_mean_annual_runoff(
    adm2,
    annual,
):
    """
    Map long-term mean annual runoff by ADM2.
    """

    mean_values = (
        annual
        .groupby(
            "adm2_id"
        )[
            "annual_runoff_mm"
        ]
        .mean()
        .reset_index()
        .rename(
            columns={
                "annual_runoff_mm":
                "mean_annual_runoff_mm"
            }
        )
    )

    map_data = adm2.merge(
        mean_values,
        on="adm2_id",
        how="left",
    )

    fig, ax = plt.subplots(
        figsize=(10, 11)
    )

    map_data.plot(
        column="mean_annual_runoff_mm",
        legend=True,
        ax=ax,
        edgecolor="black",
        linewidth=0.4,
        missing_kwds={
            "label": "No data",
        },
    )

    ax.set_title(
        "Mean Annual Runoff by ADM2",
        fontsize=15,
    )

    ax.set_axis_off()

    plt.tight_layout()

    plt.savefig(
        FIGURE_02,
        dpi=300,
        bbox_inches="tight",
    )

    plt.close()

    print(
        f"Saved figure:\n{FIGURE_02}"
    )


# ============================================================
# HEATMAP HELPER
# ============================================================

def plot_monthly_heatmap(
    climatology,
    value_column,
    title,
    output_file,
):
    """
    Create an ADM2 × month heatmap.
    """

    pivot = climatology.pivot_table(
        index="adm2_name",
        columns="month",
        values=value_column,
        aggfunc="mean",
    )

    pivot = pivot.reindex(
        columns=range(
            1,
            13
        )
    )

    # Sort counties by average magnitude
    pivot["average"] = (
        pivot.mean(
            axis=1
        )
    )

    pivot = pivot.sort_values(
        "average",
        ascending=False,
    )

    pivot = pivot.drop(
        columns="average"
    )

    figure_height = max(
        8,
        len(pivot) * 0.22,
    )

    fig, ax = plt.subplots(
        figsize=(
            12,
            figure_height,
        )
    )

    image = ax.imshow(
        pivot.values,
        aspect="auto",
    )

    month_labels = [
        "Jan",
        "Feb",
        "Mar",
        "Apr",
        "May",
        "Jun",
        "Jul",
        "Aug",
        "Sep",
        "Oct",
        "Nov",
        "Dec",
    ]

    ax.set_xticks(
        np.arange(12)
    )

    ax.set_xticklabels(
        month_labels
    )

    ax.set_yticks(
        np.arange(
            len(pivot.index)
        )
    )

    ax.set_yticklabels(
        pivot.index,
        fontsize=7,
    )

    ax.set_xlabel(
        "Month"
    )

    ax.set_ylabel(
        "ADM2 County"
    )

    ax.set_title(
        title,
        fontsize=14,
    )

    colorbar = fig.colorbar(
        image,
        ax=ax,
    )

    colorbar.set_label(
        "mm"
    )

    plt.tight_layout()

    plt.savefig(
        output_file,
        dpi=300,
        bbox_inches="tight",
    )

    plt.close()

    print(
        f"Saved figure:\n{output_file}"
    )


# ============================================================
# FIGURE 5
# RAINFALL-RUNOFF SCATTER
# ============================================================

def plot_rainfall_runoff_scatter(
    daily,
):
    """
    Plot daily rainfall against daily runoff using a
    random sample when the full dataset is large.
    """

    plot_data = daily[
        [
            "rainfall_mm",
            "runoff_mm",
        ]
    ].dropna()

    if len(plot_data) > 30000:

        plot_data = plot_data.sample(
            n=30000,
            random_state=42,
        )

    correlation = (
        plot_data[
            "rainfall_mm"
        ]
        .corr(
            plot_data[
                "runoff_mm"
            ]
        )
    )

    fig, ax = plt.subplots(
        figsize=(9, 7)
    )

    ax.scatter(
        plot_data[
            "rainfall_mm"
        ],
        plot_data[
            "runoff_mm"
        ],
        s=8,
        alpha=0.25,
    )

    ax.set_xlabel(
        "Daily Rainfall (mm/day)"
    )

    ax.set_ylabel(
        "Daily Runoff (mm/day)"
    )

    ax.set_title(
        "ADM2 Daily Rainfall vs Runoff\n"
        f"Pearson correlation = "
        f"{correlation:.3f}"
    )

    ax.grid(
        alpha=0.25
    )

    plt.tight_layout()

    plt.savefig(
        FIGURE_05,
        dpi=300,
        bbox_inches="tight",
    )

    plt.close()

    print(
        f"Saved figure:\n{FIGURE_05}"
    )

    return correlation


# ============================================================
# FIGURE 6
# NATIONAL MONTHLY CLIMATOLOGY
# ============================================================

def plot_national_monthly_climatology(
    climatology,
):
    """
    Average ADM2 monthly climatology across South Sudan.
    """

    national = (
        climatology
        .groupby(
            "month"
        )
        .agg(
            rainfall_mm=(
                "mean_daily_rainfall_mm",
                "mean",
            ),
            runoff_mm=(
                "mean_daily_runoff_mm",
                "mean",
            ),
        )
        .reset_index()
    )

    month_labels = [
        "Jan",
        "Feb",
        "Mar",
        "Apr",
        "May",
        "Jun",
        "Jul",
        "Aug",
        "Sep",
        "Oct",
        "Nov",
        "Dec",
    ]

    fig, ax = plt.subplots(
        figsize=(10, 6)
    )

    ax.plot(
        national["month"],
        national["rainfall_mm"],
        marker="o",
        label="Rainfall",
    )

    ax.plot(
        national["month"],
        national["runoff_mm"],
        marker="o",
        label="Runoff",
    )

    ax.set_xticks(
        range(
            1,
            13
        )
    )

    ax.set_xticklabels(
        month_labels
    )

    ax.set_xlabel(
        "Month"
    )

    ax.set_ylabel(
        "Mean Daily Value (mm/day)"
    )

    ax.set_title(
        "South Sudan Monthly Rainfall and Runoff Climatology\n"
        "ADM2-based spatial analysis"
    )

    ax.legend()

    ax.grid(
        alpha=0.25
    )

    plt.tight_layout()

    plt.savefig(
        FIGURE_06,
        dpi=300,
        bbox_inches="tight",
    )

    plt.close()

    print(
        f"Saved figure:\n{FIGURE_06}"
    )


# ============================================================
# PEAK MONTH MAP HELPER
# ============================================================

def plot_peak_month_map(
    adm2,
    climatology,
    value_column,
    title,
    output_file,
):
    """
    Map the month with the highest climatological value
    for each ADM2.
    """

    peak_indices = (
        climatology
        .groupby(
            "adm2_id"
        )[
            value_column
        ]
        .idxmax()
    )

    peak = climatology.loc[
        peak_indices,
        [
            "adm2_id",
            "month",
        ],
    ].copy()

    peak = peak.rename(
        columns={
            "month":
            "peak_month"
        }
    )

    map_data = adm2.merge(
        peak,
        on="adm2_id",
        how="left",
    )

    fig, ax = plt.subplots(
        figsize=(10, 11)
    )

    map_data.plot(
        column="peak_month",
        legend=True,
        ax=ax,
        edgecolor="black",
        linewidth=0.4,
        vmin=1,
        vmax=12,
    )

    ax.set_title(
        title,
        fontsize=15,
    )

    ax.set_axis_off()

    plt.tight_layout()

    plt.savefig(
        output_file,
        dpi=300,
        bbox_inches="tight",
    )

    plt.close()

    print(
        f"Saved figure:\n{output_file}"
    )


# ============================================================
# FIGURE 9
# ERA5 GRID COVERAGE MAP
# ============================================================

def plot_grid_coverage(
    adm2,
    coverage,
):
    """
    Map the number of ERA5 grid-cell centres assigned
    to each ADM2.
    """

    map_data = adm2.merge(
        coverage[
            [
                "adm2_id",
                "grid_cell_count",
            ]
        ],
        on="adm2_id",
        how="left",
    )

    fig, ax = plt.subplots(
        figsize=(10, 11)
    )

    map_data.plot(
        column="grid_cell_count",
        legend=True,
        ax=ax,
        edgecolor="black",
        linewidth=0.4,
    )

    ax.set_title(
        "ERA5 Grid-Cell Coverage by ADM2",
        fontsize=15,
    )

    ax.set_axis_off()

    plt.tight_layout()

    plt.savefig(
        FIGURE_09,
        dpi=300,
        bbox_inches="tight",
    )

    plt.close()

    print(
        f"Saved figure:\n{FIGURE_09}"
    )


# ============================================================
# BASIC QUALITY CHECK
# ============================================================

def run_quality_check(
    daily,
):
    """
    Print basic data-quality and descriptive statistics.
    """

    print_header(
        "BASIC QUALITY CHECK"
    )

    rainfall_missing = (
        daily[
            "rainfall_mm"
        ]
        .isna()
        .mean()
        * 100
    )

    runoff_missing = (
        daily[
            "runoff_mm"
        ]
        .isna()
        .mean()
        * 100
    )

    print(
        f"Missing rainfall values: "
        f"{rainfall_missing:.2f}%"
    )

    print(
        f"Missing runoff values: "
        f"{runoff_missing:.2f}%"
    )

    print(
        "\nRainfall descriptive statistics:"
    )

    print(
        daily[
            "rainfall_mm"
        ]
        .describe()
        .round(3)
    )

    print(
        "\nRunoff descriptive statistics:"
    )

    print(
        daily[
            "runoff_mm"
        ]
        .describe()
        .round(3)
    )


# ============================================================
# MAIN
# ============================================================

def main():

    print_header(
        "ADM2 ERA5 HYDROLOGY ANALYSIS"
    )

    print(
        f"Project root:\n"
        f"{PROJECT_ROOT}"
    )

    print(
        f"\nERA5 input:\n"
        f"{ERA5_RAW_DIR}"
    )

    print(
        f"\nADM2 boundary:\n"
        f"{ADM2_FILE}"
    )

    print(
        f"\nProcessed data output:\n"
        f"{PROCESSED_OUTPUT_DIR}"
    )

    print(
        f"\nFigure output:\n"
        f"{FIGURE_OUTPUT_DIR}"
    )

    # --------------------------------------------------------
    # Validate input paths
    # --------------------------------------------------------

    if not ERA5_RAW_DIR.exists():

        raise FileNotFoundError(
            "ERA5 rainfall/runoff folder not found:\n"
            f"{ERA5_RAW_DIR}\n\n"
            "Expected:\n"
            "raw_data/rainfall_and_runoff/"
        )

    if not ADM2_FILE.exists():

        raise FileNotFoundError(
            "ADM2 boundary file not found:\n"
            f"{ADM2_FILE}"
        )

    # --------------------------------------------------------
    # Find ERA5 files
    # --------------------------------------------------------

    era5_files = sorted(
        ERA5_RAW_DIR.glob(
            "ERA5_*.nc"
        )
    )

    if len(era5_files) == 0:

        raise FileNotFoundError(
            "No ERA5_*.nc files found in:\n"
            f"{ERA5_RAW_DIR}"
        )

    print_header(
        "ERA5 FILES DETECTED"
    )

    for era5_file in era5_files:

        print(
            f" - {era5_file.name}"
        )

    print(
        f"\nTotal files: "
        f"{len(era5_files)}"
    )

    # --------------------------------------------------------
    # Load ADM2
    # --------------------------------------------------------

    adm2 = load_adm2()

    # --------------------------------------------------------
    # Create ERA5 → ADM2 spatial mapping
    # --------------------------------------------------------

    (
        grid_assignment,
        grid_coverage,
        reference_latitudes,
        reference_longitudes,
    ) = build_grid_assignment(
        era5_files[0],
        adm2,
    )

    number_of_grid_cells = (
        len(reference_latitudes)
        * len(reference_longitudes)
    )

    weight_matrix = build_weight_matrix(
        grid_assignment,
        number_of_grid_cells,
        len(adm2),
    )

    # --------------------------------------------------------
    # Process all ERA5 years
    # --------------------------------------------------------

    all_year_results = []

    for era5_file in era5_files:

        year_result = process_one_era5_file(
            era5_file,
            adm2,
            weight_matrix,
            reference_latitudes,
            reference_longitudes,
        )

        all_year_results.append(
            year_result
        )

    # --------------------------------------------------------
    # Combine years
    # --------------------------------------------------------

    print_header(
        "COMBINING ALL YEARS"
    )

    daily = pd.concat(
        all_year_results,
        ignore_index=True,
    )

    daily["date"] = pd.to_datetime(
        daily["date"]
    )

    daily = (
        daily
        .sort_values(
            [
                "date",
                "adm2_name",
            ]
        )
        .reset_index(
            drop=True
        )
    )

    # --------------------------------------------------------
    # Add temporal columns
    # --------------------------------------------------------

    daily["year"] = (
        daily[
            "date"
        ].dt.year
    )

    daily["month"] = (
        daily[
            "date"
        ].dt.month
    )

    daily["month_name"] = (
        daily[
            "date"
        ].dt.strftime(
            "%b"
        )
    )

    daily["day_of_year"] = (
        daily[
            "date"
        ].dt.dayofyear
    )

    # --------------------------------------------------------
    # Save daily dataset
    # --------------------------------------------------------

    daily.to_csv(
        DAILY_OUTPUT,
        index=False,
    )

    print(
        f"\nDaily dataset saved:\n"
        f"{DAILY_OUTPUT}"
    )

    print(
        f"\nDataset shape: "
        f"{daily.shape}"
    )

    print(
        f"Date range: "
        f"{daily['date'].min().date()} "
        f"-> "
        f"{daily['date'].max().date()}"
    )

    print(
        f"ADM2 units: "
        f"{daily['adm2_name'].nunique()}"
    )

    # --------------------------------------------------------
    # Monthly summary
    # --------------------------------------------------------

    print_header(
        "BUILDING MONTHLY SUMMARY"
    )

    monthly = build_monthly_summary(
        daily
    )

    monthly.to_csv(
        MONTHLY_OUTPUT,
        index=False,
    )

    print(
        f"Saved:\n{MONTHLY_OUTPUT}"
    )

    # --------------------------------------------------------
    # Annual summary
    # --------------------------------------------------------

    print_header(
        "BUILDING ANNUAL SUMMARY"
    )

    annual = build_annual_summary(
        daily
    )

    annual.to_csv(
        ANNUAL_OUTPUT,
        index=False,
    )

    print(
        f"Saved:\n{ANNUAL_OUTPUT}"
    )

    # --------------------------------------------------------
    # Monthly climatology
    # --------------------------------------------------------

    print_header(
        "BUILDING MONTHLY CLIMATOLOGY"
    )

    climatology = (
        build_monthly_climatology(
            monthly
        )
    )

    climatology.to_csv(
        MONTHLY_CLIMATOLOGY_OUTPUT,
        index=False,
    )

    print(
        f"Saved:\n"
        f"{MONTHLY_CLIMATOLOGY_OUTPUT}"
    )

    # --------------------------------------------------------
    # Quality check
    # --------------------------------------------------------

    run_quality_check(
        daily
    )

    # --------------------------------------------------------
    # Figures
    # --------------------------------------------------------

    print_header(
        "GENERATING FIGURES"
    )

    plot_mean_annual_rainfall(
        adm2,
        annual,
    )

    plot_mean_annual_runoff(
        adm2,
        annual,
    )

    plot_monthly_heatmap(
        climatology,
        "mean_monthly_rainfall_mm",
        "ADM2 Monthly Rainfall Climatology",
        FIGURE_03,
    )

    plot_monthly_heatmap(
        climatology,
        "mean_monthly_runoff_mm",
        "ADM2 Monthly Runoff Climatology",
        FIGURE_04,
    )

    correlation = (
        plot_rainfall_runoff_scatter(
            daily
        )
    )

    plot_national_monthly_climatology(
        climatology
    )

    plot_peak_month_map(
        adm2,
        climatology,
        "mean_monthly_rainfall_mm",
        "Peak Rainfall Month by ADM2",
        FIGURE_07,
    )

    plot_peak_month_map(
        adm2,
        climatology,
        "mean_monthly_runoff_mm",
        "Peak Runoff Month by ADM2",
        FIGURE_08,
    )

    plot_grid_coverage(
        adm2,
        grid_coverage,
    )

    # --------------------------------------------------------
    # Seasonal results
    # --------------------------------------------------------

    print_header(
        "ADM2-BASED SEASONAL SUMMARY"
    )

    national_climatology = (
        climatology
        .groupby(
            [
                "month",
                "month_name",
            ]
        )
        .agg(
            rainfall_mm=(
                "mean_daily_rainfall_mm",
                "mean",
            ),
            runoff_mm=(
                "mean_daily_runoff_mm",
                "mean",
            ),
        )
        .reset_index()
    )

    wettest_row = national_climatology.loc[
        national_climatology[
            "rainfall_mm"
        ].idxmax()
    ]

    driest_row = national_climatology.loc[
        national_climatology[
            "rainfall_mm"
        ].idxmin()
    ]

    highest_runoff_row = (
        national_climatology.loc[
            national_climatology[
                "runoff_mm"
            ].idxmax()
        ]
    )

    lowest_runoff_row = (
        national_climatology.loc[
            national_climatology[
                "runoff_mm"
            ].idxmin()
        ]
    )

    print(
        "Wettest month by mean daily rainfall: "
        f"{wettest_row['month_name']} "
        f"({wettest_row['rainfall_mm']:.3f} mm/day)"
    )

    print(
        "Driest month by mean daily rainfall: "
        f"{driest_row['month_name']} "
        f"({driest_row['rainfall_mm']:.3f} mm/day)"
    )

    print(
        "Highest-runoff month: "
        f"{highest_runoff_row['month_name']} "
        f"({highest_runoff_row['runoff_mm']:.3f} mm/day)"
    )

    print(
        "Lowest-runoff month: "
        f"{lowest_runoff_row['month_name']} "
        f"({lowest_runoff_row['runoff_mm']:.3f} mm/day)"
    )

    print(
        "\nADM2 daily rainfall-runoff correlation: "
        f"{correlation:.3f}"
    )

    # --------------------------------------------------------
    # Top rainfall events
    # --------------------------------------------------------

    print_header(
        "TOP 10 ADM2 DAILY RAINFALL EVENTS"
    )

    top_rainfall = (
        daily[
            [
                "date",
                "adm1_name",
                "adm2_name",
                "rainfall_mm",
                "runoff_mm",
            ]
        ]
        .sort_values(
            "rainfall_mm",
            ascending=False,
        )
        .head(10)
    )

    print(
        top_rainfall.to_string(
            index=False
        )
    )

    # --------------------------------------------------------
    # Complete
    # --------------------------------------------------------

    print_header(
        "ANALYSIS COMPLETE"
    )

    print(
        "\nProcessed data saved to:"
    )

    print(
        PROCESSED_OUTPUT_DIR
    )

    print(
        "\nFigures saved to:"
    )

    print(
        FIGURE_OUTPUT_DIR
    )

    print(
        "\nGenerated processed datasets:"
    )

    print(
        " - ADM2_daily_rainfall_runoff.csv"
    )

    print(
        " - ADM2_monthly_rainfall_runoff.csv"
    )

    print(
        " - ADM2_annual_rainfall_runoff.csv"
    )

    print(
        " - ADM2_monthly_climatology.csv"
    )

    print(
        " - ERA5_grid_ADM2_assignment.csv"
    )

    print(
        " - ADM2_ERA5_grid_coverage.csv"
    )

    print(
        "\nGenerated figures:"
    )

    print(
        " - 01_adm2_mean_annual_rainfall_map.png"
    )

    print(
        " - 02_adm2_mean_annual_runoff_map.png"
    )

    print(
        " - 03_adm2_monthly_rainfall_heatmap.png"
    )

    print(
        " - 04_adm2_monthly_runoff_heatmap.png"
    )

    print(
        " - 05_adm2_rainfall_runoff_scatter.png"
    )

    print(
        " - 06_national_monthly_climatology.png"
    )

    print(
        " - 07_adm2_peak_rainfall_month_map.png"
    )

    print(
        " - 08_adm2_peak_runoff_month_map.png"
    )

    print(
        " - 09_adm2_grid_coverage_map.png"
    )

    print(
        "\nRecommended next step:"
    )

    print(
        "Build ADM2 flood occurrence and flooded-area "
        "indicators, then merge them with "
        "ADM2_daily_rainfall_runoff.csv."
    )


# ============================================================
# RUN SCRIPT
# ============================================================

if __name__ == "__main__":

    main()