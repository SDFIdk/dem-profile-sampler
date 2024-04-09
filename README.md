# dem-profile-sampler
Tool to sample profiles from raster DEMs along river centerlines

## Installation

An appropriate environment (here named `profile`) can be created with:

```
conda env create -n profile -f environment.yml
```

The following assumes that this environment is activated.

The tool supports editable installation using `pip`. To install it this way,
use the following command in the root directory:

```
pip install -e .
```

You may want to test your installation by calling `pytest` in the root
directory.

## Usage

```
sample_profiles [-h] [--interval INTERVAL] [--length LENGTH] [--sample-count SAMPLE_COUNT] [--progress] input_centerlines input_dem output_profiles
```

| Parameter | Description |
| --------- | ----------- |
| `input_centerlines` | Input centerline datasource (OGR-readable path or connection string) |
| `input_dem` | Input DEM raster dataset to sample Z from (GDAL-readable path or connection string) |
| `output_profiles` | Desired path to output file. Will be written as GeoPackage (GPKG) |
| `--interval INTERVAL` | Preferred interval (along centerlines) between profiles, in georeferenced units |
| `--length LENGTH` | Length of profiles in georeferenced units |
| `--sample-count SAMPLE_COUNT` | Number of samples to take along each profile |
| `-h` | Print help and exit |

## Example

This will create profiles across the linestrings in `rivers.gpkg`, sampling from the raster DEM in `dem.vrt`, writing the output profiles to `profiles.gpkg`:

```
sample_profiles rivers.gpkg dem.vrt profiles.gpkg
```
