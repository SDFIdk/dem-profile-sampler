from osgeo import gdal, ogr
from tqdm import tqdm

import argparse
from pathlib import Path

import profilesample
import profilesample.sampler

gdal.UseExceptions()
ogr.UseExceptions()

def main():
    argument_parser = argparse.ArgumentParser()
    argument_parser.add_argument('input_centerlines', type=str, help='input centerlines to take profiles across')
    argument_parser.add_argument('input_dem', type=str, help='input DEM to sample from')
    argument_parser.add_argument('output_profiles', type=str, help='path to which sampled output profiles will be written (GPKG)')
    argument_parser.add_argument('--interval', type=float, default=100.0, help='preferred distance between profiles')
    argument_parser.add_argument('--length', type=float, default=100.0, help='horizontal length of output profiles')
    argument_parser.add_argument('--sample-count', type=int, default=101, help='number of samples to take per profile')
    argument_parser.add_argument('--progress', action='store_true', help='show progress bar while processing')

    input_arguments = argument_parser.parse_args()
    centerline_path = input_arguments.input_centerlines
    dem_path = input_arguments.input_dem
    profile_path = input_arguments.output_profiles
    profile_interval = input_arguments.interval
    profile_length = input_arguments.length
    sample_count = input_arguments.sample_count

    do_show_progress = input_arguments.progress

    centerline_datasrc = ogr.Open(centerline_path)
    centerline_layer = centerline_datasrc.GetLayer()
    srs = centerline_layer.GetSpatialRef()

    dem_dataset = gdal.Open(dem_path)

    output_driver = ogr.GetDriverByName('GPKG')
    output_datasrc = output_driver.CreateDataSource(profile_path)
    output_layer = output_datasrc.CreateLayer('profiles', srs, ogr.wkbLineString25D)

    if do_show_progress:
        centerline_iter = tqdm(centerline_layer, unit='feat', ascii=True)
    else:
        centerline_iter = centerline_layer
    for centerline_feature in centerline_iter:
        centerline_geometry = centerline_feature.GetGeometryRef()
        profilesample.sampler.get_profiles(centerline_geometry, dem_dataset, output_layer, profile_interval=profile_interval, profile_length=profile_length, sample_count=sample_count)

