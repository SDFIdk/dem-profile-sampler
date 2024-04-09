from osgeo import gdal, ogr, osr
import numpy as np
from scipy.interpolate import RegularGridInterpolator

from collections import namedtuple

gdal.UseExceptions()
ogr.UseExceptions()
osr.UseExceptions()

BoundingBox = namedtuple(
    'BoundingBox',
    ['x_min', 'x_max', 'y_min', 'y_max'],
)

def get_raster_window(dataset: gdal.Dataset, bbox: BoundingBox) -> gdal.Dataset:
    """
    Return a window of the input raster dataset, containing at least the
    provided bounding box.
    """

    input_geotransform = dataset.GetGeoTransform()

    if input_geotransform[2] != 0.0 or input_geotransform[4] != 0.0:
        raise ValueError("geotransforms with rotation are unsupported")

    input_offset_x = input_geotransform[0]
    input_offset_y = input_geotransform[3]
    input_pixelsize_x = input_geotransform[1]
    input_pixelsize_y = input_geotransform[5]

    # We want to find window coordinates that:
    # a) are aligned to the source raster pixels
    # b) contain the requested bounding box plus at least one pixel of "padding" on each side, to allow for small floating-point rounding errors in X/Y coordinates
    # 
    # Recall that the pixel size in the geotransform is commonly negative, hence all the min/max calls.
    pad_amount = 1

    raw_x_min_col_float = (bbox.x_min - input_offset_x) / input_pixelsize_x
    raw_x_max_col_float = (bbox.x_max - input_offset_x) / input_pixelsize_x
    raw_y_min_row_float = (bbox.y_min - input_offset_y) / input_pixelsize_y
    raw_y_max_row_float = (bbox.y_max - input_offset_y) / input_pixelsize_y

    col_min = int(np.floor(min(raw_x_min_col_float, raw_x_max_col_float))) - pad_amount
    col_max = int(np.ceil(max(raw_x_min_col_float, raw_x_max_col_float))) + pad_amount
    row_min = int(np.floor(min(raw_y_min_row_float, raw_y_max_row_float))) - pad_amount
    row_max = int(np.ceil(max(raw_y_min_row_float, raw_y_max_row_float))) + pad_amount

    x_col_min = input_offset_x + input_pixelsize_x * col_min
    x_col_max = input_offset_x + input_pixelsize_x * col_max
    y_row_min = input_offset_y + input_pixelsize_y * row_min
    y_row_max = input_offset_y + input_pixelsize_y * row_max

    # Padded, georeferenced window coordinates. The target window to use with gdal.Translate().
    padded_bbox = BoundingBox(
        x_min=min(x_col_min, x_col_max),
        x_max=max(x_col_min, x_col_max),
        y_min=min(y_row_min, y_row_max),
        y_max=max(y_row_min, y_row_max),
    )

    # Size in pixels of destination raster
    dest_num_cols = col_max - col_min
    dest_num_rows = row_max - row_min

    translate_options = gdal.TranslateOptions(
        width=dest_num_cols,
        height=dest_num_rows,
        projWin=(padded_bbox.x_min, padded_bbox.y_max, padded_bbox.x_max, padded_bbox.y_min),
        resampleAlg=gdal.GRA_NearestNeighbour,
    )

    # gdal.Translate() needs a destination *name*, not just a Dataset to
    # write into. Create a temporary file in GDAL's virtual filesystem as a
    # stepping stone.
    window_dataset_name = "/vsimem/temp_window.tif"
    window_dataset = gdal.Translate(
        window_dataset_name,
        dataset,
        options=translate_options
    )

    return window_dataset

def get_raster_interpolator(dataset: gdal.Dataset) -> RegularGridInterpolator:
    """
    Return a scipy.interpolate.RegularGridInterpolator corresponding to a GDAL
    raster.
    """

    geotransform = dataset.GetGeoTransform()
    band = dataset.GetRasterBand(1)
    nodata_value = band.GetNoDataValue()
    z_grid = band.ReadAsArray()
    num_rows, num_cols = z_grid.shape

    if geotransform[2] != 0.0 or geotransform[4] != 0.0:
        raise ValueError("geotransforms with rotation are unsupported")

    # X and Y values for the individual columns/rows of the raster. The 0.5 is
    # added in order to obtain the coordinates of the cell centers rather than
    # the corners.
    x_values = geotransform[0] + geotransform[1]*(0.5+np.arange(num_cols))
    y_values = geotransform[3] + geotransform[5]*(0.5+np.arange(num_rows))

    # RegularGridInterpolator requires the x and y arrays to be in strictly
    # increasing order, accommodate this
    if geotransform[1] > 0.0:
        col_step = 1
    else:
        col_step = -1
        x_values = np.flip(x_values)

    if geotransform[5] > 0.0:
        row_step = 1
    else:
        row_step = -1
        y_values = np.flip(y_values)

    # NODATA values must be replaced with NaN for interpolation purposes
    z_grid[z_grid == nodata_value] = np.nan

    # The grid must be transposed to swap (row, col) coordinates into (x, y)
    # order.
    interpolator = RegularGridInterpolator(
        points=(x_values, y_values),
        values=z_grid[::row_step, ::col_step].transpose(),
        method='linear',
        bounds_error=False,
        fill_value=np.nan,
    )

    return interpolator

def get_2d_normals(vectors: np.array) -> np.array:
    raw_normals = np.column_stack([vectors[:, 1], -vectors[:, 0]])

    # normalize to unit length
    unit_normals = raw_normals / np.hypot(raw_normals[:, 0], raw_normals[:, 1])[:, np.newaxis]

    return unit_normals

def get_profiles(centerline: ogr.Geometry, raster: gdal.Dataset, output_layer: ogr.Layer, profile_interval: float, profile_length: float, sample_count: int) -> np.array:
    centerline_xyz = np.array(centerline.GetPoints())
    centerline_xy = centerline_xyz[:,:2]

    x_min = min(centerline_xyz[:,0])
    x_max = max(centerline_xyz[:,0])
    y_min = min(centerline_xyz[:,1])
    y_max = max(centerline_xyz[:,1])

    geometry_bbox = BoundingBox(x_min=x_min, x_max=x_max, y_min=y_min, y_max=y_max)

    window_dataset = get_raster_window(raster, geometry_bbox)
    window_interpolator = get_raster_interpolator(window_dataset)

    centerline_length = centerline.Length()
    segment_vectors = np.column_stack([centerline_xy[1:, 0] - centerline_xy[:-1, 0], centerline_xy[1:,1] - centerline_xy[:-1, 1]])
    segment_lengths = np.hypot(segment_vectors[:, 0], segment_vectors[:, 1])

    # Chainage for each point of this centerline, starting with 0.0
    local_chainages = np.concatenate([[0.0], np.cumsum(segment_lengths)])

    num_samples = max(1, int(np.ceil(centerline_length / profile_interval)))
    actual_sampling_dist = centerline_length / num_samples
    sampling_local_chainages = 0.5*actual_sampling_dist + np.linspace(0.0, centerline_length, num_samples, endpoint=False)

    sampling_indices = np.searchsorted(local_chainages, sampling_local_chainages) - 1 # insertion would happen at index + 1

    # The centerline coordinates through which we will put the profiles
    sampling_centers_x = np.interp(sampling_local_chainages, local_chainages, centerline_xy[:, 0])
    sampling_centers_y = np.interp(sampling_local_chainages, local_chainages, centerline_xy[:, 1])
    sampling_centers = np.column_stack([sampling_centers_x, sampling_centers_y])

    # Local along-centerline vectors in the sampling locations
    sampling_vectors = segment_vectors[sampling_indices, :]
    sampling_normals = get_2d_normals(sampling_vectors)

    profile_left = sampling_centers - 0.5 * profile_length * sampling_normals
    profile_right = sampling_centers + 0.5 * profile_length * sampling_normals

    # linspace generates [place_on_profile, profile_index, xy], swap axes to [profile_index, place_on_profile, xy]
    profiles = np.linspace(profile_left, profile_right, sample_count).swapaxes(0, 1)

    output_layer_defn = output_layer.GetLayerDefn()

    for profile_xy in profiles:
        profile_z = window_interpolator(profile_xy)
        profile_geometry = ogr.Geometry(ogr.wkbLineString25D)
        for xy, z in zip(profile_xy, profile_z):
            profile_geometry.AddPoint(xy[0], xy[1], z)
        profile_feature = ogr.Feature(output_layer_defn)
        profile_feature.SetGeometry(profile_geometry)
        output_layer.CreateFeature(profile_feature)
