#!/usr/bin/env python3
# Author: Piero Toffanin
# License: AGPLv3

import argparse
import rasterio
import fiona
import numpy as np
import numpy.ma as ma

parser = argparse.ArgumentParser(description='Fix rooftops in a DEM')
parser.add_argument('dem',
                type=str,
                help='Path to input DEM')
parser.add_argument('cutlines',
                type=str,
                help='Path to a GDAL-readable vector file containing line strings with rooftop cutlines')
parser.add_argument('output',
                    type=bool,
                    help="Output DEM")

args = parser.parse_args()

cutlines_vector = fiona.open(args.cutlines, 'r')
dem_raster = rasterio.open(args.dem, 'r')

class GeoPoint:
    def __init__(self, x, y):
        self.x = x
        self.y = y
    
    def raster_coords(self):
        row, col = dem_raster.index(self.x, self.y)
        return (col, row)

    def inside_raster(self):
        col, row = self.raster_coords()
        return row >= 0 and col >= 0 and row < dem_raster.height and col < dem_raster.width

if not dem_raster.crs:
    print("DEM has no CRS (we need that!)")
    exit(1)

print("DEM CRS: %s" % dem_raster.crs)
print("Cutlines CRS: %s" % cutlines_vector.crs.get('init').upper())

if str(dem_raster.crs) != cutlines_vector.crs.get('init').upper():
    print("CRSes do not match!")
    exit(1)

for cutline in cutlines_vector:
    geom = cutline.get('geometry')
    line_id = geom.get('id')

    if geom.get('type') != 'LineString':
        print("Skipping feature %s (not linestring)" % line_id)
        continue
    
    line = geom.get('coordinates')

    # For each line segment
    for i in range(len(line) - 1):
        a, b = GeoPoint(*line[i]), GeoPoint(*line[i + 1])

        if a.inside_raster() and b.inside_raster():
            print(a.raster_coords(), b.raster_coords())

            print("TODO: extract buffer window")
        else:
            print("Line %s outside of DEM raster bounds" % line_id)



dem_raster.close()
cutlines_vector.close()