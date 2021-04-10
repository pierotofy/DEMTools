#!/usr/bin/env python3
# Author: Piero Toffanin
# License: AGPLv3

import argparse
import rasterio
import fiona
from scipy import ndimage
import numpy as np
import math
import numpy.ma as ma
from scipy import interpolate

parser = argparse.ArgumentParser(description='Fix rooftops in a DEM')
parser.add_argument('dem',
                type=str,
                help='Path to input DEM')
parser.add_argument('cutlines',
                type=str,
                help='Path to a GDAL-readable vector file containing line strings with rooftop cutlines')
parser.add_argument('output',
                    type=str,
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
    
    def normalize(self):
        length = math.sqrt(self.x ** 2 + self.y ** 2)
        if length != 0:
            return GeoPoint(self.x / length, self.y / length)
        return GeoPoint(0, 0)
    
    def __mul__(self, factor):
        return GeoPoint(self.x * factor, self.y * factor)

    def __add__(self, other):
        return GeoPoint(self.x + other.x, self.y + other.y)

    def __str__(self):
        return "(%s %s)" % (self.x, self.y)

class GeoLine:
    def __init__(self, x1, y1, x2, y2):
        self.a = GeoPoint(x1, y1)
        self.b = GeoPoint(x2, y2)
    
    def inside_raster(self):
        return self.a.inside_raster() and self.b.inside_raster()

    def stretch(self, length):
        length_ab = math.sqrt((self.a.x - self.b.x)**2 + (self.a.y - self.b.y)**2) 
        f = length_ab * length

        b_x = self.b.x + (self.b.x - self.a.x) / f
        b_y = self.b.y + (self.b.y - self.a.y) / f

        a_x = self.a.x + (self.a.x - self.b.x) / f
        a_y = self.a.y + (self.a.y - self.b.y) / f

        return GeoLine(a_x, a_y, b_x, b_y)
    
    def compute_normals(self, scale = 1):
        dx = self.b.x - self.a.x
        dy = self.b.y - self.a.y

        n1 = GeoPoint(-dy, dx)
        n2 = GeoPoint(dy, -dx)

        return [n1.normalize() * scale, n2.normalize() * scale]

    def expand_to_box(self, length):
        n1, n2 = self.compute_normals(length)
        p1 = (n1 + self.a)
        p2 = (n1 + self.b)
        p3 = (n2 + self.b)
        p4 = (n2 + self.a)

        return GeoRectangle(p1.x, p1.y,
                        p2.x, p2.y,
                        p3.x, p3.y,
                        p4.x, p4.y)
    
    def expand_to_boxes(self, length):
        n1, n2 = self.compute_normals(length)
        p1 = (n1 + self.a)
        p2 = (n1 + self.b)
        p3 = (n2 + self.b)
        p4 = (n2 + self.a)

        return [GeoRectangle(
                p1.x, p1.y,
                p2.x, p2.y,
                self.a.x, self.a.y,
                self.b.x, self.b.y
            ), GeoRectangle(
                self.b.x, self.b.y,
                self.a.x, self.a.y,
                p3.x, p3.y,
                p4.x, p4.y
            )]

    def buffer_rectangle(self, length):
        return self.expand_to_box(length)

    def buffer_rectangles(self, length):
        return self.expand_to_boxes(length)

    def __str__(self):
        return "[%s, %s]" % (self.a, self.b)

class GeoRectangle:
    @staticmethod
    def triangle_area(a, b, c):
        return abs((b[0] * a[1] - a[0] * b[1]) + (c[0] * b[1] - b[0] * c[1]) + (a[0] * c[1] - c[0] * a[1])) / 2.0

    def __init__(self, *coords):
        num_coords = len(coords)
        if num_coords != 8:
            raise ValueError("Wrong number of coords (%s)" % num_coords)
        
        self.coords = []
        for i in range(0, num_coords - 1, 2):
            self.coords.append(GeoPoint(coords[i], coords[i + 1]))
    
    def inside_raster(self):
        return all([p.inside_raster() for p in self.coords])

    def raster_mask(self):
        mask = np.zeros((dem_raster.shape[0], dem_raster.shape[1]), dtype=np.bool)
        bbox = self.bbox().raster_box()

        for y in range(mask.shape[0]):
            for x in range(mask.shape[1]):
                if bbox.contains(x, y):
                    if self.contains_raster_coords(x, y):
                        mask[y][x] = True

        return mask

    def raster_coords(self):
        return list(map(GeoPoint.raster_coords, self.coords))
    
    def contains_raster_coords(self, x, y):
        a, b, c, d = self.raster_coords()
        p = (x, y)

        total_area = self.triangle_area(a, p, d) + \
                     self.triangle_area(d, p, c) + \
                     self.triangle_area(c, p, b) + \
                     self.triangle_area(p, b, a)
        rect_area = self.triangle_area(a, b, d) + self.triangle_area(b, c, d)
        if total_area <= rect_area:
            return True

        return False

    def bbox(self):
        x_coords = [p.x for p in self.coords]
        y_coords = [p.y for p in self.coords]

        return GeoBox(GeoPoint(min(x_coords), min(y_coords)),
                      GeoPoint(max(x_coords), max(y_coords)))

    def __str__(self):
        return "[" + ", ".join(map(str, self.coords))  + "]"

class GeoBox:
    def __init__(self, p1, p2):
        self.p1 = p1
        self.p2 = p2
    
    def raster_box(self):
        rp1 = self.p1.raster_coords()
        rp2 = self.p2.raster_coords()

        return RasterBox(rp1[0], rp1[1], rp2[0], rp2[1])

class RasterBox:
    def __init__(self, x1, y1, x2, y2):
        self.minx = min(x1, x2)
        self.miny = min(y1, y2)
        self.maxx = max(x1, x2)
        self.maxy = max(y1, y2)
    
    def contains(self, x, y):
        return x >= self.minx and x <= self.maxx and \
               y >= self.miny and y <= self.maxy

    def __str__(self):
        return "[(%s, %s), (%s, %s)]" % (self.minx, self.miny, self.maxx, self.maxy)

if not dem_raster.crs:
    print("DEM has no CRS (we need that!)")
    exit(1)

print("DEM CRS: %s" % dem_raster.crs)
print("Cutlines CRS: %s" % cutlines_vector.crs.get('init').upper())

if str(dem_raster.crs) != cutlines_vector.crs.get('init').upper():
    print("CRSes do not match!")
    exit(1)

dem = dem_raster.read()[0]

def plot(img):
    import matplotlib.pyplot as pl
    pl.figure(figsize=(20, 10))
    pl.title('Image')
    pl.imshow(img)
    pl.show()

def gapfill(dem, buffer_mask, cutoff, interpolation_type):
    masked_dem = ma.array(dem, mask=~buffer_mask)

    if interpolation_type == 'high':
        value_mask = buffer_mask & (masked_dem < cutoff)
    else:
        value_mask = buffer_mask & (masked_dem >= cutoff)

    coords = np.array(np.nonzero(~value_mask)).T
    values = masked_dem[~value_mask].compressed()

    it = interpolate.LinearNDInterpolator(coords, values, fill_value=np.nan)
    i, j = np.where(buffer_mask)
    indices = np.nonzero(buffer_mask)
    filled = it(list(zip(i, j)))
    filled_mask = ~np.isnan(filled)

    indices = (indices[0][filled_mask], indices[1][filled_mask])
    dem[indices] = filled[filled_mask]
    # dem[value_mask] = 0 if interpolation_type == 'high' else 999
    return dem

def identify_high_low(dem, buffers):
    if len(buffers) != 2:
        raise Exception("Buffers != 2")

    m1 = ma.array(dem, mask=~buffers[0].raster_mask())
    m2 = ma.array(dem, mask=~buffers[1].raster_mask())

    c1 = np.percentile(m1.compressed(), 20)
    c2 = np.percentile(m2.compressed(), 20)

    r1 = {'buffer': buffers[0], 'cutoff': c1}
    r2 = {'buffer': buffers[1], 'cutoff': c2}

    if c1 > c2:
        return r1, r2
    else:
        return r2, r1

for cutline in cutlines_vector:
    geom = cutline.get('geometry')
    line_id = geom.get('id')

    if geom.get('type') != 'LineString':
        print("Skipping feature %s (not linestring)" % line_id)
        continue
    
    line_coords = geom.get('coordinates')

    # For each line segment
    for i in range(len(line_coords) - 1):
        line = GeoLine(*line_coords[i], *line_coords[i + 1])
        print("Correcting %s" % line)
        
        buffers = line.buffer_rectangles(1)

        high, low = identify_high_low(dem, buffers)

        if high['buffer'].inside_raster():
            dem = gapfill(dem, high['buffer'].raster_mask(), high['cutoff'], 'high')
        else:
            print("Buffered line %s outside of DEM raster bounds" % line_id)

        if low['buffer'].inside_raster():
            dem = gapfill(dem, low['buffer'].raster_mask(), low['cutoff'], 'low')
        else:
            print("Buffered line %s outside of DEM raster bounds" % line_id)

profile = {
    'driver': 'GTiff',
    'width': dem.shape[1],
    'height': dem.shape[0],
    'count': 1,
    'dtype': dem.dtype.name,
    'transform': dem_raster.transform,
    'nodata': dem_raster.nodata,
    'crs': dem_raster.crs
}

with rasterio.open(args.output, 'w', **profile) as wout:
    wout.write(dem, 1)

dem_raster.close()
cutlines_vector.close()