# The print function allows us to print messages and information to the screen
print ( "Hello and welcome to my assignment for EGM722 - Programming for GIS and Remote Sensing"
        "Designated Sites such as Special Areas of Conservation (SAC) 'and' Special Protection Areas"
        "ensure the adequate conservation of habitats where many of our plants and animals live."
        "The following code will review SAC and SPA data" )

# First  import geopandas and load the data:
import geopandas as gpd

sac_data = gpd.read_file (
    'C:\EGM_722\egm722\project\data_files/sac_ITM.shp' )  # you will need to create your own file path here
print ( sac_data.head () )
spa_data = gpd.read_file (
    'C:\EGM_722\egm722\project\data_files/spa_ITM.shp' )  # you will need to create your own file path here
print ( spa_data.head () )
# The data is stored in a table (a GeoDataFrame), much like the attribute table in ArcMap.
# Next, you can discover how many rows of each feature there is.
# This will display the numbers of SACs and SPAs in Northern Ireland
rows, cols = sac_data.shape  # get the number of rows in the table,
# this gives you the count of the SAC features in Northern Ireland
print ( 'Number of SAC features: {}'.format ( rows ) )
rows, cols = spa_data.shape  # get the number of rows in the table,
# this gives you the count of the SPA features in Northern Ireland
print ( 'Number of SPA features: {}'.format ( rows ) )
# _______________________________________________________________________________________________________________________
#Convert csv file to shapefiles. Here Historical Land Use for Northern Ireland will be investigated and
#converted into a shapefile
import pandas as pd
import geopandas as gpd
from shapely.geometry import Point
import cartopy.crs as ccrs

df = pd.read_csv('C:\EGM_722\egm722\Project\Data_Files\Historical_Landuse_Dataset.csv')

df.head() #this will let you look at the DataFrame you just loaded

# You only have point information (a single Lat/Lon coordinate) for each land use,
# so it makes sense to create a Point object for each feature using that point.
# Do this by first using the python built-in zip,
# then the apply method of the DataFrame to create a point object from the list of coordinates.
df['geometry'] = list(zip(df['x'], df['y'])) # Zip is an iterator, so use list to create
                                             # something that pandas can use.
df['geometry'] = df['geometry'].apply(Point) # using the 'apply' method of the dataframe,
                                             # turn the coordinates column
                                             # into points using the x, y coordinates
gdf = gpd.GeoDataFrame(df)
gdf.set_crs("EPSG:2157", inplace=True) # This sets the coordinate reference system to epsg:2157,
                                       # Irish Transverse Mercator lat/lon

print(gdf)
gdf.to_file ('Historical_Landuse_Dataset.shp')
# Writes the csv into to a shapefile
# _____________________________________________________________________________________________________________
# This allows the use of figures interactively
import geopandas as gpd
import matplotlib.lines as mlines
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
from cartopy.feature import ShapelyFeature

plt.ion() # make the plotting interactive
# generate matplotlib handles to create a legend of the features we put in our map.
def generate_handles(labels, colors, edge='k', alpha=1):
    lc = len(colors)  # get the length of the color list
    handles = []
    for i in range(len(labels)):
        handles.append(mpatches.Rectangle((0, 0), 1, 1, facecolor=colors[i % lc], edgecolor=edge, alpha=alpha))
    return handles
# create a scale bar of length 20 km in the upper right corner of the map
def scale_bar(ax, location=(0.92, 0.95)):
    llx0, llx1, lly0, lly1 = ax.get_extent(ccrs.PlateCarree())
    sbllx = (llx1 + llx0) / 2
    sblly = lly0 + (lly1 - lly0) * location[1]

    tmc = ccrs.TransverseMercator(sbllx, sblly)
    x0, x1, y0, y1 = ax.get_extent(tmc)
    sbx = x0 + (x1 - x0) * location[0]
    sby = y0 + (y1 - y0) * location[1]

    plt.plot([sbx, sbx - 20000], [sby, sby], color='k', linewidth=9, transform=tmc)
    plt.plot([sbx, sbx - 10000], [sby, sby], color='k', linewidth=6, transform=tmc)
    plt.plot([sbx-10000, sbx - 20000], [sby, sby], color='w', linewidth=6, transform=tmc)

    plt.text(sbx, sby-4500, '20 km', transform=tmc, fontsize=8)
    plt.text(sbx-12500, sby-4500, '10 km', transform=tmc, fontsize=8)
    plt.text(sbx-24500, sby-4500, '0 km', transform=tmc, fontsize=8)
# Most of the modules are now imported and a few helper functions defined,
# Now load the data. To load the shapefile data, use GeoPandas, an open-source package designed
# to make working with geospatial data in python easier
# load the datasets
outline = gpd.read_file('C:\EGM_722\egm722\Project\data_files/NI_outline.shp')
towns = gpd.read_file('C:\EGM_722\egm722\Project\data_files/Towns.shp')
water = gpd.read_file('C:\EGM_722\egm722\Project\data_files/Water.shp')
rivers = gpd.read_file('C:\EGM_722\egm722\Project\data_files/Rivers.shp')
counties = gpd.read_file('C:\EGM_722\egm722\Project\data_files/Counties.shp')
SACs = gpd.read_file('C:\EGM_722\egm722\Project\data_files/sac_ITM.shp')
SPAs = gpd.read_file('C:\EGM_722\egm722\Project\data_files/spa_ITM.shp')

#Create a figure of size 10x10 (representing the page size in inches)
myFig = plt.figure(figsize=(10, 10))
myCRS = ccrs.UTM(29)  #Create a Universal Transverse Mercator reference system
ax = plt.axes(projection=ccrs.Mercator())  # Creates an axes object in the figure, using a Mercator
# projection, where that data will be plotted.
# Add the outline of Northern Ireland using cartopy's ShapelyFeature
outline_feature = ShapelyFeature(outline['geometry'], myCRS, edgecolor='k', facecolor='w')
xmin, ymin, xmax, ymax = outline.total_bounds
ax.add_feature(outline_feature) # add the features we've created to the map.
# using the boundary of the shapefile features, zoom the map to our area of interest
ax.set_extent([xmin, xmax, ymin, ymax], crs=myCRS) # because total_bounds gives output as xmin, ymin, xmax, ymax,

# Here, set the edge color to be the same as the face color.
water_feat = ShapelyFeature ( water['geometry'], myCRS,
                                  edgecolor='mediumblue',
                                  facecolor='mediumblue',
                                  linewidth=1 )
ax.add_feature ( water_feat )

river_feat = ShapelyFeature ( rivers['geometry'], myCRS,
                                  edgecolor='royalblue',
                                  linewidth=0.2 )

ax.add_feature ( river_feat )

SACs_feat = ShapelyFeature ( SACs['geometry'], myCRS,
                                  edgecolor='darkorchid',
                                  facecolor='darkorchid',
                                  linewidth=0.5 )

ax.add_feature ( SACs_feat )

SPAs_feat = ShapelyFeature ( SPAs['geometry'], myCRS,
                                  edgecolor='fuchsia',
                                  facecolor='fuchsia',
                                  linewidth=0.5 )

ax.add_feature ( SPAs_feat )
# ShapelyFeature creates a polygon, so for point data we can just use ax.plot()
myFig # to show the updated figure
town_handle = ax.plot(towns.geometry.x, towns.geometry.y, 's', color='0.5', ms=3, transform=myCRS)
# note: if you change the color you use to display lakes, you'll want to change it here, too
water_handle = generate_handles(['Lakes'], ['mediumblue'])
# note: if you change the color you use to display rivers, you'll want to change it here, too
river_handle = [mlines.Line2D([], [], color='royalblue')]  # have to make this a list
# get a list of unique names for the county boundaries
county_names = list(counties.CountyName.unique())
county_names.sort() # sort the counties alphabetically by name
# update county_names to take it out of uppercase text
nice_names = [name.title() for name in county_names]
# generate a list of handles for the county datasets
county_colors = ['k']
county_handles = generate_handles(counties.CountyName.unique(), county_colors, alpha=0.25)
# generate handles for SPAs
spa_handle = [mlines.Line2D([], [], color='fuchsia')]
sac_handle = [mlines.Line2D([], [], color='orchid')]
# ax.legend() takes a list of handles and a list of labels corresponding to the objects you want to add to the legend
handles = county_handles + water_handle + river_handle + town_handle + sac_handle + spa_handle
labels = nice_names + ['Lakes', 'Rivers', 'Towns', 'Special Areas of Conservation', 'Special Protection Areas']
leg = ax.legend(handles, labels, title='Legend', title_fontsize=4,
                 fontsize=2, loc='upper left', frameon=True, framealpha=1)
gridlines = ax.gridlines(draw_labels=True,
                         xlocs=[-8, -7.5, -7, -6.5, -6, -5.5],
                         ylocs=[54, 54.5, 55, 55.5])
gridlines.left_labels = False
gridlines.bottom_labels = False
ax.set_extent([xmin, xmax, ymin, ymax], crs=myCRS)
# add the text labels for the towns
for i, row in towns.iterrows():
    x, y = row.geometry.x, row.geometry.y
    plt.text(x, y, row['TOWN_NAME'].title(), fontsize=4, transform=myCRS) # use plt.text to place a label at x,y

myFig.savefig( 'map.png', bbox_inches='tight', dpi=300 )
#_____________________________________________________________________________________________________
# You need to get the conifer forestry from the raster layer
# and convert it to a shapefile as there is no shapefile data
# avaialable for forestry in Northern Ireland


import rasterio as rio
import matplotlib.pyplot as plt

plt.rcParams.update({'font.size': 22}) # update the font size for our plots to be size 22

# open the land cover raster and read the data
with rio.open('C:\EGM_722\egm722\Week5\data_files/LCM2015_Aggregate_100m.tif') as dataset:
    xmin, ymin, xmax, ymax = dataset.bounds
    crs = dataset.crs
    landcover = dataset.read(1)
    affine_tfm = dataset.transform

#Polygonize a raster using Geospatial Data Abstraction Library (GDAL)
from osgeo import gdal, ogr
import sys
# This allows GDAL to throw Python Exceptions
gdal.UseExceptions()

# Get raster datasource
src = 'src_filename'

src_ds = gdal.Open( "LCM2015_Aggregate_100m.tif" )
if src_ds is None:
    print ('Unable to open {}'.format('src_filename'))
    sys.exit(1)

try:
    srcband = src_ds.GetRasterBand(3)
except RuntimeError as e:
# for example, try GetRasterBand(2)
    print ('Band ( %i ) not found')
    print (e)
    sys.exit(1)

# Create output datasource
dst_layername = "Conifer_Forest_Polygonized"
drv = ogr.GetDriverByName("ESRI Shapefile")
dst_ds = drv.CreateDataSource( dst_layername + "Conifer_Forest_Polygonized.shp" )
dst_layer = dst_ds.CreateLayer(dst_layername, srs = None )

gdal.Polygonize( srcband, None, dst_layer, -1, [], callback=None )
#_____________________________________________________________________________________________________________

#Create a buffer from polygonized features
import ogr, os

def createBuffer(inputfn, outputBufferfn, bufferDist):
    inputds = ogr.Open(inputfn)
    inputlyr = inputds.GetLayer()

    shpdriver = ogr.GetDriverByName('Conifer_Forest_Polygonized')
    if os.path.exists(outputBufferfn):
        shpdriver.DeleteDataSource(outputBufferfn)
    outputBufferds = shpdriver.CreateDataSource(outputBufferfn)
    bufferlyr = outputBufferds.CreateLayer(outputBufferfn, geom_type=ogr.wkbPolygon)
    featureDefn = bufferlyr.GetLayerDefn()

    for feature in inputlyr:
        ingeom = feature.GetGeometryRef()
        geomBuffer = ingeom.Buffer(bufferDist)

        outFeature = ogr.Feature(featureDefn)
        outFeature.SetGeometry(geomBuffer)
        bufferlyr.CreateFeature(outFeature)
        outFeature = None

def main(inputfn, outputBufferfn, bufferDist):
    createBuffer(inputfn, outputBufferfn, bufferDist)


if __name__ == "__Conifer Forest__":
    inputfn = 'Conifer_Forest_Polygonied.shp'
    outputBufferfn = '3km_Conifer_Forest_Polygonied.shp'
    bufferDist = 3000.0

    main(inputfn, outputBufferfn, bufferDist)
#_____________________________________________________________________
#Select SACs and SPAs that are located within 3km buffer from coniferous forest
import numpy as np

from matplotlib.widgets import PolygonSelector
from matplotlib.path import Path


class SelectFromCollection:
    """
    Select indices from a matplotlib collection using `PolygonSelector`.

    Selected indices are saved in the `ind` attribute. This tool fades out the
    points that are not part of the selection (i.e., reduces their alpha
    values). If your collection has alpha < 1, this tool will permanently
    alter the alpha values.

    Note that this tool selects collection objects based on their *origins*
    (i.e., `offsets`).

    Parameters
    ----------
    ax : `~matplotlib.axes.Axes`
        Axes to interact with.
    collection : `matplotlib.collections.Collection` subclass
        Collection you want to select from.
    alpha_other : 0 <= float <= 1
        To highlight a selection, this tool sets all selected points to an
        alpha value of 1 and non-selected points to *alpha_other*.
    """

    def __init__(self, ax, collection, alpha_other=0.3):
        self.canvas = ax.figure.canvas
        self.collection = collection
        self.alpha_other = alpha_other

        self.xys = collection.get_offsets()
        self.Npts = len(self.xys)

        # Ensure that we have separate colors for each object
        self.fc = collection.get_facecolors()
        if len(self.fc) == 0:
            raise ValueError('Collection must have a facecolor')
        elif len(self.fc) == 1:
            self.fc = np.tile(self.fc, (self.Npts, 1))

        self.poly = PolygonSelector(ax, self.onselect)
        self.ind = []

    def onselect(self, verts):
        path = Path(verts)
        self.ind = np.nonzero(path.contains_points(self.xys))[0]
        self.fc[:, -1] = self.alpha_other
        self.fc[self.ind, -1] = 1
        self.collection.set_facecolors(self.fc)
        self.canvas.draw_idle()

    def disconnect(self):
        self.poly.disconnect_events()
        self.fc[:, -1] = 1
        self.collection.set_facecolors(self.fc)
        self.canvas.draw_idle()


if __name__ == '__main__':
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots()
    grid_size = 5
    grid_x = np.tile(np.arange(grid_size), grid_size)
    grid_y = np.repeat(np.arange(grid_size), grid_size)
    pts = ax.scatter(grid_x, grid_y)

    selector = SelectFromCollection(ax, pts)

    print("Select points in the figure by enclosing them within a polygon.")
    print("Press the 'esc' key to start a new polygon.")
    print("Try holding the 'shift' key to move all of the vertices.")
    print("Try holding the 'ctrl' key to move a single vertex.")

    plt.show()

    selector.disconnect()

    # After figure is closed print the coordinates of the selected points
    print('\nSelected points:')
    print(selector.xys[selector.ind])
# Congratulations, you are now finished coding________________________________________________________________________________________