
# coding: utf-8

# # Out-of-bag (OOB) plot

# In[1]:

import numpy as np
import os

from osgeo import gdal
from sklearn import metrics
from sklearn.ensemble import RandomForestClassifier

from matplotlib import pyplot as plt
get_ipython().magic('matplotlib inline')

# A list of "random" colors (for a nicer output)
COLORS = ["#000000", "#FFFF00", "#1CE6FF", "#FF34FF", "#FF4A46", "#008941"]


# define some useful functions that we are going to be using later. They are making heavy use of the GDAL api to manipulate raster and vector data

# In[2]:

def create_mask_from_vector(vector_data_path, cols, rows, geo_transform, projection, target_value=1):
    """Rasterize the given vector (wrapper for gdal.RasterizeLayer)."""
    data_source = gdal.OpenEx(vector_data_path, gdal.OF_VECTOR)
    layer = data_source.GetLayer(0)
    driver = gdal.GetDriverByName('MEM')  # In memory dataset
    target_ds = driver.Create('', cols, rows, 1, gdal.GDT_UInt16)
    target_ds.SetGeoTransform(geo_transform)
    target_ds.SetProjection(projection)
    gdal.RasterizeLayer(target_ds, [1], layer, burn_values=[target_value])
    return target_ds
def vectors_to_raster(file_paths, rows, cols, geo_transform, projection):
    """Rasterize all the vectors in the given directory into a single image."""
    labeled_pixels = np.zeros((rows, cols))
    for i, path in enumerate(file_paths):
        label = i+1
        ds = create_mask_from_vector(path, cols, rows, geo_transform, projection, target_value=label)
        band = ds.GetRasterBand(1)
        labeled_pixels += band.ReadAsArray()
        ds = None     
    return labeled_pixels
def write_geotiff(fname, data, geo_transform, projection):
    """Create a GeoTIFF file with the given data."""
    driver = gdal.GetDriverByName('GTiff')
    rows, cols = data.shape
    dataset = driver.Create(fname, cols, rows, 1, gdal.GDT_Byte)
    dataset.SetGeoTransform(geo_transform)
    dataset.SetProjection(projection)
    band = dataset.GetRasterBand(1)
    band.WriteArray(data)
    dataset = None  # Close the file


# In[3]:

raster_data_path = "7040/PixelAggregate/Resolution0.01/5bandsFINALImage.tif"
#output_fname = "classification.tiff"
train_data_path = "7040/PixelAggregate/Resolution0.01/ROITraining/"


# Define our input and output.

# In[4]:

#raster_data_path = "data/image/2298119ene2016recorteTT.tif"
#output_fname = "classification.tiff"
#train_data_path = "data/test/"
#validation_data_path = "data/train/"


# # Training

# Now, we will use the GDAL api to read the input GeoTiff: extract the geographic information and transform the band’s data into a numpy array.

# In[4]:

raster_dataset = gdal.Open(raster_data_path, gdal.GA_ReadOnly)
geo_transform = raster_dataset.GetGeoTransform()
proj = raster_dataset.GetProjectionRef()
bands_data = []
for b in range(1, raster_dataset.RasterCount+1):
    band = raster_dataset.GetRasterBand(b)
    bands_data.append(band.ReadAsArray())

bands_data = np.dstack(bands_data)
rows, cols, n_bands = bands_data.shape


# Process the training data: project all the vector data, in the training dataset, into a numpy array. Each class is assigned a label (a number between 1 and the total number of classes). If the value v in the position (i, j) of this new array is not zero, that means that the pixel (i, j) must be used as a training sample of class v.
# training_samples is the list of pixels to be used for training. In our case, a pixel is a point in the 7-dimensional space of the bands.
# training_labels is a list of class labels such that the i-th position indicates the class for i-th pixel in training_samples.

# In[5]:

files = [f for f in os.listdir(train_data_path) if f.endswith('.shp')]
classes = [f.split('.')[0] for f in files]
shapefiles = [os.path.join(train_data_path, f) for f in files if f.endswith('.shp')]

labeled_pixels = vectors_to_raster(shapefiles, rows, cols, geo_transform, proj)
is_train = np.nonzero(labeled_pixels)
training_labels = labeled_pixels[is_train]
training_samples = bands_data[is_train]


# In[ ]:

#classifier = RandomForestClassifier(n_jobs=-1)
#classifier.fit(training_samples, training_labels)


# # OOB Errors for Random Forests

# In[6]:

clf = RandomForestClassifier(warm_start=True, oob_score=True, max_features="sqrt")

#dict of (estimator, error_rate)
error_rate = {}

#explore max 200
min_estimators = 10
max_estimators = 1500

for i in range(min_estimators, max_estimators + 1):
    clf.set_params(n_estimators=i)
    clf.fit(training_samples, training_labels)
    # Record the OOB error for each `n_estimators=i` setting.
    oob_error = 1 - clf.oob_score_
    error_rate[i] = oob_error
    
# Generate the "OOB error rate" vs. "n_estimators" plot.
er = sorted(error_rate.items())
x, y = zip(*er)
fig = plt.figure(figsize=(10,5))
plt.plot(x, y)

plt.xlim(min_estimators, max_estimators)
plt.xlabel("n_estimators")
plt.ylabel("OOB error rate")
plt.show()
fig.savefig('7040/PixelAggregate/Resolution0.01/n1500.png', dpi = 300)


# In[ ]:



