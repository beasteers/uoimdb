# UO Image Processing and Tagging App

Hi! This package contains a web-based tagging application built in Flask that can be used to tag objects in images where the temporal component is a relevant feature of the dataset. The app works by loading images sequentially and laying them out as a video so that the user can traverse the images and observe changes between frames. It also performs on-the-fly image processing so things like background subtraction can be performed without having to waste disk space by pre-computing and saving all background subtracted images.

The package also features the image processing pipelines used for performing the on-the-fly processing.

To install: 
```
pip install git+https://github.com/bensteers/uoimdb.git # only python 3
```

This documentation contains information on:
- [Using the App](using_the_app.md)
- [Using uoimdb Pipelines](uoimdb.ipynb)
- [Customizing the App](customizing_the_app.md)
- [General Tools](tools.md)

## The Project
This package contains two major parts: the image labeling app and the underlying image processing pipelines.

### The App
The images are grouped into days and months and displayed in a calendar-like view.

![Calendar View](https://github.com/bensteers/uoimdb/raw/master/docs/assets/calendar_demo.gif)

By clicking on a day, you can view all of the corresponding images and draw bounding boxes over objects that you spot.

![Object Labeling](https://github.com/bensteers/uoimdb/raw/master/docs/assets/tagging_demo.gif)

Read the documentation above to get a better idea of how to use the app!

### Image Processing Pipelines
Handles the loading of images and their pipelines so that you can process images by just chaining operations together.

```python
imdb = uo.uoimdb() # loading the image database
print('Image DataFrame shape', imdb.df.shape)
imdb.df.head() # just a pandas dataframe
```

That creates an image database based on information from your `config.yaml` file. Basically, it's just an index of all of your images and their file timestamps stored in a pandas DataFrame.

Then, you can go and create image processing pipelines from the images contained in that database. It provides a nice abstract way to construct your pipeline.

```python
srcs = imdb.df.index[:15] # select the first 15 images

# is it just me or is method chaining really nice :)
pipeline = (imdb.load_images(srcs)      # creates pipeline with image srcs fed in
                .grey()                 # convert to greyscale
                .crop(y1=0.1, y2=0.9)   # crop off 10% of the top and bottom of the image
                .bgsub(window=(3,3))    # perform background subtraction using 3 images on either side
                .invert()               # subtract from 255 (now white is zero and black is 255)
                .cmap('bone'))          # apply bone colormap (see matplotlib colormaps)
              
# pipeline is a generator so images are returned as they are computed
# this means that you could run this on your 1 million image dataset, no problem.
for im in pipeline: 
  plt.imshow(im)
  plt.show()
```
