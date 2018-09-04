# UO Image Processing and Tagging App

Hi! This package contains a web-based tagging application that can be used to tag objects in images where the temporal component is a relevant feature of the dataset. The app works by loading images sequentially and laying them out as a video so that the user can traverse the images and observe changes between frames. It also performs on-the-fly image processing so things like background subtraction can be performed without having to waste disk space by pre-computing and saving all background subtracted images.

The package also features the image processing pipelines used for performing the on-the-fly processing.

This documentation contains information on:
- [Using the App](using_the_app.md)
- [Using uoimdb](uoimdb.ipynb)
- [Customizing the App](customizing_the_app.md)
- [General Tools](tools.md)
