# Urban Observatory Image Processing & Tagging App

## Installing

```
pip install git+https://github.com/bensteers/uoimdb.git # I'll push to pip eventually
```
Only Python 3 is supported. (sorry not sorry ¯\\_(ツ)_/¯).

## Quickstart

#### Starting the app
```
uo-tagger
```
Then just go to `http://localhost:5000/` or `http://localhost:5000/cal/` to get started

#### Modifying the config options

```
python -m uoimdb.config create # copy them to your local directory
sublime config.yaml # or your preferred editor, obvs
```

#### Creating a gif

```python
import uoimdb as uo

# instantiate the image database. could be slow depending on the number of images as it needs to read the image timestamps.
imdb = uo.uoimdb()
print(imdb.df.shape)
imdb.df.head()
```

```python
# generate background subtracted gif
pipeline = (imdb.load_images(imdb.df.index[:20]) # load the first 20 images and create a pipeline
		.bgsub2() # perform background subtraction across the series of images
		.to_gif('my-background-subtracted-gif')) # save images as a gif
Image(url=pipeline()) # gifs/my-background-subtracted-gif.gif
```

## Configuration

The options are defined in a `config.yaml` file. You can override configuration options by running `python -m uoimdb.config create` to get a copy of the config file to modify. If the config file exists in a location other than `./config.yaml` you can specify it like so: `imdb = uo.uoimdb(cfg='my-other-config.yaml')` (or `uo-tagger --cfg my-other-config.yaml` for the app). You can also pass individual config options into the uoimdb instantiator, so if you just want to change the image location, just do (for example): `uo.uoimdb(IMAGE_BASE_DIR='my-custom/image/path', IMAGE_FILE_PATTERN='*/*.png')` which will look for only `.png` images under `my-custom/image/path/*/*.png`.

Some configurations of interest may be: 
```yaml
SAVE_EXT: '.jpg' # the default extension that pipelines save images as
SAVE_DIR: './images' # the root directory where images and gifs are saved. images are saved under a name directory.
IMAGE_BASE_DIR: 'images' # the common directory which all images are stored under
IMAGE_FILE_PATTERN: '*' # the glob file pattern for getting all images. relative to IMAGE_DIR

APP_RUN: # the keyword arguments to pass to the Flask app.run(...)
  debug: True 
  port: 5000
  # ...
	
USERS: # TODO: don't store passwords in plain-text lol.
  username: password
	
LABELS:
 - object 1
 - object 2
 - object 3
 
COLORS:
 - #cc0000 # object 1 color
 - #00cc00 # object 2 color
 - #0000cc # object 3 color
```
You also have control over some of the default image processing parameters.


## Customizing The Tagging App

This also provides an application designed to make tagging objects across consecutive images really easy. To start the app, just run: `uo-tagger`.

The configuration options are also found inside `config.yaml` so the methods of extending configuration options are the same as stated above.

Beyond the config file, you can also interface with the app directly if you want to implement custom functionality.

So if you want to implement some custom image processing functionality, you could do:

```python
from uoimdb.tagging.app import TaggingApp, ImageProcessor


class MyImageProcessor(ImageProcessor):
	'''
	This needs 3 things:
		__init__(imdb): creates all of the filters
		filters: contains an (ordered) dictionary containing {name: pipeline}
		process_image(filename, filter): loads and processes image. returns image array (or None)

	'''
	def __init__(self, imdb):
		'''Here we just define our set of filters.'''
		super(MyImageProcessor, self).__init__(imdb)

		self.filters += [
			'Red Image': imdb.pipeline().pipe(lambda im: im[:,:,0])
		]

	# def process_images(self, filename, filter):
	# 	return super(MyImageProcessor, self).process_images(filename, filter)


app = TaggingApp(
	IMAGE_BASE_DIR='my-image-folder',
	LABELS=[
		'cat',
		'bird',
		'sandwich',
	],
	image_processor_class=MyImageProcessor
)

app.run()
```


And the `TaggingApp` class is also relatively easy to extend. There may be some issues with getting Flask to load any custom templates so that will be something to figure out in the future.

```python 
from uoimdb.tagging.app import TaggingApp

class MyTaggingApp(TaggingApp):

    def define_routes(self):
        super(MyTaggingApp, self).define_routes()
        self.my_new_routes()

    def my_new_routes(self):

        @self.app.route('/hello/<name>')
        def hello(name):
            html = '\n'.join([
                '<div><p>{}</p><img src="/filter/Greyscale/{}" /></div>'.format(src, src)
                for src in self.imdb.df.index[:10]
            ])

            return "Hello {}! Here's your first 10 images, in black and white. {}".format(name, html)

if __name__ == '__main__':
    app = MyTaggingApp()
    app.run()
```


