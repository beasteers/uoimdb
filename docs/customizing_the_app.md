# Customizing the App

This app was designed to be extensible to other applications aside from labeling plumes. There are a few ways of modifying the app to suit your own use cases. 

## Configurations

The options are defined in a `config.yaml` file. You can override configuration options by running: 
```
python -m uoimdb.config create
``` 
This creates a copy of the config.yaml file locally so that you can override specific options. Some options of interest may be: 
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
 - '#cc0000' # object 1 color
 - '#00cc00' # object 2 color
 - '#0000cc' # object 3 color
```

You also have control over some of the default image processing parameters.

If the config file exists in a location other than `./config.yaml` you can specify it like so: `imdb = uo.uoimdb(cfg='my-other-config.yaml')` (or `uo-tagger --cfg my-other-config.yaml` for the app). You can also pass individual config options into the uoimdb instantiator, so if you just want to change the image location, just do (for example): `uo.uoimdb(IMAGE_BASE_DIR='my-custom/image/path', IMAGE_FILE_PATTERN='*/*.png')` which will look for only `.png` images under `my-custom/image/path/*/*.png`.

For security purposes, the YAML document is divided into multiple documents. The second section will contain any information that is needed in the frontend. All of this will be accessible by the user so nothing sensitive should be found here. All sensitive information should be found in the first section. (further sections could be added and used for other special purposes if needed, but I don't see a relevant need for that at the moment.) The sections are defined by YAML document markers `---`. If this is not present in the document, then the entire configuration file will be sent to the frontend and user authentication information will be displayed plainly. Don't do this unless the app is purely for personal use or you don't care about authentication at all.

## Custom Image Processor

Beyond the config file, you can also interface with the app directly if you want to implement custom functionality.

So if you want to implement some custom image processing functionality, you could do:

```python
from uoimdb.tagging.app import TaggingApp, ImageProcessor


class MyImageProcessor(ImageProcessor):
	'''
	There are 3 core parts to an ImageProcessor:
		__init__(imdb): initializes all of the image pipelines
		filters: contains an (ordered) dictionary containing {name: pipeline}
		process_image(filename, filter): loads and processes image. returns image array (or None for failure)

	'''
	def __init__(self, imdb):
		'''Here we just define our set of filters.'''
		super(MyImageProcessor, self).__init__(imdb)

		self.filters += [
			'Red Channel': imdb.pipeline().pipe(lambda im: im[:,:,0])
		]

	# def process_images(self, filename, filter):
	# 	return super(MyImageProcessor, self).process_images(filename, filter)


# create app with custom configurations and image processing
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

## Custom App Routes

And the `TaggingApp` class is also relatively easy to extend. To load local templates, the easiest way is to just use jinja directly. 

```python 
from uoimdb.tagging.app import TaggingApp
import jinja2

class MyTaggingApp(TaggingApp):

    def define_routes(self):
        super(MyTaggingApp, self).define_routes()
        self.my_new_routes()

    def my_new_routes(self):
        @self.app.route('/hello/<name>')
        def hello(name):
            with open('templates/my-template.j2', 'r') as f:
            	return jinja2.Template(f.read()).render(name=name, srcs=self.imdb.df.index[:10])

if __name__ == '__main__':
    app = MyTaggingApp()
    app.run()
```

And your template is just a jinja file at `./templates/my-template.j2`. 

```html
Hello {{ name }}! Here's your first 10 images, in black and white. 

{% for src in srcs %}
	<div>
		<p>{{ src }}</p>
		<img src='/filter/Greyscale/{{ src }}' alt='' />
	</div>
{% endfor %}

```

