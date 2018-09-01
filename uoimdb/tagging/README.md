# Plume Tagging Tool

This is a Flask app meant to help in labeling instances of plumes across consecutive images.

## Current State:
* Loads images into a grid (currently all the same image)
* Can draw bounding boxes
* Can save the bounding boxes to a csv
* Can select from a few image filters
* Calendar page displays all days with images (`/calendar`)
* Table layout to see all of the plume locations recorded. Or you could just open the csv lol. (`/plume-list`)
* Can select a specific date to view images like so (`/?date=%Y-%m-%d`, e.g. `/?date=2018-04-13`)
* Background subtraction

## Next Steps:
* Add more useful image filters
* Add blob detection

## Running the app
Make sure the required packages are installed (`flask`, `pandas`, `opencv-python`). You can just run `pip install -r requirements.txt`.

To run the app:
```bash
export FLASK_APP=app.py; python -m flask run
```

Assuming your flask is set up as default, it should be available at `http://localhost:5000/`.

To run in debug mode (i.e. automatic refresh for app.py):
```bash
export FLASK_DEBUG=1; export FLASK_APP=app.py; python -m flask run
```

Personally, I just put both of these in my `.bashrc`.

## Drawing a Bounding Box
* Click on the image to place the centroid of the bounding box. 
* Drag the point to move it around the image. 
* Drag it outside the bounds of the image to delete it. 
* Hold down shift and drag the center to expand the bounding box. 

There is a bug where (shift) dragging the bounding box to the top/left causes it to grow at half the distance to the mouse as it does when dragging to the bottom/right (which follows the mouse movement). This could be considered a feature, where dragging to the top/left can be used for more granular scrubbing. 
