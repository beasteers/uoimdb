# Using the UO Tagging App

## Startup

To start the app, just run:

```
uo-tagger

# you can also specify a custom config file
uo-tagger --cfg custom-config.yaml
```

Then just go to `http://localhost:5000/` or `http://localhost:5000/cal/` to get started.

## Login

The first page you should see is a login page: `http://localhost:5000/login`

By default the username and password are admin:admin, but this can be changed via the config file.

![Login: http://localhost:5000/login/](assets/login.png)

## Calendar

The calendar can be accessed via `http://localhost:5000/` or `http://localhost:5000/cal/`. This contains a summary of the number of images and labels per day. Click a date to access the images for that day.

![Calendar: http://localhost:5000/cal/](assets/calendar_demo.gif)

Click on the `Collected Labels` button for a quick view of all collected labels.

![Collected Labels: http://localhost:5000/label-list/](assets/label_list.gif)

## Random Samples

### Create a Random Sample

To create a single random sample, go to: `/random/<name>/create/` where name is the name of your sample. 

To specify the number of images in your random sample, use `/random/<name>/create/<n>` instead. By default, `n` will use the value of `SAMPLE_SIZE` in your config file.

#### Other filters that are applied:

* **time_range**: only sample from a specific time of day. For example `?time_range=5-18` will only select images between 5am and 6pm. Defaults to `FILTER_TIME_RANGE` in your config file.
* **distance_to_gap**: only sample from images far enough from a time gap. Avoids choosing images where the background subtraction could be invalid. For example `?distance_to_gap=5` will only select images between 5am and 6pm. Defaults to `DISTANCE_FROM_GAP` in your config file.
* **overlap_existing**: by default, a sample will only be created from images that don't exist in another sample. Use this flag if you want to sample from the entire dataset.


#### Creating multiple samples at once

* **n_samples**: To create a number of samples with some percentage of overlap. For 5 overlapping samples use `/random/<name>/create/?n_samples=5`. This will create 5 samples named `<name>-0`, `<name>-1`, etc...
  
* **overlap_ratio**: the proportion that each sample should overlap (only for `n_samples`). For a 20% overlap use `?overlap_ratio=0.2`.

### Delete Random Sample

To delete a random sample, use `/random/<name>/delete/`. 

You can delete multiple samples using a glob pattern. So for example, say you ran `/random/sample/create/?n_samples=5`, you could run `/random/sample-*/delete/` to delete all of the samples you just created.

### Listing a Random Sample

#### Routes

To go to an arbitrary random sample page, go to `/random/`. You can select another random sample from the dropdown in the bottom right of the page.

To see all images in a specific random sample, go to `/random/<name>/`. 

To handle samples with a large number of images, the sample is broken up into pages with a certain number of images per page. This is determined through your config file. To view a specific page of a sample, go to `/random/<name>/p/<page>` (page index starts at 1).

To view another user's ordering, use `/random/<name>/user/<user>/p/<page>`.

#### Request Arguments

* **has_status**: filter by status (for current user). e.g. `?has_status=reviewed`
* **\<user>_has_status**: filter by status for specific user. e.g. `?otherperson_has_status=reviewed`
* **not_status**: filter out status (for current user). e.g. `?not_status=reviewed`
* **\<user>_not_status**: filter out status for specific user. e.g. `?otherperson_not_status=reviewed`
* **only_cached**: only list images that are cached (only checks for filter `DEFAULT_FILTER` in config). e.g. `?only_cached`
* **chronological**: display images ordered by date. e.g. `?chronological`
* **per_page**: temporarily change the number of samples per page.


### Viewing an Image from a Random Sample

Images are specified using their index in the sample. This is specific to the order of a user. To view a specific sample, use `/random/<name>/i/<i>/` to access the `i`th sample (zero-based index). 

To access the indexing of another user, use `/random/<name>/user/<user>/i/<i>/`. 


## Video

### Navigating the Images

The images are loaded sequentially. At the top of the window is a video timeline that shows the current image index that is currently being displayed. Directly below the timeline is a stack of boxes indicating the number of labels gathered at that frame in the image. They only update when the boxes are saved and the page is refreshed. Just below that is the title bar which shows the filename and timestamp of the current image.

To jump to a point in the day, click on the corresponding point in the timeline. 

![video timeline](assets/video-timeline.png)

At the bottom of the window are some video controls. The double arrows are jump backwards and jump forwards respectively. The default step is 10 frames, but this is configurable. The single arrows are step back and step forwards 1 frame, respectively. The play/pause button plays and pauses the video. Not that any of that needed to be spelled out. The image icon toggles the current filter to and from the Original image for easy switching between the two. The icon on the left expands the controls menu.

The controls menu has a slider for changing the video playback speed. Moving left to right speeds up the video. The image filter dropdown contains a variety of filters that can be applied to the image, defined in the image processor class. 

![video controls](assets/controls.png)

### Drawing Labels

To draw a label, click anywhere on the image. That will draw an empty box. To expand the box, Shift-click and drag until it is the right size. To adjust the position, just drag the box around the image. To change the label, change the value of the dropdown directly above the box. To keep a box selected, click on it and you will see that even when you move the cursor out of the box, the center handle remains visible. This indicates that a box is selected. To select multiple boxes, just hold down shift when clicking. To delete a box, simply drag it outside of the bounds of the image.

![Labeling: http://localhost:5000/video/\<date\>](assets/tagging_demo.gif)

### Keyboard Shortcuts

- `Space`: play/pause video
- `Left Arrow`: step back 1 image
- `Right Arrow`: step forward 1 image
- `cmd-A, ctl-A`: select all boxes
- `cmd-Delete, ctl-Delete`: delete all selected boxes
- `Up Arrow`: toggle between `Original` filter and the previously selected filter
- `Enter`: carry over ghostboxes. turn into 'real' boxes
- `cmd-S, ctl-S`: save all modified/created boxes
- `1-9`: with boxes selected, click any digit to change the label to the label with that index. see label dropdown for the ordering.
- `cmd-U, ctl-U`: clear the current image status. Use if you visit an image, but don't want it marked as reviewed.
- `Shift-Left Arrow`: Go to the previous sample
- `Shift-Right Arrow`: Go to next sample


