# uoimdb Tools

## Configuration

These are all of the available utilities from the config file script, `uoimdb/config.py`.

```bash
# create a local copy of the config file to overwrite options
python -m uoimdb.config create 
sublime config.yaml

# print out base config file to the console
python -m uoimdb.config print

# print out a config value
python -m uoimdb.config get -k my.dot.separated.key
```

## Dataset Generation

There are a few helpful ways to generate dummy datasets, handled by the `uoimdb/tools/generate.py` script.

One way is to randomly sample images from a folder. This can be done like so:

```bash
python -m uoimdb.tools.generate random sample-images/*.jpg
```

Another way is to extract frames out of a video. 

```bash
python -m uoimdb.tools.generate video sample-videos/my-video.jpg -n 1000 # limit 1000 frames
```

There are options for setting the timestamps as well.
- `--start, -s`: the start datetime
- `--end, -e`: the end datetime
- `-n`: the number of images to create
- `--frequency, -f`: the frequency string to select images at. e.g. 1h15m10s

Obviously, specifying all of them would give an overconstrained result. 

```bash
python -m uoimdb.tools.generate random *.jpg --start "2018-01-01 05:00:06" --end "2018-01-01 05:00:06" -f 10m
```

