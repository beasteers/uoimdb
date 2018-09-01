import os
import yaml
from .utils import easydict as edict
import pkg_resources

# default config location
__BASE_CONFIG__ = 'config.yaml'
__DEFAULT_CONFIG__ = __BASE_CONFIG__

cfg = None
loaded_configs = {}


def get_config_text(filename=None, check_local=True):
	'''Load the text content of a config file.
	If filename exists, it will load that text content.
	If filename was ommitted it will look for the default config file in the current working directory: config.yaml.
	If that doesn't exist, it will get the default config file from the package directory.
	'''
	if filename is None: # get default config file (stored inside package)
		filename = __DEFAULT_CONFIG__

	if check_local and os.path.isfile(filename):
		with open(filename, 'rt') as f:
			return f.read()
	else:
		return pkg_resources.resource_string(__name__, filename)
		

def get_config(filename=None):
	global cfg

	if isinstance(filename, dict):
		cfg = filename # filename is already a loaded config
		return cfg

	# if no filename specified, will return the last retrieved config if it exists
	if filename is None and cfg is not None:
		return cfg 

	# get cached config file if it exists
	if filename in loaded_configs: 
		return loaded_configs[filename]




	# get base config
	cfg = edict(**yaml.load(pkg_resources.resource_string(__name__, __BASE_CONFIG__)))

	if filename is None: 
		filename = __DEFAULT_CONFIG__

	# get locally defined configuration options
	if os.path.isfile(filename): 
		with open(filename, 'rt') as f:
			cfg.update(**yaml.load(f))




	# get image base directory from secret location
	if not cfg.IMAGE_BASE_DIR:
		image_dir = None

		env_name = cfg.get('IMAGE_BASE_DIR_ENVVAR', 'UOIMAGES_DIR')
		if env_name:
			image_dir = os.getenv(env_name)

		if not image_dir:
			try:
				dir_loc = cfg.get('IMAGE_BASE_DIR_LOC', '{}.txt'.format(env_name))
				with open(dir_loc, 'r') as f:
					image_dir = f.read().strip()
			except OSError:
				raise ValueError('Missing image location. Please specify the path as IMAGE_BASE_DIR in your config file, export it as ${}, or dump the value in {}.'.format(env_name, dir_loc))

		cfg.IMAGE_DIR = base_dir


	loaded_configs[filename] = cfg
	return cfg


# def add_cl_args(args_list):
# 	pass


def main():
	import argparse
	parser = argparse.ArgumentParser(description='')
	parser.add_argument('action', help='create|...')
	parser.add_argument('-f', '--file', default=None, help='the config file to use (input file to read)')
	parser.add_argument('-o', '--out', default='config.yaml', help='the config file to write to (output file path)')
	args = parser.parse_args()

	if args.action == 'create':
		text = pkg_resources.resource_string(__name__, __BASE_CONFIG__).decode('utf-8')
		text = '\n'.join([
			'# ' + line
			for line in text.split('\n')
		])

		if not text:
			print('Config file is empty.')
		with open(args.out, 'w') as f:
			f.write(text)

if __name__ == '__main__':
	main()

