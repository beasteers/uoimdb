import os
import yaml
from .utils import easydict as edict
import pkg_resources

# default config location
__BASE_CONFIG__ = 'config.yaml'
__DEFAULT_CONFIG__ = __BASE_CONFIG__

cfg = None
loaded_configs = {}


		

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
	import json
	import argparse
	parser = argparse.ArgumentParser(description='')
	parser.add_argument('action', help='create: copy a config file locally|print: print base config|get: get config value...')
	parser.add_argument('-f', '--file', default=None, help='the config file to use (input file to read)')
	parser.add_argument('-o', '--out', default='config.yaml', help='the config file to write to (output file path)')
	parser.add_argument('-k', '--key', default='', help='the keys to get. i.e. BG.WINDOW')
	args = parser.parse_args()

	print()

	if args.action == 'print':
		print(pkg_resources.resource_string(__name__, __BASE_CONFIG__).decode('utf-8'))

	elif args.action == 'create':
		text = pkg_resources.resource_string(__name__, __BASE_CONFIG__).decode('utf-8')
		text = '\n'.join([
			'# ' + line
			for line in text.split('\n')
		])

		if not text:
			print('Config file is empty.')
		with open(args.out, 'w') as f:
			f.write(text)

	elif args.action == 'get':
		cfg = get_config(args.file)

		print('getting cfg.{}:'.format(args.key))

		v = cfg
		for k in args.key.split('.'):
			v = v[k]
		print(json.dumps(v, indent=4))


if __name__ == '__main__':
	main()

