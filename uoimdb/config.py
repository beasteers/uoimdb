import os
import yaml
from .utils import easydict as edict
import pkg_resources

# default config location
__BASE_CONFIG__ = 'config.yaml'
__DEFAULT_CONFIG__ = __BASE_CONFIG__

cfg = None
loaded_configs = {}


def load_config(f, cfg=None):
	'''Load a multisection config file. Use the second section as the front-end data. Any other section is for private settings.
	Arguments:
		f (text/file): the yaml config file
		cfg (easydict or None): the existing config obj to add to, if exists.
	'''
	if cfg is None:
		cfg = edict()
		
	cfg_docs = list(yaml.load_all(f))
	for doc in cfg_docs:
		if doc:
			cfg.update(**doc)
	
	# config that goes to frontend. If it exists, then extend rather than replace.
	frontend = cfg_docs[1] if len(cfg_docs) > 1 else cfg_docs[0]
	cfg.frontend = cfg.frontend or edict()
	if frontend:
		cfg.frontend.update(**frontend)
	
	return cfg

def get_config(filename=None, **kw):
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
	cfg = load_config(pkg_resources.resource_string(__name__, __BASE_CONFIG__))
	

	if filename is None: 
		filename = __DEFAULT_CONFIG__

	# get locally defined configuration options
	if os.path.isfile(filename): 
		with open(filename, 'rt') as f:
			cfg = load_config(f, cfg)
	
	cfg.update(**kw)

	# get image base directory from secret location. was requested for UO images.
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

		
	cfg.LABELS_LOCATION = os.path.join(cfg.DATA_LOCATION, cfg.LABELS_LOCATION)
	cfg.IMAGE_CACHE_LOCATION = os.path.join(cfg.DATA_LOCATION, cfg.IMAGE_CACHE_LOCATION)
	cfg.RANDOM_SAMPLE_LOCATION = os.path.join(cfg.DATA_LOCATION, cfg.RANDOM_SAMPLE_LOCATION)

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
	parser.add_argument('-o', '--out', default=__DEFAULT_CONFIG__, help='the config file to write to (output file path)')
	parser.add_argument('-k', '--key', default='', help='the keys to get. i.e. BG.WINDOW')
	args = parser.parse_args()

	print()

	if args.action == 'print':
		print(pkg_resources.resource_string(__name__, args.file or __BASE_CONFIG__).decode('utf-8'))

	elif args.action == 'create':
		text = pkg_resources.resource_string(__name__, args.file or __BASE_CONFIG__).decode('utf-8')
		text = '\n'.join([
			'# ' + line if not line.startswith('---') else line
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

