import setuptools

setuptools.setup(name='uoimdb',
                 version='1.0.0',
                 description='UO Image Pipelines and Annotator App',
                 # long_description=open('README.md').read().strip(),
                 author='Ben Steers',
                 author_email='ben@bensteers.me',
                 # url='http://path-to-my-packagename',
                 packages=setuptools.find_packages(),
                 py_modules=['packagename'],
                 package_data={'uoimdb': {'*.yaml', 'tagging/static/*/*', 'tagging/templates/*'}},
                 # include_package_data=True,
                 install_requires=[
                        'opencv-python', 
                        'dill', 'imageio', 'numpy', 'pandas', 'matplotlib', 'filelock', # uoimdb
                        'PyYAML', # config.py
                        # 'Flask', 
                        'Flask-Login', 'pyopenssl', # app.py
                        'scikit-learn',
                 ],
                 entry_points={
                        'console_scripts': [
                                'uo-tagger = uoimdb.tagging.app:run'
                        ],
                 },
                 license='MIT License',
                 zip_safe=False,
                 keywords='urban observatory image database annotator annotation app tagging tag')
