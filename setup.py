from distutils.core import setup

setup (name = "gconcord",
       version = '1.0.0',
       description = "Python package of graphical CONCORD.",
       author = "Zhipu Zhou",
       packages = ['gconcord'],  # the folder parallel to setup.py file
       package_dir = {'graphical_concord_':'graphical_concord'},  # under the above folder, the location of funcs
       package_data = {'gconcord':['sharedlib.so']},
      )