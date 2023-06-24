


修改`lsh/cMinhash.cpp`文件：
- 将exc_type改为curexc_type
- 将exc_value改为curexc_value
- 将exc_traceback改为curexc_traceback




```
> python setup.py install
running install
/usr/local/lib/python3.8/dist-packages/setuptools/command/install.py:34: SetuptoolsDeprecationWarning: setup.py install is deprecated. Use build and pip and other standards-based tools.
  warnings.warn(
/usr/local/lib/python3.8/dist-packages/setuptools/command/easy_install.py:144: EasyInstallDeprecationWarning: easy_install command is deprecated. Use build and pip and other standards-based tools.
  warnings.warn(
running bdist_egg
running egg_info
writing lsh.egg-info/PKG-INFO
writing dependency_links to lsh.egg-info/dependency_links.txt
writing requirements to lsh.egg-info/requires.txt
writing top-level names to lsh.egg-info/top_level.txt
reading manifest file 'lsh.egg-info/SOURCES.txt'
adding license file 'LICENSE'
writing manifest file 'lsh.egg-info/SOURCES.txt'
installing library code to build/bdist.linux-x86_64/egg
running install_lib
running build_py
running build_ext
building 'lsh.cMinhash' extension
x86_64-linux-gnu-gcc -pthread -Wno-unused-result -Wsign-compare -DNDEBUG -g -fwrapv -O2 -Wall -g -fstack-protector-strong -Wformat -Werror=format-security -g -fwrapv -O2 -g -fstack-protector-strong -Wformat -Werror=format-security -Wdate-time -D_FORTIFY_SOURCE=2 -fPIC -I/usr/local/lib/python3.8/dist-packages/numpy/core/include -I/usr/include/python3.8 -c lsh/cMinhash.cpp -o build/temp.linux-x86_64-3.8/lsh/cMinhash.o
In file included from /usr/local/lib/python3.8/dist-packages/numpy/core/include/numpy/ndarraytypes.h:1960,
                 from /usr/local/lib/python3.8/dist-packages/numpy/core/include/numpy/ndarrayobject.h:12,
                 from /usr/local/lib/python3.8/dist-packages/numpy/core/include/numpy/arrayobject.h:5,
                 from lsh/cMinhash.cpp:304:
...
creating dist
creating 'dist/lsh-0.3.0-py3.8-linux-x86_64.egg' and adding 'build/bdist.linux-x86_64/egg' to it
removing 'build/bdist.linux-x86_64/egg' (and everything under it)
Processing lsh-0.3.0-py3.8-linux-x86_64.egg
creating /usr/local/lib/python3.8/dist-packages/lsh-0.3.0-py3.8-linux-x86_64.egg
Extracting lsh-0.3.0-py3.8-linux-x86_64.egg to /usr/local/lib/python3.8/dist-packages
Adding lsh 0.3.0 to easy-install.pth file

Installed /usr/local/lib/python3.8/dist-packages/lsh-0.3.0-py3.8-linux-x86_64.egg
Processing dependencies for lsh==0.3.0
Searching for Cython==0.29.34
Best match: Cython 0.29.34
Adding Cython 0.29.34 to easy-install.pth file
Installing cygdb script to /usr/local/bin
Installing cython script to /usr/local/bin
Installing cythonize script to /usr/local/bin

Using /usr/local/lib/python3.8/dist-packages
Searching for numpy==1.22.2
Best match: numpy 1.22.2
Adding numpy 1.22.2 to easy-install.pth file
Installing f2py script to /usr/local/bin
Installing f2py3 script to /usr/local/bin
Installing f2py3.8 script to /usr/local/bin

Using /usr/local/lib/python3.8/dist-packages
Finished processing dependencies for lsh==0.3.0
```
