CC=clang

libreaddata.dylib: readdata.o
	$(CC) -shared -o $@ $< -L/Users/jra/prj/bqn/CBQN -lcbqn -lhdf5

readdata.o: readdata.c
	$(CC) -c -o $@ $< -fPIC

#libpyfuns.dylib: pyfuns.o
#	$(CC) -shared -o $@ $< -L/Users/jra/prj/bqn/CBQN -lcbqn -lhdf5

pyfuns: pyfuns.o
	$(CC) -o $@ $< -L/Library/Frameworks/Python.framework/Versions/3.10/lib -L/Users/jra/prj/bqn/CBQN -lpython3.10 -lcbqn

pyfuns.o: pyfuns.c
	$(CC) -c -o $@ $< -fPIC -I/Library/Frameworks/Python.framework/Versions/3.10/include/python3.10

#/Library/Frameworks/Python.framework/Versions/3.10/lib
#libpython3.10.dylib
#/Library/Frameworks/Python.framework/Versions/3.10/include/python3.10
#Python.h

libmagickwand.dylib: magickwand.o
	PKG_CONFIG_PATH=/usr/local/Cellar/imagemagick/7.1.0-32/lib/pkgconfig $(CC) -shared -o $@ $< `pkg-config --libs MagickWand`

magickwand.o: magickwand.c
	PKG_CONFIG_PATH=/usr/local/Cellar/imagemagick/7.1.0-32/lib/pkgconfig $(CC) -c -o $@ $< -fPIC `pkg-config --cflags MagickWand`
