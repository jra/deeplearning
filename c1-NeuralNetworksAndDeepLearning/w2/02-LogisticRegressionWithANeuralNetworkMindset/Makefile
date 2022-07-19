CC=clang

libreaddata.dylib: readdata.o
	$(CC) -shared -o $@ $< -L/Users/jra/prj/bqn/CBQN -lcbqn -lhdf5

readdata.o: readdata.c
	$(CC) -c -o $@ $< -fPIC
