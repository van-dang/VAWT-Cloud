CFLAGS  = `pkg-config --cflags dolfin`
LIBS    = `pkg-config --libs dolfin`
CXX     = `pkg-config --variable=compiler dolfin` -g

DEST    = vawt_prepro
OBJECTS = main.o ufc2/NSE.o NodeNormal.o AdaptiveRefinement.o AdaptiveRefinementProjectScalar.o  AdaptiveRefinementProjectVector.o 

all: $(DEST)

install:

clean:
	-rm -f *.o core *.core $(OBJECTS) $(DEST)

$(DEST): $(OBJECTS)
	$(CXX) -o $@ $(OBJECTS) $(CFLAGS) $(LIBS)

.cpp.o:
	$(CXX) $(CFLAGS) -c $< -o $@
