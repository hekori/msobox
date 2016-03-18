
all: CXXFLAGS += -O3
all: CCFLAGS += -O3
all: FFLAGS += -O3
all: shlib

debug: CXXFLAGS += -DDEBUG -g -O0
debug: CCFLAGS += -DDEBUG -g -O0
debug: FFLAGS += -DDEBUG -g -O0
debug: shlib 

FC         = gfortran
LIB        = so
MAKELIB    = $(CXX) -shared -o
COMPILE_C  = $(CC) $(CXXFLAGS)    -c -o $@  -fPIC
COMPILE_F  = $(FC) $(FFLAGS) -c -o $@ -fPIC -fno-second-underscore -frecursive -finit-local-zero -fdefault-real-8 -fdefault-double-8

%.o : %.c
	$(COMPILE_C) -fPIC $<
%.o : %.f
	$(COMPILE_F) $<

CSRCS=$(wildcard *.c)
COBJS=$(CSRCS:.c=.o)
FSRCS=$(wildcard *.f)
FOBJS=$(FSRCS:.f=.o)

shlib: $(FOBJS) $(COBJS)
	$(MAKELIB) libproblem.$(LIB) *.o -lgfortran

clean:
	rm -f *.o
	rm -f *.so

