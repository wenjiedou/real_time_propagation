CC=mpiicc
CFLAGS= -O2 -msse4.2 -std=c99 -qopenmp
CFLAGS_DEBUG= -O0 -g -qopenmp 
LIB = -L/opt/intel/compilers_and_libraries_2018.2.199/linux/mkl/lib/intel64 -lmkl_intel_lp64 -lmkl_sequential -lmkl_core -lm

ROOTDIR=.
SRCDIR=$(ROOTDIR)/src
BLDDIR=$(ROOTDIR)/obj

INSTALL_DIR=$(ROOTDIR)/bin

CFLAGSLIB=-shared

$(BLDDIR):
	@mkdir -p $(BLDDIR)

$(BLDDIR)/io.o: | $(BLDDIR)
	$(CC) $(CFLAGS) -c $(SRCDIR)/io.c -o $@

$(BLDDIR)/self_energy.o: | $(BLDDIR)
	$(CC) $(CFLAGS) -c $(SRCDIR)/self_energy.c -o $@

$(BLDDIR)/propagate.o: | $(BLDDIR)
	$(CC) $(CFLAGS) -c $(SRCDIR)/propagate.c -o $@

$(BLDDIR)/main.o: | $(BLDDIR)
	$(CC) $(CFLAGS) -c $(SRCDIR)/main.c -o $@
        

$(BLDDIR)/gf2_mpi: $(BLDDIR)/io.o $(BLDDIR)/propagate.o $(BLDDIR)/main.o $(BLDDIR)/self_energy.o 
	$(CC) $(CFLAGS) -o $@ $^ $(LIB)

all: $(BLDDIR)/gf2_mpi 

clean:
	rm -f $(BLDDIR)/io.o $(BLDDIR)/propagate.o $(BLDDIR)/main.o $(BLDDIR)/self_energy.o
	rm -f $(BLDDIR)/gf2_mpi

install:
	cp -f $(BLDDIR)/gf2_mpi $(INSTALL_DIR)/gf2_mpi

.PHONY: all
