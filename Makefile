# Term Project for CSCI5576 -- parallel neural networks
#
# Targets:
#   make            build the serial trainer and every parallel driver
#   make serial     build only the serial trainer
#   make parallel   build the MPI and OpenMP drivers
#   make check      build everything and run a short smoke test
#   make clean      remove build products
#
# Requires: a C compiler, Open MPI (mpicc/mpirun) and, for the OpenMP object,
# a libomp installation.
# On macOS with Homebrew, install them with:
#   brew install open-mpi libomp
#
# libomp is keg-only, so its include and library directories are passed
# explicitly rather than relying on the default search path.

CC      ?= cc
MPICC   ?= mpicc

CFLAGS  ?= -O2 -Wall -g
LDLIBS  ?= -lm

# OpenMP support. Apple clang needs -Xpreprocessor -fopenmp plus the explicit
# libomp paths; the leading -I/-L flags are harmless on toolchains that carry
# OpenMP support natively.
OMP_PREFIX ?= $(shell brew --prefix libomp 2>/dev/null)
OMP_CFLAGS ?= -Xpreprocessor -fopenmp -I$(OMP_PREFIX)/include
OMP_LDFLAGS ?= -L$(OMP_PREFIX)/lib -lomp

# Shared library sources.
LIB_SRCS   = nn.c pprintf.c

SERIAL_BIN = train
PARALLEL_BINS = train_pp train_pp_m train_pp_mpe

# Objects that are compiled but not linked into a driver: the OpenMP library
# variant and the standalone training helpers.
EXTRA_OBJS = nn_mp.o neural_nets.o

.PHONY: all serial parallel check clean

all: serial parallel extra

serial: $(SERIAL_BIN)

parallel: $(PARALLEL_BINS)

extra: $(EXTRA_OBJS)

# Serial trainer: nn.c plus the XOR driver.
$(SERIAL_BIN): train.c $(LIB_SRCS) nn.h pprintf.h
	$(CC) $(CFLAGS) -o $@ train.c $(LIB_SRCS) $(LDLIBS)

# Data-parallel and model-parallel drivers.
train_pp: train_pp.c $(LIB_SRCS) nn.h pprintf.h
	$(MPICC) $(CFLAGS) -o $@ train_pp.c $(LIB_SRCS) $(LDLIBS)

train_pp_m: train_pp_m.c $(LIB_SRCS) nn.h pprintf.h
	$(MPICC) $(CFLAGS) -o $@ train_pp_m.c $(LIB_SRCS) $(LDLIBS)

# MPE tracing is MPICH-only. Define HAVE_MPE to compile the tracing in on a
# system that ships mpe.h; without it the MPE calls become no-ops.
train_pp_mpe: train_pp_mpe.c $(LIB_SRCS) nn.h pprintf.h
	$(MPICC) $(CFLAGS) -o $@ train_pp_mpe.c $(LIB_SRCS) $(LDLIBS)

# The OpenMP variant of the library. It is a library object rather than a
# driver; link it in place of nn.o for an OpenMP build.
nn_mp.o: nn_mp.c nn.h
	$(CC) $(CFLAGS) $(OMP_CFLAGS) -c -o $@ nn_mp.c

# Standalone training helpers (not linked into any driver).
neural_nets.o: neural_nets.c nn.h
	$(CC) $(CFLAGS) -c -o $@ neural_nets.c

check: all
	@echo "--- smoke test: serial trainer, 2 samples x 4 hidden neurons x 100 epochs ---"
	./$(SERIAL_BIN) 2 4 100
	@echo "--- MPI driver: 2 ranks (build check only, requires 2 slots) ---"
	-mpirun -np 2 ./train_pp 2 4 10

clean:
	rm -f $(SERIAL_BIN) $(PARALLEL_BINS) $(EXTRA_OBJS) *.o
	rm -rf *.dSYM
