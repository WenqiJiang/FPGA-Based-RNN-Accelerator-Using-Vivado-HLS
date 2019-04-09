#V=1

# Define V=1 for a more verbose compilation
ifndef V
	QUIET_AR            = @echo 'MAKE:' AR $@;
	QUIET_BUILD         = @echo 'MAKE:' BUILD $@;
	QUIET_C             = @echo 'MAKE:' CC $@;
	QUIET_CXX           = @echo 'MAKE:' CXX $@;
	QUIET_CHECKPATCH    = @echo 'MAKE:' CHECKPATCH $(subst .o,.c,$@);
	QUIET_CHECK         = @echo 'MAKE:' CHECK $(subst .o,.c,$@);
	QUIET_LINK          = @echo 'MAKE:' LINK $@;
	QUIET_CP            = @echo 'MAKE:' CP $@;
	QUIET_MKDIR         = @echo 'MAKE:' MKDIR $@;
	QUIET_MAKE          = @echo 'MAKE:' MAKE $@;
	QUIET_INFO          = @echo -n 'MAKE:' INFO '';
	QUIET_RUN           = @echo 'MAKE:' RUN '';
	QUIET_CLEAN         = @echo 'MAKE:' CLEAN ${PWD};
endif

TARGET = c-rnn

all: release

CC = gcc

INCDIR :=
INCDIR += -I../inc

C_FLAGS :=
C_FLAGS += -Wall

LD_FLAGS :=

LD_LIBS :=
LD_LIBS += -lm

release: C_FLAGS += -O3
release: $(TARGET)

debug: C_FLAGS += -O0
debug: C_FLAGS += -DVERBOSE
debug: C_FLAGS += -g
debug: $(TARGET)

# TODO: generate call graph with -O2
profile: C_FLAGS += -O0
profile: C_FLAGS += -fno-omit-frame-pointer
profile: C_FLAGS += -fno-inline-functions
profile: C_FLAGS += -pg
profile: C_FLAGS += -g
profile: LD_FLAGS += -pg
profile: $(TARGET)

VPATH :=
VPATH += ../inc
VPATH += ../src

C_SOURCES :=
C_SOURCES += fc.c
C_SOURCES += rnn.c
C_SOURCES += main.c
C_SOURCES += softmax.c
C_SOURCES += load_data.c
C_SOURCES += activation.c
C_SOURCES += params_init.c
#C_SOURCES += main_double.c

C_HEADERS :=
C_HEADERS += fc.h
C_HEADERS += rnn.h
C_HEADERS += softmax.h
C_HEADERS += constants.h
C_HEADERS += activation.h
C_HEADERS += params_init.h

.SUFFIXES: .c .h .o

C_OBJECTS := $(C_SOURCES:.c=.o)

$(C_OBJECTS): $(C_HEADERS)

$(TARGET): $(C_OBJECTS)
	$(QUIET_LINK)$(CC) -o $@ $(C_OBJECTS) ${LD_LIBS} ${LD_FLAGS}

.c.o:
	$(QUIET_C)$(CC) $(C_FLAGS) ${INCDIR} -c $<

run-debug: debug
	$(QUIET_RUN)./$(TARGET)
.PHONY: run-debug

run-release: release
	$(QUIET_RUN)./$(TARGET)
.PHONY: run-release

run-profile: profile
	$(QUIET_RUN)./$(TARGET)
.PHONY: run-profile

valgrind: debug
	$(QUIET_RUN)valgrind -v --leak-check=full --max-stackframe=16777216 ./$(TARGET)
.PHONY: valgrind

gdb: $(LOG_DIR) debug
	$(QUIET_RUN)gdb ./$(TARGET)
.PHONY: gdb

# See: https://ftp.gnu.org/old-gnu/Manuals/gprof-2.9.1/html_chapter/gprof_6.html
# In the following target:
# - gmon.out is the gprof output file
# - gmon.sum is the gprof cumulative output file
gprof: run-profile
ifeq (,$(wildcard ./gmon.sum))
# If the cumulative-output file does not exist, create it
	$(QUIET_RUN)mv gmon.out gmon.sum
else
# Keep adding-up profiling results
	$(QUIET_RUN)gprof -s ./$(TARGET) gmon.out gmon.sum
endif
# Analyze the cumulative results
	$(QUIET_RUN)gprof ./$(TARGET) gmon.sum > gprof.log
	$(QUIET_RUN)gprof2dot gprof.log -n0 -e0 -s -w -c color -o gprof.dot
	$(QUIET_RUN)dot gprof.dot -Tsvg -o gprof.svg
.PHONY: gprof

view-gprof: gprof
	eog gprof.svg
.PHONY: view-gprof

clean:
	$(QUIET_CLEAN)rm -rf *.o $(TARGET) *.log *.out *.svg *.pdf *.dot *.sum
.PHONY: clean
