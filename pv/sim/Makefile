#V=1

# Define V=1 for a more verbose compilation
ifndef V
	QUIET_AR            = @echo 'MAKE:' AR $@;
	QUIET_BUILD         = @echo 'MAKE:' BUILD $@;
	QUIET_C             = @echo 'MAKE:' CC $@;
	QUIET_CXX           = @echo 'MAKE:' CXX $@;
	QUIET_CHECKPATCH    = @echo 'MAKE:' CHECKPATCH $(subst .o,.cc,$@);
	QUIET_CHECK         = @echo 'MAKE:' CHECK $(subst .o,.cc,$@);
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

CC = clang++

INCDIR :=
INCDIR += -I../inc

CXX_FLAGS :=
CXX_FLAGS += -Wall
CXX_FLAGS += -pedantic

LD_FLAGS :=

LD_LIBS :=
LD_LIBS += -lm

release: CXX_FLAGS += -O3
release: $(TARGET)

debug: CXX_FLAGS += -O0
debug: CXX_FLAGS += -DVERBOSE
debug: CXX_FLAGS += -g
debug: $(TARGET)

# TODO: generate call graph with -O2
profile: CXX_FLAGS += -O0
profile: CXX_FLAGS += -fno-omit-frame-pointer
profile: CXX_FLAGS += -fno-inline-functions
profile: CXX_FLAGS += -pg
profile: CXX_FLAGS += -g
profile: LD_FLAGS += -pg
profile: $(TARGET)

VPATH :=
VPATH += ../inc
VPATH += ../src

CXX_SOURCES :=
# CXX_SOURCES += fc.cc
CXX_SOURCES += fc_hls.cc
CXX_SOURCES += rnn.cc
CXX_SOURCES += init.cc
CXX_SOURCES += main.cc
CXX_SOURCES += utils.cc
CXX_SOURCES += softmax.cc
CXX_SOURCES += activation.cc

CXX_HEADERS :=
CXX_HEADERS += fc.h
CXX_HEADERS += rnn.h
CXX_HEADERS += init.h
CXX_HEADERS += types.h
CXX_HEADERS += config.h
CXX_HEADERS += softmax.h
CXX_HEADERS += constants.h
CXX_HEADERS += activation.h

.SUFFIXES: .cc .h .o

CXX_OBJECTS := $(CXX_SOURCES:.cc=.o)

$(CXX_OBJECTS): $(CXX_HEADERS)

$(TARGET): $(CXX_OBJECTS)
	$(QUIET_LINK)$(CC) -o $@ $(CXX_OBJECTS) ${LD_LIBS} ${LD_FLAGS}

.cc.o:
	$(QUIET_C)$(CC) $(CXX_FLAGS) ${INCDIR} -c $<

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
	$(QUIET_RUN)gprof2dot gprof.log -n0 -e0 -w -c color -o gprof.dot
	$(QUIET_RUN)dot gprof.dot -Tsvg -o gprof.svg
.PHONY: gprof

view-gprof:
ifeq (,$(wildcard ./gprof.svg))
	$(QUIET_INFO)echo "You should \"make gprof\" first."
else
	eog gprof.svg
endif
.PHONY: view-gprof

clean:
	$(QUIET_CLEAN)rm -rf *.o $(TARGET) *.log *.out *.svg *.pdf *.dot *.sum
.PHONY: clean
