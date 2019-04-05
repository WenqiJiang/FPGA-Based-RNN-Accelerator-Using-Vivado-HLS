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
debug: C_FLAGS += -g
debug: $(TARGET)

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

valgrind: debug
	$(QUIET_RUN)valgrind -v --leak-check=full --max-stackframe=16777216 ./$(TARGET)
.PHONY: valgrind

gdb: $(LOG_DIR) debug
	$(QUIET_RUN)gdb ./$(TARGET)
.PHONY: gdb

clean:
	$(QUIET_CLEAN)rm -rf *.o $(TARGET)
.PHONY: clean
