# See: SDSCC/SDS++ Compiler Commands
# https://www.xilinx.com/html_docs/xilinx2018_3/sdaccel_doc/axl1504034384447.html

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
	QUIET_CLEAN         = @echo 'MAKE:' CLEAN $(PWD);
endif

TARGET_BOARD := zcu102

SD_CARD_DIR := ./sd_card

EXECUTABLE := c-rnn.elf

VPATH :=
VPATH += ./src

TOP_HW_FUNCTION := "fc<>"
TOP_HW_FUNCTION_SOURCES := src/fc.cc

INCDIR :=
INCDIR += -I./src

CXX_HOST_SOURCES :=
CXX_HOST_SOURCES += src/fc.cc
CXX_HOST_SOURCES += src/rnn.cc
CXX_HOST_SOURCES += src/init.cc
CXX_HOST_SOURCES += src/main.cc
CXX_HOST_SOURCES += src/utils.cc
CXX_HOST_SOURCES += src/softmax.cc
CXX_HOST_SOURCES += src/activation.cc

SDS_FLAGS :=
SDS_FLAGS += -sds-pf $(TARGET_BOARD)
SDS_FLAGS += -sds-hw $(TOP_HW_FUNCTION) $(TOP_HW_FUNCTION_SOURCES) -sds-end

CXX := sds++ $(SDS_FLAGS)

CXX_FLAGS :=
CXX_FLAGS += -O3
CXX_FLAGS += -Wall
CXX_FLAGS += -verbose

LD_FLAGS :=
LD_FLAGS += -lm
LD_FLAGS += -sdcard $(SD_CARD_DIR)

CXX_OBJECTS := $(CXX_HOST_SOURCES:.c=.o)

#PLATFORM_OBJECTS := $(PLATFORM_SOURCES:.c=.o)

all: $(SD_CARD_DIR) $(EXECUTABLE)
.PHONY: all

$(SD_CARD_DIR):
	mkdir -p sd_card

$(EXECUTABLE): $(CXX_OBJECTS)
	$(CXX) $(CXX_OBJECTS) $(LD_FLAGS) -o $@

%.o: %.c
	$(CXX) -c $(CXX_FLAGS) $(INCDIR) $< -o $@

#clean:
#	$(RM) $(EXECUTABLE) $(CXX_OBJECTS) *.bit

#ultraclean: clean
#	$(RM) -rf _sds sd_card


