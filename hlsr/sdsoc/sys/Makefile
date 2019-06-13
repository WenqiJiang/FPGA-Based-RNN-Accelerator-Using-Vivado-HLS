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
VPATH += ../src

##########################################
TOP_HW_FUNCTION := wrapper_rnn_fc
TOP_HW_FUNCTION_SOURCES := wrapper.cc
#SUB_HW_FUNCTION_SOURCES :=
#SUB_HW_FUNCTION_SOURCES += fc.cc 
#SUB_HW_FUNCTION_SOURCES += rnn.cc
##########################################

INCDIR :=
INCDIR += -I../src

CXX_HOST_SOURCES :=
CXX_HOST_SOURCES += activation.cc
#CXX_HOST_SOURCES += fc.cc
CXX_HOST_SOURCES += init.cc
CXX_HOST_SOURCES += main.cc
#CXX_HOST_SOURCES += rnn.cc
CXX_HOST_SOURCES += softmax.cc
CXX_HOST_SOURCES += utils.cc
CXX_HOST_SOURCES += wrapper.cc

SDS_FLAGS :=
SDS_FLAGS += -sds-pf $(TARGET_BOARD)
SDS_FLAGS += -sds-hw $(TOP_HW_FUNCTION) $(TOP_HW_FUNCTION_SOURCES) -sds-end
#SDS_FLAGS += -sds-hw $(TOP_HW_FUNCTION) $(TOP_HW_FUNCTION_SOURCES) -files ../src/rnn.cc,../src/fc.cc -sds-end
#SDS_FLAGS += -sds-hw rnn rnn.cc -sds-end
#SDS_FLAGS += -sds-hw fc fc.cc -sds-end
CXX := sds++ $(SDS_FLAGS)

CXX_FLAGS :=
CXX_FLAGS += -O3
CXX_FLAGS += -Wall
CXX_FLAGS += -verbose

LD_FLAGS :=
LD_FLAGS += -lm
LD_FLAGS += -sdcard $(SD_CARD_DIR)

CXX_OBJECTS := $(CXX_HOST_SOURCES:.cc=.o)

#PLATFORM_OBJECTS := $(PLATFORM_SOURCES:.c=.o)

all: $(SD_CARD_DIR) $(EXECUTABLE)
.PHONY: all

$(SD_CARD_DIR):
	$(QUIET_MKDIR)mkdir -p $@

$(EXECUTABLE): $(CXX_OBJECTS)
	$(QUIET_CXX)$(CXX) $(CXX_OBJECTS) $(LD_FLAGS) -o $@

%.o: %.cc
	$(QUIET_CXX)$(CXX) -c $(CXX_FLAGS) $(INCDIR) $< -o $@

clean:
	$(QUIET_CLEAN)$(RM) $(EXECUTABLE) $(CXX_OBJECTS) *.bit *.elf

ultraclean: clean
	$(QUIET_CLEAN)$(RM) -rf _sds $(SD_CARD_DIR) .Xil


