# TARGET NAME
TARGET = hub_firmware
TARGET1a = gpc_firmware

# COMPILER
CLANG = clang
CLANG_FLAGS = -m32 -emit-llvm -S -O1
ENVYAS = envyas
ENVYAS_FLAGS = -m fuc -V fuc3
LLC = llc
LLC_FLAGS = -march=nvfc

# DIR
NVFC_ROOT = ../../../nvfc
SCRIPT_DIR = $(NVFC_ROOT)/script
GDEV_PSCNV_DIR = ../../mod/pscnv/

# SCRIPT
LLVM2ENVYAS = $(SCRIPT_DIR)/llvm2envyas
HEX2BIN = $(SCRIPT_DIR)/hex2bin
HEX2BYTELIST = $(SCRIPT_DIR)/hex2bytelist
BIN2CTXCTLH = $(SCRIPT_DIR)/datalist

# LOADER
LOADER = $(NVFC_ROOT)/loader_firmware/fucload
CAT = cat
OPT = opt

all:
	make 09
	make 1a

	
09: hub.c
	$(CLANG) $< -o $(TARGET).bc $(CLANG_FLAGS)
	$(LLC) $(TARGET).bc -o $(TARGET).s $(LLC_FLAGS)
	$(LLVM2ENVYAS) $(TARGET).s -c $(TARGET).tmp -d $(TARGET).data.S
	$(CAT) boot.S $(TARGET).tmp > $(TARGET).S

1a: gpc.c
	$(CLANG) $< -o $(TARGET1a).bc $(CLANG_FLAGS)
	$(LLC) $(TARGET1a).bc -o $(TARGET1a).s $(LLC_FLAGS)
	$(LLVM2ENVYAS) $(TARGET1a).s -c $(TARGET1a).tmp -d $(TARGET1a).data.S
	$(CAT) boot.S $(TARGET1a).tmp > $(TARGET1a).S

clean:
	rm -f $(TARGET).* $(TARGET1a).* hub_ctxctl.h.data gpc_ctxctl.h.data nvc0_ctxctl.h*  *~

h:
	$(ENVYAS) $(TARGET).S $(ENVYAS_FLAGS) > $(TARGET).bytelist
	$(ENVYAS) $(TARGET1a).S $(ENVYAS_FLAGS)  > $(TARGET1a).bytelist
	$(HEX2BYTELIST) $(TARGET).data.S $(TARGET).data.bytelist
	$(HEX2BYTELIST) $(TARGET1a).data.S $(TARGET1a).data.bytelist
	$(BIN2CTXCTLH) $(TARGET).data.bytelist $(TARGET).bytelist -m hub
	$(BIN2CTXCTLH) $(TARGET1a).data.bytelist $(TARGET1a).bytelist -m gpc
	echo > nvc0_ctxctl.h
	$(CAT) nvc0_ctxctl.h.hub >> nvc0_ctxctl.h
	$(CAT) nvc0_ctxctl.h.gpc >> nvc0_ctxctl.h
	make install

install :
	cp nvc0_ctxctl.h $(GDEV_PSCNV_DIR)

run:
	$(ENVYAS) -i $(TARGET).S $(ENVYAS_FLAGS)   > $(TARGET).bin
	$(ENVYAS) -i $(TARGET1a).S $(ENVYAS_FLAGS)   > $(TARGET1a).bin
	$(HEX2BIN) $(TARGET).data.S $(TARGET).data.bin
	$(HEX2BIN) $(TARGET1a).data.S $(TARGET1a).data.bin
	sudo $(LOADER) -09c $(TARGET).bin -09d $(TARGET).data.bin -1ac $(TARGET1a).bin -1ad $(TARGET1a).data.bin

#install:
#	cp $(TARGET).bin /lib/firmware/nouveau/fuc409c
#	cp $(TARGET).data.bin /lib/firmware/nouveau/fuc409d
