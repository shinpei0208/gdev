SUBDIRS = calg2 callgpu libsrv_test

all: $(SUBDIRS)

.PHONY: $(SUBDIRS)

$(SUBDIRS):
	$(MAKE) -C $@ $(TARGET) BUILD_DIR=$(BUILD_DIR)

clean:
	$(MAKE) all TARGET=clean
