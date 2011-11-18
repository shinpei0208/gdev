#SUBDIRS = gaes test raid6
SUBDIRS = gaes test

all: $(SUBDIRS)

.PHONY: $(SUBDIRS)

$(SUBDIRS):
	$(MAKE) -C $@ $(TARGET) kv=$(kv) BUILD_DIR=$(BUILD_DIR)

clean:
	$(MAKE) all TARGET=clean
