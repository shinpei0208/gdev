# Makefile

CC		= gcc
CXX		= g++
TARGET		= ocelot hydrazine
CFLAGS          = -O3 -Wall -I $(GDEVDIR)/include -I ../..
CXXFLAGS        = -std=c++0x

ZOMBIE  = $(wildcard ./*~)

%.build:
	$(MAKE) -C $(basename $@)

%.clean:
	$(MAKE) -C $(basename $@) clean

%.all:
	@for subdir in $(TARGET); do \
		$(MAKE) $$subdir.$(basename $@); \
	done

all: build.all

clean: clean.all

.PHONY: $(TARGET)

$(TARGET):
	$(MAKE) $@.build
