# Makefile

#include ../../../API.mk

CC		= gcc
CXX		= g++
YACC		= bison
LEX		= flex
CFLAGS		= -O3 -Wall -I $(GDEVDIR)/include -I ../..
CXXFLAGS	= -std=c++0x

ifeq ($(ENABLE_CUBIN_MODULE),1)
	CXXFLAGS += -DENABLE_CUBIN_MODULE
endif

OBJS		= PTXLexer.o PTXParser.o ptx.o ptxgrammar.o

ZOMBIE		= $(wildcard ./*~)

all: $(OBJS)

implementation/%.cpp:implementation/%.yy
	(TEMPFILE=`mktemp`; \
	$(YACC) -o $@ --defines=$$TEMPFILE $^; \
	if [ $$? -ne 0 ] ; then unlink $$TEMPFILE; exit 1; fi; \
	cat $$TEMPFILE | sed 's/^int yyparse (.*);$$/\/\/\0/g' > ../../$(basename $(notdir $@)).hpp; \
	unlink $$TEMPFILE)

implementation/%.cpp:implementation/%.ll
	$(LEX) -o$@ $^

%.o:implementation/%.cpp
	$(CXX) -fPIC -c $^ -o $@ $(CXXFLAGS) $(CFLAGS)

%.o:implementation/%.c
	$(CC) -fPIC -c $^ -o $@ $(CFLAGS)

../../ptxgrammar.hpp: implementation/ptxgrammar.cpp

implementation/PTXLexer.cpp: ../../ptxgrammar.hpp

clean: ptxgrammar-clean
	rm -f $(OBJS) $(ZOMBIE)

ptxgrammar-clean:
	rm -f ../../ptxgrammar.hpp implementation/ptxgrammar.cpp
