# Makefile

CC		= gcc
CXX		= g++
YACC		= bison
LEX		= flex
CFLAGS          = -O3 -Wall -I $(GDEVDIR)/include -I ../..
CXXFLAGS        = -std=c++0x

OBJS 	= PTXLexer.o PTXParser.o ptx.o ptxgrammar.o
ZOMBIE  = $(wildcard ./*~)

all: $(OBJS)

implementation/%.cpp:implementation/%.yy
	$(YACC) -o $@ $^

implementation/%.cpp:implementation/%.ll
	$(LEX) -o$@ $^

%.o:implementation/%.cpp
	$(CXX) -fPIC -c $^ -o $@ $(CXXFLAGS) $(CFLAGS)

%.o:implementation/%.c
	$(CC) -fPIC -c $^ -o $@ $(CFLAGS)

clean:
	rm -f $(OBJS) $(ZOMBIE)

