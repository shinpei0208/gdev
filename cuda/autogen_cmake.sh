#!/bin/sh
cat $1 | sed -e "s/^#ifdef\ YYPARSE_PARAM/#if 0\n\0/g" | sed -e "s/^#endif \/\* ! YYPARSE_PARAM \*\//\0\n#endif/g" > a.hpp
cat a.hpp > $1
rm a.hpp
