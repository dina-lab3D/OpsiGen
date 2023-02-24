#! /bin/bash
FIRST="Suzy is fat"
SECOND="Sara"
res=${FIRST//Suzy/$SECOND}

echo $res

FIRST="(123456)"
SECOND="\("
FIRST=${FIRST//(/"\("}
res=${FIRST//)/"\)"}

echo $res
