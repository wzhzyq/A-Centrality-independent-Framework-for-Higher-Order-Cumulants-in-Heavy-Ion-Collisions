#!/bin/bash

filenames=$(ls rootfiles/*)
mkdir -p EachBinDistributionFromRef3
# mkdir -p EachBinDistributionFromNpart
for filename in $filenames
do
    root -l -b -q First_GetPDFEachBin.C\(\"$filename\"\)
done

# root -l -b -q Second_define_centrality.c\(\"$filename\"\)
