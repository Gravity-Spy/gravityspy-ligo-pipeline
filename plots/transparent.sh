#!bin/bash

magick convert H1_D8LC2PvaimingzzYvEPo_H1-GDS-CALIB_STRAIN_spectrogram_1.0.png -crop 575x480+100+60 -alpha set -channel A -evaluate multiply 0.4 -channel G -evaluate multiply 0.2 -channel B -evaluate multiply 0.2 half_transparent.png
magick composite half_transparent.png  H1_D8LC2PvaimingzzYvEPo_H1-ASC-AS_A_RF45_Q_PIT_OUT_DQ_spectrogram_1.0.png -gravity center -geometry -13 -background none modified.png
rm half_transparent.png