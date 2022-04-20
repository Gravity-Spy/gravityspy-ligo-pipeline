#!bin/bash

magick convert H1_D8LC2PvaimingzzYvEPo_H1-GDS-CALIB_STRAIN_spectrogram_1.0.png -crop 575x480+100+60 -alpha set -channel A -evaluate multiply 0.4 -channel G -evaluate multiply 0.2 -channel B -evaluate multiply 0.2 half_transparent.png

magick composite half_transparent.png  H1_D8LC2PvaimingzzYvEPo_H1-ASC-AS_A_RF45_Q_PIT_OUT_DQ_spectrogram_1.0.png -gravity center -geometry -13 -background none modified1.png
magick composite half_transparent.png  H1_D8LC2PvaimingzzYvEPo_H1-ASC-AS_A_RF45_Q_YAW_OUT_DQ_spectrogram_1.0.png -gravity center -geometry -13 -background none modified2.png
magick composite half_transparent.png  H1_D8LC2PvaimingzzYvEPo_H1-LSC-POP_A_LF_OUT_DQ_spectrogram_1.0.png -gravity center -geometry -13 -background none modified3.png

magick montage -geometry 800x600 -tile 3x \
H1_D8LC2PvaimingzzYvEPo_H1-GDS-CALIB_STRAIN_spectrogram_1.0.png H1_D8LC2PvaimingzzYvEPo_H1-GDS-CALIB_STRAIN_spectrogram_1.0.png H1_D8LC2PvaimingzzYvEPo_H1-GDS-CALIB_STRAIN_spectrogram_1.0.png \
H1_D8LC2PvaimingzzYvEPo_H1-ASC-AS_A_RF45_Q_PIT_OUT_DQ_spectrogram_1.0.png H1_D8LC2PvaimingzzYvEPo_H1-ASC-AS_A_RF45_Q_YAW_OUT_DQ_spectrogram_1.0.png H1_D8LC2PvaimingzzYvEPo_H1-LSC-POP_A_LF_OUT_DQ_spectrogram_1.0.png \
modified1.png modified2.png modified3.png \
subject_file/subject_file3x3_transparent.png

rm modified1.png modified2.png modified3.png 

