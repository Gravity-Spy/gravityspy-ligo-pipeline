#!bin/bash

magick convert H1_D8LC2PvaimingzzYvEPo_H1-GDS-CALIB_STRAIN_spectrogram_1.0.png -crop 575x480+100+60 -alpha set -channel A -evaluate multiply 0.4 -channel G -evaluate multiply 0.2 -channel B -evaluate multiply 0.2 half_transparent.png

magick composite half_transparent.png  H1_D8LC2PvaimingzzYvEPo_H1-ASC-AS_A_RF45_Q_PIT_OUT_DQ_spectrogram_1.0.png -gravity center -geometry -13 -background none isOverlap1.png
magick composite half_transparent.png  H1_D8LC2PvaimingzzYvEPo_H1-ASC-AS_A_RF45_Q_YAW_OUT_DQ_spectrogram_1.0.png -gravity center -geometry -13 -background none isOverlap2.png
magick composite half_transparent.png  H1_D8LC2PvaimingzzYvEPo_H1-LSC-POP_A_LF_OUT_DQ_spectrogram_1.0.png -gravity center -geometry -13 -background none isOverlap3.png

magick convert -size 800x600 xc:none -stroke red -strokewidth 5 -fill none -draw "rectangle 100,350 674,540" box.png

magick composite box.png H1_D8LC2PvaimingzzYvEPo_H1-GDS-CALIB_STRAIN_spectrogram_1.0.png isBox.png
magick composite box.png H1_D8LC2PvaimingzzYvEPo_H1-ASC-AS_A_RF45_Q_PIT_OUT_DQ_spectrogram_1.0.png isBox1.png
magick composite box.png H1_D8LC2PvaimingzzYvEPo_H1-ASC-AS_A_RF45_Q_YAW_OUT_DQ_spectrogram_1.0.png isBox2.png
magick composite box.png H1_D8LC2PvaimingzzYvEPo_H1-LSC-POP_A_LF_OUT_DQ_spectrogram_1.0.png isBox3.png

magick montage -geometry 800x600 -tile 3x \
isBox.png isBox.png isBox.png \
isBox1.png isBox2.png isBox3.png \
isOverlap1.png isOverlap2.png isOverlap3.png \
subject_file/subject_file3x3_combination.png

rm box.png
rm isBox.png isBox1.png isBox2.png isBox3.png
rm half_transparent.png
rm isOverlap1.png isOverlap2.png isOverlap3.png 
