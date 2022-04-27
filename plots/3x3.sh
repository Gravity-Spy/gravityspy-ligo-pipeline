#!bin/bash

magick convert xc:none -fill white blank.png

magick montage -geometry 800x600 -tile 3x3 \
blank.png H1_D8LC2PvaimingzzYvEPo_H1-GDS-CALIB_STRAIN_spectrogram_1.0.png blank.png \
H1_D8LC2PvaimingzzYvEPo_H1-ASC-AS_A_RF45_Q_PIT_OUT_DQ_spectrogram_1.0.png H1_D8LC2PvaimingzzYvEPo_H1-ASC-AS_A_RF45_Q_YAW_OUT_DQ_spectrogram_1.0.png H1_D8LC2PvaimingzzYvEPo_H1-LSC-POP_A_LF_OUT_DQ_spectrogram_1.0.png \
H1_D8LC2PvaimingzzYvEPo_H1-LSC-REFL_A_LF_OUT_DQ_spectrogram_1.0.png H1_D8LC2PvaimingzzYvEPo_H1-LSC-REFL_A_RF45_I_ERR_DQ_spectrogram_1.0.png H1_D8LC2PvaimingzzYvEPo_H1-SUS-PR3_M3_OPLEV_PIT_OUT_DQ_spectrogram_1.0.png \
subject_file/subject_file3x3_blank.png


