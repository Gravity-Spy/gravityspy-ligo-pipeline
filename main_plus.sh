#!bin/bash
  
magick montage -geometry 800x600 -tile x2 H1_D8LC2PvaimingzzYvEPo_H1-ASC-AS_A_RF45_Q_PIT_OUT_DQ_spectrogram_1.0.png H1_D8LC2PvaimingzzYvEPo_H1-ASC-AS_A_RF45_Q_YAW_OUT_DQ_spectrogram_1.0.png temporary1x2.png
magick montage -mode Concatenate -tile 3x H1_D8LC2PvaimingzzYvEPo_H1-LSC-POP_A_LF_OUT_DQ_spectrogram_1.0.png H1_D8LC2PvaimingzzYvEPo_H1-LSC-REFL_A_LF_OUT_DQ_spectrogram_1.0.png H1_D8LC2PvaimingzzYvEPo_H1-LSC-REFL_A_RF45_I_ERR_DQ_spectrogram_1.0.png temporary3x1.png
magick montage -mode Concatenate -tile 2x \( H1_D8LC2PvaimingzzYvEPo_H1-GDS-CALIB_STRAIN_spectrogram_1.0.png -resize 200% \) temporary1x2.png temporary3x1.png subject_file/subject_file_plus.png
rm temporary1x2.png
rm temporary3x1.png
