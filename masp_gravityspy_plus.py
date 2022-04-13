from gravityspy_ligo.subject.subject import GravitySpySubject 
from gravityspy_ligo.utils import utils   
config = utils.GravitySpyConfigFile(plot_time_ranges=[8.0, 4.0, 1.0])  
sub = GravitySpySubject(event_time=1253940741.813, ifo="H1", manual_list_of_auxiliary_channel_names=["H1:ASC-AS_A_RF45_Q_YAW_OUT_DQ", "H1:LSC-REFL_A_LF_OUT_DQ"])
#sub = GravitySpySubject(event_time=1253940741.813, ifo="H1", manual_list_of_auxiliary_channel_names=["H1:ASC-AS_A_RF45_Q_YAW_OUT_DQ", "H1:LSC-REFL_A_LF_OUT_DQ", "H1:ASC-AS_A_RF45_Q_PIT_OUT_DQ", "H1:SUS-PR3_M3_OPLEV_PIT_OUT_DQ", "H1:LSC-REFL_A_RF45_I_ERR_DQ", "H1:LSC-POP_A_LF_OUT_DQ"])

sub.make_omega_scans(verbose=True, config=config)
#sub.combine_images_for_subject_upload()
sub.upload_to_zooniverse(subject_set_id=103434)

