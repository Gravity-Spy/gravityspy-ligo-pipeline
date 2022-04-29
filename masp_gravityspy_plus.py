# from gravityspy_ligo.subject.subject import GravitySpySubject 
# from gravityspy_ligo.utils import utils   
# config = utils.GravitySpyConfigFile(plot_time_ranges=[8.0, 4.0, 1.0])  
# # sub = GravitySpySubject(event_time=1253940741.813, ifo="H1", manual_list_of_auxiliary_channel_names=["H1:ASC-AS_A_RF45_Q_YAW_OUT_DQ", "H1:LSC-REFL_A_LF_OUT_DQ"])
# #sub = GravitySpySubject(event_time=1253940741.813, ifo="H1", manual_list_of_auxiliary_channel_names=["H1:ASC-AS_A_RF45_Q_YAW_OUT_DQ", "H1:LSC-REFL_A_LF_OUT_DQ", "H1:ASC-AS_A_RF45_Q_PIT_OUT_DQ", "H1:SUS-PR3_M3_OPLEV_PIT_OUT_DQ", "H1:LSC-REFL_A_RF45_I_ERR_DQ", "H1:LSC-POP_A_LF_OUT_DQ"])
# sub = GravitySpySubject(event_time=1238210670.782, ifo="H1")

# sub.make_omega_scans(verbose=True, config=config)
# #sub.combine_images_for_subject_upload()
# sub.upload_to_zooniverse(subject_set_id=103434)

from gravityspy_ligo.subject.subject import GravitySpySubject
from gravityspy_ligo.table.events import Events
from gravityspy_ligo.utils import utils

### Select the parameters of the spectrograms/q_transforms you will be plotting (including all of the different plotting windows your would like
config = utils.GravitySpyConfigFile(plot_time_ranges=[8.0, 4.0, 1.0])

### Get list of GPS times which correspond with a glitch occur in the main channel. This can either be done manually, querying omicron directly, or uses hveto's glitches list for a given day.
start_time = 1262790978
end_time = 1262822978

list_of_glitch_times = Events.get_triggers(start=start_time, end=end_time, channel='H1:GDS-CALIB_STRAIN', dqflag=None, algorithm='hveto', verbose=True)

for event_time in list_of_glitch_times['time']:
    sub = GravitySpySubject(event_time=event_time, ifo="H1", auxiliary_channel_correlation_algorithm='hveto')

sub.make_omega_scans(verbose=True, config=config)
sub.save_omega_scans(verbose=True, config=config)
breakpoint()
#sub.upload_to_zooniverse(subject_set_id=103434)