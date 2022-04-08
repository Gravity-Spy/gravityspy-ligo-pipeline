
from gravityspy_ligo.subject.subject import GravitySpySubject

sub = GravitySpySubject(event_time=1253940741.813, ifo="H1", manual_list_of_auxiliary_channel_names=["H1:ASC-AS_A_RF45_Q_YAW_OUT_DQ", "H1:LSC-REFL_A_LF_OUT_DQ"])

sub.make_omega_scans(verbose=True)
sub.upload_to_zooniverse(subject_set_id=103434)

