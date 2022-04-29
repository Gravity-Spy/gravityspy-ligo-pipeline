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

