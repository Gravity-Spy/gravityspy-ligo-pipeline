# Workflow:

# Modified files:
subject.py, plot.py, utils.py, hveto_parser.py, table/events.py

# APIs: 

# sub = GravitySpySubject(event_time=event_time, ifo="H1", auxiliary_channel_correlation_algorithm={'hveto':round_number}, number_of_aux_channels_to_show=2)
Define a subject class instance for a gravityspy event on an interferometer. The mandatory parameters are event_time and ifo. In the Hveto correlation algorithm, it will parse auxiliary channels in the specified round that have highest correlation significance to the main channel from Hveto using hveto_parser.py. A number of top N=number_of_aux_channels_to_show auxiliary channels name that have the highest significance is parsed from the SVG file.
## parameters:
    event_time (float): The GPS time at which an excess noise event occurred
    ifo (str): What interferometer had this an excess noise event
    config
    event_generator (str)[optional, None]: The algorithm that tells us an excess noise event occurred
    auxiliary_channel_correlation_algorithm (str)[optional, None]: The algorithm that tells us the names of the top X correlated auxiliary channels with respect to h(t).
    number_of_aux_channels_to_show (int)[optional, None]: This number will determine the top N number of channels from the list provided by the auxiliary_channel_correlation_algorithm that will be kept and shown for this Subject.
    manual_list_of_auxiliary_channel_names (list)[optional, None]: This will override any auxiliary channel list that might have been supplied by the auxiliary_channel_correlation_algorithm and force this to be the auxiliary channels that are associated with this Subject.


Or iterate with event_time and round in the list of event_time:
## start_time = [a gpstime]
## end_time = [a gpstime]
## list_of_glitch_times = Events.get_triggers(start=start_time, end=end_time, channel='H1:GDS-CALIB_STRAIN', dqflag=None, algorithm='hveto', verbose=True)
## list_of_glitch_times = list_of_glitch_times[0:2]
## for event_time, round_number in zip(list_of_glitch_times['time'], list_of_glitch_times['hveto_round']):

# make_omega_scans(verbose=True):
For the main channel and top N auxiliary channels individually, use subject.make_omega_scans to get the raw time-series data from Ligo and plot spectrogram with specified plot_time_ranges.
## parameters:
    verbose:
    nproc:    
# save_omega_scans(verbose=True):
## config = utils.GravitySpyConfigFile(plot_time_ranges=[8.0, 4.0, 1.0])
Save the spectrogram as the png file.

## parameters:
    verbose (bool)[optional, False]:
    nproc (int)[optional, 1]:
    pool (???)[optional, None]:
    plot_directory = kwargs.pop('plot_directory', os.path.join(os.getcwd(), 'plots', time.from_gps(self.event_time).strftime('%Y-%m-%d'), str(self.event_time)))
    The pool is to use the multiprocessing with nproc number of processors.


# combine_images_for_subject_upload(self, number_of_rows=3, **kwargs):
Concatenate the main channel spectrogram with auxiliary channel spectrograms vertically (main channel spectrogram with number_of_rows auxiliary channel spectrograms for each time duration at a time) to get combined images.

## parameters:
    number_of_rows:
    


# upload_to_zooniverse(self, subject_set_id, project='9979'):
upload the images as well as the metadata that contains the date, filenames, subject_id, main channel name and auxiliary channel names.

## parameters:
    subject_set_id:
    project: