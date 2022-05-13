# Workflow:

# Modified files:
subject.py, plot.py, utils.py, hveto_parser.py, table/events.py

# APIs in Example Run: 

# sub = GravitySpySubject(event_time=event_time, ifo="H1", auxiliary_channel_correlation_algorithm={'hveto':round_number}, number_of_aux_channels_to_show=2)
Define a subject class instance for a gravityspy event on an interferometer. The mandatory parameters are event_time and ifo. In the Hveto correlation algorithm, it will parse auxiliary channels in the specified round that have highest correlation significance to the main channel from Hveto using hveto_parser.py. A number of top N=number_of_aux_channels_to_show auxiliary channels name that have the highest significance is parsed from the SVG file.
## parameters:
    event_time (float): The GPS time at which an excess noise event occurred.
        Or iterate with event_time and round in the list of event_time:
            ### start_time = [a gpstime]
            ### end_time = [a gpstime]
            ### list_of_glitch_times = Events.get_triggers(start=start_time, end=end_time, channel='H1:GDS-CALIB_STRAIN', dqflag=None, algorithm='hveto', verbose=True)
            ### list_of_glitch_times = list_of_glitch_times[0:2]
            ### for event_time, round_number in zip(list_of_glitch_times['time'], list_of_glitch_times['hveto_round']):
            ### sub = GravitySpySubject(**kwargs)
    ifo (str): What interferometer had this an excess noise event.
    config(object): The object of class GravitySpyConfigFile which could be eidt by utils.GravitySpyConfigFile(*args). Parameters and default values:
        sample_frequency=16384, 
        block_time=64,
        search_frequency_range=(10, 2048),
        search_q_range=(4, 64), 
        plot_time_ranges=[0.5, 1.0, 2.0, 4.0],
        plot_normalized_energy_range=(0, 25.5)
    event_generator (str)[optional, None]: The algorithm that tells us an excess noise event occurred.
    auxiliary_channel_correlation_algorithm (str)[optional, None]: The algorithm that tells us the names of the top X correlated auxiliary channels with respect to h(t)
    number_of_aux_channels_to_show (int)[optional, None]: This number will determine the top N number of channels from the list provided by the auxiliary_channel_correlation_algorithm that will be kept and shown for this Subject.
    manual_list_of_auxiliary_channel_names (list)[optional, None]: This will override any auxiliary channel list that might have been supplied by the auxiliary_channel_correlation_algorithm and force this to be the auxiliary channels that are associated with this Subject.

# make_omega_scans(verbose=True):
For the main channel and top N auxiliary channels individually, use subject.make_omega_scans to get the raw time-series data from Ligo and plot spectrogram with specified plot_time_ranges.
## parameters:
    verbose (bool)[optional, False]: Whether the log be printed on the terminal.
    nproc (int)[optional, 1]: The number of processor assigned to the multiprocessing module for parallel processing the individual _save_q_scans function.

# save_omega_scans(verbose=True):
Save the image as the .png file.
## parameters:
    plot_directory (path)[optional, os.path.join(os.getcwd(), 'plots', time.from_gps(self.event_time).strftime('%Y-%m-%d'), str(self.event_time))]: The directory image file saved to.
    verbose (bool)[optional, False]: Whether the log be printed on the terminal.
    nproc (int)[optional, 1]: The number of processor assigned to the multiprocessing module for parallel processing the individual _save_q_scans function.
    pool (object)[optional, None]: To specify the pool of processes by passing the pre-defined object instead of using the multiprocessing module. It supports asynchronous results with timeouts and callbacks and has a parallel map implementation.

# combine_images_for_subject_upload(self, number_of_rows=3, **kwargs):
Concatenate the main channel spectrogram with auxiliary channel spectrograms vertically (main channel spectrogram with number_of_rows auxiliary channel spectrograms for each time duration at a time) to get combined images. Also name this image with the ifo_event_subjectpartnum_duration.
## parameters:
    plot_directory (path)[optional, os.path.join(os.getcwd(), 'plots', time.from_gps(self.event_time).strftime('%Y-%m-%d'), str(self.event_time))]: The directory image file saved to.
    number_of_rows: the number of rows of spectrogram except for the main channel one in a single combined image. The spectrogram of a auxiliary channel occupies a row from top to bottom.

# upload_to_zooniverse(self, subject_set_id, project='9979'):
upload the images as well as the metadata that contains the date of upload, filenames, subject_id, main channel name and auxiliary channel names.
## parameters:
    subject_set_id: The id of subjectset in the Zooniverse Project the images uploaded to. The id of Gravity Spy Plus Test Subject Set is 103434.
    project: The id of Zooniverse Project the images uploaded to. The id of GravitySpy Plus project is 9979.