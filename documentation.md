# Modified files: 

subject.py, plot.py, utils.py, hveto_parser.py, table/events.py

# APIs: 
# sub = GravitySpySubject(event_time=event_time, ifo="H1", auxiliary_channel_correlation_algorithm={'hveto':round_number}, number_of_aux_channels_to_show=2)
Define a subject class instance with an event time, ifo, auxiliary_channel_correlation_algorithm and round number, number_of_auxiliary_channels_to_show

## parameters:
    event_time:
    ifo:
    auxiliary_channel_correlation_algorithm:
        If choosing the ‘hveto’ as the auxiliary_channel_correlation_algorithm, parse auxiliary channels in the specified round that have highest correlation significance to the main channel from Hveto using hveto_parser.py. A number of top N (6 by default) auxiliary channels name that have the highest significance is parsed from the SVG file.
    number_of_aux_channels_to_show:


Or loop with event_time in the:
## list_of_glitch_times = Events.get_triggers(start=start_time, end=end_time, channel='H1:GDS-CALIB_STRAIN', dqflag=None, algorithm='hveto', verbose=True)


# make_omega_scans(verbose=True, config=config):
For the main channel and top N auxiliary channels individually, use subject.make_omega_scans to get the raw time-series data from Ligo and plot spectrogram with specified plot_time_ranges.
## parameters:



# save_omega_scans(verbose=True, config=config):
## config = utils.GravitySpyConfigFile(plot_time_ranges=[8.0, 4.0, 1.0])
Save the spectrogram as the png file.
## parameters:
    The pool is to use the multiprocessing with nproc number of processors.


# combine_images_for_subject_upload(self, number_of_rows=3, **kwargs):
Using subject.combine_images_for_subject_upload, concatenate the main channel spectrogram with auxiliary channel spectrograms vertically (main channel spectrogram with number_of_rows auxiliary channel spectrograms for each time duration at a time) to get combined images

## parameters:


# upload_to_zooniverse(self, subject_set_id, project='9979'):
Using subject.upload_to_zooniverse to upload the images as well as the metadata that contains the date, filenames, subject_id, main channel name and auxiliary channel names

## parameters:
