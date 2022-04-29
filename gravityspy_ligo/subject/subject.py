# -*- coding: utf-8 -*-
# Copyright (C) Scott Coughlin (2017-)
#
# This file is part of gravityspy.
#
# gravityspy is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# gravityspy is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with gravityspy.  If not, see <http://www.gnu.org/licenses/>.

from ..utils import utils, hveto_parser
from ..table.events import id_generator
from gwpy import time
import multiprocessing
import os
import datetime
import panoptes_client
import glob

class GravitySpySubject:
    """The frame work for thinking about a single Gravity Spy subject
    """
    def __init__(self, event_time, ifo, gravityspy_id=None, event_generator=None, auxiliary_channel_correlation_algorithm=None, number_of_aux_channels_to_show=None, manual_list_of_auxiliary_channel_names=None):
        """Example of docstring on the __init__ method.
        Args:
            event_time (float): The GPS time at which an excess noise event occurred.
            ifo (str): What interferometer had this an excess noise event
            event_generator (str): The algorithm that tells us an excess noise event occurred
            auxiliary_channel_correlation_algorithm (str): The algorithm that tells us the names of the top X correlated auxiliary channels with respect to h(t).
            number_of_aux_channels_to_show (int): This number will determine the top N number of channels from the list provided by the auxiliary_channel_correlation_algorithm that will be kept and shown for this Subject.
            manual_list_of_auxiliary_channel_names (list): This will override any auxiliary channel list that might have been supplied by the auxiliary_channel_correlation_algorithm and force this to be the auxiliary channels that are associated with this Subject.
        """
        self.event_time = event_time
        if gravityspy_id is None:
            gravityspy_id = id_generator(event_time, size=20)
        self.gravityspy_id = gravityspy_id
        self.ifo = ifo
        self.main_channel = '{0}:GDS-CALIB_STRAIN'.format(ifo)
        self.event_generator = event_generator
        self.all_channels = []
        self.frametypes = []
        self.q_values = []
        self.q_transforms = []
        self.ldvw_glitchdb_image_filenames = []
        self.zooniverse_subject_image_filenames = []

        # If a manual list of auxiliary channels were provided, we can set a lot fo these attributes right now.
        if manual_list_of_auxiliary_channel_names is not None:
            # Check to make sure that channel names were supplied with names <ifo>:
            for channel in manual_list_of_auxiliary_channel_names:
                if '{0}:'.format(ifo) not in channel:
                    raise ValueError("Please supply aux channel name with leading `<ifo>:`")

            self.list_of_auxiliary_channel_names = manual_list_of_auxiliary_channel_names

            # no matter what was passed, override these to be None
            self.number_of_aux_channels_to_show = None
            self.auxiliary_channel_correlation_algorithm = None
        elif 'hveto' in auxiliary_channel_correlation_algorithm.keys():

            # if we passed 'hveto' as our algorith, we must also have identified what round this glitch is associated with.
            round_number = auxiliary_channel_correlation_algorithm['hveto']

            self.auxiliary_channel_correlation_algorithm = auxiliary_channel_correlation_algorithm
            self.number_of_aux_channels_to_show = number_of_aux_channels_to_show

            # based on the start and end time, produce a date timestamp with YYYYMMDD
            # convert start time from gps to regular date
            event_time_in_date_format = time.from_gps(self.event_time)

            event_time_in_YYYYMMDD_format = event_time_in_date_format.strftime("%Y%m%d")

            each_rounds_svg_file = glob.glob("/home/detchar/public_html/hveto/day/{0}/latest/plots/*ROUND_{1}*.svg".format(event_time_in_YYYYMMDD_format, round_number))

            for each_round_svg in each_rounds_svg_file:
                auxiliary_channels_ordered_by_signifigance = hveto_parser.hveto_parser(each_round_svg)

            self.list_of_auxiliary_channel_names = auxiliary_channels_ordered_by_signifigance[0:number_of_aux_channels_to_show]
        else:
            raise ValueError("You supplied a auxiliary_channel_correlation_algorithm that is not recognized")

        # create the final list of all channels and frametypes associated with this subject
        # Append main channel name and frametype
        self.all_channels.append(self.main_channel)
        self.frametypes.append('{0}_HOFT_C00'.format(ifo))
        # Append aux channel names and frametype
        for aux_channel in self.list_of_auxiliary_channel_names:
            self.frametypes.append('{0}_R'.format(ifo))
        self.all_channels.extend(self.list_of_auxiliary_channel_names)

    def make_omega_scans(self, pool=None, **kwargs):
        # Parse key word arguments
        config = kwargs.pop('config', utils.GravitySpyConfigFile())
        verbose = kwargs.pop('verbose', False)
        nproc = kwargs.pop('nproc', 1)

        inputs = ((self.event_time, config, channel_name, frametype, verbose)
                  for channel_name, frametype, in zip(self.all_channels, self.frametypes))

        # make q_scans
        if (pool is None) and (nproc > 1):
            with multiprocessing.Pool(nproc) as pool:
                output = pool.map(utils._make_single_qscan,
                               inputs)
        elif (pool is None) and (nproc == 1):
            output = list(map(utils._make_single_qscan, inputs))
        elif pool is not None:
            output = pool.map(utils._make_single_qscan,
                              inputs)

        # raise exceptions (from multiprocessing, single process raises inline)
        for event_time, q_transform, q_value in output:
            self.q_values.append(q_value)
            self.q_transforms.append(q_transform)

        box_x =  self.q_transforms[0].box_x
        box_y =  self.q_transforms[0].box_y
        box_x_dur =  self.q_transforms[0].box_x_dur
        box_y_dur =  self.q_transforms[0].box_y_dur
        for q_scan in self.q_transforms:
            setattr(q_scan, 'box_x', box_x)
            setattr(q_scan, 'box_y', box_y)
            setattr(q_scan, 'box_x_dur', box_x_dur)
            setattr(q_scan, 'box_y_dur', box_y_dur)

    def save_omega_scans(self, pool=None, **kwargs):
        # Parse key word arguments
        config = kwargs.pop('config', utils.GravitySpyConfigFile())
        plot_directory = kwargs.pop('plot_directory', os.path.join(os.getcwd(), 'plots', time.from_gps(self.event_time).strftime('%Y-%m-%d'), str(self.event_time)))
        verbose = kwargs.pop('verbose', False)
        nproc = kwargs.pop('nproc', 1)

        inputs = inputs = ((self.event_time, self.ifo, '{0}_{1}'.format(self.gravityspy_id, channel_name), config, plot_directory, channel_name, frametype, verbose, q_transform) for channel_name, frametype, q_transform in zip(self.all_channels, self.frametypes, self.q_transforms))

        # make q_scans
        if (pool is None) and (nproc > 1):
            with multiprocessing.Pool(nproc) as pool:
                output = pool.map(utils._save_q_scans,
                               inputs)
        elif (pool is None) and (nproc == 1):
            output = list(map(utils._save_q_scans, inputs))
        elif pool is not None:
            output = pool.map(utils._save_q_scans,
                              inputs)

        for event_time, individual_image_filenames, combined_image_filename in output:
            self.ldvw_glitchdb_image_filenames.append(combined_image_filename)
            self.zooniverse_subject_image_filenames.extend(individual_image_filenames)

    def combine_images_for_subject_upload(self):
        for image_filename in self.zooniverse_subject_image_filenames:
            breakpoint()


    def upload_to_zooniverse(self, subject_set_id, project='9979'):
        """Obtain omicron triggers to run gravityspy on

        Parameters:
            subject_set_id (optional, int) : subject set id to upload to

        Returns:
            `Events` table
        """
        subject = panoptes_client.Subject()
        subject.links.project = project
        subject.metadata['date'] = datetime.datetime.now().strftime('%Y%m%d')
        subject.metadata['subject_id'] = str(self.gravityspy_id)
        for idx, image in enumerate(self.zooniverse_subject_image_filenames):
            subject.add_location(str(image))
            subject.metadata['Filename{0}'.format(idx+1)] = image.split('/')[-1]
        subject.save()
        self.zooniverse_id = int(subject.id)
        for idx, image in enumerate(self.ldvw_glitchdb_image_filenames):
            setattr(self, 'url{0}'.format(idx), subject.raw['locations'][idx]['image/png'].split('?')[0])

        subjectset = panoptes_client.SubjectSet.find(subject_set_id)
        subjectset.add(subject)
