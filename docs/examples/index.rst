.. _examples:

#########################
Querying Gravity Spy data
#########################

============
Introduction
============

We will use `gwpy <https://gwpy.github.io/>`_, the preferred detchar software utility curated by Duncan Macleod.

The method of note is `gwpy.table.EventTable.fetch <https://gwpy.github.io/docs/latest/api/gwpy.table.EventTable.html#gwpy.table.EventTable.fetch>`_

The following example will although you to query the entire gravity spy trianingset. You do *not* need to be on CIT *but* if you do not have lal installed locally it is suggested that you do this on CIT.

Ahead of time, it is encourage to set up your user environment. For LIGO users please see `Gravity Spy Authentication <https://secrets.ligo.org/secrets/144/>`_ for information concerning authentication to access certain Gravity Spy DBs.

=========================
The `glitches_v2d0` table
=========================

This database contains all of the labelled Omicron glitches in h(t) with `SNR > 7.5` and `10 < peak_frequency < 2048`

Columns in database
~~~~~~~~~~~~~~~~~~~

Below are all of the columns in the glitches_v2d0 table which you can perform filters on if you would like

``<TableColumns names=('event_time','ifo','peak_time','peak_time_ns','start_time','start_time_ns','duration',
'search','process_id','event_id','peak_frequency','central_freq','bandwidth','channel','amplitude',
'snr','confidence','chisq','chisq_dof','param_one_name','param_one_value','gravityspy_id','Air_Compressor','Blip',
'Chirp','Extremely_Loud','Helix','Koi_Fish','Light_Modulation','Low_Frequency_Burst',
'Low_Frequency_Lines','No_Glitch','None_of_the_Above','Paired_Doves','Power_Line','Repeating_Blips',
'Scattered_Light','Scratchy','Tomte','Violin_Mode','Wandering_Line','Whistle','ml_label','workflow',
'subjectset','Filename1','Filename2','Filename3','Filename4','upload_flag','1400Ripples',
'1080Lines','image_status','data_quality','citizen_score','url1','url2','url3','url4','links_subjects','q_value','ml_confidence','vco')>``

Machine Learning model Glitch Classes
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Below are all of the classes that the ml model was trained on. These strings are what you will find in the `ml_label` columns.

    * 1400Ripples
    * Extremely_Loud
    * Fast_Scattering
    * Repeating_Blips
    * None_of_the_Above
    * Power_Line
    * Air_Compressor
    * No_Glitch
    * Low_Frequency_Lines
    * 1080Lines
    * Light_Modulation
    * Helix
    * Blip_Low_Frequency
    * Koi_Fish
    * Wandering_Line
    * Whistle
    * Scratchy
    * Violin_Mode
    * Chirp
    * Tomte
    * Low_Frequency_Burst
    * Paired_Doves
    * Scattered_Light
    * Blip

=============================
Utilizing `gwpy` `EventTable`
=============================

There are so many great ways to use `EventTable <https://gwpy.github.io/docs/latest/api/gwpy.table.EventTable.html#gwpy.table.EventTable>`_ to make plotting
publication quality plots easy.

Here we mimic the `histogram <https://gwpy.github.io/docs/latest/examples/table/histogram.html?highlight=hist>`_ functionality

.. plot::
    :context: reset
    :include-source:

    >>> import matplotlib.pyplot as plt
    >>> plt.switch_backend('agg')
    >>> from gwpy.table import EventTable
    >>> blips_O2_L1 = EventTable.fetch('gravityspy', 'glitches', selection = ['ml_label="Blip"', '1200000000 > event_time > 1137250000', 'ml_confidence > 0.95', 'ifo=L1'], host='gravityspyplus.ciera.northwestern.edu')
    >>> koi_O2_L1 = EventTable.fetch('gravityspy', 'glitches', selection = ['ml_label = "Koi_Fish"', '1200000000 > event_time > 1137250000', 'ml_confidence > 0.95', 'ifo=L1'], host='gravityspyplus.ciera.northwestern.edu')
    >>> aftercomiss_koi_l1 = koi_O2_L1[koi_O2_L1['event_time']>1178841618]
    >>> beforecomiss_koi_l1 = koi_O2_L1[koi_O2_L1['event_time']<1178841618]
    >>> beforecomiss_blips_l1 = blips_O2_L1[blips_O2_L1['event_time']<1178841618]
    >>> aftercomiss_blips_l1 = blips_O2_L1[blips_O2_L1['event_time']>1178841618]
    >>> plot = aftercomiss_blips_l1.hist('snr', logbins=True, bins=50, histtype='stepfilled', label='After Commissioning')
    >>> ax = plot.gca()
    >>> ax.hist(aftercomiss_koi_l1['snr'], logbins=True, bins=50, histtype='stepfilled', label='After Commissioning Koi')
    >>> ax.hist(beforecomiss_blips_l1['snr'], logbins=True, bins=50, histtype='stepfilled', label='Before Commissioning')
    >>> ax.hist(beforecomiss_koi_l1['snr'], logbins=True, bins=50, histtype='stepfilled', label='Before Commissioning Koi')
    >>> ax.set_xlabel('Signal-to-noise ratio (SNR)')
    >>> ax.set_ylabel('Rate')
    >>> ax.set_title('Blips and Kois before and after comissioning L1')
    >>> ax.autoscale(axis='x', tight=True)
    >>> ax.set_xlim([0,1000])
    >>> plot.legend()

