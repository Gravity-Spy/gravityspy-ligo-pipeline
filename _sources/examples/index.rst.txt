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

=======================
The `glitches` database
=======================

This database contains all of the labelled Omicron glitches in h(t) with `SNR > 7.5` and `10 < peak_frequency < 2048`

.. code-block:: bash

   $ ipython

    >>> from gwpy.table import EventTable
    >>> blips_O1 = EventTable.fetch('gravityspy','glitches',selection='"Label"="Blip" & 1137250000 > "peakGPS" > 1126400000 & "ImageStatus" = "Retired"')
    >>> koi_fish_O1 = EventTable.fetch('gravityspy','glitches',selection='"Label"="Koi_Fish" & 1137250000 > "peakGPS" > 1126400000 & "ImageStatus" = "Retired"')
    >>> whistle_O1 = EventTable.fetch('gravityspy','glitches',selection='"Label"="Whistle" & 1137250000 > "peakGPS" > 1126400000 & "ImageStatus" = "Retired" & "ifo" = "L1"')
    >>> koi_fish_O1.write('O1_Koi_Fish.csv')
    >>> blips_O1.write('O1_Blips.csv')
    >>> whistle_O1["peakGPS","peak_frequency", "snr"].write('{0}-triggers-{1}-{2}.csv'.format())



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
    >>> blips_O2_L1 = EventTable.fetch('gravityspy', 'glitches', selection = ['"Label" = "Blip"', '1200000000 > "peakGPS" > 1137250000', '"Blip" > 0.95', 'ifo=L1'])
    >>> koi_O2_L1 = EventTable.fetch('gravityspy', 'glitches', selection = ['"Label" = "Koi_Fish"', '1200000000 > "peakGPS" > 1137250000', '"Koi_Fish" > 0.95', 'ifo=L1'])
    >>> aftercomiss_koi_l1 = koi_O2_L1[koi_O2_L1['peakGPS']>1178841618]
    >>> beforecomiss_koi_l1 = koi_O2_L1[koi_O2_L1['peakGPS']<1178841618]
    >>> beforecomiss_blips_l1 = blips_O2_L1[blips_O2_L1['peakGPS']<1178841618]
    >>> aftercomiss_blips_l1 = blips_O2_L1[blips_O2_L1['peakGPS']>1178841618]
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

