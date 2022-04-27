import matplotlib.pyplot as plt
plt.switch_backend('agg')
from gwpy.table import EventTable
blips_O2_L1 = EventTable.fetch('gravityspy', 'glitches', selection = ['"Label" = "Blip"', '1200000000 > "peakGPS" > 1137250000', '"Blip" > 0.95', 'ifo=L1'])
koi_O2_L1 = EventTable.fetch('gravityspy', 'glitches', selection = ['"Label" = "Koi_Fish"', '1200000000 > "peakGPS" > 1137250000', '"Koi_Fish" > 0.95', 'ifo=L1'])
aftercomiss_koi_l1 = koi_O2_L1[koi_O2_L1['peakGPS']>1178841618]
beforecomiss_koi_l1 = koi_O2_L1[koi_O2_L1['peakGPS']<1178841618]
beforecomiss_blips_l1 = blips_O2_L1[blips_O2_L1['peakGPS']<1178841618]
aftercomiss_blips_l1 = blips_O2_L1[blips_O2_L1['peakGPS']>1178841618]
plot = aftercomiss_blips_l1.hist('snr', logbins=True, bins=50, histtype='stepfilled', label='After Commissioning')
ax = plot.gca()
ax.hist(aftercomiss_koi_l1['snr'], logbins=True, bins=50, histtype='stepfilled', label='After Commissioning Koi')
ax.hist(beforecomiss_blips_l1['snr'], logbins=True, bins=50, histtype='stepfilled', label='Before Commissioning')
ax.hist(beforecomiss_koi_l1['snr'], logbins=True, bins=50, histtype='stepfilled', label='Before Commissioning Koi')
ax.set_xlabel('Signal-to-noise ratio (SNR)')
ax.set_ylabel('Rate')
ax.set_title('Blips and Kois before and after comissioning L1')
ax.autoscale(axis='x', tight=True)
ax.set_xlim([0,1000])
plot.legend()
