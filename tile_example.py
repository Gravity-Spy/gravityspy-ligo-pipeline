from numpy.random import normal 
from scipy.signal import gausspulse
from gwpy.timeseries import TimeSeries
from gwpy.table import EventTable

noise = TimeSeries(normal(loc=1, size=4096*4), sample_rate=4096, epoch=-2)
glitch = TimeSeries(gausspulse(noise.times.value, fc=500) * 4, sample_rate=4096)
data = noise + glitch

q = data.q_transform()
q_gram = data.q_gram()
q_gram_plot = EventTable(q_gram[q_gram['energy'].argmax()]).tile('time', 'frequency', 'duration', 'bandwidth')
q_gram_ax = q_gram_plot.gca()
q_scan_plot = q.plot()
q_scan_ax = q_scan_plot.gca()
q_scan_ax.set_xlim(-.2, .2)
q_scan_ax.set_epoch(0)

q_gram_ax.set_xlim(-.2, .2)
q_gram_ax.set_epoch(0)

# q_scan_plot.save()
q_gram_plot.show()


