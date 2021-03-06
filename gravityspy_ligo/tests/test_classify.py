from gwpy.timeseries import TimeSeries
from gwpy.table import EventTable

from gravityspy_ligo.classify import classify
import os
import pandas

MODEL_NAME_CNN = os.path.join(os.path.split(__file__)[0], '..', '..', 'models',
                              'O3-multiview-classifer.h5')

SCRATCHY_TIMESERIES_PATH = os.path.join(os.path.split(__file__)[0], 'data',
                                        'timeseries',
                                        'scratchy_timeseries_test.h5')


SCRATCHY_TIMESERIES = TimeSeries.read(SCRATCHY_TIMESERIES_PATH)
EVENT_TIME = 1127700030.877928972

RESULTS_TABLE = EventTable.read(os.path.join(os.path.split(__file__)[0], 'data',
                                        'table',
                                        'scratchy_results_table.h5'), format='hdf5')

class TestUtils(object):
    """`TestCase` for the GravitySpy
    """
    def test_make_q_scans(self):

        results = classify(event_time=EVENT_TIME,
                           channel_name='L1:GDS-CALIB_STRAIN',
                           path_to_cnn=MODEL_NAME_CNN,
                           timeseries=SCRATCHY_TIMESERIES)

        results.convert_unicode_to_bytestring()
        pandas.testing.assert_frame_equal(results.to_pandas(),
                                          RESULTS_TABLE.to_pandas(),
                                          check_dtype=False,
                                          check_less_precise=True)
