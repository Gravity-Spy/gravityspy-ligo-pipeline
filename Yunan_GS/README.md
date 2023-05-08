# GravitySpy New Classification Model

You can test the model by running [test_classify.py](https://github.com/Gravity-Spy/gravityspy-ligo-pipeline/blob/Yunan/Yunan_GS/test_classify.py).
The new model weights is [O4_attention-classifier.hdf5](https://github.com/Gravity-Spy/gravityspy-ligo-pipeline/tree/Yunan/Yunan_GS/models)

Changes: 
1. [classify.py](https://github.com/Gravity-Spy/gravityspy-ligo-pipeline/blob/Yunan/Yunan_GS/gravityspy/classify/classify.py): line 93, 
```
utils.label_q_scans_new
```

2. [read_image.py](https://github.com/Gravity-Spy/gravityspy-ligo-pipeline/blob/Yunan/Yunan_GS/gravityspy/ml/read_image.py): line 36,
```
read_data_new
```

3. [labelling_test_glitches.py](https://github.com/Gravity-Spy/gravityspy-ligo-pipeline/blob/Yunan/Yunan_GS/gravityspy/ml/labelling_test_glitches.py): line 131
```
label_glitches_new
```

4.  [GS_Model.py](https://github.com/Gravity-Spy/gravityspy-ligo-pipeline/blob/Yunan/Yunan_GS/gravityspy/ml/GS_Model.py): the architecture of the model

5. [utils.py](https://github.com/Gravity-Spy/gravityspy-ligo-pipeline/blob/Yunan/Yunan_GS/gravityspy/utils/utils.py): line 131
```
label_q_scans_new
```
