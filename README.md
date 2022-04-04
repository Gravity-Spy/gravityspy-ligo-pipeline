# Gravity Spy LIGO Pipeline
Installing Gravity Spy

```
git clone https://github.com/Gravity-Spy/gravityspy-ligo-pipeline.git
cd gravityspy-ligo-pipeline
conda create --name gravityspy-plus-py38 -c conda-forge gwpy python-ldas-tools-frameapi python-ldas-tools-framecpp pandas scikit-image python-lal python-ligo-lw python=3.8 --yes
conda activate gravityspy-py38
python -m pip install .
```


# Example
```
from gravityspy_ligo.subject.subject import GravitySpySubject

sub = GravitySpySubject(event_time=1253940741.813, ifo="H1", manual_list_of_auxiliary_channel_names=["H1:ASC-AS_A_RF45_Q_YAW_OUT_DQ", "H1:LSC-REFL_A_LF_OUT_DQ"])

sub.make_omega_scans(verbose=True)
sub.upload_to_zooniverse(subject_set_id=103434)
```
