# RAFT-SS-PIV

todo

## Requirement

'''shell
pip install tfrecord protobuf=3.20.1
pip install --extra-index-url https://developer.download.nvidia.com/compute/redist --upgrade nvidia-dali-cuda110
conda install numpy matplotlib scipy jupyter
<!-- conda install -c conda-forge openpiv -->
# torch todo
'''

## Composition of datasets

### Datasets for training and validation

Dataset for training model are avaiablie on https://doi.org/10.5281/zenodo.4432496.
This dataset is stored in the format of `TFRecord`, whose basic elements are dictionary.
Each data element has 3 sub-members: target, flow, label.

* target: particle image pairs, 1x2xWxH
* flow: flow field ground truth, 1x2xWxH
* label: dumped

*Note: go to [testTfrecord.py](testTfrecord.ipynb) for more information*

### Datasets for evalution

Dataset for evaluating models is comprised of groups of sin flow and lamb-oseen flow with different parameter set.
For both sin and lamb-oseen, each parameter set contains 10 samples.
Similar to training dataset, evaluating dataset is in the form of dictionary, which has 4 keys:
img1, img2, u, v.

* img1, particle image one, Bx1xWxH
* img2, particle image two, Bx1xWxH
* u, x componet of the flow ground truth, Bx1xWxH
* v, y componet of the flow ground truth, Bx1xWxH