# NeuralTF Module
This is the official implementation of:
[Leveraging Self-Supervised Pre-Trained Vision Transformers for Neural Transfer Function Design](https://arxiv.org/abs/2309.01408) by [Dominik Engel](https://dominikengel.com), [Leon Sick](https://leonsick.github.io) and [Timo Ropinski](https://viscom.uni-ulm.de/members/timo-ropinski/).

The implementation is an external module for the [Inviwo Visualization Framework](https://inviwo.org) ([GitHub](https://github.com/inviwo/inviwo)).
To build this module, follow the instructions for [Building Inviwo](https://inviwo.org/manual-gettingstarted-build.html) and add the path to this repository to the `IVW_EXTERNAL_MODULES` CMake flag.
Note that you have to build Inviwo with the Python module and that, in order for our module to work, the Python installation used by Inviwo needs to have the following dependencies installed:
- [PyTorch](https://pytorch.org) >= 1.7
- [NumPy](https://numpy.org) >= 1.24.2
- [Scikit-learn](https://scikit-learn.org) >= 1.3.0

Accompanying this repository there is the [Neural TF Design Repository](https://github.com/xeTaiz/neural-tf-design) containing the Python scripts not relevant to the Inviwo side of things, such as utils, tests, conversion utilities, evaluation scripts etc.

## Usage
There are three main processors to use our approach:
- DinoSimilarities: This is a Python processor and computes the actual similarity maps and refinement of our method. It receives the raw volume data as input through its `VolumeInport` and outputs a similarity volume for each class. Its `Update Similarities` property needs to be linked to the `DINOVolumeRenderer`'s `Update Similarities` property.
- DINOVolumeRenderer: This renderer takes as input the raw volume, entry and exit points, and a set of similarity volumes and performs the actual rendering of the similarity maps. Its properties allow adding/removing and configuring transfer functions for different classes, as well as camera and rendering parameters. The output is a rendered image.
- BrushVolumeSliceGL: This processor slices an input volume along a specified axis and allows placing and displaying of annotations through singular points or brushing. This processor is used for each axis, for both raw and a current similarity map

The accompanying workspace in `neuraltf/data/workspaces/ntf.inv` shows an example workspace configuration to use our approach.
