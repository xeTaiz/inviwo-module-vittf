# Leveraging Self-Supervised Vision Transformers for Segmentation-based Transfer Function Design
[Paper](https://arxiv.org/abs/2309.01408) $~~~$ [Project Page](https://dominikengel.com/vit-tf)

This is the external [Inviwo Module](https://github.com/inviwo/inviwo) of the paper by [Dominik Engel](https://dominikengel.com), [Leon Sick](https://leonsick.github.io) and
[Timo Ropinski](https://viscom.uni-ulm.de/members/timo-ropinski/).

This repository contains the Inviwo integration of our work and needs to be compiled together with Inviwo.

[![Demonstration of our Method](https://img.youtube.com/vi/kTPBCYJtEJc/0.jpg)](https://www.youtube.com/watch?v=kTPBCYJtEJc)

# Cite our Paper
```bibtex
@misc{engel2024vittf,
 abstract = {In volume rendering, transfer functions are used to classify structures of interest, and to assign optical properties such as color and opacity. They are commonly defined as 1D or 2D functions that map simple features to these optical properties. As the process of designing a transfer function is typically tedious and unintuitive, several approaches have been proposed for their interactive specification. In this paper, we present a novel method to define transfer functions for volume rendering by leveraging the feature extraction capabilities of self-supervised pre-trained vision transformers. To design a transfer function, users simply select the structures of interest in a slice viewer, and our method automatically selects similar structures based on the high-level features extracted by the neural network. Contrary to previous learning-based transfer function approaches, our method does not require training of models and allows for quick inference, enabling an interactive exploration of the volume data. Our approach reduces the amount of necessary annotations by interactively informing the user about the current classification, so they can focus on annotating the structures of interest that still require annotation. In practice, this allows users to design transfer functions within seconds, instead of minutes. We compare our method to existing learning-based approaches in terms of annotation and compute time, as well as with respect to segmentation accuracy. Our accompanying video showcases the interactivity and effectiveness of our method.},
 author = {Engel, Dominik and Sick, Leon and Ropinski, Timo},
 doi = {10.48550/arXiv.2309.01408},
 publisher = {arXiv},
 title = {Leveraging Self-Supervised Vision Transformers for Segmentation-based Transfer Function Design},
 year = {2023}
}
```

# Installation
1. You need to build [Inviwo](https://github.com/inviwo/inviwo) from source. The latest `main` *should* work. The exact version used
by us [can be found here](https://github.com/xetaiz/inviwo-ntf). Please consult the [Build Guide](https://inviwo.org/manual-gettingstarted-build.html)
to do this. We recommend using `Qt6`, as well as `vcpkg` for dependencies.
2. Make sure you build Inviwo with Python support. The Python installation used by Inviwo will require
    - [PyTorch](https://pytorch.org) >= 1.7
    - [NumPy](https://numpy.org) >= 1.24.2
    - [Scikit-learn](https://scikit-learn.org) >= 1.3.0
3. Add this repository to the `IVW_EXTERNAL_MODULES` variable in CMake, configure, and check the `IVW_MODULE_NEURALTF` boolean to
include our module into the build process
4. Open Inviwo and open the Workspace from this repository in `neuraltf/data/workspaces/ntf.inv` as a starting point

# Usage
There are three main processors to use our approach:
- DinoSimilarities: This is a Python processor and computes the actual similarity maps and refinement of our method. It receives the raw volume data as input through its `VolumeInport` and outputs a similarity volume for each class. Its `Update Similarities` property needs to be linked to the `DINOVolumeRenderer`'s `Update Similarities` property.
- DINOVolumeRenderer: This renderer takes as input the raw volume, entry and exit points, and a set of similarity volumes and performs the actual rendering of the similarity maps. Its properties allow adding/removing and configuring transfer functions for different classes, as well as camera and rendering parameters. The output is a rendered image.
- BrushVolumeSliceGL: This processor slices an input volume along a specified axis and allows placing and displaying of annotations through singular points or brushing. This processor is used for each axis, for both raw and a current similarity map
