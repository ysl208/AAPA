# Action-aware Perceptual Anchoring (AAPA)

This is an implementation of the paper [Maintaining a Reliable World Model using Action-aware Perceptual Anchoring](https://arxiv.org/abs/2107.03038).

AAPA is a rule-based approach that considers inductive biases to perform high-level reasoning over the results from low-level object detection and improves the robot's perceptual capability for complex tasks. It can handle object detection errors, out of view, and more complex scenarios such as invisible displacements by considering agent actions.

~NOTE: This repo still under development.~

## Requirements
You will need to run an object detector such as as Faster R-CNN that produces a list of bounding boxes from a 2D image.


### Conda Environment

Set up the Conda environment using a quick installation:

```
conda env create -f environment.yml
```

### Code description

- `scripts/data_processor.py`: DataProcessor class that reads in detected object locations and relations from a csv file in the `data` folder

- `scripts/aapa.py`: main script for maintaining a world state using object permanence from the given data

