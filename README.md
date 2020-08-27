# unetdl4j
U-Net is a convolutional neural network that was developed for biomedical image segmentation which yields precise segmentation upon training over even a small image dataset. This network gets its U-shaped architecture from a contracting path, which follows the typical architecture of a convolutional network and an expanding path, which consists of an upsampling of the feature map followed by a convolution per step.


This project is the implementation of U-Net using the Deeplearning4j library. Deeplearning4j is the first commercial-grade, open-source, distributed deep-learning library written for Java.


The base dataset consists of wide-field epifluorescent images of mouse neuroblastoma cells (cultured neurons) with cytoplasmic (phalloidin) stain and a set of manual segmentations.

Featured project: Autosegmentation of cultured neurons

Link: http://flagella.crbs.ucsd.edu/images/CCDB_6843

The actual dataset used for this project is comprised of microscopic images of single cells cropped from the images in the base dataset.

## Sample Datasets

### For Training

- 100 microscopic cellular images with the corresponding labels: https://drive.google.com/drive/folders/1u3SgJYb1LObpboEKkURQr3Mh7FrPrf_8?usp=sharing

- 300 microscopic cellular images with the corresponding labels: https://drive.google.com/drive/folders/1UUq6W-3P7Mg-eSE6_UJSCQaC8Xazc3zH?usp=sharing

- 44 microscopic cellular images with the corresponding labels (easy training): https://drive.google.com/drive/folders/1Ox0fi1V9dwBXPHisgLc9kjaIfZFZ27dy?usp=sharing

 - 44 microscopic cellular images with the corresponding labels (split for cross validation): https://drive.google.com/drive/folders/1eyRLg1s110ID-T8Oa4Je0xB9fqyy-zZr?usp=sharing

### For Testing

 - 20 microscopic cellular images: https://drive.google.com/drive/folders/1lNphWDWUDq6U4K25kP-zHL8U4ETawuDE?usp=sharing


## How to compile
Instructions on:

- Building
```sh
% git clone https://github.com/Medha-B/unetdl4j.git
```
- Compile
```sh
% mvn compile
% mvn package   # create jar
```

## How to run
```sh
% java -cp target/segmentation-1.0-SNAPSHOT-bin.jar org.sbml.spatial.segmentation.TrainUnetModel
```
