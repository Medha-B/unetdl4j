# unetdl4j

U-Net is a convolutional neural network developed for biomedical image segmentation and yields precise segmentation upon training over even a small image dataset. This network gets its U-shaped architecture from a contracting path, which follows the typical architecture of a convolutional network and an expanding path, which consists of an upsampling of the feature map followed by a convolution per step.


This project is the implementation of U-Net using the Deeplearning4j library. Deeplearning4j is the first commercial-grade, open-source, distributed deep-learning library written for Java.


The base dataset consists of wide-field epifluorescent images of mouse neuroblastoma cells (cultured neurons) with cytoplasmic (phalloidin) stain and a set of manual segmentations.

Featured project: [Autosegmentation of cultured neurons](http://flagella.crbs.ucsd.edu/images/CCDB_6843)

The actual dataset used for this project is comprised of microscopic images of single cells cropped from the images in the base dataset.

- [Cell images](https://drive.google.com/drive/folders/17EMpftIiXo9IWT-0YyqIaiBNzIKzD3Ii?usp=sharing)
- [Ground Truth images](https://drive.google.com/drive/folders/12PB5LtSkFJSRyLGk7rmL8NZSQJe3Km06?usp=sharing)

## Sample Datasets

### For Training

- [100 microscopic cellular images with the corresponding labels](https://drive.google.com/drive/folders/1u3SgJYb1LObpboEKkURQr3Mh7FrPrf_8?usp=sharing)

- [300 microscopic cellular images with the corresponding labels](https://drive.google.com/drive/folders/1UUq6W-3P7Mg-eSE6_UJSCQaC8Xazc3zH?usp=sharing)

- [44 microscopic cellular images with the corresponding labels (easy training)](https://drive.google.com/drive/folders/1Ox0fi1V9dwBXPHisgLc9kjaIfZFZ27dy?usp=sharing)

- [44 microscopic cellular images with the corresponding labels (split for cross validation)](https://drive.google.com/drive/folders/1eyRLg1s110ID-T8Oa4Je0xB9fqyy-zZr?usp=sharing)

### For Testing

 - [20 microscopic cellular images](https://drive.google.com/drive/folders/1lNphWDWUDq6U4K25kP-zHL8U4ETawuDE?usp=sharing)


## How to compile
Instructions on:

- Building
```sh
% git clone https://github.com/Medha-B/unetdl4j.git
```
- Compiling
```sh
% mvn compile
% mvn package   # create jar
```

## How to run
Intructions on running the class:

- UnetTrainAndTest.java

This class is used for training the U-Net model on a custom dataset and subsequently inferring one or more images from the trained model. The training and testing datasets can be saved in the 'dataset' folder under the working directory. Similarly, the inferred image gets saved in the 'output' folder. The [sample dataset](https://drive.google.com/drive/folders/1Ox0fi1V9dwBXPHisgLc9kjaIfZFZ27dy?usp=sharing) can be downloaded and saved in the [small_dataset](https://github.com/Medha-B/unetdl4j/tree/master/dataset/small_dataset) under [dataset](https://github.com/Medha-B/unetdl4j/tree/master/dataset) folder in local workspace to run the code successfully.

```sh
% java -cp target/segmentation-1.0-SNAPSHOT-bin.jar org.sbml.spatial.segmentation.UnetTrainAndTest </path/to/small_dataset>
```

- UnetTrainAndSave.java

This class is used for training the U-Net model on a custom dataset and subsequently saving the model weights. The training dataset and the trained model can be saved in the 'dataset' and 'saveModel' folders respectively in the working directory.

```sh
% java -cp target/segmentation-1.0-SNAPSHOT-bin.jar org.sbml.spatial.segmentation.UnetTrainAndSave </path/to/dataset>
```

- UnetLoadAndTest.java

This class is used for inferring images by loading the saved and trained U-Net model. The testing dataset and models weights can be saved in the 'dataset' and 'saveModel' folders respectively in the working directory.

```sh
% java -cp target/segmentation-1.0-SNAPSHOT-bin.jar org.sbml.spatial.segmentation.UnetLoadAndTest </path/to/test_image>
```

- UnetIOU.java

This class is used to obtain IOU values and Dice Coefficient for one or more images inferred from the U-Net model and the associated ground truth images. The dataset containing inferred and corresponding ground truth images can be saved in the 'dataset' folder in the working directory.

```sh
% java -cp target/segmentation-1.0-SNAPSHOT-bin.jar org.sbml.spatial.segmentation.UnetIOU </path/to/ground_truth_image> </path/to/inferred_image> 
```

- TrainUnetModel

This class is used for training the U-Net model for single channel (grayscale) images. 

```sh
% java -cp target/segmentation-1.0-SNAPSHOT-bin.jar org.sbml.spatial.segmentation.TrainUnetModel </path/to/test_image>
```
## Documentation

Javadocs of unetdl4j are present in the [doc](https://github.com/Medha-B/unetdl4j/tree/master/doc) directory in the repository.

## Licensing

unetdl4j is licensed under the Apache License, Version 2.0. See [LICENSE](https://github.com/Medha-B/unetdl4j/blob/master/LICENSE.txt) for the complete text.
