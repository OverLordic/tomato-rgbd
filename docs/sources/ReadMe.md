# 4th International Autonomous Greenhouse Challenge: Vision challenge

This dataset was generated for the 4th International Autonomous Greenhouse Challenge in the experimental facilities of
the Greenhouse Horticulture Business Unit in Bleiswijk, The Netherlands in 2024.

Information regarding the competition can be found here http://www.autonomousgreenhouses.com/. The technical document describing the competition is included in the dataset as it provides context for the included imaging.

## Summary and objective

This dataset consists of images of dwarf tomato plants. The images are taken with a RGBD camera under defined conditions
and contain images of individual dwarf tomato plants of different varieties, in different growth stages and grown in
different growing conditions. Each image is connected to information on the manually measured plant traits: plant
height, plant fresh weight, leaf area and the number of red fruits. Teams use the images to develop a computer vision
algorithm. This algorithm will have to be able to estimate the plant traits of a set of unseen dwarf tomato plant
images, provided after the preparation phase, in a limited timeframe.

## Dataset split: train, val(s), test

This dataset is released over the course of the challenge. In the first phase, a training set with ground truth
measurements is released (50% of the data). Later, on a weekly basis, 6 small validation sets are released to measure the teams'
progress (6 batches, in total 30% of the data). Finally, a test set is used for the final evaluation of the teams (20% of the data)

For the initial training set, the ground truth measurements are released directly. For the validation sets, the ground
truth measurements are only provided after all team's results are provided.

## Folder structure

Each subset for training, validation and testing contains following files and folders:

- **depth**: folder with all depth-related images.
- **rgb**: folder with RGB images in `.png` format.
- **ground_truth_{subset}.json**: `.json` file with all ground truth measurement information.
- **anon_keys_{subset}.json**: `.json` file with the original file name. There were not provided to the teams during the challenge.
- **oak-d-s2-poe_conf.json**: `.json` file with the configuration information of used camera. A single camera was used
  for the entire data set collection, so this file can be used for all images in all subsets.


## Ground truth measurement

This dataset contains images and measured data on dwarf tomato plants growing in controlled greenhouse conditions. There
different tomato varieties and samples of the crop are destructively measured after imaging.

The tomato plants were grown under different lighting and EC treatments to evaluate the effects crop characteristics.
For the challenge, images of the dataset
are not linked to the used treatments. Information on the plant can be obtained from the original image filename.

  The filename, e.g., `camir_27_2023_11_23_11_23_13_nc6l6_rgb`, contains the camera number (`camir_27`), imaging date and time (CET) `2023_11_23_11_23_13`, and the plant id (`nc6l6`). The plant id is generated as follows:
  - The variety: nc (non-commercial), pl (plum), ch (cherry) and he (heart).
  - The EC treatment: 3 (EC3) or 6 (EC6)
  - The lighting strategy: h (high light), m (medium light), l (low light), n (no light)
  - The plant number: [1-6] 6 plants for each combination of variety, EC and light are measured.

The sampled plants were destructively measured for the following crop traits:

- `fw_plant` (gram): A head of lettuce, harvested from a hydroponic cultivation system has two parts, the 'root' and
  the 'shoot' The 'shoot' is the top part, being the edible part of the plant, starting at the attachment point of the
  first leaves. The trait is measured in gram/plant.
- `height` (cm): The height of the highest part of the plant, measured vertically from the soil level, measured in cm.
- `leaf_area` (cm2): For the destructive measurements, leaves are cut from the stem and their surface. The total area
  of all leaves is recorded as `leaf_area`.
- `number_of_red_fruits`(count): The number of ripe tomatoes counted, based on their color. A distinction was made
  between red, orange and green fruits. Only ripe, red tomatoes are counted.

Missing values are recorded as `NaN` in the json file, and only occur a few times. In the evaluation, the predictions
for which the ground truth is missing are ignored. All ground truth measurements are registered manually, and can therefore
contain errors. Height and number of red fruits are inspected by hand/eye, so are observer dependent to some degree.

## Imaging

Images were taken with one Oak-D S2 POE camera and the dataset includes the collected RGB and Depth data. The depth
images were aligned to the color images. Consequently, the described camera intrinsics of both the depth and color
images and resolution (3840 x 2160) are similar. The depth images can be converted to 3D point clouds using the
intrinsics in the file `oak-d-s2-poe_conf.json` (`color_int`).

The images can be linked together by their filename. In the `ground_truth.json`, the key is the filename, and matches to
images in the respective image folders.
