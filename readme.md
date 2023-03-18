# What is this?

This is a simple bird classifier written for COMP-4301 (Computer Vision) at Memorial University of Newfoundland.
It was written in python using the TensorFlow library.

## What is nl_birds?

This is a simple dataset of various birds found in Newfoundland and Labrador. We curated the dataset
using Google Images, where we cleaned the dataset manually for any pictures that weren't a clear representation
of the bird species we were trying to predict.

## Example Usage

Usages:

` python bird.py create ` 

Creates the CNN for the bird image classifier using the nl_birds dataset.
Saves to the working directory as "my_model.h5".

` python bird.py test "Image/Path" `

Prints prediction using the model and the image from "image_name" as input.

Example input/output:

Input:
` bird.py test .\test_images\test_seagull.jpg `

Output:
```
1/1 [==============================] - 0s 367ms/step
[[ 0.64453435 -3.483947    0.46864077 -0.98530567  4.7151265   0.18946311
   2.1591187 ]]
Seagull
```