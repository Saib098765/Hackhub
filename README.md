# Hackhub
VGG based No2 Predictor using Sentinal Satellite Images

Having precise air quality predictions is essential with growing emissions and harmful substances being released into the air. Current resources for air quality data typically provide information only for larger regions, with limited availability for smaller, localized areas. This model helps predicting NO₂ levels for specific terrain types in small, targeted regions.

Function:
The model takes in an image of a small region and predicts the NO₂ levels of that area in ppm. It is a CNN-based model, trained on several images of various terrains along with their NO₂ values and validated against their corresponding ground data. The model reads the input image and its features to provide an accurate NO₂ level prediction. It gives accurate prediction for 15-56 ppm and with 15 different classes.

Steps to building a functioning model:
1. Initialize images: Since the dataset for mapping No2 values from satellite images is not freely available for training this kind of CNN model, we created this dataset ourselves. No2 maps are available specific to london region (with 15 different classes of NO2). No2 image maps from 5 geographical strata were obtained from https://www.londonair.org.uk/london/asp/annualmaps.asp.
Satellite images from Sentinel V2 (via google earth) were obtained corresponding to the selected NO2 maps.

Files: input_images, no2_images, sat_images

2. Generating data to make the model more senstive: Since the No2 maps are only scalable upto 5-6 regions geographically, its essential that our data set contains a large number of values to train from; in order to eliminate bias, or overfitting issues. To address this, we split each no2 and corresponding satellite image into 100 sub patches (10x10) to enlargen our dataset and include more features to train the model on.

Files: Split_images.py
This code generates no2_patches, sat_patches directories with the split images, training_data.csv files.

3. Generating average no2 values: From the no2 patch images generated, each pixel is mapped to the 15 no2 (ppm) classes as referred from the LondonAir no2 map; and an average no2 value is appended to dataset.

Files: avg_no2_val.py
Generates a training_data_with_avg.csv file.

4. Remove Zero values: If avg_no2 value is zero, we eliminate the corresponding no2, satellite images from the dataset. [Outliers]

Files: Remove_zero_avg_values.py
Generates training_data_final.csv. This is our final dataset used to train the model.

5. VGG model: We have engaged in transfer learning on a pre trained VGG model.

Files: VGG_model.py
Generates a vgg16_no2_regressor.pth file with the saved state of the model. MSE loss function and Adam optimizer have been used on a training:validation (80:20) dataset. The model performs with a 87.6 % accuracy on 20 epochs. (Also converges with a similar value around epoch 14).

6. Testing: Test the model on a new image

Files: Prediction.py
Generates predicted No2 value to the corresponding satellite image.

7. User-Interface: To get a more simplified visual of the model using grandio.

Files: UI.py





