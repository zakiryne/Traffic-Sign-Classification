# **Traffic Sign Recognition** 

## Writeup

### You can use this file as a template for your writeup if you want to submit it as a markdown file, but feel free to use some other method and submit a pdf if you prefer.

---

**Build a Traffic Sign Recognition Project**

The goals / steps of this project are the following:
* Load the data set (see below for links to the project data set)
* Explore, summarize and visualize the data set
* Design, train and test a model architecture
* Use the model to make predictions on new images
* Analyze the softmax probabilities of the new images
* Summarize the results with a written report


[//]: # (Image References)

[image1]: ./examples/Hist.png "Visualization"
[image2]: ./examples/Gray&Color "Grayscaling"
[image3]: ./examples/random_noise.jpg "Random Noise"
[image4]: ./examples/road-sign-speed-limit-30-kmh.jpg "Traffic Sign 1"
[image5]: ./examples/Do-Not-Enter.jpg "Traffic Sign 2"
[image6]: ./examples/Yield.jpg "Traffic Sign 3"
[image7]: ./examples/Stop.jpg "Traffic Sign 4"
[image8]: ./examples/mandatory-direction-up.png "Traffic Sign 5"

## Rubric Points
### Here I will consider the [rubric points](https://review.udacity.com/#!/rubrics/481/view) individually and describe how I addressed each point in my implementation.  

---
### Writeup / README

#### 1. Provide a Writeup / README that includes all the rubric points and how you addressed each one. You can submit your writeup as markdown or pdf. You can use this template as a guide for writing the report. The submission includes the project code.

You're reading it! and here is a link to my [project code](https://github.com/udacity/CarND-Traffic-Sign-Classifier-Project/blob/master/Traffic_Sign_Classifier.ipynb)

### Data Set Summary & Exploration

#### 1. Provide a basic summary of the data set. In the code, the analysis should be done using python, numpy and/or pandas methods rather than hardcoding results manually.

I used the pandas library to calculate summary statistics of the traffic
signs data set:

* The size of training set is 34799
* The size of the validation set is 4410
* The size of test set is 12630
* The shape of a traffic sign image is (32, 32, 3)
* The number of unique classes/labels in the data set is 43

#### 2. Include an exploratory visualization of the dataset.

Here is an exploratory visualization of the data set. It is a bar chart showing how the data are not evenly distributed against the label (43 total)

![alt text][image1]

### Design and Test a Model Architecture

#### 1. Describe how you preprocessed the image data. What techniques were chosen and why did you choose these techniques? Consider including images showing the output of each preprocessing technique. Pre-processing refers to techniques such as converting to grayscale, normalization, etc. (OPTIONAL: As described in the "Stand Out Suggestions" part of the rubric, if you generated additional data for training, describe why you decided to generate additional data, how you generated the data, and provide example images of the additional data. Then describe the characteristics of the augmented training set like number of images in the set, number of images for each class, etc.)

As a first step, I decided to convert the images to grayscale because it takes less computation than color images.

Here is an example of a traffic sign image before and after grayscaling.

![alt text][image2]

As a last step, I normalized the image data because it is usually much easier to optimize when the input is normalized. You can except an improvement in both speed of convergence and in the accuracy of the output.

There are other preprocessing techniques like rotation, flip, color contrast, zoom in/out etc. Which make the model more accurate. Since I dont have time to do these, I left only with normalization and Gray scale conversion.




#### 2. Describe what your final model architecture looks like including model type, layers, layer sizes, connectivity, etc.) Consider including a diagram and/or table describing the final model.

My final model consisted of the following layers:

| Layer         		|     Description	        					| 
|:---------------------:|:---------------------------------------------:| 
| Input         		| 32x32x1 Gray image   							| 
| Convolution 	     	| 1x1 stride, VALID padding, outputs 28x28x6 	|
| RELU                  |                                               |
| Max pooling	      	| 2x2 stride,  SAME padding, outputs 14x14x6 	|
| Convolution 		    | 1x1 stride, VALID padding, outputs 10x10x16   |
| RELU                  |                                               |
| Max pooling           | 2x2 stride,  SAME padding, outputs 5x5x16     |
| Fully connected		| Input 400, Output 120        					|
| RELU					|         										|
| DropOut				|												|
| Fully connected		| Input 120, Output 84							|
| RELU                  |                                               |
| DropOut               |                                               |
| Fully connected       | Input 84, Output 43                           |


#### 3. Describe how you trained your model. The discussion can include the type of optimizer, the batch size, number of epochs and any hyperparameters such as learning rate.

To train the model, I used Adam Optmizer, Batch size 100, EPOCH size 60, learning rate 0.001.

#### 4. Describe the approach taken for finding a solution and getting the validation set accuracy to be at least 0.93. Include in the discussion the results on the training, validation and test sets and where in the code these were calculated. Your approach may have been an iterative process, in which case, outline the steps you took to get to the final solution and why you chose those steps. Perhaps your solution involved an already well known implementation or architecture. In this case, discuss why you think the architecture is suitable for the current problem.

My final model results were:

* validation set accuracy of 0.954
* test set accuracy of 0.942

If an iterative approach was chosen:
* What was the first architecture that was tried and why was it chosen?
I tried with simple LeNet to be familiar with all the process begining from data sets accumulation, plotting images, changing to gray scale and normilization.
Then training and calculating accuracy for the very first time.

* What were some problems with the initial architecture?
With the initial architecture, I was only getting 0.89 accuracy.

*How was the architecture adjusted and why was it adjusted? Typical adjustments could include choosing a different model architecture, adding or taking away layers (pooling, dropout, convolution, etc), using an activation function or changing the activation function. One common justification for adjusting an architecture would be due to overfitting or underfitting. A high accuracy on the training set but low accuracy on the validation set indicates over fitting; a low accuracy on both sets indicates under fitting.
To boost up the accuracy I included drop out layers.

* Which parameters were tuned? How were they adjusted and why?
I also tuned some hyper parameter along with dropout layer addition. I changed EPOCH from 10 to 60 and decrease the batch size from 128 to 100.

* What are some of the important design choices and why were they chosen? For example, why might a convolution layer work well with this problem? How might a dropout layer help with creating a successful model?
The convolution layer is needed to capture the traffic signal any where in the whole image. The dropout is regulate the optimization method.


If a well known architecture was chosen:
* What architecture was chosen?
May be googlenet

* Why did you believe it would be relevant to the traffic sign application?
googlenet is well known for image classification.

* How does the final model's accuracy on the training, validation and test set provide evidence that the model is working well?
Because the model apply prediction using the trained weights and then calculate the accuracy which clearly indicates that model will work well most of the time.
 

### Test a Model on New Images

#### 1. Choose five German traffic signs found on the web and provide them in the report. For each image, discuss what quality or qualities might be difficult to classify.

Here are five German traffic signs that I found on the web:

![alt text][image4] ![alt text][image5] ![alt text][image6] 
![alt text][image7] ![alt text][image8]

The first image might be difficult to classify because ...

#### 2. Discuss the model's predictions on these new traffic signs and compare the results to predicting on the test set. At a minimum, discuss what the predictions were, the accuracy on these new predictions, and compare the accuracy to the accuracy on the test set (OPTIONAL: Discuss the results in more detail as described in the "Stand Out Suggestions" part of the rubric).

Here are the results of the prediction:

| Image			        |     Prediction	        					| 
|:---------------------:|:---------------------------------------------:| 
| Speed limit (30km/h)  | Speed limit (30km/h)  						| 
| No entry     			| No entry 										|
| Yield					| Yield											|
| Stop Sign	      		| Stop Sign					 					|
| Ahead Only			| Ahead Only      								|


The model was able to correctly guess 5 of the 5 traffic signs, which gives an accuracy of 100%. This compares favorably to the accuracy on the test set of 94.2%

#### 3. Describe how certain the model is when predicting on each of the five new images by looking at the softmax probabilities for each prediction. Provide the top 5 softmax probabilities for each image along with the sign type of each probability. (OPTIONAL: as described in the "Stand Out Suggestions" part of the rubric, visualizations can also be provided such as bar charts)

The code for making predictions on my final model is located in the 11th cell of the Ipython notebook.

For the first image, the model is relatively sure that this is a Speed limit (30km/h) (probability of 0.99), and the image does contain a Speed limit (30km/h) sign. The top five soft max probabilities were

| Probability         	|     Prediction	        					| 
|:---------------------:|:---------------------------------------------:| 
| 9.99e-01         		| Speed limit (30km/h)   						| 
| 8.68e-07     			| Speed limit (50km/h) 							|
| 1.56e-09				| Speed limit (80km/h)							|
| 1.88e-10	      		| End of speed limit (80km/h)					|
| 4.35e-12				| Speed limit (20km/h)     						|


For the second image is as follows

| Probability         	|     Prediction	        					| 
|:---------------------:|:---------------------------------------------:| 
| 1.00e+00         		| No entry   									| 
| 1.78e-12     			| Stop 											|
| 9.38e-19				| Speed limit (20km/h)							|
| 7.88e-19	      		| Roundabout mandatory							|
| 1.86e-19				| Go straight or right     						|

And so on....

### (Optional) Visualizing the Neural Network (See Step 4 of the Ipython notebook for more details)
#### 1. Discuss the visual output of your trained network's feature maps. What characteristics did the neural network use to make classifications?


