# Preparations and setups
## Installing python 3.7

Open your terminal by **command + space** + terminal and check for the python version by:
```
python --version
```
You'll most likely see something like this
```
out: Python 2.7.16
```
We need to upgrade this to python 3.7 by following these steps consecutively
```
/usr/bin/ruby -e "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/master/install)"
```
In this step, you're asked to enter a password, it is the same password, as the computer login. This step will take a minute. 
If the download was successful, proceed with the following steps.
```
export PATH="/usr/local/opt/python/libexec/bin:/usr/local/bin:$PATH"

brew install python
```
If you check for the version again you should see this:
```
out: Python 3.7.5
```
We successfully upgraded python! To use python properly, we need to download and create a virtual environment. To do so, please follow those last steps

```
pip install virtualenv

virtualenv pytorch

source pytorch/bin/activate
```

If you've successfully activate your environment you should see it in your terminal like this:
```
(pytorch)
```

## Installing Pycharm

[Click here](https://www.jetbrains.com/pycharm/download/download-thanks.html?platform=mac&code=PCC) to download Pycharm.

When opening Pycharm, make sure to not import any settings. We will do that manually
<img width="422" alt="pycharm" src="https://user-images.githubusercontent.com/48200405/68656268-c7180d00-0531-11ea-8b87-923d627952c4.png">

To see if everything works as expected, we try to run a small test code with Pycharm.
Go to your applications and open Pycharm. 
**This step is important, as we need to shift the working directory from our local machine to the TRANSFER server**. To do this, press 'open' and navigate via the transfer to the NN folder, as shown in the pictures: 
<img width="400" alt="open" src="https://user-images.githubusercontent.com/48200405/68580283-54962700-0476-11ea-9ecf-f78ff60952fb.png"> <img width="400" alt="open" src="https://user-images.githubusercontent.com/48200405/68584186-2ff27d00-047f-11ea-9a3a-0ad6e0433a45.png">

Make sure to open NN in pycharm, as this will create a new project containing all the python files.

In the left bottom corner you'll find a panel \<No interpreter\>, click on it and press **add interpreter**.
<img width="302" alt="add_int" src="https://user-images.githubusercontent.com/48200405/68657768-9ab1c000-0534-11ea-92ae-a3ae32465ff7.png">


Add the newly created environment to pycharm as shown in the picture below
<img width="847" alt="env" src="https://user-images.githubusercontent.com/48200405/68657897-c765d780-0534-11ea-8e19-910241d749dc.png">

If everything worked as expected, you should see the activated environment in the bottom right corner:
<img width="327" alt="Screenshot 2019-11-12 at 10 05 11" src="https://user-images.githubusercontent.com/48200405/68658090-17449e80-0535-11ea-8667-86b52d652079.png">

As a last step, open the terminal in the bottom left cornor and type:
```
pip install -r requirenments.txt
```
  
Now open the test.py file in the test folder and and run it using **control + r**

If you see 'Hi there, thanks for using my program :)' written to your console, everything worked fine and you're ready to go

# Workflow overview
<img width="809" alt="Screenshot 2019-11-12 at 17 04 13" src="https://user-images.githubusercontent.com/48200405/68688361-e1241080-056e-11ea-8f6a-d914049fdbe1.png">

# Workflow without training

## Data preparation

In this section, there are two major steps you have to perform
1. Merge the channels from the raw images and convert them to RGB using fiji
2. Square and downscale images with padding and resizing operations using python
### Convert to RGB
For the first step you should prepare a folder with all the raw images you want to analyse using the Neural Network. Now navigate to the fiji folder as shown below and open the script 'ConvertToRGB_v2.jim' (drag it into fiji not just double clicking). 
Note: the images are huge, it will take a while to convert them all, use the tatooine (or another) server, for faster processing.
<img width="1069" alt="fiji" src="https://user-images.githubusercontent.com/48200405/68586000-f3755000-0483-11ea-866f-fc4472a30222.png">
<img width="450" alt="fiji2" src="https://user-images.githubusercontent.com/48200405/68587929-8adca200-0488-11ea-9573-dfa42b8eded8.png"> <img width="400" alt="unpro" src="https://user-images.githubusercontent.com/48200405/68587842-5c5ec700-0488-11ea-8f3e-c73f190ac4f7.png"> 
Run the fiji macro, you'll be asked, what input and output folder to use, use the folder with the raw images as input folder and 'imgs_unprocessed' as output folder and start converting.
### Pad and downscale images
In the same manner as described before, open 'NN' via transfer in pycharm and open the script 'process_images.py'.
Run the script and wait until all the images are processed (you will see which file is being processed in the console). 
When all the images have been processed, your dataset should be ready in the NN -> dataset -> imgs folder. 

## Make prediction
Now for the fun part! You will now run your images through the neural network. In pycharm, open the file 'make_prediction.py' and run it. Every image in the imgs folder will be predicted and saved to the prediction_output folder in the dataset directory. 

## Make ground truth data
<img width="425" alt="fiji3" src="https://user-images.githubusercontent.com/48200405/68600252-7a392580-04a2-11ea-880a-1c018149fd7b.png"> <img width="425" alt="fiji4" src="https://user-images.githubusercontent.com/48200405/68600259-7b6a5280-04a2-11ea-969d-9ddb084ab63a.png">
To correct the pancreatic area from the NN predictions, use the fiji macro 'adjust_prediction_v1.jim', which you'll find in the Fiji folder of the NN directory.

Open it dragging it into fiji and run the script. You are asked to select the original and the segmented image. 

Make sure to select the raw nd2 image and the predicted output from the neural network respectively and specify an output folder, where the ground truth data and results will be saved into. 

It's important to tick the **Downscale image** box, as it is mandatory to downscale the original image to the size of the NN segmentation mask

# Workflow with training
In case you have a new batch of images it may make sense to train the model again. The more images we have for training, the better and more general the prediction will work afterwards. 

## Data preparation for training
To process the original images, follow the same steps as in the Data preparation without training section.

In contrast to the previous section, we also need to process the ground truth data.

Copy the ground truth data into the folder **masks_unprocessed** in the NN directory of the TRANSFER server. 

In Pycharm run the script **process_images.py**. You're data is now ready in the **imgs** and **masks** folder of the NN/dataset directory. 

## Training

In Pycharm, open the script **UNet_main.py** and run it. This will take ~5-10h, depending on how big your dataset is. The output file will be saved in the NN folder, with the name **"pancreas_\[today's date\]_unet_best_model.pth"**

## Make prediction

This step is the same as in the **No training** section. The only difference is that the model we use for prediction here is different (the one you just trained before). 

## Make ground truth data

Follow the same steps as in the **No training** section.
