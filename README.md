# Assignment 5: Multi-Class Classification of Impressionist Painters

### Description of task: Classifiying artistis from painting using Logistic Regression and a Neural Network <br>
This assignment was assigned by the course instructor as “Assignment 5 – CNNs on Cultural Image Data”. The purpose of the assignment was to build a deep convolutional neural network classifier that is able to predict impressionist artists from paintings, in order to demonstrate our understanding of how to preprocess real-world, complex image data to be compatible with deep convolutional neural networks, as well as our understanding of how to build and train a deep convolutional neural network.  


### Content and Repository Structure <br>
The repository follows the overall structure below. The python ```cnn-artistis.py``` is located in the ```src``` folder. The outputs produced when running the scripts can be found within the ```output``` folder. The ```data``` folder contains a subset of the full dataset. If the user wishes to obtain the full dataset on which the model was trained, it is available on [Kaggle](https://www.kaggle.com/delayedkarma/impressionist-classifier-data). To obtain the full dataset, I suggest downloading it from Kaggle and uploading it to the data folder as a zip-file and then unzipping it via the command line. Alternatively, I recommend setting up the Kaggle command-line which is explained in this [article](https://necromuralist.github.io/kaggle-competitions/posts/set-up-the-kaggle-command-line-command/). 

| Folder | Description|
|--------|:-----------|
| ```data``` | A folder containing a subset of the full dataset.
| ```src``` | A folder containing the python scripts for the particular assignment.
| ```output``` | A folder containing the outputs produced when running the python scripts within the src folder.
| ```requirements.txt```| A file containing the dependencies necessary to run the python script.
| ```create_cnn_venv.sh```| A bash-file that creates a virtual environment in which the necessary dependencies listed in the ```requirements.txt``` are installed. This script should be run from the command line.
| ```LICENSE``` | A file declaring the license type of the repository.

### Usage and Technicalities <br>
If the user wishes to engage with the code and reproduce the obtained results, this section includes the necessary instructions to do so. It is important to remark that all the code that has been produced has only been tested in Linux and MacOS. Hence, for the sake of convenience, I recommend using a similar environment to avoid potential problems. 
If the user wishes to engage with the code and reproduce the obtained results, this section includes the necessary instructions to do so. First, the user will have to create their own version of the repository by cloning it from GitHub. This is done by executing the following from the command line: 

```
$ git clone https://github.com/sofieditmer/CNN.git
```

Once the user has cloned the repository, a virtual environment must be set up in which the relevant dependencies can be installed. To set up the virtual environment and install the relevant dependencies, a bash-script is provided, which automatically creates and installs the dependencies listed in the ```requirements.txt``` file when executed. To run the bash-script that sets up the virtual environment and installs the relevant dependencies, the user must execute the following from the command line. 

```
$ cd CNN
$ bash create_cnn_venv.sh 
```

Once the virtual environment has been set up and the relevant dependencies listed in the ```requirements.txt``` have been installed within it, the user is now able to run the ```cnn-artists.py``` script provided in the ```src``` folder in the repository directly from the command line. The user has the option of specifying additional arguments, however, this is not required to run the script. 
In order to run the script, the user must first activate the virtual environment in which the script can be run. Activating the virtual environment is done as follows.

```
$ source cnn_venv/bin/activate
```

Once the virtual environment has been activated, the user is now able to run ```cnn-artists.py``` script.

```
(cnn_venv) $ python cd src
(cnn_venv) $ python cnn-artists.py
```

For the ```cnn-artists.py``` script, the user is able to modify the following parameters, however, as mentioned this is not compulsory:

```
-t, --train_data, str <name-of-training-data-directory>, default = "training"
-te, --test_data: str <name-of-validation-data-directory>, default = "validation"
-e, --n_epochs: int <number-of-epochs>, default = 100
-b, --batch_size: int <size-of-batches>, default = 32
-o, --output: str <name-of-output-file>, default “cnn_classification_report.txt”
```
The abovementioned parameters allow the user to adjust the pipeline, if necessary, but because default parameters have been set, it makes the script run without explicitly specifying these arguments.  

### Output <br>
When running the ```cnn-artists.py```script, the following files will be saved in the ```output``` folder: 
1. ```LeNet_model.png``` Visualization of the LeNet model structure.
2. ```model_summary.txt``` Summary of the LeNet model structure.
3. ```model_history.png``` Visualization showing loss/accuracy of the model during training.
4. ```classification_report.txt``` Classification report.

### Discussion of Results <br>
The LeNet model was trained on the full dataset as well as the augmented, artificial data, and achieved weighted average accuracy score of 44% (see [Classification Report](https://github.com/sofieditmer/cnn/blob/main/output/cnn_classification_report.txt)). Moreover, the classification report reveals that the F1-scores for the 10 impressionist artists range between an accuracy of 31% to 51%, suggesting that it is difficult for the model to learn the data and reach a high performance on all classes. This is likely due to the size of the dataset, which does not meet that standard of what a CNN model usually requires to be able to reach high performance. Increasing the number of paintings for each artist would most likely make a difference in classification performance, given that the model would have more data to learn from.
When assessing the loss and accuracy learning curves of the model, it is evident that the model needs more data to be able to increase its performance (see [Loss/Accuracy Plot](https://github.com/sofieditmer/cnn/blob/main/output/model_loss_accuracy_history.png)). The training and validation accuracy curves steadily increase, but reaches a plateau around 12 epochs, where they are no longer able to learn more from the data. Similarly, the training and validation loss curves steadily decrease, suggesting that the model is learning from the data, but these also slowly flatten. All in all, the loss and accuracy learning curves of the model clearly suggest that increasing the amount of data would benefit the model performance. I tried to address this problem by implementing data augmentation, but this only increased the performance slightly. The problem with data augmentation is the validity of the artificial data being produced, and since I did not want to create data that the model would never encounter, I only increased the training data to a limited extent. The most optimal would be to gather more real-world data to inform the model.

### License <br>
This project is licensed under the MIT License - see the [LICENSE](https://github.com/sofieditmer/cnn/blob/main/LICENSE) file for details.

### Contact Details <br>
If you have any questions feel free to contact me on [201805308@post.au.dk](201805308@post.au.dk)
