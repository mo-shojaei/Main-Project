# Table of Contents
* [Mobile Application User Review Classification](#mobile-application-user-review-classification)
* [Installation](#Installation)
* [Prerequisites](#Prerequisites)
* [Running the tests](#running-the-tests)
# Mobile Application User Review Classification

The popularity of mobile apps continues to grow over recent years. Mobile app stores (such as the Google Play Store and Apple's App Store) provide a unique user feedback mechanism to application developers through the possibility of posting application reviews in text format. A vast amount of user reviews for mobile applications make the usage of them for developers hard. Not all reviews are helpful to a developer for updating and making their app better. 
The goal of this project is to gather user reviews, analyze them and separate them based on their usefulness for software developer. Until now, that this readme is writen, the process compeletely finished until the binary classification to find the useful reviews.

Here you can see the domain model for this project:
![Screenshot](Capture.PNG)

And here is the architecture:
![Screenshot](Architecture.png)


## Installation
All parts of this system is written in Python. Below are some tools that you can download before started with this system:
| Name of the tools | 
| ------------- |
|[Python download](https://www.python.org/downloads/)|
|[Visual Studio Code](https://code.visualstudio.com/download)|
|[Jupyter Notebook](https://jupyter.org/install)|
|[Google Colab](https://colab.research.google.com/notebooks/welcome.ipynb)|


### Prerequisites

There are some libraries you need yo install before using each part of this system.

To use the first crawler, you need to install these libraries:
```
pip install csv
pip install codecs
pip install selenium 
pip install collections
pip install time
pip install re
```
To use the second crawler, which is my main crawler which I used to gather my dataset, you need to install these libraries:
```
pip install json
pip install pandas
pip install tqdm 
pip install seaborn
pip install matplotlib.pyplot
pip install pygments
pip install google_play_scraper
```
To use the data cleaner, you need to install these libraries:
```
pip install nltk
pip install inflect
```
There are a few other libraries that you need to be able to run the data cleaner, but you already installed them for the crawlers.
After installing all these libraries, you are ready to use each part of this system.

To use the BERT you need to install these libraries:
pip install math
pip install tensorflow
pip install tensorflow_hub

To use the feature extraction and classification model you need to install this library:

pip install scikit-learn


## Running the tests

There are some tests provided in Unit Test directory. Until now, the unit tests check to see if each function of data cleaner works in a right way. For each function, I write tests to check how does it respond in differenet situation. To run them you just need to install one library and add the data cleaner directory to your PYTHONPATH:
```
pip install unittest
```


## Authors

* **Mohammadreza Shojaei Kol Kachi** 

