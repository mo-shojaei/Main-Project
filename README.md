# Table of Contents
* [Mobile Application User Review Classification](#mobile-application-user-review-classification)
* [Installation](#Installation)
* [Prerequisites](#Prerequisites)
# Mobile Application User Review Classification

The popularity of mobile apps continues to grow over recent years. Mobile app stores (such as the Google Play Store and Apple's App Store) provide a unique user feedback mechanism to application developers through the possibility of posting application reviews in text format. A vast amount of user reviews for mobile applications make the usage of them for developers hard. Not all reviews are helpful to a developer for updating and making their app better. 
The goal of this project is to gather user reviews, analyze them and separate them based on their usefulness for software developer. Until now, that this readme is writen, the process compeletely finished until the feature extraction.

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

## Running the tests

There are some tests provided in Unit Test directory. Until now, the unit tests check to see if each function of data cleaner works in a right way. For each function, I write tests to check how does it respond in differenet situation. To run them you just need to install one library and add the data cleaner directory to your PYTHONPATH:
```
pip install unittest
```
## Deployment

Add additional notes about how to deploy this on a live system

## Built With

* [Dropwizard](http://www.dropwizard.io/1.0.2/docs/) - The web framework used
* [Maven](https://maven.apache.org/) - Dependency Management
* [ROME](https://rometools.github.io/rome/) - Used to generate RSS Feeds

## Contributing

Please read [CONTRIBUTING.md](https://gist.github.com/PurpleBooth/b24679402957c63ec426) for details on our code of conduct, and the process for submitting pull requests to us.

## Versioning

We use [SemVer](http://semver.org/) for versioning. For the versions available, see the [tags on this repository](https://github.com/your/project/tags). 

## Authors

* **Billie Thompson** - *Initial work* - [PurpleBooth](https://github.com/PurpleBooth)

See also the list of [contributors](https://github.com/your/project/contributors) who participated in this project.

## License

This project is licensed under the MIT License - see the [LICENSE.md](LICENSE.md) file for details

## Acknowledgments

* Hat tip to anyone whose code was used
* Inspiration
* etc
