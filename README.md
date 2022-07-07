## Disaster Response Pipeline

#### Project Description

In a disaster management system tracking messages and delivering it to relevant agencies promptly is crucial. Sometimes these messages need to be alerted to multiple authorities. This project attempts to do this and the real dataset which is provided by 'FigureEight' is used to perform ETL and train a ML pipeline.  

However, out of the 3 genre types of messages (direct, social, and news), looks like it includes fewer messages from social media compared to the others. It would be great in the next release if they can include similar numbers for all 3 classes.  

## Libraries Used

Following libraries were used for the analysis

```bash
python=3.6.15  
scikit-learn=0.23.2
matplotlib=3.3.4
seaborn=0.11.2
plotly=5.5.0  
pandas=1.1.5
numpy=1.19.5
flask=2.0.2
jinja2=3.0.3
nltk=3.6.6 
```
## Installation

If you don't have above packages then you can use following commands to install them

```bash
conda install 'missing package name'
```

## Files Descriptions

```bash
data:   
 →  process_data.py  
 →  disaster_categories.csv  
 →  disaster_messages.csv  
 →  DisasterResponse.db

  
models:  
 →  train_classifier.py  
 →  classifier.pkl.zip (need to unzip if you want only to run the dashboard)

app:   
 →  run.py  
 →  templates (go.html, master.html)
```

## How to Interact with this project

```bash
'
1. Run the following commands in the project\s root directory to set up your database and model.

    - To run ETL pipeline that cleans data and stores in database
        `python data/process_data.py data/disaster_messages.csv data/disaster_categories.csv data/DisasterResponse.db`
    - To run ML pipeline that trains classifier and saves
        `python models/train_classifier.py data/DisasterResponse.db models/classifier.pkl`


2. Go to `app` directory: `cd app`

3. Run your web app: `python run.py`
```


## Data Source
[FigureEight](https://appen.com/pre-labeled-datasets/)

## Acknowledgements
I want to thank FigureEight for sharing dataset and for the Udacity for the guidance and template code.

## License
[MIT](https://choosealicense.com/licenses/mit/)

