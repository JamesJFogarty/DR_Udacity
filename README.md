# Disaster Response Pipeline Project
This NLP pipeline looks at the Udacity Project for predicting what Response is needed to a given disaster given the messages sent. This was done in different genres (direct to emergency services, through news or social media)

## ML Notes
This Classifier uses a RandomF with a basic gris search - Other methods I recomend trying are Logistic regression, Naive Bayes and LinearSVC. These all work really well for this type of classification.
To impove this model including the Genre category as a feature variable could be a good idea as different infomation outlets provide different issues and can could help improve the outcome (you can use Feature Union to include this within the models/train_classifier.py file within the Pipeline function).

### Instructions:
1. Run the following commands in the project's root directory to set up your database and model.

    - To run ETL pipeline that cleans data and stores in database
        `python data/process_data.py data/disaster_messages.csv data/disaster_categories.csv data/DisasterResponse.db`
    - To run ML pipeline that trains classifier and saves
        `python models/train_classifier.py data/DisasterResponse.db models/classifier.pkl`

2. Run the following command in the app's directory to run your web app.
    `python run.py`

3. Go to http://0.0.0.0:3001/

#### Web app
In the web app once this is run you will initially have some visuals outlining the data once you input a line of text and hit the search button you will then recive predictions for the disaster highlighted in green