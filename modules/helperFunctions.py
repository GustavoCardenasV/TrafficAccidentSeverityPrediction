import pickle
import numpy as np
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn import metrics
from sklearn.naive_bayes import GaussianNB
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import confusion_matrix
from sklearn import preprocessing
from keras.models import Sequential
from keras.layers import Dense
import numpy as np


def readFile(file):
    DataFrame = pd.read_csv(file,
                            error_bad_lines=False,
                            header=[0],
                            sep=';')
    return DataFrame


def checkMissingValues(DataFrame):
  percent_missing = DataFrame.isnull().sum() * 100 / len(DataFrame)
  missing_value_df = pd.DataFrame({'column_name': DataFrame.columns,
                                 'percent_missing': percent_missing})
  return missing_value_df


## It can be observed that the only missing values are for the feature "Light" with 1.2% of missing values and "Pedestrian Activity"
## with an non significant percentage.
## Because of the non representative percentage of missing values, it is decided to drop all NA values and after that check again
## implementing the "cleanMissingValues()" function.
def cleanMissingValues(DataFrame):
  DataFrame = DataFrame.dropna(axis=0, how='any')

  return DataFrame


# Here we create a "cleanDataFrame()" function.
# First drop the columns that are not going to be considered for further analisys due to the nature of the predictive problem.
# Then, it is necessary to rearange the features order to have an easier way to study them
def cleanDataFrame(DataFrame):
    DataFrame = DataFrame.drop(['Region',
                                'Year',
                                'Communication Video Equipment',
                                'Parking Lot Flag',
                                'Crash Count',
                                'Número de registros',
                                'Hit And Run Indicator',
                                'Total Casualty',
                                'Total Vehicles Involved',
                                'On Road Flag',
                                'Diagram',
                                'Type Collision 2',
                                'Pedestrian Activity',
                                'Speed Zone',
                                'Animal Flag',
                                'Distracted Involved',
                                'Impaired Involved',
                                'Speed Involved',
                                'Driver In Ext Distraction',
                                'Driver Inattentive',
                                'Driving Too Fast',
                                'Driving Without Due Care',
                                'Exceeding Speed',
                                'Excessive Speed',
                                ], axis=1)

    DataFrame = DataFrame[['Month',
                           'Weather',
                           'Light',
                           'Alcohol Involved',
                           'Drug Involved',
                           'Fell Asleep',
                           'Cyclist Involved',
                           'Pedestrian Involved',
                           'Motorcycle Involved',
                           'Land Use',
                           'Speed Advisory',
                           'Traffic Control',
                           'Traffic Flow',
                           'Road Class',
                           'Road Character',
                           'Road Condition',
                           'Road Surface',
                           'Crash Type',
                           ]]

# On a previous version of the EDA, I ran some experiments without the 'unknown' values, but after that turned back off except for
# the 'Road Surface' feature in order to get rid of a single corrupt piece of data causing a bug.
    # DataFrame = DataFrame[~DataFrame['Weather'].isin(['Unknown'])]
    # DataFrame = DataFrame[~DataFrame['Land Use'].isin(['Unknown'])]
    # DataFrame = DataFrame[~DataFrame['Traffic Control'].isin(['Unknown'])]
    # DataFrame = DataFrame[~DataFrame['Traffic Flow'].isin(['Unknown'])]
    # DataFrame = DataFrame[~DataFrame['Road Class'].isin(['Unknown'])]
    # DataFrame = DataFrame[~DataFrame['Road Character'].isin(['Unknown'])]
    # DataFrame = DataFrame[~DataFrame['Road Condition'].isin(['Unknown'])]
    DataFrame = DataFrame[~DataFrame['Road Surface'].isin(['Unknown'])]


    DataFrame = DataFrame.dropna(axis=0, how='any')
    return DataFrame

## Define a function con convert original categorical values into the categorical engineering values
def convertCategories(DataFrame):
    categ_var = {

        "Month": {"January": 0,
                  "February": 1,
                  "March": 2,
                  "April": 3,
                  "May": 4,
                  "June": 5,
                  "July": 6,
                  "August": 7,
                  "September": 8,
                  "October": 9,
                  "November": 10,
                  "December": 11},
        "Weather": {"Clear": 0,
                    "Raining": 1,
                    "Cloudy": 2,
                    "Snow/Sleet": 3,
                    "Fog": 4,
                    "Smog / Smoke": 5,
                    "Strong Wind": 6,
                    "Other": 0,
                    "Hail": 7,
                    "Unknown": 0
                    },
        "Light": {"Daylight": 0,
                  "Dark / Some Illumination": 1,
                  "Dark / Full Illumination": 1,
                  "Dark / No Illumination": 1,
                  "Dawn": 2,
                  "Dusk": 3,
                  "Other": 0},
        "Alcohol Involved": {"No": 0,
                             "Yes": 1},
        "Drug Involved": {"No": 0,
                          "Yes": 1},
        "Fell Asleep": {"No": 0,
                        "Yes": 1},
        "Cyclist Involved": {"No": 0,
                             "Yes": 1},
        "Pedestrian Involved": {"No": 0,
                                "Yes": 1},
        "Motorcycle Involved": {"No": 0,
                                "Yes": 1},
        "Land Use": {"Urban Residential": 0,
                     "Business/Shopping": 1,
                     "Agricultural/Undeveloped": 2,
                     "Industrial/Manufacturing": 3,
                     "Rural Residential": 4,
                     "Apartment Residential": 5,
                     "School/Playground": 6,
                     "Recreational/Park/Camping": 7,
                     "Other": 0,
                     "Unknown": 0
                     },
        "Speed Advisory": {"Not Applicable": 0,
                           "Advisory - 10 Km/H": 1,
                           "Advisory - 20 Km/H": 2,
                           "Advisory - 30 Km/H": 3,
                           "Advisory - 40 Km/H": 4,
                           "Advisory - 50 Km/H": 5,
                           "Advisory - 60 Km/H": 6,
                           "Advisory - 70 Km/H": 7,
                           "Advisory - 80 Km/H": 8,
                           "Advisory - 90 Km/H": 9,
                           "Advisory - 100 Km/H": 10,
                           "Advisory - 110 Km/H": 11},
        "Traffic Control": {"None": 0,
                            "Stop Sign": 1,
                            "Yield Sign": 2,
                            "Officer/Flagman/School Guard": 3,
                            "Railroad Crossing Sign": 4,
                            "Lane Use Turn Control Sign": 5,
                            "Traffic Signal - Red": 6,
                            "Traffic Signal - Yellow": 6,
                            "Traffic Signal - Green": 6,
                            "Traffic Signal W/Adv Flash - Red": 6,
                            "Traffic Signal W/Adv Flash- Yellow": 6,
                            "Traffic Signal W/Adv Flash - Green": 6,
                            "Tra Signal W/Adv Flash - Green": 6,
                            "Tra Signal W/Adv Flash - Red": 6,
                            "Tra Signal W/Adv Flash- Yellow": 6,
                            "Flashing Signal - Red": 7,
                            "Flashing Signal - Yellow": 7,
                            "Flashing Signal - Green": 7,
                            "Lane Use Signal - Red": 8,
                            "Lane Use Signal - Yellow": 8,
                            "Lane Use Signal - Green": 8,
                            "Not Applicable": 9,
                            "Other": 0,
                            "Unknown": 0
                            },
        "Traffic Flow": {"TwoWayTraffic": 1,
                         "OneWayTraffic": 0,
                         "Other": 0,
                         "Unknown": 0
                         },
        "Road Class": {
            "Two Lanes, Undivided": 1,
            "Four Lanes, Divided": 4,
            "Two Lanes, Divided": 1,
            "Not Applicable": 1,
            "Four Lanes, Undivided": 4,
            "One Lane, Undivided": 0,
            "Six Lanes, Divided": 6,
            "Three Lanes, Divided": 3,
            "Three Lanes, Undivided": 3,
            "One Lane, Divided": 0,
            "Five Lanes, Divided": 5,
            "Six Lanes, Undivided": 6,
            "Five Lanes, Undivided": 5,
            "One Lane, Ramp": 0,
            "Seven Lanes, Divided": 7,
            "Two Lanes, Ramp": 1,
            "Other": 1,
            "Seven Lanes, Undivided": 7,
            "Three Lanes, Ramp": 3,
            "Four Lanes, Ramp": 4,
            "Six Lanes, Ramp": 6,
            "Five Lanes, Ramp": 5,
            "Seven Lanes, Ramp": 7,
            "Unknown": 1
        },
        "Road Character": {
            "Straight - Flat": 0,
            "Straight - Some Grade": 0,
            "Straight - Steep Grade": 0,
            "Straight - Hillcrest": 0,
            "Straight - Sag": 0,
            "Single Curve - Flat": 1,
            "Single Curve - Some Grade": 1,
            "Single Curve - Steep Grade": 1,
            "Single Curve - Hillcrest": 1,
            "Single Curve - Sag": 1,
            "Sharp Curve - Flat": 2,
            "Sharp Curve - Some Grade": 2,
            "Sharp Curve - Steep Grade": 2,
            "Sharp Curve - Hillcrest": 2,
            "Sharp Curve - Sag": 2,
            "Switchback - Flat": 3,
            "Switchback - Some Grade": 3,
            "Switchback - Steep Grade": 3,
            "Switchback - Hillcrest": 3,
            "Switchback - Sag": 3,
            "Winding Curve - Flat": 4,
            "Winding Curve - Some Grade": 4,
            "Winding Curve - Steep Grade": 4,
            "Winding Curve - Hillcrest": 4,
            "Winding Curve - Sag": 4,
            "Reverse Curve - Flat": 5,
            "Reverse Curve - Some Grade": 5,
            "Reverse Curve - Steep Grade": 5,
            "Reverse Curve - Hillcrest": 5,
            "Reverse Curve - Sag": 5,
            "Other": 6,
            "Unknown": 0
        },
        "Road Condition": {
            "Dry": 0,
            "Wet": 1,
            "Snow": 2,
            "Ice": 3,
            "Slush": 4,
            "Other": 5,
            "Muddy": 6,
            "Unknown": 0
        },
        "Road Surface": {
            "Asphalt": 0,
            "Concrete": 1,
            "Gravel": 2,
            "Earth": 3,
            "Brick/Stone": 4,
            "Wood": 5,
            "Oiled Gravel": 6,
            "Other": 7
            # "Unknown": 0
        },
        "Crash Type": {"Property damage only": 0,
                       "Casualty crash": 1
                       }
    }
    DataFrame = DataFrame.replace(categ_var)
    return DataFrame

def useLabelEncoder(DataFrame):
  #creating labelEncoder
  le = preprocessing.LabelEncoder()

  # Converting string labels into numbers.
  DataFrame['Month']=le.fit_transform(DataFrame['Month'])
  DataFrame['Weather']=le.fit_transform(DataFrame['Weather'])
  DataFrame['Land Use']=le.fit_transform(DataFrame['Land Use'])
  DataFrame['Speed Advisory']=le.fit_transform(DataFrame['Speed Advisory'])
  DataFrame['Traffic Control']=le.fit_transform(DataFrame['Traffic Control'])
  DataFrame['Traffic Flow']=le.fit_transform(DataFrame['Traffic Flow'])
  DataFrame['Road Class']=le.fit_transform(DataFrame['Road Class'])
  DataFrame['Road Character']=le.fit_transform(DataFrame['Road Character'])
  DataFrame['Road Condition']=le.fit_transform(DataFrame['Road Condition'])
  DataFrame['Road Surface']=le.fit_transform(DataFrame['Road Surface'])

  return DataFrame