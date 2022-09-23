# Automatisation of hardware usage visualisations

## automate_daily_treated_nagios.py <br /> 
This python script automates the data retrieval and treatement from mysql database to an elasticsearch cluster. It provides the treated data originated from from nagios checkers to perform hardware usage visualisations. <br />
Its current setup is on the linux enowip16dsy machine with a daily cronojob at 23h55 that collects the data from 00h of the day in the matter till 00h of the next day. <br />

## automate_hourly_raw_nagios.py <br /> 
This python script automates the data retrieval and insertion from mysql databate to elasticsearch. Untreated data retrieved from the last hour.

In order to install the requirements, please create and activate a virtual environment, and install the requirements.txt as the following: <br /> 
$ virtualenv env <br /> 
$ source env/bin/activate <br /> 
$ pip3 install -r requirements.txt <br /> 