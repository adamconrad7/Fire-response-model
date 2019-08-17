#import os
import io
import os.path
import requests
import PyPDF2
import numpy as np
import pandas as pd
from sklearn import model_selection
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
from sklearn.neighbors import KNeighborsClassifier
from requests.adapters import HTTPAdapter
from requests.packages.urllib3.util.retry import Retry
#from urllib.parse import urljoin
from bs4 import BeautifulSoup
#from pathlib import Path
from PyPDF2 import PdfFileReader
from datetime import timedelta
from sklearn.externals import joblib

def requests_retry_session(
    retries=3,
    backoff_factor=0.3,
    status_forcelist=(500, 502, 504),
    session=None,
):
    session = session or requests.Session()
    retry = Retry(
        total=retries,
        read=retries,
        connect=retries,
        backoff_factor=backoff_factor,
        status_forcelist=status_forcelist,
    )
    adapter = HTTPAdapter(max_retries=retry)
    session.mount('http://', adapter)
    session.mount('https://', adapter)
    return session

def get_weather(n_years):
    print('Getting weather data...')
    year0 = 2009
    dates = []
    for i in range(0,n_years):
        dates.append(str(year0))
        year0 += 1
    frames = []
    for i in range(len(dates)):
        print('Getting year: ', dates[i], '...')
        url="https://www1.ncdc.noaa.gov/pub/data/uscrn/products/daily01/" + dates[i]
        response = requests.get(url)
        year_html= BeautifulSoup(response.text, "html.parser")
        sitelist = []
        for link in year_html.select("a[href$='.txt']"):
            str_to_find = 'CRND0103-' + dates[i] + '-'
            link_s = str(link)
            link_str = link_s.split(str_to_find)
            #need to check if data is good, only include results with a 'C'
            if link_str[1][:2] == 'WA' or link_str[1][:2] == 'OR':
                    full_url = url + '/' + link.text
                    res = requests.get(full_url)
                    soup= str(BeautifulSoup(res.text, "html.parser")).split()
                    list = []
                    for j in soup:
                        if j == 'R' or j == 'C' or j == 'U': continue
                        list.append(float(j))
                    df = pd.DataFrame(np.array(list).reshape(-1,27))
                    names=["station_id", "Date", "version", "long", 'lat', 'airtemp_max', 'airtemp_min', 'airtemp_diff', 'airtemp_avg', 'precipitation', 'radiation', 'irtemp_max', 'irtemp_min', 'irtemp_avg', 'humidity_max', 'humidity_min', 'humidity_avg', 'soil_moisture_5cm', 'soil_moisture_10cm', 'soil_moisture_20cm', 'soil_moisture_50cm', 'soil_moisture_100cm', 'soil_temp_5cm', 'soil_temp_10cm', 'soil_temp_20cm', 'soil_temp_50cm', 'soil_temp_100cm']
                    df.columns = names
                    del df['station_id']
                    del df['version']
                    del df['long']
                    del df['lat']
                    sitelist.append(df)
        sites_averaged = sum(sitelist)/len(sitelist)
        frames.append(sites_averaged)
    print('Done.')
    #print(pd.concat(frames))
    return pd.concat(frames)

def get_NWPLs(n_years):
    print('Getting PL data...')
    start_date = '2009-01-01'
    end_date = str(2009+n_years) + '-12-31'
    daterange = pd.date_range(start_date, end_date)
    list = []
    c = 0
    for single_date in daterange:
        if c % 365 == 0:
            print('Getting year: ', 2009 + (c / 365), '...' )
        #we can loop trhough the urls and check if a 200 status code is recivied, then get pl and date from pdf.
        url='https://www.predictiveservices.nifc.gov/IMSR/' + single_date.strftime('%Y')+ '/' + single_date.strftime("%Y-%m-%d").replace('-', '') + 'IMSR.pdf'
        res = requests_retry_session().get(url)
        if res.status_code ==200:
            f = io.BytesIO(res.content)
            reader = PdfFileReader(f)
            for i in range(0, reader.getNumPages()):
                try:
                    page = reader.getPage(i).extractText()
                except Exception as e:
                    continue
                str_to_find = 'Northwest Area (PL '
                split_page = page.split(str_to_find)
                if len(split_page) > 1:
                    NWPL = split_page[1].strip()[0]
                    break
        else:
            NWPL = 1
        list.append(float(int(single_date.strftime("%Y-%m-%d").replace('-', ''))))
        list.append(float(int(NWPL)))
        c+=1
    df = pd.DataFrame(np.array(list).reshape(-1,2))
    df.columns = ['Date', 'NWPL']
    print('Done.')
    return df

def collect_data():
    n_years = int(input('How many years of data do you want?: '))
    df2 = get_weather(n_years)
    df1 = get_NWPLs(n_years)

    data = df2.merge(df1,on=['Date'])
    pd.DataFrame(data).to_csv('dynamic_data.csv', index=None)

def train():
    print('Training model...')
    dataset = pd.read_csv('dynamic_data.csv')

    # Split-out validation dataset
    array = dataset.values
    X = array[:,0:23]
    Y = array[:,23]
    validation_size = 0.1
    seed = 7
    X_train, X_validation, Y_train, Y_validation = model_selection.train_test_split(X, Y, test_size=validation_size, random_state=seed)
    scoring = 'accuracy'

    # Make predictions on validation dataset
    knn = KNeighborsClassifier()
    knn.fit(X_train, Y_train)
    predictions = knn.predict(X_validation)
    print(accuracy_score(Y_validation, predictions))
    print(confusion_matrix(Y_validation, predictions))
    print(classification_report(Y_validation, predictions))
    joblib.dump(knn,'NWPL_model')

def load_model():
    return joblib.load('NWPL_model')

def get_todays_weather():
    url="https://www1.ncdc.noaa.gov/pub/data/uscrn/products/daily01/" + pd.to_datetime('today').strftime('%Y')
    response = requests.get(url)
    year_html= BeautifulSoup(response.text, "html.parser")
    sitelist = []
    for link in year_html.select("a[href$='.txt']"):
        str_to_find = 'CRND0103-' + pd.to_datetime('today').strftime('%Y') + '-'
        link_s = str(link)
        link_str = link_s.split(str_to_find)
        #need to check if data is good, only include results with a 'C'
        if link_str[1][:2] == 'WA' or link_str[1][:2] == 'OR':
                full_url = url + '/' + link.text
                res = requests.get(full_url)
                soup= str(BeautifulSoup(res.text, "html.parser")).split()
                list = []
                for j in soup:
                    if j == 'R' or j == 'C' or j == 'U': continue
                    list.append(float(j))
                df = pd.DataFrame(np.array(list).reshape(-1,27))
                names=["station_id", "Date", "version", "long", 'lat', 'airtemp_max', 'airtemp_min', 'airtemp_diff', 'airtemp_avg', 'precipitation', 'radiation', 'irtemp_max', 'irtemp_min', 'irtemp_avg', 'humidity_max', 'humidity_min', 'humidity_avg', 'soil_moisture_5cm', 'soil_moisture_10cm', 'soil_moisture_20cm', 'soil_moisture_50cm', 'soil_moisture_100cm', 'soil_temp_5cm', 'soil_temp_10cm', 'soil_temp_20cm', 'soil_temp_50cm', 'soil_temp_100cm']
                df.columns = names
                del df['station_id']
                del df['version']
                del df['long']
                del df['lat']
                sitelist.append(df)
    sites_averaged = sum(sitelist)/len(sitelist)
    return sites_averaged.iloc[[-1]]

def predict_yesterday():
    model = load_model()
    today = get_todays_weather()
    return model.predict(today)

def main():
    if os.path.exists('NWPL_model'):
        print(predict_yesterday())
    elif os.path.exists('dynamic_data.csv'):
        train()
        print(predict_yesterday())
    else:
        collect_data()
        train()
        print(predict_yesterday())

main()
