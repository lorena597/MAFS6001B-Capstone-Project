import random
import traceback
from selenium import webdriver
from time import sleep
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.common.by import By
import pandas as pd
import pickle
import time

def getMultiplePage(dates):
    crypto_list = []
    to_return = []

    url = 'https://coinmarketcap.com/historical/{date}/'
            
    chrome_options = Options()
    chrome_options.add_argument("--headless")

    driver = webdriver.Chrome(options = chrome_options)

    for date in dates:
        print(f"================  Get {url.format(date = date)} =================")
        try:
            df = getSinglePage(driver , url.format(date = date)) 
           
            to_return.append(
                {
                    "date" : date ,
                    "df" : df
                }
                )
        except:
            print(traceback.format_exc())
            print(f"================  Get {url.format(date = date)} fail! ===============")
        sleep(1 + random.random()*4)


    return to_return, crypto_list

def getSinglePage(driver , url):
    driver.get(url)

    t = driver.find_element_by_tag_name('html')
    
    
    last_height = driver.execute_script("return document.body.scrollHeight")

    height = 0
    while height < last_height:
        driver.execute_script(f"scroll(0, {height});")
        height += 500
        sleep(0.5)
    #last_height = driver.execute_script("return document.body.scrollHeight")
    
    a = t.get_attribute('innerHTML')
    df = pd.read_html(a)[-1]

    return df

if __name__ == "__main__":
    temp_range = pd.bdate_range(start='1/1/2020', end='05/15/2022', freq='W-SUN')
    temp_range = [str(date)[:10].replace('-','') for date in temp_range]
    date_range = []
    last_date = ''
    for i in range(len(temp_range)):
        if i == 0:
            last_date = temp_range[i]
            date_range.append(temp_range[i])
            continue
        elif temp_range[i][:6] == last_date[:6]:
            continue
        else:
            date_range.append(temp_range[i])
            last_date = temp_range[i]
    to_return, cryptolist = getMultiplePage(date_range)

    for d in to_return:
        d['df'].to_csv(f"raw_data/{d['date']}.csv")


