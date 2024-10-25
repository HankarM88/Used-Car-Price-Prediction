import requests
import pandas as pd 
from bs4 import BeautifulSoup

# Scrap data from one url 
def scrap(url):
	req = requests.get(url)
	data = req.text
	soup = BeautifulSoup(data, "html.parser")
	li_tags = soup.findAll("li",{'class':["sc-qmn92k-0 bdihtW"]})
	div_tags = soup.findAll("div",{'class':["sc-6p5md9-2 dhFfvW"]})
	p_tags = soup.findAll("p",{'class':["sc-1x0vz2r-0 bGMGAj"]})
	span_tags = soup.findAll("span",{'class':["sc-1x0vz2r-0 gCIGeB"]})
	results = []
	for tag in li_tags:
		results.append(':'.join(tag.findAll(text=True)).replace('\n',''))
	for tag in div_tags:
		results.append(':'.join(tag.findAll(text=True)).replace('\n',''))
	for tag in p_tags:
		results.append(':'.join(tag.findAll(text=True)).replace('\n',''))
	for tag in span_tags:
		results.append(':'.join(tag.findAll(text=True)).replace('\n',''))
	return results

if __name__ == '__main__':
    # Get the urls and save them in a file
    counter = 0
    
    df_urls =  pd.read_csv('../data/car_urls.csv',names=['url'])
    urls = list(df_urls.url)
    list_cars = []
    col_names = ['Operation','Sector','Car_State','Mileage','Year','Mark','Model','N_Doors',
                'Origin','First_Hand','Fuel_type', 'Fiscal_power','Gear_Type','Price','City','Date']
    # Loop over the urlss dataframe
    print('SCRAPING...')
    for url in urls[:]:
        data = scrap(str(url))
        list_cars.append(data)
        counter += 1
        if counter%100 == 0:
            page = int(round(counter/100,0))
            print(f"PAGE: {page}")
            df_cars = pd.DataFrame(data = list_cars)
            #save to csv
            df_cars.to_csv('../data/raw_used_cars.csv',mode='a',index=False)
            print(f'{100 * page} LINKS ADDED.') 
            # empty the list every 100 scrapings 
            del list_cars[:]
    
    
