from bs4 import BeautifulSoup
import requests
import pandas as pd
# Define a function to get page urls 
def get_urls(main_url):
	list_urls = []
	# Loop over the paginated urls
	for i in range(600,700):
		# Get the page url
		url = main_url+str(i)
		# Get the request response
		req  = requests.get(url)
		page = req.content
		# Transform it to bs object
		soup = BeautifulSoup(page, "lxml")
		# Loop over page links
		for div in soup.findAll('div', {'class': ["oan6tk-0 dLOfLV"]}):
			a = div.findAll('a')[0]
			car_url = a.get('href')
			list_urls.append(car_url)
	# Save urls to csv file 
	df_urls = pd.DataFrame(data={"url": list_urls})
	df_urls.to_csv('../data/urls_2022.csv',mode='a',header=False,index=False)


if __name__ == '__main__':
	# Get the car urls and save them in a file
	counter = 0
	main_url = 'https://www.avito.ma/fr/maroc/voitures-%C3%A0_vendre?o='
	print('CRAWLING URLS...')
	get_urls(main_url)
	
	
	
