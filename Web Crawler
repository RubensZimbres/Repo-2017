import urllib
from bs4 import BeautifulSoup
url=["https://www.isango.com/rome/rome-open-tour-and-colosseum-tour-and-vatican-museums_24323",'https://www.isango.com/rome/pompeii-half-day-tour-from-rome-by-high-speed-train_25749']

for i in range(0,len(url)):
    html = urllib.request.urlopen(url[i]).read()
    soup = BeautifulSoup(html,'html.parser')
    
    ### TOUR NAME
    soup.title.string
    
    ### NUMBER OF REVIEWS
    a3=soup.find(onclick="$('.ReviewWrap > a').click()")
    chars_to_remove = ['<a class="aLink" href="javascript:void(0)" onclick="$','</a>']
    sc = set(chars_to_remove)
    
    ### PRICE OK
    soup.find_all('del')
    
    a2=soup.find(itemprop="lowPrice")
    chars_to_remove = ['<em itemprop="lowPrice">£ ', '</em>']
    sc = set(chars_to_remove)
    
    ### RATING OK
    for link in soup.find_all('span')[12:13]:
        aa=(link.get('class'))
    aa[4]
    chars_to_remove = ['w', '-']
    sc = set(chars_to_remove)
    
    print('TITLE:',soup.title.string)
    print('PRICE = ',''.join([c for c in a2 if c not in sc]))
    print('RATING = ', int(''.join([c for c in aa[4] if c not in sc]))/10)
    print('REVIEWS = ',''.join([c for c in a3 if c not in sc]))
    print('\n')    
