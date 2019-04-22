import requests
from bs4 import BeautifulSoup
import pandas as pd
import time

max_results = 20
city_set = ['Los+Angeles','Austin']
columns = ["city", "job_title", "company_name", "location", "summary"]

df = []
for city in city_set:
    for start in range(0, max_results, 1):
        page = requests.get('https://www.indeed.com/jobs?q=computer+science&l=' + str(city) + '&start=' + str(start))
        time.sleep(1)
        soup = BeautifulSoup(page.text, "lxml")
        for div in soup.find_all(name="div", attrs={"class":"row"}):
            job_post = []
            job_post.append(city)
            for a in div.find_all(name="a", attrs={"data-tn-element":"jobTitle"}):
                job_post.append(a["title"])
            company = div.find_all(name="span", attrs={"class":"company"})
            if len(company) > 0:
                for b in company:
                    job_post.append(b.text.strip())
            else:
                sec_try = div.find_all(name="span", attrs={"class":"result-link-source"})
                for span in sec_try:
                    job_post.append(span.text)

            c = div.findAll(name='span', attrs={'class': 'location'})
            for span in c:
                 job_post.append(span.text)
            d = div.findAll('div', attrs={'class': 'summary'})
            for span in d:
                job_post.append(span.text.strip())
        df.append(job_post)
        
df00=pd.DataFrame(df)
df00.columns=columns
df00.to_csv("jobs_report.csv",index=False)
