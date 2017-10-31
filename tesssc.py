import urllib.request
from bs4 import BeautifulSoup
import re
import datetime
import random

def downloadPic(picUrl):
    listChar=picUrl.split('/')
    picName=listChar[-1]
    pic=urlopen(picUrl)
    f=open(picName,"wb")
    f.write(pic.read())
    f.close()

url = "https://images.google.com"
data = urllib.request.urlopen(url).read()
data = data.decode('UTF-8')
print(data)