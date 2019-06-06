#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Apr 12 14:05:43 2019

@author: ashleshashinde
"""

from bs4 import BeautifulSoup
import requests
import re
import pandas as pd

URL = 'https://genius.com/Ed-sheeran-perfect-lyrics'
page = requests.get(URL)
html = BeautifulSoup(page.text, "html.parser") # Extract the page's HTML as a string

# Scrape the song lyrics from the HTML
lyrics = html.find("div", class_="lyrics").get_text()
print(lyrics)