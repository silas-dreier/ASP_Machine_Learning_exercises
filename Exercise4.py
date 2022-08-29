#!/usr/bin/env python3
# coding: utf-8
# Author:   Silas Dreier <silas.dreier@ifw-kiel.de>
"""Exercise 4"""

import pandas as pd
import numpy as np
from pathlib import Path
import nltk
from sklearn.feature_extraction.text import TfidfVectorizer
from string import digits, punctuation
import pickle
from scipy.spatial.distance import pdist
from scipy.cluster.hierarchy import linkage, dendrogram
import matplotlib.pyplot as plt
import re


CONSTANT1 = Path("./Asp_Machine_Leaning_exercises")

# 1: Completed. Arrived at https://regexone.com/problem/complete?

# 2
# (a)
file_list = sorted(Path('./data/speeches').glob('R0*'))
corpus = []
for file_path in file_list:
    try:
       with open(file_path, encoding="utf8") as f_input:
            corpus.append(f_input.read())
    except UnicodeDecodeError:
        print(file_path)

# (b)
# Stemming, Tokenizing, Stopwords
_stemmer = nltk.snowball.SnowballStemmer("english")
_stopwords = nltk.corpus.stopwords.words("english")
def tokenize_and_stem(text):
    d = {p: "" for p in digits + punctuation}
    text = text.translate(str.maketrans(d))
    return [_stemmer.stem(t) for t in nltk.word_tokenize(text)]
# Vectorizing
tfidf = TfidfVectorizer(df_min=5, tokenizer=tokenize_and_stem, stop_words=_stopwords, ngram_range=(1,3) )
# df_min=5 added to reduce number of vectors & prevent PC from freezing
tfidf_matrix = tfidf.fit_transform(corpus)
# Creating Sparse Matrix
df_count = pd.DataFrame(tfidf_matrix.todense().T,
                        index=tfidf.get_feature_names_out())
# Pickle Data
with open("./output/speech_matrix.pk", 'wb') as handle:
    pickle.dump(df_count, handle, protocol=pickle.HIGHEST_PROTOCOL)
# Save Output as csv
df_count.to_csv("./output/terms.csv")

# 3 Speeches II
# a)
pickle_in = open("./output/speech_matrix.pk", "rb")
loaded_matrix = pickle.load(pickle_in)

# b)
array = np.array(loaded_matrix)

y = pdist(array, metric='cosine') #causes PC to freeze when df_min=5 is not introduced in the tfidf vectorization
Z = linkage(y, method='complete')
dn = dendrogram(Z, no_labels=True)


# 4 Job Ads
# a)
raw = []
f = open('data/Stellenanzeigen.txt', 'r', encoding='utf-8')
try:
    for line in f:
        raw.append(line.strip())
except UnicodeDecodeError:
    pass
df_raw = pd.DataFrame(raw)
df_raw.replace('', np.nan, inplace=True)
df_raw = df_raw.dropna()

# Columns
df_raw["Newspaper"] = df_raw[0].str.extract('(Tages-Anzeiger|Neue Zürcher Zeitung)')
df_raw["Date"] = df_raw[0].str.extract('(\d{1,2}\.\s*\w*[äüö]?\w*\s*\d{4})')
df_raw["Newspaper"]=df_raw["Newspaper"].fillna(method='ffill')
df_raw["Date"]=df_raw["Date"].fillna(method='ffill')
df_raw["Ad"] = df_raw[0]
df_raw["Ad"].iloc[3] = "Innovationsfreude und psychologisches Geschick, Ihre Kombination? " \
                       "Unsere Klientin, ein erfolgreiches Dienstleistungsunternehmen im Non Profit Bereich, " \
                       "mit zirka 60 Geschäftsstellen in der Schweiz, geniesst eine ausgezeichnete Reputation. " \
                       "Als wesentlicher Erfolgsfaktor des Unternehmens gelten die hochmotivierten Mitarbeiter/innen, " \
                       "deren fachliche Weiterbildung und persönliche Entwicklung laufend gefördert wird. " \
                       "Neue Formen der Beratung der Kunden und deren Betreuung sind die zukunftsorientierten " \
                       "Strategien im Markt und Ihre Chance, in einem anspruchsvollen Projekt mitzuwirken. " \
                       "In ein gut eingespieltes Team suchen wir zur Ergänzung eine/n Projektmitarbeiter/in 80% " \
                       "für herausforderndes Pilotprojekt in verschiedenen Kantonen. " \
                       "Sie sind der Geschäftsleitung unterstellt und zuständig für die konzeptionelle Ausrichtung " \
                       "sowie die Koordination und Evaluation des Projektes im Sozialbereich und " \
                       "arbeiten aktiv an dessen Umsetzung mit. Ihr Arbeitsbereich umfasst die Schaffung von " \
                       "Arbeitsplätzen für behinderte Menschen. Es ist eine spannende Aufgabe mit zukunftsweisenden " \
                       "Möglichkeiten und Chancen, sehr viel dazuzulernen. Es ist eine anspruchsvolle " \
                       "Drehscheibenfunktion für engagierte Mitarbeiter/innen mit einer abgeschlossenen " \
                       "Ausbildung (Bereich Administration, Betriebswirtschaft, Sozialarbeit, eventuell Erfahrung " \
                       "in einem öffentlichen Arbeitsvermittlungszentrum). Sie sind kommunikativ, teamfähig und haben " \
                       "Freude am Kontakt mit Menschen. Ihr Arbeitsplatz befindet sich in Zürich. Bitte " \
                       "senden Sie Ihre Bewerbung an die beauftragte Personalberatung. Wir freuen uns sehr, " \
                       "Sie bald persönlich kennenzulernen, um mit Ihnen weitere Details zu besprechen. " \
                       "Diskretion sichern wir Ihnen zu. " \
                       "[foto] BUCHER CONSULT management consulting services " \
                       "[adr], 8008 Zürich, [tel], [email], [wwwadr]."
df_raw["Ad"].iloc[4] = "Innovationsfreude und psychologisches Geschick, Ihre Kombination? " \
                       "Unsere Klientin, ein erfolgreiches Dienstleistungsunternehmen im Non Profit Bereich, " \
                       "mit zirka 60 Geschäftsstellen in der Schweiz, geniesst eine ausgezeichnete Reputation. " \
                       "Als wesentlicher Erfolgsfaktor des Unternehmens gelten die hochmotivierten Mitarbeiter/innen, " \
                       "deren fachliche Weiterbildung und persönliche Entwicklung laufend gefördert wird. " \
                       "Neue Formen der Beratung der Kunden und deren Betreuung sind die zukunftsorientierten " \
                       "Strategien im Markt und Ihre Chance, in einem anspruchsvollen Projekt mitzuwirken. " \
                       "In ein gut eingespieltes Team suchen wir zur Ergänzung eine/n Projektmitarbeiter/in 80% " \
                       "für herausforderndes Pilotprojekt in verschiedenen Kantonen. " \
                       "Sie sind der Geschäftsleitung unterstellt und zuständig für die konzeptionelle Ausrichtung " \
                       "sowie die Koordination und Evaluation des Projektes im Sozialbereich und " \
                       "arbeiten aktiv an dessen Umsetzung mit. Ihr Arbeitsbereich umfasst die Schaffung von " \
                       "Arbeitsplätzen für behinderte Menschen. Es ist eine spannende Aufgabe mit zukunftsweisenden " \
                       "Möglichkeiten und Chancen, sehr viel dazuzulernen. Es ist eine anspruchsvolle " \
                       "Drehscheibenfunktion für engagierte Mitarbeiter/innen mit einer abgeschlossenen " \
                       "Ausbildung (Bereich Administration, Betriebswirtschaft, Sozialarbeit, eventuell Erfahrung " \
                       "in einem öffentlichen Arbeitsvermittlungszentrum). Sie sind kommunikativ, teamfähig und haben " \
                       "Freude am Kontakt mit Menschen. Ihr Arbeitsplatz befindet sich in Zürich. Bitte " \
                       "senden Sie Ihre Bewerbung an die beauftragte Personalberatung. Wir freuen uns sehr, " \
                       "Sie bald persönlich kennenzulernen, um mit Ihnen weitere Details zu besprechen. " \
                       "Diskretion sichern wir Ihnen zu. " \
                       "[foto] BUCHER CONSULT management consulting services " \
                       "[adr], 8008 Zürich, [tel], [email], [wwwadr]."
df_raw["Ad"].iloc[5] = "Innovationsfreude und psychologisches Geschick, Ihre Kombination? " \
                       "Unsere Klientin, ein erfolgreiches Dienstleistungsunternehmen im Non Profit Bereich, " \
                       "mit zirka 60 Geschäftsstellen in der Schweiz, geniesst eine ausgezeichnete Reputation. " \
                       "Als wesentlicher Erfolgsfaktor des Unternehmens gelten die hochmotivierten Mitarbeiter/innen, " \
                       "deren fachliche Weiterbildung und persönliche Entwicklung laufend gefördert wird. " \
                       "Neue Formen der Beratung der Kunden und deren Betreuung sind die zukunftsorientierten " \
                       "Strategien im Markt und Ihre Chance, in einem anspruchsvollen Projekt mitzuwirken. " \
                       "In ein gut eingespieltes Team suchen wir zur Ergänzung eine/n Projektmitarbeiter/in 80% " \
                       "für herausforderndes Pilotprojekt in verschiedenen Kantonen. " \
                       "Sie sind der Geschäftsleitung unterstellt und zuständig für die konzeptionelle Ausrichtung " \
                       "sowie die Koordination und Evaluation des Projektes im Sozialbereich und " \
                       "arbeiten aktiv an dessen Umsetzung mit. Ihr Arbeitsbereich umfasst die Schaffung von " \
                       "Arbeitsplätzen für behinderte Menschen. Es ist eine spannende Aufgabe mit zukunftsweisenden " \
                       "Möglichkeiten und Chancen, sehr viel dazuzulernen. Es ist eine anspruchsvolle " \
                       "Drehscheibenfunktion für engagierte Mitarbeiter/innen mit einer abgeschlossenen " \
                       "Ausbildung (Bereich Administration, Betriebswirtschaft, Sozialarbeit, eventuell Erfahrung " \
                       "in einem öffentlichen Arbeitsvermittlungszentrum). Sie sind kommunikativ, teamfähig und haben " \
                       "Freude am Kontakt mit Menschen. Ihr Arbeitsplatz befindet sich in Zürich. Bitte " \
                       "senden Sie Ihre Bewerbung an die beauftragte Personalberatung. Wir freuen uns sehr, " \
                       "Sie bald persönlich kennenzulernen, um mit Ihnen weitere Details zu besprechen. " \
                       "Diskretion sichern wir Ihnen zu. " \
                       "[foto] BUCHER CONSULT management consulting services " \
                       "[adr], 8008 Zürich, [tel], [email], [wwwadr]."
df_raw=df_raw.drop(0, axis=1)
df_raw=df_raw.drop_duplicates()
df_raw=df_raw.drop(0, axis=0)
df_raw=df_raw.drop(3, axis=0)
df_raw=df_raw.drop(7, axis=0)
for i in range(10, 400, 3):
    df_raw=df_raw.drop(i, axis=0)
df_final = df_raw