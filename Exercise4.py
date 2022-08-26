#!/usr/bin/env python3
# coding: utf-8
# Author:   Silas Dreier <silas.dreier@ifw-kiel.de>
"""Exercise 4"""

import pandas as pd
from pathlib import Path
from nltk.corpus.reader.plaintext import PlaintextCorpusReader
import os

CONSTANT1 = Path("./Asp_Machine_Leaning_exercises")

# 1: Completed. Arrived at https://regexone.com/problem/complete?

# 2
file_list = sorted(Path('./data/speeches').glob('R0*'))
corpus = []
    try:
        for file_path in file_list:
            with open(file_path, encoding="utf8", errors="surrogateescape") as f_input:
            corpus.append(f_input.read())
    except: UnicodeDecodeError

