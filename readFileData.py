"""
Copyright (c) 2022
Written by : Mehak Gagneja
Description: Read the content of files with csv, text, json, and excel extensions
"""

# import the required library: here I would need panda package for reading the files
import pandas as pd
    
# function to read the CSV file
# fileName: parameter to accept the file name
# fileSeperator: parameter to indicate how the data is seperated
# indexCol: parameter to identify the index column
def readCSVFile(fileName, fileSeperator, indexCol):
    # Check if the fileName is not null and then proceed further
    if fileName is not None:
        #if the fileSeperator is not provided it will be defaulted to ',' seperated
        csvFileContent = pd.read_csv(
            fileName,
            sep=fileSeperator if fileSeperator is not None else ',',
            index_col=indexCol,
            encoding='utf-8',
            on_bad_lines='warn'
            )
        print(csvFileContent)
        return csvFileContent
    else:
        print('Please provide a valid File Name')