import csv
import cv2

def readCSV():
    url = open("./data/list.csv", "r")
    read_file = csv.reader(url)
    return read_file
    url.close()

def writeCSV(arrList):
    url = open("./data/list.csv", "a+", newline='')
    write_file = csv.writer(url)
    write_file.writerow(arrList)
    url.close()
