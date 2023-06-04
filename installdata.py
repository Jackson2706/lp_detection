import os
import urllib.request as request
import zipfile

data_dir = "../data"

if not os.path.exists(data_dir):
    os.mkdir(data_dir)

url = "https://app.roboflow.com/ds/YeInb4iKqT?key=4HHD9n731g"
target_dir = os.path.join(data_dir, "License_Plate.v4i.voc.zip")

if not os.path.exists(target_dir):
    request.urlretrieve(url=url, filename=target_dir)

    zip = zipfile.ZipFile(target_dir)
    zip.extractall(data_dir)
    zip.close

os.remove(target_dir)