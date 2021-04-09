import zipfile

with zipfile.ZipFile("Path to Task3_Images.zip", 'r') as zip_ref:
  zip_ref.extractall("Datasets/Task3 Images")



#/root/Unzipped/wav/.....
