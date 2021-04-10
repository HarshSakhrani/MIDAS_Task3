import zipfile

with zipfile.ZipFile("Path to Task3_Images.zip", 'r') as zip_ref:       #Please add path to Task3_Images.zip
  zip_ref.extractall("path_to_add/MIDAS_Task3/Datasets/Task3 Images")	#Please add path_to_add for your local machine



#/root/Unzipped/wav/.....
