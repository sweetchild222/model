import gdown
import os
import zipfile

dir = "clip_data"
os.mkdir(dir)

file_id = "1tO1c277Z001SHWZdA_2CyDf4gwDNW4mb"
file_path = dir + "/clip_data.zip"
gdown.download(id=file_id, output=file_path, quiet=False)


zip_file = zipfile.ZipFile(file_path)
zip_file.extractall(path=dir)