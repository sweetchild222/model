import tensorflow as tf
from tensorflow.python.platform import build_info as build


print(f"cuda ver: {build.build_info['cuda_version']}") 

print(f"cudnn ver: {build.build_info['cudnn_version']}") 
