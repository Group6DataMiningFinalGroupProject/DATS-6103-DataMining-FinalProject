#Image information from dataset is fetched
#URL is validated and image present in given URL is downloaded
#Loaded image is then processed to unique aspect ratio to maintain uniformity
#Resized image is placed in Resized_image Folder
#Errored out images are moved to Resized image1 folder
#Import necessary Packages
import requests, shutil
import os
from skimage import io
import numpy as np
from skimage.transform import resize
from skimage import img_as_ubyte
new_size = (256, 256)
errored_id = []
def download_prep(im_info,loc):
    try:
        response=requests.get(im_info.iloc[1], stream=True)
        open('./Resized_image1/'+str(im_info.iloc[0])+'.jpg','wb').write(response.content)
        img1 = io.imread('./Resized_image1/'+str(im_info.iloc[0])+'.jpg')
        io.imsave('./Resized_image/'+str(loc)+'/'+str(im_info.iloc[0])+'.jpg',img_as_ubyte(np.array(resize(img1,new_size,mode='reflect', anti_aliasing = True,anti_aliasing_sigma=None))))
        os.remove('./Resized_image1/'+str(im_info.iloc[0])+'.jpg')
    except:
        print(im_info.iloc[0])
        errored_id.append(im_info.iloc[0])