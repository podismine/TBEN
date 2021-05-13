import numpy as np
import operator
import scipy.ndimage as nd
import random
from random import gauss
from transformations import rotation_matrix
from scipy.ndimage.interpolation import map_coordinates

def dataaug(data):
    return coordinateTransformWrapper(data,maxDeg=20,maxShift=5, mirror_prob = 0)

def num2vect(bin_range = [14, 94], bin_step = 2, sigma = 1):

    bin_start = bin_range[0]
    bin_end = bin_range[1]
    bin_length = bin_end - bin_start
    bin_number = int(bin_length / bin_step)
    bin_centers = bin_start + float(bin_step) / 2 + bin_step * np.arange(bin_number)

    for i in range(bin_number):
        x1 = bin_centers[i] - float(bin_step) / 2
        x2 = bin_centers[i] + float(bin_step) / 2
    return bin_centers


# this code is modified from ****, link: ********
def make_rand_vector(dims):
    vec = [gauss(0, 1) for i in range(dims)]
    mag = sum(x**2 for x in vec) ** .5
    return [x/mag for x in vec]

def generate_label(label,sigma = 2, bin_step = 1):
    labelset = np.array([i * bin_step + 14 for i in range(int(84 / bin_step))])

    dis = np.exp(-1/2. * np.power((labelset - label)/sigma/sigma, 2))
    dis = dis / dis.sum()
    return dis, labelset

def flip_sagital(img,prob):
    if random.random() < prob:
        for batch in range(img.shape[2]):
            img[:,:,batch] = np.flipud(img[:,:,batch])
    return img

def coordinateTransformWrapper(X_T1,maxDeg=0,maxShift=7.5,mirror_prob = 0.5):
    #X_T1 = flip_sagital(X_T1, mirror_prob)
    randomAngle = np.radians(maxDeg*2*(random.random()-0.5))
    unitVec = tuple(make_rand_vector(3))
    shiftVec = [maxShift*2*(random.random()-0.5),
                maxShift*2*(random.random()-0.5),
                maxShift*2*(random.random()-0.5)]
    X_T1 = coordinateTransform(X_T1,randomAngle,unitVec,shiftVec)
    return X_T1

def coordinateTransform(vol,randomAngle,unitVec,shiftVec,order=1,mode='constant'):

    ax = (list(vol.shape))
    ax = [ ax[i] for i in [1,0,2]]
    coords=np.meshgrid(np.arange(ax[0]),np.arange(ax[1]),np.arange(ax[2]))

    xyz=np.vstack([coords[0].reshape(-1)-float(ax[0])/2,
               coords[1].reshape(-1)-float(ax[1])/2,
               coords[2].reshape(-1)-float(ax[2])/2,
               np.ones((ax[0],ax[1],ax[2])).reshape(-1)])
    
    mat=rotation_matrix(randomAngle,unitVec)
    transformed_xyz=np.dot(mat, xyz)

    x=transformed_xyz[0,:]+float(ax[0])/2+shiftVec[0]
    y=transformed_xyz[1,:]+float(ax[1])/2+shiftVec[1]
    z=transformed_xyz[2,:]+float(ax[2])/2+shiftVec[2]
    x=x.reshape((ax[1],ax[0],ax[2]))
    y=y.reshape((ax[1],ax[0],ax[2]))
    z=z.reshape((ax[1],ax[0],ax[2]))
    new_xyz=[y,x,z]
    new_vol=map_coordinates(vol,new_xyz, order=order,mode=mode)
    return new_vol
