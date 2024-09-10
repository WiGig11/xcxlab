# THis is a readme file for the work titiled:
Frequency-Aware Facial Image Shadow Removal through Skin Color and Texture Learning

skin color : low frequency
texture : high frequency

a frequency-domain image decomposition network (FDecomposeNet) to decompose the image into a high-frequency part and a low-frequency part, producing a skin color map and a texture map that are unrelated to shadows.

# 2024910 15:03
In the first read, overall strcture seems to contain steps:
1. divide the image into human face and background
2. face uses a shared encoder and two individual decoder to divide face into skin color and texture
3. skin color and texture are encoded through two encoder and individual fuse module to fuse the information into the removel process
4. inpput is passed through a encoder and a fusing process to join texutre and skin color to produce output
5. output is joined with backgroiund to form res
6. use a discriminator to judge.

# 2024910 19:35
In the second read with code, overall strcture seems to contain steps:
1. divide the image into human face and background (seems right)
2. face uses a shared encoder and two individual decoder to divide face into skin color and texture (this is pretrained ! )
3. skin color and texture are encoded through two encoder and individual fuse module to fuse the information into the removel process (yes in real use)
4. inpput is passed through a encoder and a fusing process to join texutre and skin color to produce output (yes)
5. output is joined with backgroiund to form res (maybe)
6. use a discriminator to judge. (yes)
the FDblock is actually two individual pathes but shares some common parts (not reasonable!!)

high-frequency:use Laplacian filter to take texture map as texture GT
low-frequency:use YCrCb to extract color information and a blur operation 


### !!!!
**The skin color and texture is pretrained!**
feature map : edge map from edge branch
face feature map : skin color map from skin color branch