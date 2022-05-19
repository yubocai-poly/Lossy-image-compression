# CSE102-Advanced-Programming-Final-Project

## Project Instruction
The goal of this project is to develop a lossy image compression format close to JPEG. Lossy compression algorithms allow to greatly reduce the size of image files at the price of loosing some data from the original image. In general, a lossy image compression works by pruning the information the human eye is not sensible to. This is the case, for instance, in the JPEG file format.

The JPEG compression algorithm can be decomposed into several steps:

- In theory, JPEG-based compression schemes are independent of the image color-space (e.g. Red-Green-Blue). However the best rates of compression are obtained in a color space that separates the light-intensity (luminance) from the color information (chrominance). Indeed, the human eye is quite sensitive to luminance but little to chrominance and we can then afford to lose more chrominance information.

- Hence, the algorithm starts by converting the original image from its initial RGB (Red-Green-Blue) colorimetric model towards the YCbCr model of type luminance/chrominance. In this model, Y is the information of luminance, and Cb and Cr give chrominance values, respectively the blue minus Y and the red minus Y.

- This low sensitivity of the human eye to chrominance is exploited by performing a subsampling of the color signals. The principle of the operation is to reduce the size of several blocks of chrominance in a single value.

- The image is then split into blocks of 8×8 subpixels.

- A 2D DCT (Discrete Cosine Transform) is then applied to each block. The DCT is a variant of the Fourier transform. It decomposes a block, considered as a function in two variables, into a sum of cosine functions. Each block is thus described in a map of frequencies and amplitudes rather than in pixels and color information.

- Then, a quantization step takes place. The quantization consists in dividing the DCT matrix by another one, called quantization matrix. The goal here is to attenuate the high frequencies, i.e. those to which the human eye is very insensitive. These frequencies have low amplitudes, and they are further attenuated by quantization (some coefficients are even often reduced to 0).

- Finally, each block is coded using a variation of the Run-length encoding associated with an arithmetic coding – both are lossless data compression algorithm.

## Project Schedule
The goal of this project is to write a JPEG-like image encoder/decoder. This is a 3-week project:

- during the first week, we will focus on:
    - loading / saving image files,
    - converting images between the RGB & the YCbCr color space, and
    - performing channels subsampling.

- during the second week, we will focus on the DCT transform, and

- during the third week, we will consider the RLE & arithmetic coding.




## Contributors

This project exists thanks to all the people who contribute. 
<a href="https://github.com/yubocai-poly/CSE102-Advanced-Programming-Fianl-Project/network/dependencies)"><img src="https://opencollective.com/standard-readme/contributors.svg?width=890&button=false" /></a>
