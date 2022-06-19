"""
Yubo Cai
CSE102 - Advanced Programming
Final Project - Lossy image compression test file
2022.05.16 - 2022.06.20
"""

import myjpeg as jpeg
import math
import random

""" Check list !

Exercise 1_1 - ppm_tokenize(stream) - check
Exercise 1_2 - ppm_load(stream) - check
Exercise 1_3 - ppm_save(w, h, img, outfile) - check

Exercise 2_1 - RGB2YCbCr(r, g, b) - PASS
Exercise 2_2 - YCbCr2RGB(Y, Cb, Cr) - PASS
Exercise 2_3 - img_RGB2YCbCr(img) - check
Exercise 2_4 - img_YCbCr2RGB(Y, Cb, Cr) - check

Exercise 3_1 - subsampling(w, h, C, a, b) - check
Exercise 3_2 - extrapolate(w, h, C, a, b) - check

Exercise 4_1 - block_splitting(10, 9, C) - PASS

Exercise 5_1 - DCT(v) - PASS
Exercise 5_2 - IDCT(v) - PASS
Exercise 5_3 - DCT2(m, n, A) - PASS
Exercise 5_4 - IDCT2(m, n, A) - PASS
Exercise 5_5 - redalpha(i) - check
Exercise 5_6 - ncoeff8(i, j) - check
Exercise 5_7 - DCT_Chen(A) - check
Exercise 5_8 - IDCT_Chen(A) - check

Exercise 6_1 - quantization(A, Q) - PASS
Exercise 6_2 - quantizationI(A, Q) - PASS
Exercise 6_3 - Qmatrix(isY, phi) - check
 
Exercise 7_1 - zigzag(A) - PASS
Exercise 7_2 - rle0(g) - PASS
"""


# test1_1
def test1_1():
    with open('test.ppm') as stream:
        for token in jpeg.ppm_tokenize(stream):
            print(token)


# test1_2
def test1_2():
    with open('test.ppm') as stream:
        w, h, img = jpeg.ppm_load(stream)
    print(w)
    print(h)
    print(img)


# test1_3
def test1_3():
    with open('test.ppm') as stream:
        w, h, img = jpeg.ppm_load(stream)
    with open('output_test.ppm', 'w') as output:
        jpeg.ppm_save(w, h, img, output)


# test2_1 - check
def test2_1():
    a = jpeg.RGB2YCbCr(255, 0, 0)
    print(a)


# test2_2 - check
def test2_2():
    a = jpeg.YCbCr2RGB(255, 0, 0)
    print(a)


# test2_3 - check
def test2_3():
    with open('test.ppm') as stream:
        w, h, img = jpeg.ppm_load(stream)
    Y, Cb, Cr = jpeg.img_RGB2YCbCr(img)
    print(jpeg.img_RGB2YCbCr(img))


# test2_4  - check
def test2_4():
    with open('test.ppm') as stream:
        w, h, img = jpeg.ppm_load(stream)
    print(img)
    Y, Cb, Cr = jpeg.img_RGB2YCbCr(img)
    print(jpeg.img_YCbCr2RGB(Y, Cb, Cr))

    if jpeg.img_RGB2YCbCr(img) == jpeg.img_YCbCr2RGB(Y, Cb, Cr):
        print('True')
    else:
        print('Must be the round error')


# test3_1
def test3_1():
    with open('test.ppm') as stream:
        w, h, img = jpeg.ppm_load(stream)
    Y, Cb, Cr = jpeg.img_RGB2YCbCr(img)
    print(Y)
    Y = [[76, 149, 29], [225, 255, 0], [225, 255, 0]]
    print(jpeg.subsampling(3, 3, Y, 2, 2))
    print(jpeg.subsampling(3, 2, Y, 2, 2))
    print(jpeg.subsampling(3, 2, Y, 2, 1))
    print(jpeg.subsampling(5, 5, Y, 2, 2))


# test3_2
def test3_2():
    Y = [[176, 14], [240, 0]]
    print(jpeg.extrapolate(3, 2, Y, 2, 2))
    print(jpeg.extrapolate(3, 3, Y, 2, 2))
    print(jpeg.extrapolate(4, 4, Y, 2, 2))
    print(jpeg.extrapolate(5, 5, Y, 2, 2))


# test4_1 - check
C1 = [
    [1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
    [2, 3, 4, 5, 6, 7, 8, 9, 10, 1],
    [3, 4, 5, 6, 7, 8, 9, 10, 1, 2],
    [4, 5, 6, 7, 8, 9, 10, 1, 2, 3],
    [5, 6, 7, 8, 9, 10, 1, 2, 3, 4],
    [6, 7, 8, 9, 10, 1, 2, 3, 4, 5],
    [7, 8, 9, 10, 1, 2, 3, 4, 5, 6],
    [8, 9, 10, 1, 2, 3, 4, 5, 6, 7],
    [9, 10, 1, 2, 3, 4, 5, 6, 7, 8],
]

C2 = [
    [1, 2, 3, 4, 5, 6],
    [2, 3, 4, 5, 6, 7],
    [3, 4, 5, 6, 7, 8],
    [4, 5, 6, 7, 8, 9],
    [5, 6, 7, 8, 9, 10],
    [6, 7, 8, 9, 10, 1],
    [7, 8, 9, 10, 1, 2],
]

C3 = [
    [1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
    [2, 3, 4, 5, 6, 7, 8, 9, 10, 1],
    [3, 4, 5, 6, 7, 8, 9, 10, 1, 2],
    [4, 5, 6, 7, 8, 9, 10, 1, 2, 3],
    [5, 6, 7, 8, 9, 10, 1, 2, 3, 4],
    [6, 7, 8, 9, 10, 1, 2, 3, 4, 5],
]

C4 = [
    [1, 2, 3, 4, 5, 6, 7],
    [2, 3, 4, 5, 6, 7, 8],
    [3, 4, 5, 6, 7, 8, 9],
    [4, 5, 6, 7, 8, 9, 10],
    [5, 6, 7, 8, 9, 10, 1],
    [6, 7, 8, 9, 10, 1, 2],
    [7, 8, 9, 10, 1, 2, 3],
    [8, 9, 10, 1, 2, 3, 4],
    [9, 10, 1, 2, 3, 4, 5],
]


def test4_1():
    for el in jpeg.block_splitting(10, 9, C1):
        print(el)


# test5_1
def test5_1():
    print(jpeg.DCT([8, 16, 24, 32, 40, 48, 56, 64]))


# test5_2
def test5_2():
    v = [
        float(random.randrange(-10**5, 10**5))
        for _ in range(random.randrange(1, 128))
    ]
    v2 = jpeg.IDCT(jpeg.DCT(v))
    assert (all(math.isclose(v[i], v2[i]) for i in range(len(v))))


# test5_3
A = [
    [140, 144, 147, 140, 140, 155, 179, 175],
    [144, 152, 140, 147, 140, 148, 167, 179],
    [152, 155, 136, 167, 163, 162, 152, 172],
    [168, 145, 156, 160, 152, 155, 136, 160],
    [162, 148, 156, 148, 140, 136, 147, 162],
    [147, 167, 140, 155, 155, 140, 136, 162],
    [136, 156, 123, 167, 162, 144, 140, 147],
    [148, 155, 136, 155, 152, 147, 147, 136],
]

A_hat = [
    [1210.000, -17.997, 14.779, -8.980, 23.250, -9.233, -13.969, -18.937],
    [20.538, -34.093, 26.330, -9.039, -10.933, 10.731, 13.772, 6.955],
    [-10.384, -23.514, -1.854, 6.040, -18.075, 3.197, -20.417, -0.826],
    [-8.105, -5.041, 14.332, -14.613, -8.218, -2.732, -3.085, 8.429],
    [-3.250, 9.501, 7.885, 1.317, -11.000, 17.904, 18.382, 15.241],
    [3.856, -2.215, -18.167, 8.500, 8.269, -3.608, 0.869, -6.863],
    [8.901, 0.633, -2.917, 3.641, -1.172, -7.422, -1.146, -1.925],
    [0.049, -7.813, -2.425, 1.590, 1.199, 4.247, -6.417, 0.315],
]


def test5_3():
    print(jpeg.DCT2(8, 8, A))


# test5_4
def test5_4():
    m = random.randrange(1, 5)
    n = random.randrange(1, 5)
    A = [[float(random.randrange(-10**5, 10**5)) for _ in range(n)]
         for _ in range(m)]
    A2 = jpeg.IDCT2(m, n, jpeg.DCT2(m, n, A))
    assert (all(
        math.isclose(A[i][j], A2[i][j]) for i in range(m) for j in range(n)))


# test5_5
def test5_5():
    for i in range(56):
        print(jpeg.redalpha(i))


# test5_6
def test5_6():
    M8 = [[jpeg.ncoeff8(i, j) for j in range(8)] for i in range(8)]

    def M8_to_str(M8):

        def for1(s, i):
            return f"{'+' if s >= 0 else '-'}{i:d}"

        return "\n".join(" ".join(for1(s, i) for (s, i) in row) for row in M8)

    print(M8_to_str(M8))


mat_index = [
    [+4, +4, +4, +4, +4, +4, +4, +4],
    [+1, +3, +5, +7, -7, -5, -3, -1],
    [+2, +6, -6, -2, -2, -6, +6, +2],
    [+3, -7, -1, -5, +5, +1, +7, -3],
    [+4, -4, -4, +4, +4, -4, -4, +4],
    [+5, -1, +7, +3, -3, -7, +1, -5],
    [+6, -2, +2, -6, -6, +2, -2, +6],
    [+7, -5, +3, -1, +1, -3, +5, -7],
]


# test5_7
def test5_7():
    print(jpeg.DCT_Chen(A))


# test5_8
def test5_8():
    print(jpeg.IDCT_Chen(A_hat))


# test6_1
# test6_2
# test6_3
def test6_3():
    print(jpeg.Qmatrix(isY=True, phi=50))
    print(jpeg.Qmatrix(isY=False, phi=60))


# test7_1
def test7_1():
    for el in jpeg.zigzag(A):
        print(el)


# test7_2
def test7_2():
    for el in jpeg.rle0([0, 0, 4, 0, 0, 0, 7, 1, 0, 2, 0, 0]):
        print(el)