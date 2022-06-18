"""
Yubo Cai
CSE102 - Advanced Programming
Final Project - Lossy image compression
2022.05.16 - 2022.06.20
"""

################################################ The PPM input format ################################################

import math
import random


# Exercise 1_1
def ppm_tokenize(stream):
    s = stream.readlines()
    # we want to get rid of all the content after the '#'
    lis = []
    for el in s:
        a = el.split('#')[0].strip()
        lis.append(a)

    output = []
    output.append(lis[0])
    a = lis[1].split(' ')[0]
    b = lis[1].split(' ')[1]
    output.append(a)
    output.append(b)
    output.append(lis[2])

    for el in lis[3:]:
        if el == '':
            continue
        for el1 in el.split(' '):
            if el1 == '':
                continue
            output.append(el1)

    for element in output:
        yield element
    # print all the elements in the list


# Exercise 1_2
# we create a auxiliary function to convert the list to tuple
def convert(list):
    return tuple(list)


def ppm_load(stream):
    """ 
    Input: 'test.ppm'
    Output: 3
            2
            [[(255, 0, 0), (0, 255, 0), (0, 0, 255)], [(255, 255, 0), (255, 255, 255), (0, 0, 0)]]
    """
    lis = []
    img = []
    for token in ppm_tokenize(stream):
        lis.append(token)

    weight, height = int(lis[1]), int(lis[2])

    # we want to append a 3-element list to the img from lis[4:]
    n = (len(lis) - 4) // 3
    new_lis = lis[4:]
    for i in range(n):
        img.append(new_lis[i * 3:i * 3 + 3])

    new_array = []
    for el in img:
        for element in el:
            element = int(element)
            new_array.append(element)

    new_tuple_array = []
    for i in range(len(new_array)):
        if i % 3 == 0:
            new_tuple_array.append(convert(new_array[i:i + 3]))

    # In the end we want to return a matrix of width elements and height elements
    new_matrix = []
    for i in range(len(new_tuple_array) // weight):
        new_matrix.append(new_tuple_array[i * weight:i * weight + weight])

    return (weight, height, new_matrix)


# Exercise 1_3
def ppm_save(w, h, img, outfile):
    """ 
    Input: 'test.ppm'
    Output: 'output_test.ppm'
    """
    outfile.write('P3' + '\n')
    outfile.write(f'{w} {h}\n')
    outfile.write('255' + '\n')
    for row in img:
        for el in row:
            outfile.write(f'{el[0]} {el[1]} {el[2]} ')
            outfile.write('\n')


################################################ RGB to YCbCr conversion & channel separation ################################################


# Exercise 2_1
def RGB2YCbCr(r, g, b):
    """write a function RGB2YCbCr(r, g, b) that takes a pixel's color in the RGB color space and that converts it in the YCbCr color space, returning the 3-element tuple (Y, Cb, Cr)."""
    """ 
    Input: (R, G, B)
    Output: (Y, Cb, Cr)
    """
    y = round(0.299 * r + 0.587 * g + 0.114 * b)
    cb = round(128 - 0.168736 * r - 0.331264 * g + 0.5 * b)
    cr = round(128 + 0.5 * r - 0.418688 * g - 0.081312 * b)
    # we testify whether the number is above 255 or below 0 for each
    if y > 255:  # we round the number to 255 if it is above 255
        y = 255
    if y < 0:
        y = 0
    if cb > 255:
        cb = 255
    if cb < 0:  # we round the number to 0 if it is below 0
        cb = 0
    if cr > 255:
        cr = 255
    if cr < 0:
        cr = 0
    return (y, cb, cr)


# Exercise 2_2 - check
def YCbCr2RGB(Y, Cb, Cr):
    """
    write a function YCbCr2RGB(Y, Cb, Cr) that takes a pixel's color in the YCbCr color space and that converts it in the RGB color space, returning the 3-element tuple (r, g, b).
    Input: (Y, Cb, Cr)
    Output: (R, G, B)
    """
    r = round(Y + 1.402 * (Cr - 128))
    g = round(Y - 0.344136 * (Cb - 128) - 0.714136 * (Cr - 128))
    b = round(Y + 1.772 * (Cb - 128))
    if r > 255:  # we round the number to 255 if it is above 255
        r = 255
    if r < 0:  # we round the number to 0 if it is below 0
        r = 0
    if g > 255:
        g = 255
    if g < 0:
        g = 0
    if b > 255:
        b = 255
    if b < 0:
        b = 0
    return (r, g, b)


# Exercise 2_3
def img_RGB2YCbCr(img):
    """ 
    Input: [[(255, 0, 0), (0, 255, 0), (0, 0, 255)], [(255, 255, 0), (255, 255, 255), (0, 0, 0)]]
    Output: ([[76, 150, 29], [226, 255, 0]], [[85, 44, 255], [0, 128, 128]], [[255, 21, 107], [149, 128, 128]])
    """
    # we first create the matrix of Y
    Y = [0] * len(img)
    for i in range(len(img)):
        Y[i] = [0] * len(img[i])

    # same method for Cb and Cr
    Cb = [0] * len(img)
    for i in range(len(img)):
        Cb[i] = [0] * len(img[i])

    Cr = [0] * len(img)
    for i in range(len(img)):
        Cr[i] = [0] * len(img[i])

    for row in img:
        for pixel in row:
            r, g, b = pixel
            y, cb, cr = RGB2YCbCr(r, g, b)
            index_row = img.index(row)
            index_pixel = row.index(pixel)
            Y[index_row][index_pixel], Cb[index_row][index_pixel], Cr[
                index_row][index_pixel] = y, cb, cr

    return (Y, Cb, Cr)


# Exercise 2_4
def img_YCbCr2RGB(Y, Cb, Cr):
    """ 
    Input: ([[76, 150, 29], [226, 255, 0]], 
            [[85, 44, 255], [0, 128, 128]], 
            [[255, 21, 107], [149, 128, 128]])
    Output: [[(254, 0, 0), (0, 255, 1), (0, 0, 254)], [(255, 255, 0), (255, 255, 255), (0, 0, 0)]]
    """
    # inverse function of img_RGB2YCbCr
    len_row = len(Y)  # 2
    len_column = len(Y[0])  # 3
    img = [0] * len_row
    for i in range(len_row):
        img[i] = [0] * len_column

    for i in range(len_row):
        for j in range(len_column):
            y, cb, cr = Y[i][j], Cb[i][j], Cr[i][j]
            r, g, b = YCbCr2RGB(y, cb, cr)
            img[i][j] = (r, g, b)

    return img


################################################ Subsampling ################################################
# Exercise 3_1
def sub_matrix(w, h, C):
    # this function help make matrix C to the matrix size of w and h, the empty part we replace as 0
    mat = [-1] * h
    for i in range(w):
        mat[i] = [-1] * w

    for i in range(len(C)):
        for j in range(len(C[i])):
            mat[i][j] = C[i][j]

    return mat


def ave_num(sum, w, h):
    nodes = w * h
    return round(sum / (nodes))


def subsampling(w, h, C, a, b):
    """ 
    Input: [[76, 150, 29], 
            [226, 255, 0]]
    Output: subsampling(3, 3, Y, 2, 2)  [[176, 14], [240, 0]]
            subsampling(3, 2, Y, 2, 2)  [[176, 14]]
            subsampling(3, 2, Y, 2, 2)  The size of the image is not enough
    """
    C_width = len(C[0])
    C_height = len(C)
    C_index_width = C_width // a + 1
    C_index_height = C_height // b + 1
    if w >= C_index_width * a or h >= C_index_height * b:
        print('The size of the image is not enough')
        return None

    if w >= a:
        out_index_width = w - a + 1
    else:
        out_index_width = 1

    if h >= b:
        out_index_height = h - b + 1
    else:
        out_index_height = 1

    matrix = []

    row = 0
    col = 0
    while row < h:
        matrix.append([])
        while col < w:
            sum = 0
            if col + a > w or row + b > h:
                min_w = min(w, col + a)
                min_h = min(h, row + b)
                for i in range(row, min_h):
                    for j in range(col, min_w):
                        sum += C[i][j]
                w1 = min_w - col
                h1 = min_h - row
                matrix[-1].append(ave_num(sum, w1, h1))
            else:
                for i in range(b):
                    for j in range(a):
                        sum += C[row + i][col + j]
                matrix[-1].append(ave_num(sum, a, b))
            col += a
        row += b
        col = 0

    return matrix


# Exercise 3_2
def extrapolate(w, h, C, a, b):
    """
    Input: Y=[[176, 14], [240, 0]] /  extrapolate(3,3,Y,2,2)
           Y=[[176, 14], [240, 0]] /  extrapolate(4,4,Y,2,2)
           Y=[[176, 14], [240, 0]] /  extrapolate(5,5,Y,2,2)
    Output: [[176, 176, 14], [176, 176, 14], [240, 240, 0]]
            [[176, 176, 14, 14], [176, 176, 14, 14], [240, 240, 0, 0], [240, 240, 0, 0]]
            [[176, 176, 14, 14, 0], [176, 176, 14, 14, 0], [240, 240, 0, 0, 0], [240, 240, 0, 0, 0], [0, 0, 0, 0, 0]]
    """
    # we firss create the matrix of zeros in the size w and h
    Mat = [0] * h
    for i in range(h):
        Mat[i] = [0] * w

    C_width = len(C[0])
    C_height = len(C)

    if w >= C_width * a and h >= C_height * b:
        for i in range(C_height):
            for j in range(C_width):
                for k in range(b):
                    for l in range(a):
                        Mat[i * b + k][j * a + l] = C[i][j]

    else:
        # if w < C_width * a or h < C_height * b
        row = 0
        col = 0
        for i in range(h):
            for j in range(w):
                index_i = i // b
                index_j = j // a
                Mat[i][j] = C[index_i][index_j]

    return Mat


################################################ Block splitting ################################################
# Exercise 4_1
def block_splitting(w, h, C):
    """
    Write a function block_splitting(w, h, C) that takes a channel C and that yield all the 8x8 subblocks of the channel, line by line, fro left to right. 
    If the channel data does not represent an integer number of blocks, then you must inplement the aforementionned padding technique.
    """
    for i in range(0, h, 8):
        for j in range(0, w, 8):
            C_block = []
            for k in range(i, i + 8):
                C_block.append([])
                for l in range(j, j + 8):
                    index_i, index_j = min(h - 1, k), min(l, w - 1)
                    C_block[-1].append(C[index_i][index_j])

            yield C_block


################################################ Discrete Cosine Transform (DCT) ################################################
"""""" """""" """ The general case """ """""" """"""


# Exercise 5_1
def DCT(v):
    """
    Write a function DCT(v) that takes a vector v and that returns its DCT-II.
    """
    N = len(v)
    C = [0] * N
    for i in range(N):
        for n in range(N):
            if i == 0:
                delta = 1 / math.sqrt(2)
            else:
                delta = 1
            C[i] += delta * math.sqrt(2 / N) * v[n] * math.cos(
                math.pi * (2 * n + 1) * i / (2 * N))
    for i in range(N):
        C[i] = round(C[i], 2)
    return C


# Exercise 5_2
def IDCT(v):
    """
    Write a function IDCT(v) that computes the inverse DCT-II of the vector v.
    """
    N = len(v)
    # we first compute the matrix of coefficients
    C_ij = []
    for i in range(N):
        value = 0
        for j in range(N):
            if j == 0:
                delta = 1 / math.sqrt(2)
            else:
                delta = 1
            value += delta * v[j] * math.sqrt(2 / N) * math.cos(
                math.pi * (i + 0.5) * j / N)
        C_ij.append(value)
    return C_ij


# Exercise 5_3
"""We first define some operations on matrixes"""


def matrix(n):
    """In this function we compute the matrix of coefficients"""
    mat = [0] * n
    for i in range(n):
        mat[i] = [0] * n

    for i in range(n):
        for j in range(n):
            if i == 0:
                delta = 1 / math.sqrt(2)
            else:
                delta = 1
            mat[i][j] = delta * math.sqrt(2 / n) * math.cos(math.pi *
                                                            (2 * j + 1) * i /
                                                            (2 * n))

    return mat


def zero_matrix(m, n):
    """In this function we create a zero matrix"""
    mat = [0] * m
    for i in range(m):
        mat[i] = [0] * n

    return mat


def matrix_round(A, m, n, r):
    """In this function we round the matrix"""
    for i in range(m):
        for j in range(n):
            A[i][j] = round(A[i][j], r)
    return A


def matrix_mult(A, B):
    """In this function we compute the multiplication of two matrixes"""
    m = len(A)
    n = len(A[0])
    p = len(B[0])
    C = zero_matrix(m, p)
    for i in range(m):
        for j in range(p):
            for k in range(n):
                C[i][j] += A[i][k] * B[k][j]

    return C


def matrix_transpose(A):
    """In this function we compute the transpose of a matrix"""
    m = len(A)
    n = len(A[0])
    C = zero_matrix(n, m)
    for i in range(m):
        for j in range(n):
            C[j][i] = A[i][j]
    return C


def DCT2(m, n, A):
    C_m = matrix(m)
    C_n = matrix(n)

    # we first compute the matrix C_m * A
    C1 = zero_matrix(m, n)
    C_n_tranpose = matrix_transpose(C_n)
    C1 = matrix_mult(C_m, A)

    # we then compute the matrix C1 * tranpose(C_n)
    C2 = zero_matrix(m, n)
    C2 = matrix_mult(C1, C_n_tranpose)

    # we round the numbers in the matrix C2 to 3 decimal places

    return C2


# Exercise 5_4
def IDCT2(m, n, A):
    """In this function we compute the inverse of A, which we have A_hat and we compute the inverse of A_hat"""
    C_m = matrix(m)
    C_n = matrix(n)
    C1 = zero_matrix(m, n)
    C1 = matrix_mult(matrix_transpose(C_m), A)

    C2 = zero_matrix(m, n)
    C2 = matrix_mult(C1, C_n)
    C2 = matrix_round(C2, m, n, 3)

    return C2


"""""" """""" """ The 8x8 DCT-II Transform & Chen's Algorithm """ """""" """"""


# Exercise 5_5
def redalpha(i):
    """
    In this function we transfer i-th cosine with the parity of cosine function:
    cos(x+pi) = -cos(x)
    cos(x+2pi) = cos(x)
    """
    a = i // 32
    i = i - 32 * a

    if i <= 8:
        s = 1
        k = i

    if i > 8 and i <= 16:
        s = -1
        k = 16 - i

    if i > 16 and i <= 24:
        s = -1
        k = i - 16

    if i > 24 and i <= 32:
        s = 1
        k = 32 - i

    return (s, k)


# Exercise 5_6
def ncoeff8(i, j):
    if i == 0:
        s, k = 1, 4
    else:
        s, k = redalpha(i * (2 * j + 1))
    return (s, k)


# Exercise 5_7
def DCT_Chen(A):
    C1 = zero_matrix(8, 8)
    count = 0
    C_8 = matrix(8)

    for i in range(8):
        if i == 0:
            for j in range(8):
                column = 0
                for sub_i in range(8):
                    column += A[sub_i][j]
                C1[i][j] = column * C_8[i][0]
                count += 1

        if i == 1:
            for j in range(8):
                C1[i][j] = C_8[i][0] * (A[0][j] - A[7][j]) + C_8[i][1] * (
                    A[1][j] - A[6][j]) + C_8[i][2] * (
                        A[2][j] - A[5][j]) + C_8[i][3] * (A[3][j] - A[4][j])
                count += 4

        if i == 2:
            for j in range(8):
                C1[i][j] = C_8[i][0] * (A[0][j] + A[7][j] - A[3][j] - A[4][j]
                                        ) + C_8[i][1] * (A[1][j] + A[6][j] -
                                                         A[2][j] - A[5][j])
                count += 2

        if i == 3:
            for j in range(8):
                C1[i][j] = C_8[i][0] * (A[0][j] - A[7][j]) + C_8[i][6] * (
                    A[6][j] - A[1][j]) + C_8[i][5] * (
                        A[5][j] - A[2][j]) + C_8[i][4] * (A[4][j] - A[3][j])
                count += 4

        if i == 4:
            for j in range(8):
                C1[i][j] = C_8[i][0] * (A[0][j] + A[3][j] + A[4][j] + A[7][j] -
                                        A[2][j] - A[1][j] - A[5][j] - A[6][j])
                count += 1

        if i == 5:
            for j in range(8):
                C1[i][j] = C_8[i][0] * (A[0][j] - A[7][j]) + C_8[i][6] * (
                    -A[1][j] + A[6][j]) + C_8[i][2] * (
                        A[2][j] - A[5][j]) + C_8[i][3] * (A[3][j] - A[4][j])
                count += 4

        if i == 6:
            for j in range(8):
                C1[i][j] = C_8[i][0] * (A[0][j] + A[7][j] - A[3][j] - A[4][j]
                                        ) + C_8[i][2] * (A[2][j] + A[5][j] -
                                                         A[1][j] - A[6][j])
                count += 2

        if i == 7:
            for j in range(8):
                C1[i][j] = C_8[i][0] * (A[0][j] - A[7][j]) + C_8[i][6] * (
                    -A[1][j] + A[6][j]) + C_8[i][2] * (
                        A[2][j] - A[5][j]) + C_8[i][4] * (-A[3][j] + A[4][j])
                count += 4

    C2 = zero_matrix(8, 8)
    C_8T = matrix_transpose(C_8)

    # Then we use the same method to compute the second part of DCT

    for i in range(8):
        if i == 0:  # check
            for j in range(8):
                column = 0
                for sub_i in range(8):
                    column += C1[j][sub_i]
                C2[j][i] = column * C_8T[0][0]
                count += 1

        if i == 1:  # checked
            for j in range(8):
                C2[j][i] = C_8T[0][i] * (C1[j][0] - C1[j][7]) + C_8T[1][i] * (
                    C1[j][1] - C1[j][6]) + C_8T[2][i] * (
                        C1[j][2] - C1[j][5]) + C_8T[3][i] * (C1[j][3] -
                                                             C1[j][4])
                count += 4

        if i == 2:
            for j in range(8):
                C2[j][i] = C_8T[0][i] * (
                    C1[j][0] + C1[j][7] - C1[j][3] - C1[j][4]) + C_8T[1][i] * (
                        C1[j][1] + C1[j][6] - C1[j][2] - C1[j][5])
                count += 2

        if i == 3:  # checked
            for j in range(8):
                C2[j][i] = C_8T[0][i] * (C1[j][0] - C1[j][7]) + C_8T[6][i] * (
                    C1[j][6] - C1[j][1]) + C_8T[5][i] * (
                        C1[j][5] - C1[j][2]) + C_8T[4][i] * (C1[j][4] -
                                                             C1[j][3])
                count += 4

        if i == 4:  # checked
            for j in range(8):
                C2[j][i] = C_8T[0][i] * (C1[j][0] + C1[j][3] + C1[j][4] +
                                         C1[j][7] - C1[j][2] - C1[j][1] -
                                         C1[j][5] - C1[j][6])
                count += 1

        if i == 5:  # checked
            for j in range(8):
                C2[j][i] = C_8T[0][i] * (C1[j][0] - C1[j][7]) + C_8T[6][i] * (
                    -C1[j][1] + C1[j][6]) + C_8T[2][i] * (
                        C1[j][2] - C1[j][5]) + C_8T[3][i] * (C1[j][3] -
                                                             C1[j][4])
                count += 4

        if i == 6:  # checked
            for j in range(8):
                C2[j][i] = C_8T[0][i] * (
                    C1[j][0] + C1[j][7] - C1[j][3] - C1[j][4]) + C_8T[2][i] * (
                        C1[j][2] + C1[j][5] - C1[j][1] - C1[j][6])
                count += 2

        if i == 7:  # checked
            for j in range(8):
                C2[j][i] = C_8T[0][i] * (C1[j][0] - C1[j][7]) + C_8T[6][i] * (
                    -C1[j][1] + C1[j][6]) + C_8T[2][i] * (
                        C1[j][2] - C1[j][5]) + C_8T[4][i] * (-C1[j][3] +
                                                             C1[j][4])
                count += 4

    C2 = matrix_round(C2, 8, 8, 3)

    return C2


"""""" """""" """ The inverse 8x8 Transform """ """""" """"""


# Exercise 5_8
def matrix_change():
    C1 = matrix(8)
    C = zero_matrix(8, 8)
    for i in range(8):
        if i == 0:
            for j in range(8):
                C[i][j] = C1[0][j % 4]

        if i == 1:
            for j in range(8):
                C[i][j] = C1[2][j % 4]

        if i == 2:
            for j in range(8):
                C[i][j] = C1[4][j % 4]

        if i == 3:
            for j in range(8):
                C[i][j] = C1[6][j % 4]

        if i == 4:
            for j in range(8):
                if j <= 3:
                    C[i][j] = C1[1][j]
                else:
                    C[i][j] = -C1[1][j - 4]

        if i == 5:
            for j in range(8):
                if j <= 3:
                    C[i][j] = C1[3][j]
                else:
                    C[i][j] = -C1[3][j - 4]

        if i == 6:
            for j in range(8):
                if j <= 3:
                    C[i][j] = C1[5][j]
                else:
                    C[i][j] = -C1[5][j - 4]

        if i == 7:
            for j in range(8):
                if j <= 3:
                    C[i][j] = C1[7][j]
                else:
                    C[i][j] = -C1[7][j - 4]

    return C


def IDCT_Chen_aux(A):
    C = matrix_change()
    mat = []
    for i in range(8):
        for j in range(8):
            if j == 0:
                num = C[0][0] * (
                    A[0] +
                    A[4]) + C[4][0] * A[1] + C[1][0] * A[2] + C[5][0] * A[
                        3] + C[6][0] * A[5] + C[3][0] * A[6] + C[7][0] * A[7]
                mat.append(num)

            if j == 1:
                num = C[0][0] * (
                    A[0] -
                    A[4]) + C[5][0] * A[1] + C[3][0] * A[2] - C[7][0] * A[
                        3] - C[4][0] * A[5] - C[1][0] * A[6] - C[6][0] * A[7]
                mat.append(num)

            if j == 2:
                num = C[0][0] * (
                    A[0] -
                    A[4]) + C[6][0] * A[1] - C[3][0] * A[2] - C[4][0] * A[
                        3] + C[7][0] * A[5] + C[1][0] * A[6] + C[5][0] * A[7]
                mat.append(num)

            if j == 3:
                num = C[0][0] * (
                    A[0] +
                    A[4]) + C[7][0] * A[1] - C[1][0] * A[2] - C[6][0] * A[
                        3] + C[5][0] * A[5] - C[3][0] * A[6] - C[4][0] * A[7]
                mat.append(num)

            if j == 4:
                num = C[0][0] * (
                    A[0] +
                    A[4]) - C[7][0] * A[1] - C[1][0] * A[2] + C[6][0] * A[
                        3] - C[5][0] * A[5] - C[3][0] * A[6] + C[4][0] * A[7]
                mat.append(num)

            if j == 5:
                num = C[0][0] * (
                    A[0] -
                    A[4]) - C[6][0] * A[1] - C[3][0] * A[2] + C[4][0] * A[
                        3] - C[7][0] * A[5] + C[1][0] * A[6] - C[5][0] * A[7]
                mat.append(num)

            if j == 6:
                num = C[0][0] * (
                    A[0] -
                    A[4]) - C[5][0] * A[1] + C[3][0] * A[2] + C[7][0] * A[
                        3] + C[4][0] * A[5] - C[1][0] * A[6] + C[6][0] * A[7]
                mat.append(num)

            if j == 7:
                num = C[0][0] * (
                    A[0] +
                    A[4]) - C[4][0] * A[1] + C[1][0] * A[2] - C[5][0] * A[
                        3] - C[6][0] * A[5] + C[3][0] * A[6] - C[7][0] * A[7]
                mat.append(num)
    return mat


def IDCT_Chen(A):
    aux = []
    for el in A:
        aux.append(IDCT_Chen_aux(el))
    mat = []
    for i in range(8):
        mat.append([])
    for j in range(8):
        vec = [el[j] for el in aux]
        vec1 = IDCT_Chen_aux(vec)
        for k in range(8):
            mat[k].append(vec1[k])

    for i in range(8):
        for j in range(8):
            mat[i][j] = round(mat[i][j], 0)

    return mat


################################################ Quantization ################################################


# Exercise 6_1 - check
def quantization(A, Q):
    # The quantization of A by Q
    Mat = zero_matrix(8, 8)
    for i in range(8):
        for j in range(8):
            Mat[i][j] = round(A[i][j] / Q[i][j])

    return Mat


# Exercise 6_2 - check
def quantizationI(A, Q):
    # The inverse quantization of A by Q
    Mat = zero_matrix(8, 8)
    for i in range(8):
        for j in range(8):
            Mat[i][j] = round(A[i][j] * Q[i][j])

    return Mat


# Exercise 6_3
def quant_Sipser(M):
    if M == True:
        quant_matrix = [
            [16, 11, 10, 16, 24, 40, 51, 61],
            [12, 12, 14, 19, 26, 58, 60, 55],
            [14, 13, 16, 24, 40, 57, 69, 56],
            [14, 17, 22, 29, 51, 87, 80, 62],
            [18, 22, 37, 56, 68, 109, 103, 77],
            [24, 35, 55, 64, 81, 104, 113, 92],
            [49, 64, 78, 87, 103, 121, 120, 101],
            [72, 92, 95, 98, 112, 100, 103, 99],
        ]

    if M == False:
        quant_matrix = [
            [17, 18, 24, 47, 99, 99, 99, 99],
            [18, 21, 26, 66, 99, 99, 99, 99],
            [24, 26, 56, 99, 99, 99, 99, 99],
            [47, 66, 99, 99, 99, 99, 99, 99],
            [99, 99, 99, 99, 99, 99, 99, 99],
            [99, 99, 99, 99, 99, 99, 99, 99],
            [99, 99, 99, 99, 99, 99, 99, 99],
            [99, 99, 99, 99, 99, 99, 99, 99],
        ]

    return quant_matrix


def Qmatrix(isY, phi):
    Mat = quant_Sipser(M=isY)
    Q = zero_matrix(8, 8)
    if phi >= 50:
        S = 200 - 2 * phi
    else:
        S = round(5000 / phi)
    for i in range(8):
        for j in range(8):
            Q[i][j] = math.ceil((50 + S * Mat[i][j]) / 100)
    return Q


################################################ Zig-Zag walk & RLE Encoding ################################################
"""""" """""" """ Zig-Zag walk """ """""" """"""


# Exercise 7_1 - check
def zigzag(A):
    lis = []
    i, j = 0, 0
    lis.append(A[i][j])
    while (i <= 7 and j <= 7):

        if j % 2 == 0 and i == 0:
            j += 1
            lis.append(A[i][j])
            for m in range(1, j + 1):
                j, i = j - 1, i + 1
                lis.append(A[i][j])

        if i % 2 == 1 and j == 0 and i < 7:
            i += 1
            lis.append(A[i][j])
            for m in range(1, i + 1):
                j, i = j + 1, i - 1
                lis.append(A[i][j])

        if i == 7 and j % 2 == 0:
            j += 1
            lis.append(A[i][j])
            for m in range(7 - j):
                j, i = j + 1, i - 1
                lis.append(A[i][j])

        if j == 7 and i % 2 == 1:
            if j == 7 and i == 7:
                break
            i += 1
            lis.append(A[i][j])
            for m in range(7 - i):
                j, i = j - 1, i + 1
                lis.append(A[i][j])

    for el in lis:
        yield el


# Exercise 7_2 - check
def rle0(g):
    pre = 0
    for el in g:
        if el == 0:
            pre += 1
        else:
            yield ((pre, el))
            pre = 0


################################################ The test file ################################################
"""
Yubo Cai
CSE102 - Advanced Programming
Final Project - Lossy image compression test file
2022.05.16 - 2022.06.20
"""


# test1_1 - check
def test1_1():
    with open('test.ppm') as stream:
        for token in ppm_tokenize(stream):
            print(token)


# test1_2 - check
def test1_2():
    with open('test.ppm') as stream:
        w, h, img = ppm_load(stream)
    print(w)
    print(h)
    print(img)


# test1_3 - check
def test1_3():
    with open('test.ppm') as stream:
        w, h, img = ppm_load(stream)
    with open('output_test.ppm', 'w') as output:
        ppm_save(w, h, img, output)


# test2_1 - check
def test2_1():
    a = RGB2YCbCr(255, 0, 0)
    print(a)


# test2_2 - check
def test2_2():
    a = YCbCr2RGB(255, 0, 0)
    print(a)


# test2_3 - check
def test2_3():
    with open('test.ppm') as stream:
        w, h, img = ppm_load(stream)
    Y, Cb, Cr = img_RGB2YCbCr(img)
    print(img_RGB2YCbCr(img))


# test2_4  - check
def test2_4():
    with open('test.ppm') as stream:
        w, h, img = ppm_load(stream)
    print(img)
    Y, Cb, Cr = img_RGB2YCbCr(img)
    print(img_YCbCr2RGB(Y, Cb, Cr))

    if img_RGB2YCbCr(img) == img_YCbCr2RGB(Y, Cb, Cr):
        print('True')
    else:
        print('Must be the round error')


# test3_1
def test3_1():
    with open('test.ppm') as stream:
        w, h, img = ppm_load(stream)
    Y, Cb, Cr = img_RGB2YCbCr(img)
    print(Y)
    Y = [[76, 149, 29], [225, 255, 0], [225, 255, 0]]
    print(subsampling(3, 3, Y, 2, 2))
    print(subsampling(3, 2, Y, 2, 2))
    print(subsampling(3, 2, Y, 2, 1))
    print(subsampling(5, 5, Y, 2, 2))


# test3_2
def test3_2():
    Y = [[176, 14], [240, 0]]
    print(extrapolate(3, 2, Y, 2, 2))
    print(extrapolate(3, 3, Y, 2, 2))
    print(extrapolate(4, 4, Y, 2, 2))
    print(extrapolate(5, 5, Y, 2, 2))


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
    for el in block_splitting(10, 9, C1):
        print(el)


# test5_1
def test5_1():
    print(DCT([8, 16, 24, 32, 40, 48, 56, 64]))


# test5_2
def test5_2():
    v = [
        float(random.randrange(-10**5, 10**5))
        for _ in range(random.randrange(1, 128))
    ]
    v2 = IDCT(DCT(v))
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
    print(DCT2(8, 8, A))


# test5_4
def test5_4():
    m = random.randrange(1, 5)
    n = random.randrange(1, 5)
    A = [[float(random.randrange(-10**5, 10**5)) for _ in range(n)]
         for _ in range(m)]
    A2 = IDCT2(m, n, DCT2(m, n, A))
    assert (all(
        math.isclose(A[i][j], A2[i][j]) for i in range(m) for j in range(n)))


# test5_5
def test5_5():
    for i in range(56):
        print(redalpha(i))


# test5_6
def test5_6():
    M8 = [[ncoeff8(i, j) for j in range(8)] for i in range(8)]

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
    print(DCT_Chen(A))


# test5_8
def test5_8():
    print(IDCT_Chen(A_hat))


# test6_1
# test6_2
# test6_3
def test6_3():
    print(Qmatrix(isY=True, phi=50))
    print(Qmatrix(isY=False, phi=60))


# test7_1
def test7_1():
    for el in zigzag(A):
        print(el)


# test7_2
def test7_2():
    for el in rle0([0, 0, 4, 0, 0, 0, 7, 1, 0, 2, 0, 0]):
        print(el)