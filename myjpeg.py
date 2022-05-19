"""
Yubo Cai
CSE102
Final Project - Part 1
2022.05.16
"""

################################################ The PPM input format ################################################

import math


# Exercise 1_1
def ppm_tokenize(stream):
    """ 
    Input: 'test.ppm'
    Output: P3
            3
            2
            255
            255
            0
            0
            0
            255
            0
            0
            0
            255
            255
            255
            0
            255
            255
            255
            0
            0
            0
    """
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


def test1_1():
    with open('test.ppm') as stream:
        for token in ppm_tokenize(stream):
            print(token)


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


def test1_2():
    with open('test.ppm') as stream:
        w, h, img = ppm_load(stream)
    print(w)
    print(h)
    print(img)


# Exercise 1_3
# 应该是ppm_save(w, h, img, stream) stream 是输入内的文件名， 然后来去输出相同的文件ppm
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


def test1_3():
    with open('test.ppm') as stream:
        w, h, img = ppm_load(stream)
    with open('output_test.ppm', 'w') as output:
        ppm_save(w, h, img, output)


################################################ RGB to YCbCr conversion & channel separation ################################################


# Exercise 2_1
def RGB2YCbCr(r, g, b):
    """.DS_Store"""
    """write a function RGB2YCbCr(r, g, b) that takes a pixel's color in the RGB color space and that converts it in the YCbCr color space, returning the 3-element tuple (Y, Cb, Cr)."""
    """ 
    Input: (R, G, B)
    Output: (Y, Cb, Cr)
    """
    y = round(0.299 * r + 0.587 * g + 0.114 * b)
    cb = round(128 - 0.168736 * r - 0.331264 * g + 0.5 * b)
    cr = round(128 + 0.5 * r - 0.418688 * g - 0.081312 * b)
    # we testify whether the number is above 255 or below 0 for each
    if y > 255:
        y = 255
    if y < 0:
        y = 0
    if cb > 255:
        cb = 255
    if cb < 0:
        cb = 0
    if cr > 255:
        cr = 255
    if cr < 0:
        cr = 0
    return (y, cb, cr)


# test
def test2_1():
    a = RGB2YCbCr(255, 0, 0)
    print(a)


# Exercise 2_2
def YCbCr2RGB(Y, Cb, Cr):
    """write a function YCbCr2RGB(Y, Cb, Cr) that takes a pixel's color in the YCbCr color space and that converts it in the RGB color space, returning the 3-element tuple (r, g, b)."""
    """ 
    Input: (Y, Cb, Cr)
    Output: (R, G, B)
    """
    r = round(Y + 1.402 * (Cr - 128))
    g = round(Y - 0.344136 * (Cb - 128) - 0.714136 * (Cr - 128))
    b = round(Y + 1.772 * (Cb - 128))
    if r > 255:
        r = 255
    if r < 0:
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


# test
def test2_2():
    a = YCbCr2RGB(255, 0, 0)
    print(a)


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


def test2_3():
    with open('test.ppm') as stream:
        w, h, img = ppm_load(stream)
    Y, Cb, Cr = img_RGB2YCbCr(img)
    print(img_RGB2YCbCr(img))


# Exercise 2_4
def img_YCbCr2RGB(Y, Cb, Cr):
    """ 
    Input: ([[76, 150, 29], 
             [226, 255, 0]], 
            
            [[85, 44, 255], 
             [0, 128, 128]], 
             
            [[255, 21, 107], 
             [149, 128, 128]])
    Output: [[(255, 0, 0), (0, 255, 0), (0, 0, 255)], [(255, 255, 0), (255, 255, 255), (0, 0, 0)]]
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
                for i in range(a):
                    for j in range(b):
                        sum += C[row + i][col + j]
                matrix[-1].append(ave_num(sum, a, b))
            col += a
        row += b
        col = 0

    return matrix


def test3_1():
    with open('test.ppm') as stream:
        w, h, img = ppm_load(stream)
    Y, Cb, Cr = img_RGB2YCbCr(img)
    print(Y)
    Y = [[76, 149, 29], [225, 255, 0], [225, 255, 0]]
    print(subsampling(3, 3, Y, 2, 2))
    print(subsampling(3, 2, Y, 2, 2))
    print(subsampling(5, 5, Y, 2, 2))


# Exercise 3_2


def extrapolate(w, h, C, a, b):
    """
    Input: Y=[[176, 14], [240, 0]] /  extrapolate(3,3,Y,2,2)
           Y=[[176, 14], [240, 0]] /  extrapolate(4,4,Y,2,2)
           Y=[[176, 14], [240, 0]] /  extrapolate(5,5,Y,2,2)
           
    Output: [[176, 176, 14], 
             [176, 176, 14], 
             [240, 240, 0]]
    
            [[176, 176, 14, 14], 
             [176, 176, 14, 14], 
             [240, 240, 0, 0], 
             [240, 240, 0, 0]]
             
            [[176, 176, 14, 14, 0], 
             [176, 176, 14, 14, 0], 
             [240, 240, 0, 0, 0], 
             [240, 240, 0, 0, 0], 
             [0, 0, 0, 0, 0]]
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


def test3_2():
    Y = [[176, 14], [240, 0]]
    print(extrapolate(3, 2, Y, 2, 2))
    print(extrapolate(3, 3, Y, 2, 2))
    print(extrapolate(4, 4, Y, 2, 2))
    print(extrapolate(5, 5, Y, 2, 2))
