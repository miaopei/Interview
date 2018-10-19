#!/usr/bin/python
#coding=utf-8

def binary_search(f_n, f_l):
    """
    :type f_n: int
    :type f_l: list
    :rtype int
    """
    low = 0
    high = len(f_l)
    while low <= high:
        mid = (low + high) / 2
        if f_l[mid] == f_n:
            return mid
        elif f_l[mid] < f_n:
            low = mid + 1
        else:
            high = mid - 1

    return -1

if __name__ == '__main__':
    f_list = [1,2,3,7,8,9,10,5]

    f_list.sort()
    print("原有序列表为：{}".format(f_list))

    try:
        f_num = int(input("请输入要查找的数："))
    except:
        print("请输入正确的数！")
        exit()

    result = binary_search(f_num, f_list)
    if result != -1:
        print("要找的元素 {} 的序号为：{}".format(f_num, result))
    else:
        print("未找到！")

