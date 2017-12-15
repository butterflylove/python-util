# coding=utf-8

"""
Author:zhangtianlong
"""


import os
import re


def get_file_total_lines(filename):
    """获取一个文本文件的总行数
    
    Args:
        filename: 文件的完整路径
    
    Returns:
        int: 文件的总行数
    """
    count = 0
    with open(filename) as f:
        for line in f:
            count += 1
    return count


def append_lines(filename, content_list):
    """以追加的方式写文件，将列表中的内容按行写入到文件中

    Args:
        filename: 写入文件的完整路径 
        content_list: 写入内容列表

    Returns:
        void
    """
    f_des = file(filename, "a")
    f_des.writelines(content_list)
    f_des.close()


def segment_2_small_files(src_filename, sharding_file_size=20000000, des_dir="/home/"):
    """将大文件切分成小文件.

    Args:
        src_filename:源文件完整路径的字符串表示 
        sharding_file_size: 切分成的小文件的大小
        des_dir: 切分出的小文件的存放目录

    Returns:
        void
    """
    source_file = open(src_filename)
    file_no = 0
    while 1:
        lines = source_file.readlines(sharding_file_size)
        if not lines:
            break
        line_list = []
        for line in lines:
            line_list.append(line)
        append_lines(des_dir + "sharding_file_" + str(file_no), line_list)
        file_no += 1
    source_file.close()


def merge_files(sharding_dir="/home/sharding_dir", target_file="/Users/zhangtianlong01/merged_file.txt", sharding_file_prefix=r"."):
    """将同一目录下 指定前缀名的多个小文件，合并成一个文本文件
    
    Args：
        sharding_dir:存放小文件的目录
        target_file:合并成的大文件的完整路径
        sharding_file_prefix:小文件的文件名前缀，默认匹配任意前缀名的文件
    
    Returns:
        Boolean:是否合并成功
    """
    if os.path.exists(sharding_dir):
        if sharding_dir[-1] != '/':
            sharding_dir += "/"
        file_list = [x for x in os.listdir(sharding_dir) if os.path.isfile(sharding_dir + x)]
        remove_pattern = re.compile(r'^\.')
        file_list = [x for x in file_list if not remove_pattern.match(x)]      # 剔除以[.]开头的文件
        prefix_pattern = re.compile(sharding_file_prefix)
        file_list = [x for x in file_list if prefix_pattern.match(x)]   # 过滤符合指定前缀名的文件
        for item in file_list:
            content_list = []
            with open(sharding_dir + item) as f:
                for line in f:
                    content_list.append(line)
                append_lines(target_file, content_list)
        return True
    else:
        return False     # 文件目录不存在


def extract_n_columns(src_file, des_file, col_no_list, delimiter='\t'):
    """从一个文件中提取出固定几列，再写入到新的文件中

    Args:
        src_file:读取文件的路径
        des_file:写入文件的路劲
        col_no_list:提取列的编号列表，编号从0开始计数，例如[0,4,6]
        delimiter:所读取文件分割列的分隔符

    Returns:
        void
    """
    if len(col_no_list) == 0:
        return
    result = []
    with open(src_file) as f:
        for line in f:
            arr = line.strip("\n").split(delimiter)
            new_line = ""
            for index in col_no_list:
                new_line += arr[index] + "\t"
            new_line = new_line[:-1] + "\n"
            result.append(new_line)
    append_lines(des_file, result)


def read_column_2_list(src_file, col_no, delimiter='\t'):
    """从文件中读取固定一列的值，存入list中

    Args:
        src_file:读取文件的路径
        col_no:列的编号,从0开始
        delimiter:文件列的分隔符

    Returns:
        固定列所有值的列表
    """
    result = []
    with open(src_file) as f:
        for line in f:
            arr = line.strip("\n").split(delimiter)
            result.append(arr[col_no])
    return result


def select_with_where_in(src_file, des_file, where_col_no, in_list, result_no_list="*", delimiter='\t'):
    """模拟文件的select where in操作, 暂时只支持int,string类型的where in 操作

    Args:
        src_file:读取文件的路径
        des_file:写入结果文件的路径
        where_col_no:进行where in 操作的列的索引号
        in_list:where in的list, 是一个值列表
        result_no_list:读取结果列的索引号列表,默认*即选择所有列,
                       如果不取默认*,则填入要结果列列表,比如[1,4]
        delimiter:文件列的分隔符

    Returns:
        void
    """
    total_count = 0
    is_int_type = False  # where in 列是否为int类型
    if isinstance(result_no_list, list) and len(result_no_list) == 0:
        return
    if len(in_list) == 0:
        return
    in_set = set()
    for x in in_list:
        in_set.add(x)
    type_where_col = type(in_list[0])
    if type_where_col == type(1):
        is_int_type = True
    result_content = []
    f = open(src_file)
    while 1:
        lines = f.readlines(10000)  # 解决在大文件的情况下内存溢出问题
        if not lines:
            break
        for line in lines:
            arr = line.strip("\n").split(delimiter)
            col_value = arr[where_col_no]
            if is_int_type:
                col_value = int(col_value)
            if col_value in in_set:
                total_count += 1
                print total_count
                if result_no_list == "*":
                    result_content.append(line)
                elif len(result_no_list) == 1:
                    result_content.append(arr[result_no_list[0]] + "\n")
                elif len(result_no_list) > 1:
                    str = ""
                    for i in range(len(result_no_list)):
                        str += arr[result_no_list[i]] + "\t"
                    str = str[:-1] + "\n"
                    result_content.append(str)
    f.close()
    append_lines(des_file, result_content)


def select_join(src_file_0, src_file_1, join_col, des_file):
    """实现两个文件根据某一列进行join操作

    Args:
        src_file_0:第一个文件
        src_file_1:第二个文件
        join_col:
        des_file:join操作后生成的文件

    Returns:
        void
    """
    x = dict()
    with open(src_file_0) as f:
        for line in f:
            arr = line.strip("\n").split("\t")
            x[arr[0]] = arr[1] + "," + arr[2]
    y = dict()
    with open(src_file_1) as f:
        for line in f:
            arr = line.strip("\n").split("\t")
            y[arr[0]] = arr[1]
    result = []
    for k, v in x.iteritems():
        try:
            v2 = y[k]
            str = v + "," + v2 + "\n"
        except KeyError as e:
            str = v + "," + "\n"
        result.append(str)
    append_lines(des_file, result)


def read_kv_file(kv_file, delimiter='\t', value2int=True):
    """将kv文件读入字典中,若存在相同的键,则覆盖原来的value

    Args:
        kv_file:文件地址
        delimiter:kv文件中key和value的分隔符
        value2int:是否将字典中的value转化为int

    Returns:
        dict:字典
    """
    kv = dict()
    with open(kv_file) as f:
        for line in f:
            arr = line.strip("\n").split(delimiter)
            key = arr[0]
            value = arr[1]
            if value2int:
                value = int(value)
            kv[key] = value
    return kv


if __name__ == "__main__":
    print "RUNNING...."
    segment_2_small_files("/home/work/tsdb/admin/output3.txt", sharding_file_size=20000000, des_dir="/home/work/tsdb/admin/sharding3/")
    segment_2_small_files("/home/work/tsdb/admin/output4.txt", sharding_file_size=20000000, des_dir="/home/work/tsdb/admin/sharding4/")
    print "FIN!!!!"

