# coding=utf-8


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


def segment_2_small_files(src_filename, sharding_file_size=40000000, des_dir="/home/"):
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


def merge_files(sharding_dir="/home/sharding_dir", target_file="/Users/zhangtianlong01/merged_file.txt"):
    """将同一目录下的多个文本小文件，合并成一个文本文件
    
    Args：
        sharding_dir:存放小文件的目录
        target_file:合并成的大文件的完整路径
    
    Returns:
        Boolean:是否合并成功
    """
    if os.path.exists(sharding_dir):
        if sharding_dir[-1] != '/':
            sharding_dir += "/"
        file_list = [x for x in os.listdir(sharding_dir) if os.path.isfile(sharding_dir + x)]
        pattern = re.compile(r'^\.')
        file_list = [x for x in file_list if not pattern.match(x)]      # 过滤以[.]开头的文件
        for item in file_list:
            content_list = []
            with open(sharding_dir + item) as f:
                for line in f:
                    content_list.append(line)
                append_lines(target_file, content_list)
        return True
    else:
        return False     # 文件目录不存在


if __name__ == "__main__":
    print "RUNNING...."
    print "FIN!!!!"

