import os
import jieba

from Tools import savefile, readfile
################################定义一个可以去除停用词的分词函数################################
def stopwordslist(filepath):
    stopwords = [line.strip() for line in open(filepath, 'r').readlines()]
    return stopwords
# 对句子进行分词
def seg_sentence(sentence):
    sentence_seged = jieba.cut(sentence.strip())
    stopwords = stopwordslist('stopwords.txt')  # 这里加载停用词的路径
    outstr = ''
    for word in sentence_seged:
        if word not in stopwords:
            if word != '\t':
                outstr += word
                outstr += " "
    return outstr

def corpus_segment(corpus_path, seg_path, lpath):
    '''
    corpus_path是未分词语料库路径
    seg_path是分词后语料库存储路径
    '''
    catelist = os.listdir(corpus_path)  # 获取corpus_path下的所有子目录

    print("正在分词...")
    # 获取每个目录（类别）下所有的文件
    for mydir in catelist:

        class_path = corpus_path + mydir + "/"  # 拼出分类子目录的路径如：train_corpus/art/
        seg_dir = seg_path + mydir + "/"  # 拼出分词后存贮的对应目录路径如：train_corpus_seg/art/

        if not os.path.exists(seg_dir):  # 是否存在分词目录，如果没有则创建该目录
            os.makedirs(seg_dir)

        file_list = os.listdir(class_path)  # 获取未分词语料库中某一类别中的所有文本

        for file_path in file_list:  # 遍历类别目录下的所有文件
            if 'pdf' in file_path:
                   gggggg=1
            else:
                if 'categories' in file_path:
                    fullname = class_path + file_path  # 拼出文件名全路径如：train_corpus/art/21.txt
                    source = open(fullname, 'r', encoding='gb18030', errors='ignore')  #
                    line = source.readline()

                    while line != "":

                         ss = line.split('**')

                         line = source.readline()
                         kk=seg_sentence(ss[1])
                         path=lpath+ ss[1] + "/"
                         if not os.path.exists(path):
                              os.makedirs(path)
                         savefile(path  +'[1].'+ss[1]+ '.txt', ''.join(kk+'').encode('utf-8'))



                else:
                    fullname = class_path + file_path  # 拼出文件名全路径如：train_corpus/art/21.txt
                    source = open(fullname, 'r', encoding='gb18030', errors='ignore')  #
                    line = source.readline()
                    content_seg = ''
                    i = 0
                    while line != "" and i < 10:
                        if '题目' in line or '摘要' in line or '关键字' in line or '*'in line:
                             content_seg=content_seg
                        else:
                             content = seg_sentence(line)
                             content_seg = content_seg + content + ' '
                        line = source.readline()
                        i = i + 1
                    savefile(seg_dir + file_path, ''.join(content_seg).encode('utf-8'))
                '''
                fullname = class_path + file_path  # 拼出文件名全路径如：train_corpus/art/21.txt
                content = readfile(fullname)  # 读取文件内容

                content = content.replace('\r\n'.encode('utf-8'), ''.encode('utf-8')).strip()  # 删除换行
                content = content.replace(' '.encode('utf-8'), ''.encode('utf-8')).strip()  # 删除空行、多余的空格
                content_seg = seg_sentence(content)  # 为文件内容分词

                savefile(seg_dir + file_path, ''.join(content_seg).encode('utf-8'))  # 将处理后的文件保存到分词后语料目录
               '''
    print("中文测试集分词结束！！！")
#if __name__ == "__main__":
def fenci():
    corpus_path = "./test/"  # 未分词分类语料库路径
    seg_path = "./testfc/"  # 分词后分类语料库路径
    lpath = "./trainfc/"
    corpus_segment(corpus_path, seg_path, lpath)
