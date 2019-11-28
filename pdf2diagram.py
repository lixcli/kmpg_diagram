from pdf2photo import *
from argparse import ArgumentParser
import os


parser = ArgumentParser(description='''
        将pdf文件转换成jpg图片然后筛选出带有表格的图片
        将文件输出到out文件夹
''')
parser.add_argument("--in_path",type=str,default="test_pdf",
                    help="pdf dir")
parser.add_argument("--out_path",type=str,default="test_img",
                    help="jpg dir")
parser.add_argument("--choise",type=str,default="test",
                    help='''
                    choise :
                        test: 框出数据表
                        diagram: 将有表格的图片框出并保存到指定文件下
                    ''')

args = parser.parse_args()

in_dir = args.in_path
out_dir = args.out_path
if __name__ == "__main__":
    if not os.path.exists(out_dir):
        os.mkdir(out_dir)

    
    pdf_to_photo(in_dir,out_dir)
    wd = os.getcwd()
    file_list=[]
    for root, dirs, files in os.walk(out_dir):
        for file in files:
            file_list.append(file)
    with open("tmp/test_img.txt",'w') as f:
        for each in file_list:
            f.write(f"{wd}/{out_dir}/{each}\n")
    # test
    os.system(f"./test.sh -p {wd+'/tmp/test_img.txt'} -c test")
