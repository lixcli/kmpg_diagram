import fitz #pymupdf的包的名称
import glob
import os
from tqdm import tqdm

def pdf_to_photo(in_dir_path,out_dir_path):
    pdfFile = glob.glob(in_dir_path+"/*.PDF")
    print("pdf转图片中...")
    
    for file in tqdm(pdfFile):
        doc = fitz.open(file)
        for pg in tqdm(range(doc.pageCount)):
            page = doc[pg]
            # 每个尺寸的缩放系数为4，zoom越大，生成图像像素越高
            zoom = 4.0
            rotate = int(0)
            trans = fitz.Matrix(zoom, zoom).preRotate(rotate)
            # 生成图片
            pm = page.getPixmap(matrix=trans, alpha=False)
            file = os.path.basename(file)
            ff = file.rstrip(".PDF")
            pm.writePNG( out_dir_path +'/' + ff + str(pg) + '.png')
        

if __name__=="__main__":
    pdf_to_photo(in_dir,out_dir)
    