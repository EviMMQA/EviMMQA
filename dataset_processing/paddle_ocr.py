import os
from paddleocr import PaddleOCR

def process_images(root_dir):
    """
    Process all JPG images in directory tree, excluding files with 'pseudo' in name,
    and save OCR results to TXT files
    """

    ocr = PaddleOCR(use_angle_cls=True, lang='en')
    
    for root, _, files in os.walk(root_dir):
        for file in files:
            if file.lower().endswith('.jpg') and 'pseudo' not in file.lower():
                img_path = os.path.join(root, file)
                txt_path = os.path.splitext(img_path)[0] + '.txt'
                
                # print(f"Processing: {img_path}")

                result = ocr.ocr(img_path, cls=True)

                extracted_text = []
                if result and result[0]:
                    for line in result[0]:
                        if line and line[1]:
                            extracted_text.append(line[1][0])

                with open(txt_path, 'w') as f:
                    f.write('\n'.join(extracted_text))
                # print(f"Saved OCR results to: {txt_path}")

if __name__ == "__main__":
    root_dir = "evidence_synthesis/articles"
    process_images(root_dir)