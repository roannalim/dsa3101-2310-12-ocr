import os
import xlsxwriter
from tesseract_ocr_model import tesseract_ocr
from pytorch_ocr_model.pytorch_ocr_model_function import pytorch_easy_ocr

answers = {"capt_after": 1310,
           "cinnamon_after": 790,
           "cinnamon_before": 370,
           "rc4_after": 1940,
           "u_town_residence_after": 210,
           "u_town_residence_before": 160}

row_id = 0
col_id = 0

model = "pytorch" #"tesseract" or "tensorflow" or "pytorch"

workbook = xlsxwriter.Workbook(f'{model}_evaluation_results.xlsx')
worksheet = workbook.add_worksheet()
headers = ["id", "area", "value", "image_name", "ocr_result", "outcome"]

for item in headers:
    worksheet.write(row_id, col_id, item)
    col_id += 1

for area, value in answers.items():
    directory = f"../../../12-ocr-image-data-flattened/{area}"

    for filename in os.scandir(directory):
        if filename.is_file():
            filepath = filename.path
            result = pytorch_easy_ocr(filepath)

            if (str(value) in result):
                outcome = "Pass"
            else:
                outcome = "Fail"

            row_id += 1
            
            to_write = [row_id, area, value, filename.name, result, outcome]
            
            col_id = 0

            for item in to_write:
                if col_id >= 6: exit()
                worksheet.write(row_id, col_id, item)
                col_id += 1
                
workbook.close()
            