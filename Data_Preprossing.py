import xlrd
import xlwt
import numpy as np

def write_excel_xls(path, sheet_name,value):
    index = len(value)  # 获取需要写入数据的行数
    workbook = xlwt.Workbook()  # 新建一个工作簿
    sheet = workbook.add_sheet(sheet_name)  # 在工作簿中新建一个表格
    for i in range(0, index):
        for j in range(0, len(value[i])):
            sheet.write(i, j, value[i][j])  # 像表格中写入数据（对应的行和列）
    workbook.save(path)  # 保存工作簿
    print("保存结果成功！")

def read_excel_xls(path):
    ls=[]
    workbook = xlrd.open_workbook(path)  # 打开工作簿
    sheets = workbook.sheet_names()  # 获取工作簿中的所有表格
    worksheet = workbook.sheet_by_name(sheets[0])  # 获取工作簿中所有表格中的的第一个表格
    for i in range(0, worksheet.nrows):
        row=[]
        for j in range(0, worksheet.ncols):
            row.append(worksheet.cell_value(i, j))  # 逐行逐列读取数据
        ls.append(row)
    return ls
if(__name__=="__main__"):
    ls=read_excel_xls("DATA/{}_{}.xls".format("000938","紫光股份"))
    counter=1
    for i in ls[2:]:
        if (i[5]>ls[counter][5]):
            i.append(1)
        else:
            i.append(0)
        counter+=1
    mat=np.array(ls[2:])
    print(mat)
    #write_excel_xls(path="DATA/{}_{}.xls".format("000938","紫光股份"),sheet_name="preprocessing",value=ls)
