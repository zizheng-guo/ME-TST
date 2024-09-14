import pandas as pd

def load_excel(dataset_name):

    # For dataset CASME3
    if(dataset_name == 'CASME_3'):
        xl = pd.ExcelFile("../dataset/"+dataset_name + '/casme3_label.xlsx')
        codeFinal = xl.parse(xl.sheet_names[0]) #Get data
        codeFinal.rename(columns={
            'Subject':'subject', 
            'Filename':'video', 
            'Onset':'onset', 
            'Apex':'apex', 
            'Offset':'offset',
            'AU':'au', 
            'Objective class':'oc',
            'emotion':'emotion'}, inplace=True)
        videoNames = []
        subjectNames = []
        for videoName in codeFinal.iloc[:,1]:
            videoNames.append(str(videoName))
        codeFinal['videoName'] = videoNames #Redundant
        codeFinal['videoCode'] = videoNames
        codeFinal['type'] = 'micro-expression'
        codeFinal['subjectCode'] = codeFinal['subject'] #Redundant

    # For dataset SAMMLV
    elif(dataset_name=='SAMMLV'):
        xl_SAMMLV = pd.ExcelFile("../dataset/"+dataset_name + '/SAMM_LongVideos_V3_Release.xlsx')
        xl_SAMM = pd.ExcelFile("../dataset/"+dataset_name + '/SAMM_Micro_FACS_Codes_v2.xlsx')
        colsName_SAMMLV = ['Subject', 'Filename', 'Inducement Code', 'Onset', 'Apex', 'Offset', 'Duration', 'Type', 'Action Units', 'Notes']
        codeFinal_SAMMLV = xl_SAMMLV.parse(xl_SAMMLV.sheet_names[0], header=None, names=colsName_SAMMLV, skiprows=[0,1,2,3,4,5,6,7,8,9])
        colsName_SAMM = ['Subject', 'Filename', 'Inducement Code', 'Onset', 'Apex', 'Offset', 'Duration', 'Type', 'Action Units', 'Emotion', 'Classes', 'Notes']
        codeFinal_SAMM = xl_SAMM.parse(xl_SAMM.sheet_names[0], header=None, names=colsName_SAMM, skiprows=[0])
        codeFinal_SAMM['Filename'] = codeFinal_SAMM['Filename'].astype('object')
        codeFinal = codeFinal_SAMMLV.merge(codeFinal_SAMM[['Filename', 'Type', 'Emotion']], on=['Filename', 'Type']) #Merge two excel files
        videoNames = []
        subjectName = []
        for videoName in codeFinal.iloc[:,1]:
            videoNames.append(str(videoName).split('_')[0] + '_' + str(videoName).split('_')[1])
            subjectName.append(str(videoName).split('_')[0])
        codeFinal['videoCode'] = videoNames
        codeFinal['subjectCode'] = subjectName
        codeFinal['Type'].replace({'Micro - 1/2': 'micro-expression'}, inplace=True)
        codeFinal.rename(columns={'Type':'type', 'Onset':'onset', 'Offset':'offset', 'Apex':'apex', 'Emotion':'emotion'}, inplace=True) 

    print(codeFinal.columns)
    return codeFinal