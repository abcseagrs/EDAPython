# CLI : shiny run
import os
from math import ceil
from typing import List
import json
import pandas
# import shinyswatch
from pathlib import Path
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import asyncio
# import pycanon 
import pycanon.anonymity as anony
from shiny import App, Inputs, Outputs, Session, reactive, render, run_app, ui

import chardet
import chardet
import csv

def guessDelimiter(filepathname):
    with open(filepathname, 'rb') as f:
        content = f.read(1 * 1024 * 1024)
        result = chardet.detect(content)
        s = csv.Sniffer()
        
        delimiter = s.sniff(content.decode(result['encoding'])).delimiter
    
        return delimiter

def guessEncoding(filepathname):
    with open(filepathname, 'rb') as f:
        content = f.read(1 * 1024 * 1024)
        result = chardet.detect(content)
        
        return result['encoding']


# set plot font, Malgun Gothic
plt.rcParams["font.family"] = "Malgun Gothic"
# sns.set(font="Nanum Gothic", rc={"axes.unicode_minus":False}, style='darkgrid')

from util.sharedStore import sharedStore

# =====================================
# Define UI
# =====================================

store = sharedStore()
SHINY_SHAREDATA = store.get("SHINY_SHAREDATA")

input_file_stype = {"style": "display:black;"}

LOADING_UI = False

app_ui = ui.page_fluid(
    {"style": "background-color: rgba(0, 128, 255, 0.1);border:0;padding:0;margin:0;font-size: 13px;"},
    ui.layout_sidebar(
        ui.panel_sidebar(
            ui.tags.div(
                {"style": "display:black;"},
                ui.input_file("file1", "분석 파일 선택 : ", multiple=False, button_label="파일 선택...", placeholder="파일 선택 바랍니다.", accept=".csv,.txt")
            ),
            ui.navset_tab_card(
                ui.nav(
                    "그래프",
                    ui.input_select(
                        "plotColumn",
                        "분석할 칼럼 선택 : ",
                        [],
                    ),
                    ui.input_numeric(
                        "plotObs", 
                        "분할 구간수 입력 : ", 10, min=1, max=100),
                    ui.input_action_button("setPlot", "그래프생성", disabled=LOADING_UI),
                    value="tabPlot"
                ),
                ui.nav(
                    "통계/샘플링 분석", 
                    ui.input_select(
                        "statColumn",
                        "분석할 칼럼 선택 : ",
                        [],
                    ),
                    ui.input_numeric(
                        "statObs", 
                        "샘플 추출 건수 입력 : ", 10, min=1, max=100),
                    ui.input_slider(
                        "statSlider", "샘플 추출 구간 설정: ", min=1, max=2000, value=(100, 500)
                    ),
                    ui.input_action_button("setStat", "데이터분석 및 추출", disabled=LOADING_UI),
                    value="tabStat",
                
                ),
                ui.nav(
                    "특이정보분석", 
                    ui.input_checkbox("chkCatAll", "전체 범주형 칼럼 대상 분석", True),
                    ui.panel_conditional(
                        "!input.chkCatAll",
                        ui.input_selectize(
                            "selCatColumn",
                            "범주형 분석 대상 칼럼을 선택 : ",
                            [],
                            multiple=True,
                        )
                    ),
                    ui.input_numeric(
                        "freqObs", 
                        "분석 빈도수 입력 : ", 10, min=1, max=100
                    ),
                    ui.input_action_button("setFreq", "빈도분석", disabled=LOADING_UI),
                    ui.tags.br(""),
                    ui.input_checkbox("chkNumAll", "전체 수치형 칼럼 대상 분석", True),
                    ui.panel_conditional(
                        "!input.chkNumAll",
                        ui.input_selectize(
                            "selNumColumn",
                            "수치형 분석 대상 칼럼을 선택 : ",
                            [],
                            multiple=True,
                        )
                    ),
                    ui.input_action_button("setStat2", "기초통계분석", disabled=LOADING_UI),
                    value="tabOutlier"),
                ui.nav(
                    "KLT 분석", 
                    ui.input_selectize(
                        "qiColumn",
                        "준식별자 칼럼을 선택 : ",
                        [],
                        multiple=True,
                    ),
                    ui.input_action_button("setQi", "준식별자 익명지표 분석", disabled=LOADING_UI),
                    ui.tags.br(""),
                    ui.input_selectize(
                        "saColumn",
                        "민감정보 칼럼을 선택 : ",
                        [],
                        multiple=True,
                    ),
                    ui.input_action_button("setSa", "민감정보 익명지표 분석", disabled=LOADING_UI),  
                    value="tabKlt"),
                id="inTabset",
            ),
        ),
        ui.panel_main(
            ui.output_ui("dyn_ui"),
        ),
    ),
    title="EDA",
)

# =====================================
# Define Server
# =====================================
def server(input, output, session):
    
    fileName = reactive.Value(None)
    df = reactive.Value(None)
    cntRows = reactive.Value(None)
    colNameList = reactive.Value(None)
    catColNameList = reactive.Value(None)
    numColNameList = reactive.Value(None)
    resultSetFreq = reactive.Value(None)
    setFreqBaseVal = reactive.Value(None)

    @reactive.Effect
    def _():
        typeTab = input.inTabset()
        # print(f"Check input data -> {typeTab}")
        
        if input.file1() is None and fileName() is None :
            jsonData = SHINY_SHAREDATA
            
            fileName2 = jsonData["file"]
            filePath = os.path.join(jsonData["dir"], jsonData["file"])
            
            if not os.path.exists(filePath) :
                return
            
            optSep = jsonData["sep"]
            optEncode = jsonData["encode"]
            
            init(filePath, optSep, optEncode)
            fileName.set(fileName2)
    
    @reactive.Effect
    @reactive.event(input.file1)
    def _():
        
        fileInfos = input.file1()
        fileinfo = fileInfos[0]
        fileName2 = fileinfo['name']
        fileSize = fileinfo['size']
        fileType = fileinfo['type']
        filePath = fileinfo['datapath']
        
        optSep = guessDelimiter(filePath)
        optEncode = guessEncoding(filePath)
        
        init(filePath, optSep, optEncode)
        fileName.set(fileName2)
        
    
    @output
    @render.ui
    def dyn_ui():
        typeTab = input.inTabset()
        if typeTab == "tabPlot":
            return  ui.tags.br(""), ui.output_text(
                        "descPlot1"
                    ),  ui.output_plot(
                            "plot1"
                        ),  ui.output_text(
                                "descPlot2"
                            ),  ui.output_plot("plot2")
        elif typeTab == "tabStat":
            return  ui.tags.br(""), ui.output_text_verbatim(
                        "descStat", placeholder=True
                    ),  ui.output_table("sample")
        elif typeTab == "tabOutlier":
            return  ui.tags.br(""), ui.navset_tab_card(
                        ui.nav(
                            "범주형 데이터 분석",
                            ui.output_text_verbatim(
                                "descFreqCat", placeholder=True
                            ),  
                            ui.input_select(
                                "freqCatColumn2",
                                "지정된 빈도 이하 보유 칼럼 목록 : ",
                                [],
                            ),  
                            ui.output_table(
                                "freqData"
                            ),
                            value="tabCat"
                        ),
                        ui.nav(
                            "수치형 데이터 분석", 
                            ui.output_text_verbatim("descNumStat", placeholder=True),
                            value="tabNum",
                        ),
                        id="mainTabSet",
                    ),
        elif typeTab == "tabKlt":
            return  ui.tags.br(""), ui.output_text_verbatim(
						"descQi", placeholder=True
					),  ui.output_text_verbatim(
							"descSa", placeholder=True
						),	ui.navset_tab_card(
								ui.nav(
									"QI Data",
									ui.output_table("qiAnony"),
                                    value="tabQi",
								),
								ui.nav(
									"QI & SA Data", 
									ui.output_table("saAnony"),
									value="tabSa",
								),
								id="mainTabSet2",
							),

    @output
    @render.text
    @reactive.event(input.setPlot)
    async def descPlot1():
        tmpCntRows = cntRows()
        with ui.Progress(min=1, max=tmpCntRows) as p:
            p.set(message="Calculation in progress", detail=None)
            msg = "데이터 처리 중...."
            msg = msg+"\n완료되면 그래프가 생성됩니다."
            for i in range(1, tmpCntRows):
                p.set(i, message=msg)
        return "빈도 기준 분석 그래프 : "
    
    @output
    @render.plot
    @reactive.event(input.setPlot)
    def plot1():
        tmpDf = df()
        tmpColName = input.plotColumn()
        tmpDf = tmpDf[[tmpColName]]
        if tmpDf[tmpColName].dtype == "object":
            tmpDf[tmpColName] = tmpDf[tmpColName].astype('category')
        
        # plt.barh(tmpColName, tmpDf)
        plt.xticks(rotation=-45)
        return sns.countplot(x=tmpColName, data=tmpDf)
        # return tmpDf[input.plotColumn()].value_counts().plot.bar()

    @output
    @render.text
    @reactive.event(input.setPlot)
    async def descPlot2():
        tmpDf = df()
        tmpColName = input.plotColumn()
        tmpDf = tmpDf[[tmpColName]]
        if tmpDf[tmpColName].dtype == "object":
            tmpDf[tmpColName] = tmpDf[tmpColName].astype('category')

        if tmpDf[tmpColName].dtype != "category":
            return "수치형 구간 나눔 분석 그래프 : "
        else:
            return
    
    @output
    @render.plot
    @reactive.event(input.setPlot)
    def plot2():
        tmpDf = df()
        tmpColName = input.plotColumn()
        tmpDf = tmpDf[[tmpColName]]
        if tmpDf[tmpColName].dtype == "object":
            tmpDf[tmpColName] = tmpDf[tmpColName].astype('category')

        if tmpDf[tmpColName].dtype != "category":
            return sns.histplot(x=tmpColName, data=tmpDf, bins=input.plotObs())
        else:
            return
    
    @output
    @render.text
    @reactive.event(input.setStat)
    def descStat():
        tmpDf = df()
        tmpColName = input.statColumn()
        tmpDf = tmpDf[[tmpColName]]
        tmpDf.columns = ["v1"]
        if tmpDf["v1"].dtype == "object":
            tmpDf["v1"] = tmpDf["v1"].astype('category')

        if tmpDf["v1"].dtype == "category":
            freq = tmpDf["v1"].value_counts()
            freq = freq.sort_values(ascending=True)
            # print(f"freq -> {freq}")
            # naList = freq.loc[freq.values <= 10].index.to_list()
            cnt = 0
            tmpDesc = "최대 5개까지 범주 데이터를 표시(빈도가 낮은 순으로 표시)"
            for index, value in enumerate(freq):
                # print(index, value)
                # print(freq.index[index], value, str(value))
                cnt = cnt + 1
                if cnt < 6:
                    tmpDesc = tmpDesc + "\n" + str(cnt) + ". " + freq.index[index] + " - " + str(value) + "건"
                else:
                    break
            return tmpDesc
        else:
            # check max, min and freq
            vMax = tmpDf["v1"].max()
            vMin = tmpDf["v1"].min()
            # print("vmax -> ", vMax)
            # aaa = tmpDf[tmpDf["v1"] >= vMax]
            # print("aaa -> ", aaa)
            # print("count aaa -> ", len(aaa))
            tmpDesc = "1. 평균 : " + str(tmpDf["v1"].mean())
            tmpDesc = tmpDesc + "\n" + "2. 중앙값 : " + str(tmpDf["v1"].median())
            tmpDesc = tmpDesc + "\n" + "3. 최댓값 : " + str(vMax) + " (총 " + str(len(tmpDf[tmpDf["v1"] >= vMax])) + "건)"
            tmpDesc = tmpDesc + "\n" + "4. 최솟값 : " + str(vMin) + " (총 " + str(len(tmpDf[tmpDf["v1"] <= vMin])) + "건)"
            tmpDesc = tmpDesc + "\n" + "5. 1사분위 : " + str(tmpDf["v1"].quantile(q=0.25))
            tmpDesc = tmpDesc + "\n" + "6. 3사분위 : " + str(tmpDf["v1"].quantile(q=0.75))
            return tmpDesc
    
    @output
    @render.table
    @reactive.event(input.setStat)
    def sample():
        # define dataframe
        tmpDf = df()
        tmpColName = input.statColumn()
        tmpDf = tmpDf[[tmpColName]]
        # tmpDf.columns = ["v1"]

        # check index range for slice
        sliderRng = list(input.statSlider()) 
        idxStart = sliderRng[0]-1
        idxEnd = sliderRng[1]
        # print(f"start -> {idxStart}, end -> {idxEnd}")

        # check show data and slice
        tmpDf["NO"] = tmpDf.index+1 
        tmpDf = tmpDf.iloc[idxStart:idxEnd,:]

        # sampling
        # print(input.statObs())
        # tmpDf = tmpDf.sample(input.statObs(), replace=False, axis=0)
        tmpDf = tmpDf.sample(input.statObs(), replace=True)
        tmpDf = tmpDf[["NO", tmpColName]].sort_index(ascending=True)
        # print(f"tmpDF -> {tmpDf.head()}")
        return tmpDf
    
    @reactive.Effect
    @reactive.event(input.setFreq)
    def _():
        # set dataframe
        tmpDf = df()
        if input.chkCatAll():
            tmpColNameList = catColNameList()
        else:
            tmpColNameList = list(input.selCatColumn())

        # check freq base value, setFreqBaseVal
        setFreqBaseVal.set(input.freqObs())
        tmpBaseVal = setFreqBaseVal()
        
        # check freq and update select list
        aColNameList = []
        aColNameListDesc = []
        for i in range(0, len(tmpColNameList)):
            freq = tmpDf[tmpColNameList[i]].value_counts()
            freq = freq.sort_values(ascending=True)
            tmpList = freq.loc[freq.values <= tmpBaseVal].index.to_list()
            if len(tmpList) >= 1:
                aColNameList.append(tmpColNameList[i])
                aColNameListDesc.append(tmpColNameList[i] + " | " + str(len(tmpList)) + "건")
        
        tmpDesc = "범주형 빈도 분석 결과"
        if len(aColNameList) >= 1:
            tmpDesc = tmpDesc + "\n기준 빈도 이하 보유 칼럼 수 : " + str(len(aColNameList))
            tmpDesc = tmpDesc + "\n아래 칼럼 목록에서 선택하면 해당 정보를 확인할 수 있습니다."
            ui.update_select(
                "freqCatColumn2",
                choices=aColNameListDesc,
                # selected=aColNameListDesc[0]
            )
        else:
            tmpDesc = tmpDesc + "기준 빈도 이하 보유 칼럼이 존재하지 않습니다."
            ui.update_select(
                "freqCatColumn2",
                choices=[],
                # selected=aColNameListDesc[0]
            )
        resultSetFreq.set(tmpDesc)

        ui.update_navs("mainTabSet", selected="tabCat")

    @output
    @render.text
    @reactive.event(input.setFreq)
    def descFreqCat():
        tmpDesc = resultSetFreq()
        return tmpDesc
    
    @output
    @render.table
    @reactive.event(input.freqCatColumn2)
    def freqData():
        # check column name
        tmpValue = input.freqCatColumn2()
        tmpValue = tmpValue.split(" | ")
        
        # check freq base value
        tmpBaseVal = setFreqBaseVal()
        # print(f"freq base value -> {tmpBaseVal}")

        # set dataframe
        tmpDf = df()
        freq = tmpDf[tmpValue[0]].value_counts()
        freq = freq.sort_values(ascending=True)
        freq = freq.loc[freq.values <= tmpBaseVal]
        freq = freq.reset_index()
        freq["NO"] = freq.index+1
        freq.columns = [tmpValue[0],"빈도","NO"]
        freq = freq[["NO",tmpValue[0],"빈도"]]
        
        return freq

    @output
    @render.text
    @reactive.event(input.setStat2)
    def descNumStat():
        # check Data and numeric columns
        tmpDf = df()
        if input.chkNumAll():
            tmpColNameList = numColNameList()
        else:
            tmpColNameList = list(input.selNumColumn())

        retDesc = None
        for i in range(0, len(tmpColNameList)):
            # define column name
            tmpColName = tmpColNameList[i]

            # check max, min and freq
            vMax = tmpDf[tmpColName].max()
            vMin = tmpDf[tmpColName].min()
            tmpDesc = str(i+1) + ". 칼럼 " + tmpColName + " 기초통계데이터"
            tmpDesc = tmpDesc + "\n    - 평균 : " + str(tmpDf[tmpColName].mean())
            tmpDesc = tmpDesc + "\n    - 중앙값 : " + str(tmpDf[tmpColName].median())
            tmpDesc = tmpDesc + "\n    - 최댓값 : " + str(vMax) + " (총 " + str(len(tmpDf[tmpDf[tmpColName] >= vMax])) + "건)"
            tmpDesc = tmpDesc + "\n    - 최솟값 : " + str(vMin) + " (총 " + str(len(tmpDf[tmpDf[tmpColName] <= vMin])) + "건)"
            tmpDesc = tmpDesc + "\n    - 1사분위 : " + str(tmpDf[tmpColName].quantile(q=0.25))
            tmpDesc = tmpDesc + "\n    - 3사분위 : " + str(tmpDf[tmpColName].quantile(q=0.75))
            if retDesc is None:
                retDesc = tmpDesc
            else:
                retDesc = retDesc + "\n\n" + tmpDesc
        
        return retDesc
    
    @reactive.Effect
    @reactive.event(input.setStat2)
    def _():
        ui.update_navs("mainTabSet", selected="tabNum")

    @output
    @render.text
    @reactive.event(input.setQi)
    def descQi():
        # check quasi-id columns
        tmpQiColumn = input.qiColumn()
        if not tmpQiColumn:
            return
        
        # check Data-Set
        tmpDf = df()
        
        # define return info
        kAnony = anony.k_anonymity(tmpDf, tmpQiColumn)
        retDesc = "1. 준식별자 대상 익명성 지표 분석"
        retDesc = retDesc + "\n  - 열정보 : " +  json.dumps(tmpQiColumn, ensure_ascii=False)
        retDesc = retDesc + "\n  - k_anonymity : " + str(kAnony)
        return retDesc

    @output
    @render.table
    @reactive.event(input.setQi)
    def qiAnony():
        # check column name
        tmpQiColumn = input.qiColumn()
        if not tmpQiColumn:
            return
        
        tmpQiColumn = list(tmpQiColumn)
        # set dataframe
        tmpDf = df()
        freq = tmpDf[tmpQiColumn].value_counts(ascending=True)
        # freq = freq.sort_values(ascending=True)
        freq = freq.reset_index()
        freq["NO"] = freq.index+1
        tmpQiColumn.extend(["빈도","NO"])
        # tmpQiColumn.append("NO")
        freq.columns = tmpQiColumn
        tmpQiColumn = tmpQiColumn[-1:]+tmpQiColumn[:-1]
        freq = freq[tmpQiColumn]
        return freq

    @reactive.Effect
    @reactive.event(input.setQi)
    def _():
        ui.update_navs("mainTabSet2", selected="tabQi")

    # @output
    # @render.text
    @reactive.Effect
    @reactive.event(input.setSa)
    # async def descSa():
    async def _():
        tmpCntRows = cntRows()
        print(f"tmpCntRows -> {tmpCntRows}")
        
        with ui.Progress(min=1, max=tmpCntRows) as p:
            p.set(message="Calculation in progress", detail=None)
            msg = "데이터 처리 중...."
            msg = msg+"\n완료되면 QI & SA Data 탭에 데이터가 표시됩니다."
            for i in range(1, tmpCntRows):
                p.set(i, message=msg)
        # return "데이터 처리 진행 중..."
    
    @output
    @render.text
    @reactive.event(input.setSa)
    def descSa():
        # check quasi-id columns
        tmpQiColumn = input.qiColumn()
        tmpSaColumn = input.saColumn()
        if not tmpQiColumn or not tmpSaColumn:
            return
        # check Data-Set
        tmpDf = df()
        
        # define return info
        lDiv = anony.l_diversity(tmpDf, tmpQiColumn, tmpSaColumn)
        entropyL = anony.entropy_l_diversity(tmpDf, tmpQiColumn, tmpSaColumn)
        tClos = anony.t_closeness(tmpDf, tmpQiColumn, tmpSaColumn)
        retDesc = "2. 민감정보 대상 익명성 지표 분석"
        retDesc = retDesc + "\n  - 열정보 : " +  json.dumps(tmpSaColumn, ensure_ascii=False)
        retDesc = retDesc + "\n  - l_diversity : " + str(lDiv)
        retDesc = retDesc + "\n  - entropy_l_diversity : " + str(entropyL)
        retDesc = retDesc + "\n  - t_closeness : " + str(tClos)

        return retDesc
    
    @output
    @render.table
    @reactive.event(input.setSa)
    def saAnony():
        # check column name
        tmpQiColumn = input.qiColumn()
        tmpSaColumn = input.saColumn()
        if not tmpQiColumn or not tmpSaColumn:
            return
        tmpQiColumn = list(tmpQiColumn)
        tmpSaColumn = list(tmpSaColumn)
        
        # check Data-Set
        tmpDf = df()
        
        # generate table on k-anony
        freq = tmpDf[tmpQiColumn].value_counts(ascending=True)
        # freq = freq.sort_values(ascending=True)
        freq = freq.reset_index()
        freq["ec_No"] = freq.index+1
        tmpQiColumn.extend(["빈도","ec_No"])
        # tmpQiColumn.append("NO")
        freq.columns = tmpQiColumn
        # tmpQiColumn = tmpQiColumn[-1:]+tmpQiColumn[:-1]
        # freq = freq[tmpQiColumn]
        print(f"k column -> {tmpQiColumn}, lt column -> {tmpSaColumn}")

        # generate table on k-anony and lt-column
        tmpDf = tmpDf[tmpQiColumn[:-2]+tmpSaColumn]
        newDf = pandas.merge(tmpDf, freq, on=tmpQiColumn[:-2])
        newDf.sort_values("ec_No", inplace=True)
        return newDf
    
    @reactive.Effect
    @reactive.event(input.setSa)
    def _():
        ui.update_navs("mainTabSet2", selected="tabSa")


    # ============================================
    # Data init~ on file change
    # ============================================
    def init(filePath, optSep, optEncode):
        # read data-set
        tmpDf = pandas.read_csv(filePath, sep=optSep , encoding=optEncode)
        df.set(tmpDf)
        cntRows.set(len(tmpDf))
        tmpColNameList = tmpDf.columns.to_list()
        colNameList.set(tmpColNameList)
        tmpCatColNameList = []
        tmpNumColNameList = []
        for i in tmpColNameList:
            if tmpDf[i].dtype == "object" or tmpDf[i].dtype == "category":
                tmpCatColNameList.append(i)
            else:
                tmpNumColNameList.append(i)
        
        catColNameList.set(tmpCatColNameList)
        numColNameList.set(tmpNumColNameList)
        # print(f"catcol list -> {catColNameList()}, numcol list -> {numColNameList()}")

        # plot area
        ui.update_select(
            "plotColumn",
            choices=tmpColNameList,
            selected=None
        )

        # if df[colNameList[0]].dtype == "object" or df[colNameList[0]].dtype == "category":
        #     print("plot1")
            
        # stat area
        ui.update_select(
            "statColumn",
            choices=tmpColNameList,
            selected=None
        )
        ui.update_slider(
            "statSlider", value=(int(len(tmpDf)*0.25), int(len(tmpDf)*0.75)), min=1, max=len(tmpDf)
        )

        # freq area
        ui.update_selectize(
            "selCatColumn",
            choices=tmpCatColNameList,
            selected=None,
            # server=True,
        )
        ui.update_select(
            "selNumColumn",
            choices=tmpNumColNameList,
            selected=None,
            # server=True,
        )

        # klt area
        ui.update_selectize(
            "qiColumn",
            choices=tmpColNameList,
            selected=None,
            # server=True,
        )
        ui.update_selectize(
            "saColumn",
            choices=tmpColNameList,
            selected=None,
            # server=True,
        )

app = App(app_ui, server)