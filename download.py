# from selenium import webdriver
# from seleniumwire import webdriver

from utils import import_yf_fundamentals2
import re
import csv
import time
import requests
import os, os.path
import access
# from browsermobproxy import Server
import time
import yfinance as yf

from io import StringIO
import pandas as pd
import numpy as np
# https://stackoverflow.com/questions/54456944/how-to-know-if-my-custom-http-headers-are-being-passed# https://financialmodelingprep.com/api/financials/income-statement/AAPL
# https://financialmodelingprep.com/developer/docs/
# https://financialmodelingprep.com/api/v3/company/stock/list

#--- Sedium Configuration ---#
# chromeOptions = webdriver.ChromeOptions()
# prefs = {"download.default_directory" :"C:/Users/yften/OneDrive/Investing/tempFiles"}
# chromeOptions.add_experimental_option("prefs",prefs)
dir = 'C:/Users/yften/OneDrive/Investing/tempFiles/'
#--- Morningstar ---#
reportTypes = ['is', 'bs', 'cf']
# reportTypes = []
tickers = ['7203.T','F'] #
nonUSTickers = []
# periods = ['12']
periods = ['12', '3']

class Download3():
    def __init__(self,tickers = tickers, nonUSTickers = nonUSTickers,reportTypes = ['is', 'bs','cf','kr']):
        self.api_key = 'ff213862e0f521134afa75c442c72d82'
        self.dir = 'C:/Users/yften/OneDrive/Investing/tempFiles/'
        self.tickers = tickers
        self.nonUSTickers = nonUSTickers
        self.reportTypes = reportTypes
        self.reportType_dict = {'is':'income-statement','bs':'balance-sheet-statement','cf':'cash-flow-statement'}
        # self.key_order_dict = {'is':['date','Revenue','Revenue Growth','Cost of Revenue','Gross Profit','R&D Expenses','SG&A Expense','Operating Expenses','Operating Income','Interest Expense','Earnings before Tax','Income Tax Expense','Net Income - Non-Controlling int','Net Income - Discontinued ops','Net Income','Preferred Dividends','Net Income Com','EPS','EPS Diluted','Weighted Average Shs Out','Weighted Average Shs Out (Dil)','Dividend per Share','Gross Margin','EBITDA Margin','EBIT Margin','Profit Margin','Free Cash Flow margin','EBITDA','EBIT','Consolidated Income','Earnings Before Tax Margin','Net Profit Margin'],
        #                        'bs':['date','Cash and cash equivalents','Short-term investments','Cash and short-term investments','Receivables','Inventories','Total current assets','Property','Goodwill and Intangible Assets','Long-term investments','Tax assets','Total non-current assets','Total assets','Payables','Short-term debt','Total current liabilities','Long-term debt','Total debt','Deferred revenue','Tax Liabilities','Deposit Liabilities','Total non-current liabilities','Total liabilities','Other comprehensive income','Retained earnings (deficit)','Total shareholders equity','Investments','Net Debt','Other Assets','Other Liabilities'],
        #                        'cf':['date','Depreciation & Amortization','Stock-based compensation','Operating Cash Flow','Capital Expenditure','Acquisitions and disposals','Investment purchases and sales','Investing Cash flow','Issuance (repayment) of debt','Issuance (buybacks) of shares','Dividend payments','Financing Cash Flow','Effect of forex changes on cash','Net cash flow / Change in cash','Free Cash Flow','Net Cash/Marketcap']}
        self.key_order_dict = {'is':["date","revenue","costOfRevenue","grossProfit","researchAndDevelopmentExpenses","generalAndAdministrativeExpenses","sellingAndMarketingExpenses","sellingGeneralAndAdministrativeExpenses","otherExpenses","operatingExpenses","costAndExpenses","operatingIncome","interestIncome","interestExpense","depreciationAndAmortization","incomeBeforeTax","incomeTaxExpense","totalOtherIncomeExpensesNet","netIncome","eps","epsdiluted","weightedAverageShsOut","weightedAverageShsOutDil","ebitdaratio","incomeBeforeTaxRatio","grossProfitRatio","netIncomeRatio","operatingIncomeRatio","ebitda"],
                               'bs':["date","cashAndCashEquivalents","shortTermInvestments","cashAndShortTermInvestments","netReceivables","inventory","otherCurrentAssets","totalCurrentAssets","propertyPlantEquipmentNet","goodwill","intangibleAssets","goodwillAndIntangibleAssets","longTermInvestments","taxAssets","otherNonCurrentAssets","totalNonCurrentAssets","otherAssets","totalAssets","accountPayables","shortTermDebt","deferredRevenue","otherCurrentLiabilities","taxPayables","totalCurrentLiabilities","longTermDebt","deferredRevenueNonCurrent","deferredTaxLiabilitiesNonCurrent","otherNonCurrentLiabilities","totalNonCurrentLiabilities","otherLiabilities","capitalLeaseObligations","totalDebt","netDebt","totalLiabilities","preferredStock","commonStock","retainedEarnings","accumulatedOtherComprehensiveIncomeLoss","othertotalStockholdersEquity","totalStockholdersEquity","totalLiabilitiesAndStockholdersEquity","minorityInterest","totalEquity","totalLiabilitiesAndTotalEquity","totalInvestments"],
                               'cf':["date","netIncome","depreciationAndAmortization","deferredIncomeTax","stockBasedCompensation","changeInWorkingCapital","accountsReceivables","inventory","accountsPayables","otherWorkingCapital","otherNonCashItems","netCashProvidedByOperatingActivities","investmentsInPropertyPlantAndEquipment","acquisitionsNet","purchasesOfInvestments","salesMaturitiesOfInvestments","otherInvestingActivites","netCashUsedForInvestingActivites","debtRepayment","commonStockIssued","commonStockRepurchased","dividendsPaid","otherFinancingActivites","netCashUsedProvidedByFinancingActivities","effectOfForexChangesOnCash","netChangeInCash","freeCashFlow","cashAtEndOfPeriod","cashAtBeginningOfPeriod","operatingCashFlow","capitalExpenditure"]}

        self.s = requests.session()
        self.headers = {'Referer': 'http://financials.morningstar.com/'}
        # self.main()
    def main(self):
        self.remove_tmp()
        self.download_yf()
        self.download_modprep()
    def download_modprep(self):
        ticker_all = self.tickers + self.nonUSTickers
        for ticker in self.tickers:
        # if ticker.rfind('.') == -1:
            tmp_ticker = ticker
            for reportType in self.reportTypes:
                print(ticker, reportType)
                if reportType == 'kr':
                    self.download_KR(ticker)
                    # url = 'https://financials.morningstar.com/finan/ajax/exportKR2CSV.html?t=' + ticker
                    # req = self.s.get(url, headers=self.headers)
                    # data = StringIO(req.text)
                    # data = data.read().splitlines()
                    # reader = csv.reader(data, delimiter=',')
                    # path = self.dir + ticker + '_kr.csv'
                    # with open(path, 'w+', newline='', encoding='utf-8') as f:
                    #     writer = csv.writer(f)
                    #     writer.writerows(reader)
                else:
                    for period in periods:
                        key_order = self.key_order_dict[reportType]
                        # key_map = self.key_map_dict[reportType]
                        if period == '3':
                            print('https://financialmodelingprep.com/api/v3/' + self.reportType_dict[reportType] +'/'+ ticker + '?period=quarter' + '&apikey=' + self.api_key)
                            data = self.s.get('https://financialmodelingprep.com/api/v3/' + self.reportType_dict[reportType] +'/'+ ticker + '?period=quarter' + '&apikey=' + self.api_key).json()

                            # print('https://financialmodelingprep.com/api/v3/financials/' + self.reportType_dict[reportType] +'/'+ ticker + '?period=quarter' + '&apikey=' + self.api_key)
                            # data = self.s.get('https://financialmodelingprep.com/api/v3/financials/' + self.reportType_dict[reportType] +'/'+ ticker + '?period=quarter' + '&apikey=' + self.api_key).json()
                        else:
                            print('https://financialmodelingprep.com/api/v3/' + self.reportType_dict[reportType] +'/'+ ticker + '?apikey=' + self.api_key)
                            data = self.s.get('https://financialmodelingprep.com/api/v3/' + self.reportType_dict[reportType] +'/'+ ticker + '?apikey=' + self.api_key).json()

                            # print('https://financialmodelingprep.com/api/v3/financials/' + self.reportType_dict[reportType] +'/'+ ticker + '?apikey=' + self.api_key)
                            # data = self.s.get('https://financialmodelingprep.com/api/v3/financials/' + self.reportType_dict[reportType] +'/'+ ticker + '?apikey=' + self.api_key).json()

                        # print(data)
                        # print(data.keys())
                        # print(data['financials'])
                        try:
                            data = pd.DataFrame.from_dict(data)
                            # data = pd.DataFrame.from_dict(data['financials'])
                            data = data.sort_values(by=['date'])
                            
                            ### Get the latest 10
                            # print(data.date)
                            data = data[-10:]
                            data = data.T
                            data = data.reindex(key_order)
                            if reportType == 'is':
                                data = data.rename(index={"revenue":"Revenue","costOfRevenue":"Cost of Revenue","grossProfit":"Gross Profit","researchAndDevelopmentExpenses":"R&D Expenses","generalAndAdministrativeExpenses":"generalAndAdministrativeExpenses","sellingAndMarketingExpenses":"SG&A Expense","sellingGeneralAndAdministrativeExpenses":"sellingGeneralAndAdministrativeExpenses","otherExpenses":"otherExpenses","operatingExpenses":"Operating Expenses","costAndExpenses":"costAndExpenses","operatingIncome":"Operating Income","interestIncome":"interestIncome","interestExpense":"Interest Expense","depreciationAndAmortization":"depreciationAndAmortization","incomeBeforeTax":"Earnings before Tax","incomeTaxExpense":"Income Tax Expense","totalOtherIncomeExpensesNet":"totalOtherIncomeExpensesNet","netIncome":"Net Income","eps":"EPS","epsdiluted":"EPS Diluted","weightedAverageShsOut":"Weighted Average Shs Out","weightedAverageShsOutDil":"Weighted Average Shs Out (Dil)","ebitdaratio":"EBITDA Margin","incomeBeforeTaxRatio":"EBIT Margin","grossProfitRatio":"Profit Margin","netIncomeRatio":"netIncomeRatio","operatingIncomeRatio":"operatingIncomeRatio","ebitda":"EBITDA"})
                            if reportType == 'bs':
                                data = data.rename(index={"cashAndCashEquivalents":"Cash and cash equivalents","shortTermInvestments":"Short-term investments","cashAndShortTermInvestments":"Cash and short-term investments","netReceivables":"Receivables","inventory":"Inventories","totalCurrentAssets":"Total current assets","propertyPlantEquipmentNet":"Property","goodwill":"Goodwill and Intangible Assets","longTermInvestments":"Long-term investments","taxAssets":"Tax assets","totalNonCurrentAssets":"Total non-current assets","totalAssets":"Total assets","accountPayables":"Payables","shortTermDebt":"Short-term debt","taxPayables":"Tax Liabilities","totalCurrentLiabilities":"Total current liabilities","longTermDebt":"Long-term debt","totalDebt":"Total debt","netDebt":"Net Debt","retainedEarnings":"Retained earnings (deficit)","totalStockholdersEquity":"Total shareholders equity","totalInvestments":"Investments","totalLiabilities":"Total liabilities"})
                            if reportType == 'cf':
                                data = data.rename(index={"depreciationAndAmortization":"Depreciation & Amortization","stockBasedCompensation":"Stock-based compensation","netCashProvidedByOperatingActivities":"Operating Cash Flow","acquisitionsNet":"Acquisitions and disposals","netCashUsedForInvestingActivites":"Investing Cash flow","debtRepayment":"Issuance (repayment) of debt","commonStockIssued":"Issuance (buybacks) of shares","dividendsPaid":"Dividend payments","netCashUsedProvidedByFinancingActivities":"Financing Cash Flow","effectOfForexChangesOnCash":"Effect of forex changes on cash","netChangeInCash":"Net cash flow / Change in cash","freeCashFlow":"Free Cash Flow","capitalExpenditure":"Capital Expenditure"})

                            path = self.dir + ticker + '_' + reportType + '_' + period + '.csv'
                            data.to_csv(path)
                        except:
                            print('error')
                            continue


                    # print(data,data.index)
            # if ticker.rfind('.') != -1:
            #     try:
            #         import_yf_fundamentals2(ticker,tmp_ticker)
            #     except:
            #         print("error import_yf_fundamentals2 for ...",ticker)
            #         pass
    def remove_tmp(self):
        filelist = [f for f in os.listdir(self.dir) if f.endswith(".csv")]
        # filelist = [f for f in os.listdir(self.dir)]
        for f in filelist:
            os.remove(os.path.join(self.dir, f))
    def download_yf(self):
        for index, ticker in enumerate(self.tickers):
            # download_yahooFinance(ticker, ticker)
            print('downlaoding...',ticker)
            df_price=yf.download(ticker)
            df_price.to_csv(dir+ticker+'.csv')
        for index, ticker in enumerate(self.nonUSTickers):
            if ticker.rfind('.') == -1:
                tmp_ticker = ticker
            if ticker.rfind('.') != -1:
                # download_yahooFinance(ticker, tmp_ticker)
                print('downlaoding...',ticker)
                df_price=yf.download(ticker)
                df_price.to_csv(dir+tmp_ticker+'.csv')
    def download_KR(self,ticker):
        url = 'https://financials.morningstar.com/finan/ajax/exportKR2CSV.html?t=' + ticker
        req = self.s.get(url, headers=self.headers)
        data = StringIO(req.text)
        data = data.read().splitlines()
        reader = csv.reader(data, delimiter=',')
        path = self.dir + ticker + '_kr.csv'
        with open(path, 'w+', newline='', encoding='utf-8') as f:
            writer = csv.writer(f)
            writer.writerows(reader)
        
class download_yahooFinance(): # Not usable anymore?
    def __init__(self,symbol,output_name,dir=dir):
        self.dir = dir
        start_date = 0
        end_date = self.get_now_epoch()
        print(symbol)
        cookie, crumb = self.get_cookie_crumb(symbol)
        self.get_data(symbol, output_name, start_date, end_date, cookie, crumb)

    def split_crumb_store(self, v):
        ### Yahoo Finance #########################################
        return v.split(':')[2].strip('"')

    def find_crumb_store(self, lines):
        for l in lines:
            if re.findall(r'CrumbStore', l):
                return l
        print("Did not find CrumbStore")

    def get_cookie_value(self, r):
        return {'B': r.cookies['B']}

    def get_page_data(self, symbol):
        url = "https://finance.yahoo.com/quote/%s/?p=%s" % (symbol, symbol)
        r = requests.get(url)
        cookie = self.get_cookie_value(r)
        lines = r.content.decode('unicode-escape').strip().replace('}', '\n')
        return cookie, lines.split('\n')

    def get_cookie_crumb(self, symbol):
        cookie, lines = self.get_page_data(symbol)
        crumb = self.split_crumb_store(self.find_crumb_store(lines))
        return cookie, crumb

    def get_data(self, symbol, output_name, start_date, end_date, cookie, crumb):
        filename = '%s%s.csv' % (self.dir,output_name)
        url = "https://query1.finance.yahoo.com/v7/finance/download/%s?period1=%s&period2=%s&interval=1d&events=history&crumb=%s" % (
            symbol, start_date, end_date, crumb)
        print(url)
        response = requests.get(url, cookies=cookie)
        with open(filename, 'wb') as handle:
            for block in response.iter_content(1024):
                handle.write(block)
    def get_now_epoch(self):
        # @see https://www.linuxquestions.org/questions/programming-9/python-datetime-to-epoch-4175520007/#post5244109
        return int(time.time())

if __name__ == "__main__":
    Download3().main()
    # Download3().download_KR('MITSF')
    # download_yahooFinance('^VIX','^VIX')
    # import_yf_fundamentals2('9843.T')
# TOdo move all aboce under __init__
 
'''
https://financials.morningstar.com/ajax/ReportProcess4CSV.html?t=CAH&reportType=cf&period=12&dataType=A&order=asc&columnYear=10&number=3
https://financials.morningstar.com/ajax/exportKR2CSV.html?t=
https://financials.morningstar.com/ajax/ReportProcess4CSV.html?&t=CAH&version=SAL&cur=&reportType=is&period=12&dataType=A&order=asc&columnYear=10&number=3
https://financials.morningstar.com/ajax/ReportProcess4CSV.html?&t=0P000000GY&region=usa&culture=en-US&version=SAL&cur=&reportType=is&period=12&dataType=A&order=asc&columnYear=10&curYearPart=1st5year&rounding=3&view=raw&r=983300&denominatorView=raw&number=3
'''


