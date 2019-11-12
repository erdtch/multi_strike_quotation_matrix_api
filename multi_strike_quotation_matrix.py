from flask import Flask, redirect, url_for, request
import math
import pandas as pd
import datetime as dt
import bs4 as bs
import re
from datetime import datetime, timedelta
import numpy as np
from scipy.stats import norm
from math import sqrt, exp
import json

quotes = ['ADVANC', 'AOT', 'BANPU', 'BBL', 'BDMS', 'BEM', 'BGRIM', 'BH', 'BJC', 'BPP', 'BTS', 'CBG', 'CPALL', 'CPF',
          'CPN', 'DELTA', 'DTAC', 'EA', 'EGCO', 'GLOBAL', 'GPSC' ,'PTT', 'GULF', 'KTC', 'SCB', 'INTUCH', 'IVL', 'LH', 'TCAP',
          'TISCO', 'TMB', 'TRUE', 'SCC', 'TOA', 'TOP', 'TU', 'MINT']

class DataSimulatorCSV:
    def __init__(self, quotes=[], start_date='2019-03-01', end_date='2019-10-01', rolling_range=60):
        self.start_index = rolling_range + 1
        self.rolling_range = rolling_range
        
        start_date = (datetime.strptime(start_date, '%Y-%m-%d') - timedelta(days=self.start_index+rolling_range/2)).date().strftime('%Y-%m-%d')
        self.quotes = quotes 
        
        self.historical_data = self.download_historical_data(start_date, end_date)
        self.data , self.ret, self.ret_sd = self.init_dataset() 

        self.index = self.start_index
    
    def edit_quotes(self, quotes, quotes_type=1):
        '''
        quotes_type : 0 = realtime
        quotes_type : 1 = historical 
        '''
        new_quotes = [] 
        for sym in quotes :
            n_sym = sym 
            if quotes_type == 0 : 
                if '.BK' in sym : 
                    n_sym = sym.replace('.BK', '')
            elif quotes_type == 1 : 
                if not ('.BK' in sym) : 
                    # n_sym = sym + '.BK' # yahoo symbol
                    n_sym = sym.replace('.BK', '') # csv symbol
            new_quotes.append(n_sym)
        return new_quotes
                 
    def download_historical_data(self, start_date, end_date):
        quotes = self.edit_quotes(self.quotes, 1)
        
        data = pd.DataFrame() 
        for quote in quotes :
            if quote == 'COM7':
                quote = 'COM_SEVEN'
            csv_file = "SET100_Dataset/"  + quote + '.csv'
            try : 
                df = pd.read_csv(csv_file)

                ticker = str(df.Ticker[0])
                if ticker == 'True' : 
                    ticker = 'TRUE'
                df[ticker] = df.Close
                df = df[['Date/Time', ticker]]
                if data.empty :
                    data = df.copy()
                else : 
                    data = data.merge(df, how='outer', on='Date/Time')
            except Exception as e : 
                print('There is no data of ', quote)
                print(e)

        if not data.empty :
            data['Date/Time'] = pd.to_datetime(data['Date/Time'], format='%m/%d/%Y')
            data = data.rename(columns={'Date/Time': 'Date'})
            data = data.sort_values('Date')
            data = data[(data.Date >= start_date) & (data.Date <= end_date)]

            data = data.fillna(method='bfill', limit=2) 
            # data = data.dropna()
            data = data.set_index('Date')
        return data
    
    def init_dataset(self):
        if not self.historical_data.empty : 
            data  = self.historical_data
            ret = data.pct_change() * 100
            ret_std = ret.rolling(self.rolling_range).std()
            return data, ret, ret_std 
        return pd.DataFrame(), pd.DataFrame(), pd.DataFrame() 

    def next_data(self):
        data = self.data.iloc[self.index-self.rolling_range+2:self.index]
        ret = self.ret.iloc[self.index-self.rolling_range+2:self.index]
        ret_sd = self.ret_sd.iloc[self.index-self.rolling_range+2:self.index]
        self.index += 1 
        return data, ret, ret_sd
    
    def reset(self):
        self.index = self.start_index

class PutOption:
    def __init__(self, ID, stock, param_list, volatility):
        #param_list = [underlyingPrice, strikePrice, interestRate ,annualDividends, daysToExpiration]
        self.ID = ID
        self.stock = stock
        self.initParam_list = param_list
        self.initVolatility = volatility
        self.init_spot = param_list[0]
        #init option
        self.putPrice , self.putDelta , self.putTheta, self.putRho, self.vega, self.gamma = (0,)*6     
        
        self.spot = 0
        self.strike = 0
        self.maturity = 0
        self.interest = 0
        self.volatility = 0
        self.dividend = 0
        self.d1 = 0
        self.d2 = 0
        
        self.put_option_update(param_list, volatility) 
        self.init_putPrice = self.putPrice
        
#         print('initial option values')
#         print('delta : ', self.putDelta)
#         print('theta : ', self.putTheta)
#         print('gamma : ', self.gamma)
#         print('vega : ', self.vega)
#         print('rho : ', self.putRho)
        
    def put_option_update(self, param_list, volatility):
        
        self.spot = param_list[0]
        self.strike = param_list[1]
        self.maturity = param_list[4] / 365
        self.interest = param_list[2] / 100
        self.volatility = volatility / 100
        self.dividend = param_list[3] / 100
        
        self.d1 = (self.volatility * sqrt(self.maturity)) ** (-1) * (
        np.log(self.spot / self.strike) + (self.interest - self.dividend + self.volatility ** 2 / 2) * self.maturity)
        self.d2 = self.d1 - self.volatility * sqrt(self.maturity)

        self.putPrice = self.price()
        self.putDelta = self.delta()
        self.putTheta = self.theta()
        self.putRho =  self.rho()
        self.vega = self.calvega()
        self.gamma =  self.calgamma()
             
    def price(self):
        return norm.cdf(-self.d2) * self.strike * exp(-self.interest * self.maturity) - norm.cdf(
            -self.d1) * self.spot * exp(-self.dividend * self.maturity)
 
    def delta(self):
        return (norm.cdf(self.d1) - 1) * exp(-self.dividend * self.maturity)
 
    def calgamma(self):
        return exp(-self.dividend * self.maturity) * norm.pdf(self.d1) / (
        self.spot * self.volatility * sqrt(self.maturity))
 
    def calvega(self):
        return self.spot * norm.pdf(self.d1) * sqrt(self.maturity) * exp(-self.dividend * self.maturity)
 
    def theta(self):
        return -exp(-self.dividend * self.maturity) * (self.spot * norm.pdf(self.d1) * self.volatility) / (
        2 * sqrt(self.maturity)) + self.interest * self.strike * exp(
            -self.interest * sqrt(self.maturity)) * norm.cdf(-self.d2) - self.dividend * self.spot * exp(
            -self.dividend * self.maturity) * norm.cdf(-self.d1)
 
    def rho(self):
        return -self.strike * self.maturity * exp(-self.interest * self.maturity) * norm.cdf(-self.d2)

class ELN:
    def __init__(self, ID=0, stock='symbol', param_list=[], volatility=15, volFactor=2, value=10000000 , spreadThreshold = 2, create_new_note=True, create_date=None,exp_date=None):
        #param_list = [underlyingPrice, strikePrice, interestRate ,annualDividends, daysToExpiration, costOfFund]
        self.ID                  = ID
        self.stock               = stock
        self.initOptionParamList = param_list
        self.initVolatility      = volatility 
        self.volFactor           = volFactor
        self.value               = value
        self.spreadThreshold     = spreadThreshold
        self.currentHedgingPrice = param_list[0]
        self.putOption           = PutOption(ID, stock, self.initOptionParamList, self.initVolatility * self.volFactor * math.sqrt(252))
        self.optionPosition      = int((value * (param_list[2]/100)) / self.putOption.putPrice)
        
        self.elnPCT = (1 / (1 + (param_list[5] * param_list[4] / 100 / 365))) - (self.putOption.putPrice / param_list[1]) 
        self.yieldPA = ((1 / self.elnPCT) - 1) * (365 / param_list[4]) * 100

app = Flask(__name__)
@app.route('/quotation_matrix/<int:tenor>', methods=['POST'])
def multi_strike_quotation_matrix(tenor):
    print(tenor)

    elnList = []
    discount_strike = [0.97, 0.95, 0.93]
    for stock in quotes:
        daysToExpiration = tenor
        underlyingPrice = prices.iloc[-1][str(stock)]
        interestRate = 1.5
        annualDividends = 1
        volatility = ret_std.iloc[-1][str(stock)] 
        volFactor = 1.5
        value = 10000000
        rf = 1.5
        costOfFund = 1.5
        tmp97, tmp95, tmp93 = 0, 0, 0

    #         print('init spot', underlyingPrice, 'init volatility : ',  volatility * volFactor * math.sqrt(252))

        for strikeFactor in discount_strike:
            strikePrice = underlyingPrice * strikeFactor
            param_list = [underlyingPrice,
                            strikePrice, 
                            interestRate,
                            annualDividends,
                            daysToExpiration,
                            rf,
                            costOfFund]

            ELN_STOCK = ELN(ID = 1, 
                            stock = stock,  
                            param_list = param_list, 
                            volatility = volatility, 
                            volFactor = volFactor,
                            value = value)

            if strikeFactor == 0.97:
                tmp97 = round(ELN_STOCK.yieldPA, 2)
            elif strikeFactor == 0.95:
                tmp95 = round(ELN_STOCK.yieldPA, 2)
            elif strikeFactor == 0.93:
                tmp93 = round(ELN_STOCK.yieldPA, 2)


        elnList.append((stock, tmp97, tmp95, tmp93))

    return pd.DataFrame(elnList, columns=['Ticker', '97%', '95%', '93%']).to_json(orient='records')

@app.route('/term_sheet', methods=['POST'])
def term_sheet():
    if request.method == 'POST':
        body = request.get_json()
        tenor = body['tenor']
        ticker = body['ticker']
        notional = body['notional']
        discount = body['discount']

        stock = ticker
        daysToExpiration = tenor
        underlyingPrice = prices.iloc[-1][str(stock)]
        interestRate = 1.5
        annualDividends = 1
        volatility = ret_std.iloc[-1][str(stock)] 
        volFactor = 1.2
        value = notional
        costOfFund = 1
        strikePrice = underlyingPrice * discount

        param_list = [underlyingPrice,
                        strikePrice, 
                        interestRate,
                        annualDividends,
                        daysToExpiration,
                        costOfFund]

        ELN_STOCK = ELN(ID = 1, 
                        stock = ticker,  
                        param_list = param_list, 
                        volatility = volatility, 
                        volFactor = volFactor,
                        value = value)

        #for term sheet

        issueDate = dt.date.today() + dt.timedelta(days=2)
        tradeDate = dt.date.today()
        effectiveDate = dt.date.today() + dt.timedelta(days=2)
        valuationDate = dt.date.today() + dt.timedelta(days=tenor - 2)
        maturityDate = dt.date.today() + dt.timedelta(days=tenor)
        term = tenor
        nominalAmount = notional
        disAmount = notional * ELN_STOCK.elnPCT
        withHoldingTax = (nominalAmount - disAmount) * 0.15
        commission = nominalAmount * 0.01
        issuePrice = disAmount + withHoldingTax + commission
        spotPrice = underlyingPrice
        percentStikeSpot = discount
        strikePrice = spotPrice * percentStikeSpot
        yieldAfterTax = ELN_STOCK.yieldPA * 0.85
        noShareAfterTax = ELN_STOCK.optionPosition

        jsonTmp = {
                "Isssue Date": issueDate,
                "Trade Date": tradeDate,
                "Effective Date": effectiveDate,
                "Valuation Date": valuationDate,
                "Maturity Date": maturityDate,
                "Term": term,
                "Nominal Amount (THB)": nominalAmount,
                "Issue Price (THB)": issuePrice,
                "With-holding Tax (15%)": withHoldingTax,
                "Spot Price (THB)": spotPrice,
                "% Strike of Spot": percentStikeSpot,
                "Strike Price (THB)": strikePrice,
                "Annualized Yield after Tax": yieldAfterTax,
                "No. of Shares after Tax & Fee": noShareAfterTax
            }

    return json.dumps(jsonTmp, indent=4, sort_keys=True, default=str)

if __name__ == '__main__':
    sim = DataSimulatorCSV(quotes, '2019-6-01', '2019-10-28')
    prices, ret, ret_std  = sim.next_data()
    app.run(host='0.0.0.0',debug = True)



