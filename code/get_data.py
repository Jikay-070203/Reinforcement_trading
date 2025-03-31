from tvDatafeed import TvDatafeed,Interval
from pathlib import Path
import os
def clean_data(df,weekend=False,timeframe='1d'):
    clone=df.copy()
    # clone=clone.asfreq(timeframe,method='ffill')
    # clone['Day']=clone.index.strftime('%A')
    # if weekend:
    #     cond=clone['Day'].isin(['Saturday','Sunday'])
    #     clone=clone[~cond]
    clone.columns=['Open','High','Low','Close']
    clone.index.names=['Date']
    return clone
companies={
        'TSLA':['NASDAQ',True],
        'MWG':['HOSE',True],
        'ETHUSD':['CRYPTO',False],
        'VCB':['HOSE',True],
        'BTCUSD':['CRYPTO',False],
        'FPT':['HOSE',True],
        'AMZN':['NASDAQ',True],
        'BRK.B':['NYSE',True],
        'MSFT':['NASDAQ',True],
        'AAPL':['NASDAQ',True],
        'QCOM':['NASDAQ',True],
        'NVDA':['NASDAQ',True],
        'XRPUSD':['CRYPTO',False],
        'SILVER':['TVC',True],
        'META':['NASDAQ',True],
        'DOGEUSD':['CRYPTO',False],
        'VNM':['HOSE',True],
        'BNBUSD':['CRYPTO',False],
        'ORCL':['NYSE',True],
        'AMD':['NYSE',True],
        'GOLD':['TVC',True],
        'SOLUSD':['CRYPTO',False],
        'ACB':['HOSE',True],
        'MBB':['HOSE',True],
        'TCB':['HOSE',True],
        'VPB':['HOSE',True],
        'SPX':['TVC',True]
    }

tv=TvDatafeed()
convert={'1min':Interval.in_1_minute,'3min':Interval.in_3_minute,'5min':Interval.in_5_minute,
         '15min':Interval.in_15_minute,'30min':Interval.in_30_minute,'45min':Interval.in_45_minute,
         '1h':Interval.in_1_hour,'2h':Interval.in_2_hour,'3h':Interval.in_3_hour,'4h':Interval.in_4_hour,
         '1d':Interval.in_daily}

if not os.path.exists(os.path.join(Path(__file__).parent,'data')):
    os.mkdir(os.path.join(Path(__file__).parent,'data'))
for symbol,exchange in companies.items():
    if not os.path.exists(os.path.join(Path(__file__).parent,f'data/{symbol}')):
        os.mkdir(os.path.join(Path(__file__).parent,f'data/{symbol}'))
    for timeframe,interval_object in convert.items():
        print(symbol,timeframe)
        try:
            df=tv.get_hist(symbol,exchange=exchange[0],interval=interval_object,n_bars=1000000)
            df.drop(columns=['symbol','volume'],inplace=True)
            df=clean_data(df,weekend=exchange[1])
            df.to_csv(os.path.join(Path(__file__).parent,f'data/{symbol}/{symbol}_{timeframe}.csv'))
        except Exception as e:
            print(f'Error on {symbol} {str(e)}')