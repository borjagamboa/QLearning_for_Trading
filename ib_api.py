from ibapi.wrapper import EWrapper
from ibapi.client import EClient
from ibapi.contract import Contract, ContractDetails
from datetime import datetime
import time, threading
import pandas as pd
import os

class IBapi(EWrapper, EClient):
    def __init__(self):
        EClient.__init__(self, self)
        self.cripto_historical = {}
        self.stock_historical = {}
        self.active_request = []

    def reset_requests(self):
        self.active_request = []

    def nextValidId(self, orderId: int):
        # logging.debug("setting nextValidOrderId: %d", orderId)
        self.nextValidOrderId = orderId
        print("NextValidId:", orderId)

    def tickPrice(self, reqId, tickType, price, attrib):
        if tickType == 2 and reqId == 1:
            print('The current ask price is: ', price)
        elif tickType == 1:
            print('The current bid price is: ', price)
        elif tickType == 14:
            print('The current open price is: ', price)
        elif tickType == 6:
            print('The current high price is: ', price)
        elif tickType == 7:
            print('The current low price is: ', price)

    def historicalData(self, reqId, bar):
        print(f'Time: {bar.date} Open: {bar.open} High: {bar.high}'
              f' Low: {bar.low} Close: {bar.close} ')
        self.active_request.append([bar.date, bar.open, bar.high, bar.low, bar.close])

    def historicalDataEnd(self, reqId:int, start:str, end:str):
        """ Marks the ending of the historical bars reception. """
        print('Historical data loaded')

    def contractDetails(self, reqId: int, contractDetails: ContractDetails):
        super().contractDetails(reqId, contractDetails)
        print(contractDetails)

    def contractDetailsEnd(self, reqId: int):
        super().contractDetailsEnd(reqId)
        print("ContractDetailsEnd. ReqId:", reqId)

    def get_news(self):
        news_contract = Contract()
        news_contract.secType = "NEWS"
        news_contract.exchange = "BRFG"  # Briefing Trader
        self.reqContractDetails(1, news_contract)
        pass


    def get_last_OHLC(self, ticker):
        contract = Contract()
        contract.symbol = ticker
        contract.secType = 'STK'
        contract.exchange = 'SMART'
        contract.currency = 'USD'
        try:
            self.reqMktData(1, contract, '', True, False, [])
        except:
            print('Unable to get last OHLC')
        return

    def get_cripto_historical(self, cripto_tickers: list, duration: str, barsize: str):
        '''
        Loads cripto historicals data in Pandas DataFrame, accessible via IBapi.cripto_historical dict
            - cripto_tickers: list
            - duration: str. Example: '100 D' See IBTWS API documentation and data limitations
            - barsize: str. Example: '1 day' See IBTWS API documentation and data limitations
        '''
        for reqId, ticker in enumerate(cripto_tickers):
            self.cripto_historical[ticker] = []
            cripto_contract = Contract()
            cripto_contract.symbol = ticker
            cripto_contract.secType = "CRYPTO"
            cripto_contract.exchange = "PAXOS" # This the exchange for all criptos in IB
            cripto_contract.currency = "USD"

            # Request Market Data for last X days in hourly bar-size
            self.reqHistoricalData(1, cripto_contract, '', duration, barsize,
                                  'AGGTRADES', 0, 2, False, [])

            time.sleep(5)  # Sleep interval to allow time for incoming price data

            # Store data in Pandas DataFrame
            df = pd.DataFrame(self.active_request, columns=['DateTime', 'Open', 'High', 'Low', 'Close'])
            df['DateTime'] = pd.to_datetime(df['DateTime'], unit='s')
            df.set_index(['DateTime'], inplace=True)
            self.cripto_historical[ticker] = df
            self.reset_requests()
        return self.cripto_historical

    def save_csv(self, ticker, df):
        '''
        Save OHLC in csv. It is saved by default in a folder called like the ticker and in the current wdir
        '''
        try:
            # Get current datetime to name the file
            dt_time = datetime.now().strftime('%y-%m-%d-%H%M')
            if os.path.join(os.getcwd(), ticker):
                pass
            else:
                os.mkdir(ticker)
            df.to_csv(f'{ticker}/{dt_time}.csv')
            print(f'New data saved into {ticker}/{dt_time}.csv')
            app.disconnect()
        except:
            print('Unable to save data')
            app.disconnect()


if __name__ == "__main__":
    def run_loop():
        app.run()

    cripto_tickers = ['BTC', 'ETH']

    app = IBapi()
    app.connect("127.0.0.1", 7497, clientId=123)

    # Start the socket in a thread
    api_thread = threading.Thread(target=run_loop, daemon=True)
    api_thread.start()
    time.sleep(1) # Sleep interval to allow time for connection to server
    print("serverVersion:%s connectionTime:%s" % (app.serverVersion(), app.twsConnectionTime()))
    app.get_last_OHLC('AAPL')

#    app.get_news()
#    app.save_csv(ticker, df)
    app.disconnect()

################### TESTS #######################