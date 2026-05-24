from sec_edgar_downloader import Downloader
dl = Downloader('FinancialRAG', 'your@email.com', './data/raw')
for ticker in ['NVDA','MSFT','GOOGL','META','AAPL']:
    dl.get('10-K', ticker, limit=3)
    dl.get('10-Q', ticker, limit=8)