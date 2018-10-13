# stock-predictor (http://stockpredictor.pythonanywhere.com)
Predicts next five days prices for NASDAQ stocks using Python 3.7, Flask and yahoo finance  libraries.
This App uses Continuous Regresson, and fetches 10 years or as long as available data to train the model.
It has function to use different k folds, which implemented manually and has different methods for model's accuracy evaluation.
App uses Flask,WTForms and jinja2 as web delivery and representation tools.

The focus is on Machine learning regression and obviously There are couple of opportunities that can make the app more robust and more convienient as well specially in development and representation parts. Here are afew ideas for whoever is interested:
- There can be a drop down box/ Auto completion text box for entering Stock names rather than symbols.
- It can be extended to train the model by defined specific begining date.
- It can get data from more reliable API.

