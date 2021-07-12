import pandas as pd
import matplotlib.pyplot as plt
import mplfinance as mpf
import seaborn as sns
import sys

sns.set_theme()

df = pd.read_csv(sys.argv[1], sep=";", decimal=",")
df.Date = pd.to_datetime(df.Date, format="%d/%m/%Y")
df = df.iloc[:, :-1] # drop last column
df.rename(columns={"Ouverture": "Open", "Fermeture": "Close", "Valeur Haute": "High", "Valeur Basse": "Low"}, inplace=True)
print(df.dtypes)
df = df.set_index("Date")
print(df)
print(df.dtypes)


mpf.plot(df, type="candle", volume=True)
plt.show()
