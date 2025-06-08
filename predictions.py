import pandas as pd
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
from sqlalchemy.engine import URL
from sqlalchemy import create_engine

DB_USER = "postgres" # <-- вставте свої дані
DB_PASSWORD = "," # <-- вставте свої дані
DB_HOST = "localhost" # <-- вставте свої дані
DB_PORT = "5432" # <-- вставте свої дані
DB_NAME = "classicmodels"

url = URL.create(
    drivername="postgresql+psycopg2",
    username=DB_USER,
    password=DB_PASSWORD,
    host=DB_HOST,
    port=DB_PORT,
    database=DB_NAME
)

dsn = url.render_as_string(hide_password=False)
print(">>> DSN для psycopg2 (з паролем):", dsn)
try:
    raw_bytes = dsn.encode("utf-8")
    print(">>> «raw_bytes» успішно закодовано в UTF-8:", raw_bytes)
except UnicodeEncodeError as uee:
    print("!!! Помилка кодування DSN у UTF-8:", repr(uee))

engine = create_engine(url)
conn = engine.connect()
print("Успішно підключилися до бази даних classicmodels!")

query = """
SELECT
    customerNumber               AS customer_id,
    AVG(amount)                  AS avgtransactionamount,
    COUNT(*)                     AS transactioncount,
    SUM(amount)                  AS totalpaymentamount,
    MIN(paymentDate)             AS firstpaymentdate,
    MAX(paymentDate)             AS lastpaymentdate
FROM ClassicModels.Payments
GROUP BY customerNumber;
"""
df = pd.read_sql(query, conn)
print("Назви колонок у DataFrame:", df.columns.tolist())

df['firstpaymentdate'] = pd.to_datetime(df['firstpaymentdate'])
df['lastpaymentdate'] = pd.to_datetime(df['lastpaymentdate'])
df['daysSinceLastPurchase'] = (pd.Timestamp.today() - df['lastpaymentdate']).dt.days
df['daysSinceFirstPurchase'] = (pd.Timestamp.today() - df['firstpaymentdate']).dt.days
df['tenureDays'] = (df['lastpaymentdate'] - df['firstpaymentdate']).dt.days

df['avgDaysBetweenPayments'] = df.apply(
    lambda row: (row['tenureDays'] / (row['transactioncount'] - 1))
    if row['transactioncount'] > 1 else row['tenureDays'],
    axis=1
)

print("\n=== Перші 5 рядків із розширеними метриками ===")
print(df[[
    'customer_id',
    'avgtransactionamount',
    'transactioncount',
    'totalpaymentamount',
    'firstpaymentdate',
    'lastpaymentdate',
    'daysSinceFirstPurchase',
    'daysSinceLastPurchase',
    'tenureDays',
    'avgDaysBetweenPayments'
]].head())

features = df[[
    'avgtransactionamount',
    'transactioncount',
    'totalpaymentamount',
    'daysSinceLastPurchase',
    'avgDaysBetweenPayments',
    'tenureDays'
]].copy()

scaler = StandardScaler()
X_scaled = scaler.fit_transform(features)

SSE = []
K_range = range(2, 8)
for k in K_range:
    km = KMeans(n_clusters=k, random_state=42).fit(X_scaled)
    SSE.append(km.inertia_)

plt.figure(figsize=(8, 5))
plt.plot(K_range, SSE, marker='o', linestyle='-')
plt.xlabel('k (кількість кластерів)')
plt.ylabel('WCSS (Inertia)')
plt.title('Elbow-метод для вибору числа кластерів')
plt.grid(True)
plt.show()

n_clusters = 3
kmeans = KMeans(n_clusters=n_clusters, random_state=42)
df['cluster'] = kmeans.fit_predict(X_scaled)

cluster_centers = kmeans.cluster_centers_
centers_original_scale = scaler.inverse_transform(cluster_centers)
centers_df = pd.DataFrame(
    centers_original_scale,
    columns=features.columns
)
centers_df['cluster'] = centers_df.index
print("\n=== Центри кластерів (у оригінальних одиницях) ===")
print(centers_df)

print("\n=== Результати кластеризації для перших 5 клієнтів ===")
print(df[[
    'customer_id',
    'avgtransactionamount',
    'transactioncount',
    'totalpaymentamount',
    'daysSinceLastPurchase',
    'avgDaysBetweenPayments',
    'tenureDays',
    'cluster'
]].head())

plt.figure(figsize=(8, 6))
plt.scatter(X_scaled[:, 0], X_scaled[:, 1], c=df['cluster'], cmap='viridis', s=50, alpha=0.7)
for i, center in enumerate(cluster_centers):
    plt.scatter(center[0], center[1], marker='X', s=200, edgecolor='k', linewidth=1.5)
plt.xlabel('avgtransactionamount (scaled)')
plt.ylabel('transactioncount (scaled)')
plt.title('Кластеризація клієнтів (avgAmt vs count)')
plt.grid(True)
plt.show()

plt.figure(figsize=(8, 6))
plt.scatter(X_scaled[:, 3], X_scaled[:, 2], c=df['cluster'], cmap='plasma', s=50, alpha=0.7)
for i, center in enumerate(cluster_centers):
    plt.scatter(center[3], center[2], marker='X', s=200, edgecolor='k', linewidth=1.5)
plt.xlabel('daysSinceLastPurchase (scaled)')
plt.ylabel('totalpaymentamount (scaled)')
plt.title('Кластеризація клієнтів (recency vs total payment)')
plt.grid(True)
plt.show()

output_table = "customer_clusters"
df_to_save = df[[
    'customer_id',
    'avgtransactionamount',
    'transactioncount',
    'totalpaymentamount',
    'firstpaymentdate',
    'lastpaymentdate',
    'daysSinceFirstPurchase',
    'daysSinceLastPurchase',
    'tenureDays',
    'avgDaysBetweenPayments',
    'cluster'
]].copy()
df_to_save.to_sql(
    name=output_table,
    con=engine,
    if_exists="replace",
    index=False
)

print(f"\nРезультати кластеризації записані в таблицю '{output_table}' бази classicmodels.")
conn.close()
