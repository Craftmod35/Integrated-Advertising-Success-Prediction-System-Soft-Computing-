import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from minisom import MiniSom
from sklearn.preprocessing import MinMaxScaler
from pylab import bone, pcolor, colorbar, plot, show

# 1. قراءة البيانات
df = pd.read_csv('data/Credit_Card_Applications.csv')
X = df.iloc[:, :-1].values 
Y = df.iloc[:, -1].values  # <<< تصليح: تعريف Y عشان الـ Markers تشتغل

# 2. التطبيع (Normalization)
sc = MinMaxScaler(feature_range=(0, 1))
X_scaled = sc.fit_transform(X)

# 3. بناء وتدريب الـ SOM
# حجم الشبكة بناءً على قاعدة 5*sqrt(N) تقريباً
som = MiniSom(x=10, y=10, input_len=X.shape[1], sigma=1.0, learning_rate=0.5)
som.random_weights_init(X_scaled)
som.train_random(data=X_scaled, num_iteration=100)

# 4. رسم الخريطة وتحديد الرموز (Visualization)
bone() 
pcolor(som.distance_map().T) 
colorbar() 

# إضافة علامات للعملاء (دائرة للأحمر/رفض، مربع للأخضر/قبول)
markers = ['o', 's']
colors = ['r', 'g']
for i, x in enumerate(X_scaled):
    w = som.winner(x) 
    plot(w[0] + 0.5,
         w[1] + 0.5,
         markers[Y[i]], 
         markeredgecolor = colors[Y[i]],
         markerfacecolor = 'None',
         markersize = 10,
         markeredgewidth = 2)
    
show()

# 5. تحديد العملاء المشكوك فيهم (Outliers)
mappings = som.win_map(X_scaled)

# ملاحظة: لازم تشوف الرسمة وتختار إحداثيات المربعات "البيضاء" جداً
# هعملك حماية هنا عشان الكود ميعملش Crash لو الإحداثيات مش موجودة
# --- After show() ---

# Extracting the customers who fell into the bright white square (5, 5)
# Note: Ensure (5, 5) matches the coordinates of the white square you saw in Figure_1.png
try:
    target_fraud_customers = mappings[(5, 5)] 
    target_fraud_customers = sc.inverse_transform(target_fraud_customers)

    print(f"\nFound {len(target_fraud_customers)} suspect customers in the white square.")
    print("List of their Customer IDs:")
    print(target_fraud_customers[:, 0]) # Printing the first column which contains the ID
except KeyError:
    print("\nCoordinates (5, 5) not found in mappings. Please check the white squares in your plot.")