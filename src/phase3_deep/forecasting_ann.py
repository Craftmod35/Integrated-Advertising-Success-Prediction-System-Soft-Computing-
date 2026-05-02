import numpy as np
import matplotlib.pyplot as plt
from ann_layer import predict, df # استيراد الموديل والبيانات الأصلية[cite: 3]

# --- 1. إعداد بيانات السلسلة الزمنية (Forecasting Logic) ---
# سنفترض أن كل سطر في Advertising.csv يمثل أسبوعاً من الحملة
sales_data = df['Sales'].values

def create_forecasting_window(data, window_size=5):
    """تحويل البيانات لشكل نافذة زمنية للتوقع المستقبلي"""
    last_window = data[-window_size:]
    return last_window

# --- 2. محاكاة التوقع للمستقبل (Future Projection) ---
print("--- Generating Future Sales Forecast ---")

# لنأخذ آخر ميزانيات تم صرفها كمدخلات للتوقع القادم
# نستخدم ميزانية افتراضية للحملة القادمة بناءً على المتوسطات
future_budget = np.array([[df['TV'].mean(), df['Radio'].mean(), df['Newspaper'].mean()]], dtype=np.float32)

# التوقع باستخدام محرك الـ ANN[cite: 4]
forecasted_sales = predict(future_budget)

print(f"Current Market Pattern detected from last {len(df)} campaigns.")
print(f"Predicted Sales for the NEXT campaign: {forecasted_sales[0][0]:.2f} units.")

# --- 3. رسم منحنى التوقع (Forecasting Visualization) ---
# سنعرض آخر 20 حملة مع التوقع الجديد كـ "نقطة مستقبلية"
plt.figure(figsize=(10, 5))
plt.plot(range(len(sales_data[-20:])), sales_data[-20:], label='Past Sales Pattern', color='blue', marker='o')
plt.plot(20, forecasted_sales, label='Future Forecast', color='green', marker='D', markersize=10)

plt.title('Time Series Forecasting: Predicting Next Campaign Success')
plt.xlabel('Timeline (Recent Campaigns)')
plt.ylabel('Sales Units')
plt.legend()
plt.grid(True)
plt.show()