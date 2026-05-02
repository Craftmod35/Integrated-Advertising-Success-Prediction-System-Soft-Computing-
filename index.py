import sys
import os

# 1. تعريف المسارات عشان السيستم يشوف كل الفولدرات
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), 'src')))

# 2. استيراد المحركات اللي تعبنا فيها
# بننادي دالة التوقع من الـ ANN ومحاكي الـ Fuzzy
from phase3_deep.ann_layer import predict
from phase6_fuzzy.fuzzy_layer import ad_simulation

print("--- Advertising AI Integrated System ---")

# 3. خطوة الـ Deep Learning (التوقع)
# هنجرب ميزانية جديدة: TV=200, Radio=40, Newspaper=10
new_campaign = [[200.0, 40.0, 10.0]]
predicted_sales = predict(new_campaign)[0][0] # هيطلع لنا رقم زي 18.5 مثلاً

print(f"[ANN] Forecasted Sales for this campaign: {predicted_sales:.2f} Units")

# 4. خطوة الـ Fuzzy Logic (اتخاذ القرار)
# بناخد الرقم اللي طلع ونبعته للفازي
ad_simulation.input['sales_forecast'] = predicted_sales
ad_simulation.input['market_engagement'] = 8.5 # قيمة جاية من الـ RL[cite: 10]

ad_simulation.compute()

# 5. الناتج النهائي
final_decision = ad_simulation.output['budget_scaling']
print(f"[FUZZY] System Decision: Scale Budget by {final_decision:.2f}%")

print("\n--- Process Finished Successfully ---")