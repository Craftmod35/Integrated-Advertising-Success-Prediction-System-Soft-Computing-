import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import mean_absolute_error, r2_score
import sys
import os

# --- خطوة 1: استيراد الموديل والبيانات من الملف التاني ---
# بنستخدم التنظيم اللي في الصورة (src.phase3_deep.ann_layer)
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../')))

try:
    # دلوقتي المتغيرات دي "بقت موجودة" فعلاً في ann_layer
    from ann_layer import forward_pass, X_test_scaled, Y_test_scaled, scaler_y
except ImportError:
    # كخطة بديلة لو شغال من الـ Root
    from src.phase3_deep.ann_layer import forward_pass, X_test_scaled, Y_test_scaled, scaler_y
print("Success! Module linked properly.")

# --- خطوة 2: الحصول على التوقعات ---
print("Calculating model predictions...")
y_pred_scaled = forward_pass(X_test_scaled) # تمرير بيانات الاختبار للشبكة

# إرجاع الأرقام لأصلها (De-normalization)
# التعديل هنا:
y_pred = scaler_y.inverse_transform(y_pred_scaled.numpy()) 
y_actual = scaler_y.inverse_transform(Y_test_scaled)

# --- خطوة 3: حساب مقاييس الأداء ---
mae = mean_absolute_error(y_actual, y_pred)
r2 = r2_score(y_actual, y_pred)

print(f"\n--- Accuracy Report ---")
print(f"Average Error (MAE): {mae:.2f} Sales Units")
print(f"Model Confidence (R2 Score): {r2*100:.2f}%")

# --- خطوة 4: الرسم البياني ---
plt.figure(figsize=(10, 6))
plt.plot(y_actual, label='Actual Sales', color='blue', linewidth=2)
plt.plot(y_pred, label='Predicted Sales', color='red', linestyle='--', linewidth=2)
plt.title('Actual vs Predicted Sales (ANN Evaluation)')
plt.xlabel('Test Samples')
plt.ylabel('Sales Value')
plt.legend()
plt.grid(True)
plt.show()