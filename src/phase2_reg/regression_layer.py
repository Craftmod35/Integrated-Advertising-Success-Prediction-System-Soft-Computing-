import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.preprocessing import MinMaxScaler



learning_rate=0.01
training_steps=5000
display_step=200
alpha_penalty = 0.1 # معامل قوة التنظيم

df = pd.read_csv('data/Advertising.csv')

X_raw = df['TV'].values.reshape(-1, 1).astype(np.float32)
Y_raw = df['Sales'].values.reshape(-1, 1).astype(np.float32)

scaler_x = MinMaxScaler()
scaler_y = MinMaxScaler()

X_scaled = scaler_x.fit_transform(X_raw).flatten()
Y_scaled = scaler_y.fit_transform(Y_raw).flatten()

threshold = np.mean(Y_scaled)
Y_class = np.array([1 if y > threshold else 0 for y in Y_scaled], dtype=np.float32)

#W :(slope) الوزن 
#b : (Intercept) الانحراف
W= tf.Variable(tf.random.normal([1]), name='Weight')
b= tf.Variable(tf.random.normal([1]), name='bias')

"""
tf.Variable: هذا الأمر يخبر TensorFlow أن هذه القيم ليست ثابتة، بل هي "متغيرات قابلة للتدريب" (Trainable). السيستم سيبدأ بتعديلها في كل دورة (Step) لتقليل الخطأ.

tf.random.normal([1]): هنا بنطلب من TensorFlow إنتاج رقم عشوائي يتبع "التوزيع الطبيعي" (Normal Distribution). الـ [1] تعني أننا نريد رقماً واحداً فقط لكل متغير.

الأسماء (name="weight"): مفيدة جداً غداً عندما تربط هذا الجزء بالـ Fuzzy Logic أو الـ ANN، لتتبع أي وزن يتغير.
"""



#(Linear Regression) دالة التنبؤ الخطي 

def linear_regression(x):
  return W*x+b

# Mean square error دالة حساب متوسط مربعات الخطأ
def mean_square(y_predicted, y_true):
  return tf.reduce_mean(tf.square(y_predicted-y_true))

"""
y_pred - y_true: الفرق بين اللي الموديل توقعه وبين المبيعات الحقيقية.

tf.square: بنربع الفرق ده عشان نلغي الإشارات السالبة ونكبر الأخطاء الكبيرة (عشان الموديل يخاف منها ويصلحها).

tf.reduce_mean: بنجيب المتوسط لكل البيانات اللي في الـ Array.
"""

def sigmoid(z):
    return 1 / (1 + tf.exp(-z))

def predict_success_probability(x):
    z = W * x + b
    return sigmoid(z)

def ridge_loss(y_pred, y_true):
    # Mean Squared Error + L2 Penalty[cite: 2]
    mse = tf.reduce_mean(tf.square(y_pred - y_true))
    penalty = alpha_penalty * tf.reduce_sum(tf.square(W))
    return mse + penalty


#Stochastic Gradient Descent لتحديد الاوزان 
#ده اللي هيعدل W و b بناءً على معدل التعلم اللي حددناه (0.01).
optimizer=tf.optimizers.SGD(learning_rate)

def run_optimization():
    with tf.GradientTape() as g:
        pred = linear_regression(X_scaled)
        loss = ridge_loss(pred, Y_scaled)
    gradients = g.gradient(loss, [W, b])
    optimizer.apply_gradients(zip(gradients, [W, b]))
    return loss
  
 
    """
 \
     tf.GradientTape():
     تخيل ده "كاميرا مراقبة". في الـ AI، عشان تعرف تعدل الـ $W$، محتاج تحسب "المشتقة"       (Derivative) بتاعة الـ Loss بالنسبة لـ $W$. الكاميرا دي بتسجل الخطوات الحسابية عشان تعرف ترجع بضهرها وتحسب المشتقة دي تلقائياً.

     g.gradient(loss, [W, b]): هنا بنطلع قيمة الميل (Slope). لو الميل موجب، الموديل بيفهم إنه محتاج ينقص $W$. ولو سالب، بيزودها.

     zip(gradients, [W, b]): الـ zip بتربط كل "تعديل" بالمتغير بتاعه. (تعديل $W$ يروح لـ $W$ وتعديل $b$ يروح لـ $b$).

    """



  
  
print("Start Training")
for step in range(1, training_steps + 1):
    current_loss = run_optimization()

    if step % display_step == 0:
        print(f"Step: {step}, Loss: {current_loss.numpy():.4f}, W: {W.numpy()[0]:.4f}, b: {b.numpy()[0]:.4f}")


def get_final_prediction(input_budget):
    # تحويل الرقم العادي لنطاق الموديل (Scale)[cite: 2]
    scaled_input = scaler_x.transform([[input_budget]])[0][0]
    
    # التوقع
    sales_scaled = linear_regression(scaled_input).numpy()
    prob = predict_success_probability(scaled_input).numpy()[0]
    
    # إرجاع القيمة لأصلها (Inverse Scale) عشان الـ Fuzzy يفهمها
    actual_sales_pred = scaler_y.inverse_transform([sales_scaled])[0][0]
    
    return actual_sales_pred, prob

test_budget = 200.0 
expected_sales, success_chance = get_final_prediction(test_budget)

print(f"\n--- Results: {test_budget}$ ---")
print(f"Expected Sales {expected_sales:.2f}")
print(f"Succes Rate (Logistic): {success_chance*100:.2f}%")




plt.scatter(X_scaled, Y_scaled, color='blue', alpha=0.5, label='Actual Data')
plt.plot(X_scaled, W.numpy() * X_scaled + b.numpy(), color='red', label='Fitted Line (Ridge)')
plt.title('TV Budget vs Sales (Real Data)')
plt.legend()
plt.show()

#python src/phase2_reg/regression_layer.py
