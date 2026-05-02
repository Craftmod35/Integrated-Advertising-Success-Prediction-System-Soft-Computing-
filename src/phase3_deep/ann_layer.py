import tensorflow as tf
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler


df = pd.read_csv('data/Advertising.csv')

# تجهيز المدخلات للـ ANN
X_ann = df[['TV', 'Radio', 'Newspaper']].values.astype(np.float32)
Y_ann = df['Sales'].values.reshape(-1, 1).astype(np.float32)

scaler_x = MinMaxScaler()
scaler_y = MinMaxScaler()
X_scaled = scaler_x.fit_transform(X_ann)
Y_scaled = scaler_y.fit_transform(Y_ann)

from sklearn.model_selection import train_test_split

# ... بعد الـ Scaling ...
# تقسيم البيانات (80% تدريب، 20% اختبار)
X_train_scaled, X_test_scaled, Y_train_scaled, Y_test_scaled = train_test_split(
    X_scaled, Y_scaled, test_size=0.2, random_state=42
)

# تعديل الـ train_step عشان تستخدم الـ Train فقط
def train_step():
    with tf.GradientTape() as tape:
        predictions = forward_pass(X_train_scaled) # التدريب على الجزء المخصص فقط
        loss = tf.reduce_mean(tf.square(predictions - Y_train_scaled))
    
    variables = [W1, b1, W2, b2]
    gradients = tape.gradient(loss, variables)
    optimizer.apply_gradients(zip(gradients, variables))
    return loss

# 2. تعريف الأوزان والـ Biases (بناءً على الرسم في صفحة 2 من الملف)
# هيدن لاير فيها 4 أعصاب، والأوتبوت فيها 1
input_size = 3
hidden_size = 4
output_size = 1

# أوزان بين الإدخال والطبقة المخفية
W1 = tf.Variable(tf.random.normal([input_size, hidden_size]), name='W1')
b1 = tf.Variable(tf.zeros([hidden_size]), name='b1')

# أوزان بين الطبقة المخفية والمخرج
W2 = tf.Variable(tf.random.normal([hidden_size, output_size]), name='W2')
b2 = tf.Variable(tf.zeros([output_size]), name='b2')

# Normalization للكل (ضروري جداً في الـ ANN)
scaler_x = MinMaxScaler()
X_ann_scaled = scaler_x.fit_transform(X_ann)


# 3. الدوال الرياضية (Forward Propagation)
def forward_pass(X):
    # الخطوة السحرية: تحويل المدخلات لـ float32 عشان تطابق الأوزان
    X = tf.cast(X, tf.float32) 
    
    # الطبقة المخفية (Hidden Layer) + Sigmoid
    h = tf.nn.sigmoid(tf.matmul(X, W1) + b1)
    
    # الطبقة النهائية (Output Layer)[cite: 3, 6]
    y_pred = tf.matmul(h, W2) + b2
    return y_pred

# 4. التدريب (Backpropagation)
optimizer = tf.optimizers.Adam(learning_rate=0.01)

def train_step():
    with tf.GradientTape() as tape:
        predictions = forward_pass(X_scaled)
        # حساب الخطأ (Loss)
        loss = tf.reduce_mean(tf.square(predictions - Y_scaled))
    
    # انتشار الخطأ للخلف وتحديث الأوزان
    variables = [W1, b1, W2, b2]
    gradients = tape.gradient(loss, variables)
    optimizer.apply_gradients(zip(gradients, variables))
    return loss


# 5. حلقة التدريب (The Actual Training)
epochs = 3000
display_step = 500

print("--- Neural Training Begun (ANN) ---")
for epoch in range(1, epochs + 1):
    current_loss = train_step()
    
    if epoch % display_step == 0 or epoch == 1:
        print(f"Epoch: {epoch}, Loss: {current_loss.numpy():.6f}")

# 6. تجربة التوقع (Testing)
def predict(input_data):
    # تحويل المدخلات لـ Scaled قبل التوقع
    scaled_input = scaler_x.transform(input_data)
    pred_scaled = forward_pass(scaled_input)
    # إرجاع النتيجة لأصلها (المبيعات الحقيقية)
    return scaler_y.inverse_transform(pred_scaled)

# تجربة بميزانية حقيقية: TV=150, Radio=30, Newspaper=20
test_ads = np.array([[150.0, 30.0, 20.0]], dtype=np.float32)
predicted_sales = predict(test_ads)
print(f"\n Expected sales for {test_ads[0]}: {predicted_sales[0][0]:.2f}")

