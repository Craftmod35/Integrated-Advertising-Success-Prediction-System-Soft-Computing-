import numpy as np

def min_max_scaler(data):
    """
    بتحول البيانات لنطاق [0, 1] بناءً على الخطة المعتمدة للمشروع.
    """
    # تحويل الداتا لـ Numpy Array لضمان سرعة الحسابات
    data = np.array(data)
    
    data_min = np.min(data)
    data_max = np.max(data)
    
    # تطبيق المعادلة: (X - min) / (max - min)
    #
    scaled_data = (data - data_min) / (data_max - data_min)
    
    return scaled_data, data_min, data_max

def inverse_scaler(scaled_data, data_min, data_max):
    """
    بترجع الأرقام لأصلها (مثلاً لو عايز تعرف المبيعات بالدولار مش بنسبة 0 لـ 1)
    """
    return scaled_data * (data_max - data_min) + data_min