import numpy as np
import skfuzzy as fuzz
from skfuzzy import control as ctrl
import matplotlib.pyplot as plt

# 1. تعريف المتغيرات (Universes)
forecast = ctrl.Antecedent(np.arange(0, 31, 1), 'sales_forecast')
engagement = ctrl.Antecedent(np.arange(0, 11, 1), 'market_engagement')
scale = ctrl.Consequent(np.arange(0, 101, 1), 'budget_scaling')

# 2. دالات الانتماء (Membership Functions)
forecast['low'] = fuzz.trimf(forecast.universe, [0, 0, 10])
forecast['medium'] = fuzz.trimf(forecast.universe, [8, 15, 22])
forecast['high'] = fuzz.trimf(forecast.universe, [18, 30, 30])

engagement['poor'] = fuzz.trimf(engagement.universe, [0, 0, 5])
engagement['average'] = fuzz.trimf(engagement.universe, [4, 5, 6])
engagement['amazing'] = fuzz.trimf(engagement.universe, [5, 10, 10])

scale['cut'] = fuzz.trimf(scale.universe, [0, 0, 20])
scale['maintain'] = fuzz.trimf(scale.universe, [15, 30, 45])
scale['aggressive'] = fuzz.trimf(scale.universe, [40, 70, 100])

# 3. القواعد (السر في حل المشكلة)[cite: 11]
# ضفنا قاعدة تغطي (Medium & Amazing) عشان الكود ميعملش KeyError
rule1 = ctrl.Rule(forecast['high'] & engagement['amazing'], scale['aggressive'])
rule2 = ctrl.Rule(forecast['low'] | engagement['poor'], scale['cut'])
rule3 = ctrl.Rule(forecast['medium'] & engagement['average'], scale['maintain'])
rule4 = ctrl.Rule(forecast['high'] & engagement['average'], scale['aggressive'])
rule5 = ctrl.Rule(forecast['medium'] & engagement['amazing'], scale['aggressive']) # القاعدة الجديدة

# 4. التنفيذ[cite: 11]
ad_decision_ctrl = ctrl.ControlSystem([rule1, rule2, rule3, rule4, rule5])
ad_simulation = ctrl.ControlSystemSimulation(ad_decision_ctrl)

# المدخلات اللي سببت المشكلة
ad_simulation.input['sales_forecast'] = 15.34 
ad_simulation.input['market_engagement'] = 9.0

ad_simulation.compute()

# التأكد من وجود المفتاح قبل الطباعة (للحماية)
if 'budget_scaling' in ad_simulation.output:
    print(f"--- Soft Computing Final Decision ---")
    print(f"ANN Forecast: 15.34 Units")
    print(f"RL Engagement: 9.0/10")
    print(f"Suggested Budget Increase: {ad_simulation.output['budget_scaling']:.2f}%")
else:
    print("Error: No rules fired for the given inputs.")

scale.view(sim=ad_simulation)
plt.show()