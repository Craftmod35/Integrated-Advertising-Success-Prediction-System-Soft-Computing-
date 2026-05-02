import numpy as np
import skfuzzy as fuzz
from skfuzzy import control as ctrl
import matplotlib.pyplot as plt

# 1. تعريف المتغيرات (Universes)
forecast = ctrl.Antecedent(np.arange(0, 11, 1), 'sales_forecast')
engagement = ctrl.Antecedent(np.arange(0, 11, 1), 'market_engagement')
scale = ctrl.Consequent(np.arange(0, 51, 1), 'budget_scaling')

forecast['low'] = fuzz.trimf(forecast.universe, [0, 0, 4])
forecast['medium'] = fuzz.trimf(forecast.universe, [3, 5, 7])
forecast['high'] = fuzz.trimf(forecast.universe, [6, 10, 10])

engagement['very_poor'] = fuzz.trimf(engagement.universe, [0, 0, 2])
engagement['poor'] = fuzz.trimf(engagement.universe, [1, 3, 5])
engagement['average'] = fuzz.trimf(engagement.universe, [4, 5, 6])
engagement['good'] = fuzz.trimf(engagement.universe, [5, 7, 9])
engagement['amazing'] = fuzz.trimf(engagement.universe, [8, 10, 10])

scale['cut_budget'] = fuzz.trimf(scale.universe, [0, 0, 10])        # تقليل الميزانية
scale['maintain'] = fuzz.trimf(scale.universe, [8, 15, 22])       # الحفاظ عليها
scale['moderate_inc'] = fuzz.trimf(scale.universe, [20, 30, 35])   # زيادة متوسطة
scale['aggressive_inc'] = fuzz.trimf(scale.universe, [30, 40, 45]) # زيادة قوية
scale['maximum_scale'] = fuzz.trimf(scale.universe, [40, 50, 50])  # أقصى زيادة

rule1 = ctrl.Rule(forecast['low'] | engagement['very_poor'], scale['cut_budget'])
rule2 = ctrl.Rule(engagement['average'], scale['maintain'])
rule3 = ctrl.Rule(engagement['amazing'] & forecast['high'], scale['maximum_scale'])
rule4 = ctrl.Rule(engagement['good'] | forecast['high'], scale['aggressive_inc'])
rule5 = ctrl.Rule(forecast['medium'] & engagement['good'], scale['moderate_inc'])

ad_decision_ctrl = ctrl.ControlSystem([rule1, rule2, rule3, rule4, rule5])
ad_simulation = ctrl.ControlSystemSimulation(ad_decision_ctrl)

# تجربة السيستم (الربط مع نواتج الـ ANN)
# نفترض الـ ANN طلعت توقع 8.5 وتفاعل السوق 9.2
ad_simulation.input['sales_forecast'] = 8.5
ad_simulation.input['market_engagement'] = 9.2

ad_simulation.compute()

print(f"Final Decision, Increasing Budget By : {ad_simulation.output['budget_scaling']:.2f}%")

#  العرض البياني
forecast.view()
engagement.view()
scale.view(sim=ad_simulation)
plt.show()