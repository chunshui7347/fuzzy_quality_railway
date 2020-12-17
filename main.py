import numpy as np
import matplotlib.pyplot as plt
import skfuzzy as fuzz

food = service = np.arange(0,11,1)
tip = np.arange(0,25.1,0.1)
bad=fuzz.trimf(food, [0,0,5])
descent=fuzz.trimf(food, [0,5,10])
great=fuzz.trimf(food, [5,10,10])
poor=fuzz.trimf(service, [0,0,5])
acceptable=fuzz.trimf(service, [0,5,10])
amazing=fuzz.trimf(service, [5,10,10])
low=fuzz.trimf(tip, [0,0,12.5])
medium=fuzz.trimf(tip, [0,12.5,25])
high=fuzz.trimf(tip, [12.5,25,25])

## Rule 1: If the food is bad, then the tip is low
is_bad = fuzz.interp_membership(food, bad, 6)
fire_rule1 = is_bad
## Rule 2: If the service is acceptable, then the tip is medium
is_acceptable = fuzz.interp_membership(service, acceptable, 8)
fire_rule2 = is_acceptable
## Rule 3: If the food is great AND the service is amazing, then the tip is high
is_great = fuzz.interp_membership(food, great, 6)
is_amazing = fuzz.interp_membership(service, amazing, 8)
fire_rule3 = min(is_great,is_amazing)
rule1_clip = np.fmin(fire_rule1,low)
rule2_clip = np.fmin(fire_rule2,medium)
rule3_clip = np.fmin(fire_rule3,high)

######### Aggregate the rules #########
temp = np.fmax(rule1_clip,rule2_clip)
fuzzy_output = np.fmax(temp,rule3_clip)
######### Defuzzification #########
tip_to_pay = fuzz.defuzz(tip, fuzzy_output, 'centroid')
print ('the tip to pay is RM', tip_to_pay)
# expected output : RM 12.853535353535325

tip_activation = fuzz.interp_membership(tip, fuzzy_output, tip_to_pay)
tip0 = np.zeros_like(tip)
fig, ax0 = plt.subplots(figsize=(8, 3))
ax0.plot(tip, low, 'b', linewidth=1.5, linestyle='--', )
ax0.plot(tip, medium, 'g', linewidth=1.5, linestyle='--')
ax0.plot(tip, high, 'r', linewidth=1.5, linestyle='--')
ax0.fill_between(tip, tip0, fuzzy_output, facecolor='Orange', alpha=0.5)
ax0.plot([tip_to_pay, tip_to_pay], [0, tip_activation], 'k', linewidth=2.5,
alpha=0.9)
ax0.get_xaxis().tick_bottom()
ax0.get_yaxis().tick_left()
ax0.set_xlim([min(tip),max(tip)])
ax0.set_ylim([0,1])
plt.xlabel('tip')
plt.ylabel('membership degree')
plt.title('Tipping problem')
plt.show()