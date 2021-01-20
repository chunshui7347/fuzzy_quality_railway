import numpy as np
import skfuzzy as fuzz
from skfuzzy import control as ctrl
import pandas as pd
from utils import plot

cleanness = np.arange(1, 11)
price = np.arange(0.00, 5.01, 0.01)
frequency = np.arange(1, 31)

dirty = fuzz.trimf(cleanness, [1, 1, 5])
normal = fuzz.trimf(cleanness, [1, 5, 10])
clean = fuzz.trimf(cleanness, [5, 10, 10])

cheap = fuzz.trimf(price, [0, 0, 2.5])
reasonable = fuzz.trimf(price, [0, 2.5, 5])
expensive = fuzz.trimf(price, [2.5, 5, 5])

less = fuzz.trimf(frequency, [1, 1, 10])
enough = fuzz.trapmf(frequency, [5, 10, 15, 20])
many = fuzz.trapmf(frequency, [12, 20, 30, 30])

# graph plotting
plot.plot_graph(cleanness, dirty, normal, clean, 'cleanness', 'membership function')
plot.plot_graph(price, cheap, reasonable, expensive, 'price', 'membership function')
plot.plot_graph(frequency, less, enough, many, 'frequency', 'membership function')

# read train input and test input
train_data = pd.read_csv("data/fuzzy_part2_train.csv")
test_data = pd.read_csv("data/fuzzy_part2_test.csv")

train_price = train_data.price.to_list()
train_frequency = train_data.frequency.to_list()
train_cleanness = train_data.cleanness.to_list()
train_output = train_data.rate_class.to_list()

test_price = test_data.price.to_list()
test_frequency = test_data.frequency.to_list()
test_cleanness = test_data.cleanness.to_list()
test_output = test_data.rate_class.to_list()

rule_set = pd.DataFrame()
# Aggregate and calculate degree

cleanness_dirty_output = []
cleanness_normal_output = []
cleanness_clean_output = []
cleanness_max = []
cleanness_max_string = []

price_cheap_output = []
price_reasonable_output = []
price_expensive_output = []
price_max = []
price_max_string = []

frequency_less_output = []
frequencyy_enough_output = []
frequency_many_output = []
frequency_max = []
frequency_max_string = []

output_degree = []

for i in range(len(train_data)):

    input_cleanness = train_cleanness[i]
    input_price = train_price[i]
    input_frequency = train_frequency[i]

    cleanness_dirty = round(fuzz.interp_membership(cleanness, dirty, input_cleanness), 2)
    cleanness_normal = round(fuzz.interp_membership(cleanness, normal, input_cleanness), 2)
    cleanness_clean = round(fuzz.interp_membership(cleanness, clean, input_cleanness), 2)
    temp_cleanness_max = max(cleanness_dirty, cleanness_normal, cleanness_clean)
    cleanness_string = ""

    if temp_cleanness_max == cleanness_dirty:
        cleanness_string = "dirty"
    elif temp_cleanness_max == cleanness_normal:
        cleanness_string = "normal"
    else:
        cleanness_string = "clean"

    price_cheap = round(fuzz.interp_membership(price, cheap, input_price), 2)
    price_reasonable = round(fuzz.interp_membership(price, reasonable, input_price), 2)
    price_expensive = round(fuzz.interp_membership(price, expensive, input_price), 2)
    temp_price_max = max(price_cheap, price_reasonable, price_expensive)
    price_string = ""

    if temp_price_max == price_cheap:
        price_string = "cheap"
    elif temp_price_max == price_reasonable:
        price_string = "reasonable"
    else:
        price_string = "expensive"

    frequency_less = round(fuzz.interp_membership(frequency, less, input_frequency), 2)
    frequency_enough = round(fuzz.interp_membership(frequency, enough, input_frequency), 2)
    frequency_many = round(fuzz.interp_membership(frequency, many, input_frequency), 2)
    temp_frequency_max = max(frequency_less, frequency_enough, frequency_many)
    frequency_string = ""

    if temp_frequency_max == frequency_less:
        frequency_string = "less"
    elif temp_frequency_max == frequency_enough:
        frequency_string = "enough"
    else:
        frequency_string = "many"

    cleanness_dirty_output.append(cleanness_dirty)
    cleanness_normal_output.append(cleanness_normal)
    cleanness_clean_output.append(cleanness_clean)
    cleanness_max.append(temp_cleanness_max)
    cleanness_max_string.append(cleanness_string)

    price_cheap_output.append(price_cheap)
    price_reasonable_output.append(price_reasonable)
    price_expensive_output.append(price_expensive)
    price_max.append(temp_price_max)
    price_max_string.append(price_string)

    frequency_less_output.append(frequency_less)
    frequencyy_enough_output.append(frequency_enough)
    frequency_many_output.append(frequency_many)
    frequency_max.append(temp_frequency_max)
    frequency_max_string.append(frequency_string)

    degree = round(temp_cleanness_max * temp_price_max * temp_frequency_max, 2)
    output_degree.append(degree)

rule_set['cleanness_dirty'] = pd.Series(cleanness_dirty_output)
rule_set['cleanness_normal'] = pd.Series(cleanness_normal_output)
rule_set['cleanness_clean'] = pd.Series(cleanness_clean_output)
rule_set['cleanness_max'] = pd.Series(cleanness_max)
rule_set['cleanness_max_string'] = pd.Series(cleanness_max_string)

rule_set['price_cheap'] = pd.Series(price_cheap_output)
rule_set['price_reasonable'] = pd.Series(price_reasonable_output)
rule_set['price_expensive'] = pd.Series(price_expensive_output)
rule_set['price_max'] = pd.Series(price_max)
rule_set['price_max_string'] = pd.Series(price_max_string)

rule_set['frequency_less'] = pd.Series(frequency_less_output)
rule_set['frequency_enough'] = pd.Series(frequencyy_enough_output)
rule_set['frequency_many'] = pd.Series(frequency_many_output)
rule_set['frequency_max'] = pd.Series(frequency_max)
rule_set['frequency_max_string'] = pd.Series(frequency_max_string)

rule_set['degree'] = pd.Series(output_degree)
rule_set['output'] = pd.Series(train_output)

rule_set.head

# Export the analysis of each rule to the Training File#
rule_set.to_csv('data/rules.csv')

# set input and output ranges
new_cleanness = ctrl.Antecedent(np.arange(1, 11, 1), 'cleanness')
new_price = ctrl.Antecedent(np.arange(0.00, 5.01, 0.01), 'price')
new_frequency = ctrl.Antecedent(np.arange(1, 31, 1), 'frequency')
quality = ctrl.Consequent(np.arange(1, 11, 1), 'quality')

# fuzzification
new_cleanness['dirty'] = fuzz.trimf(new_cleanness.universe, [1, 1, 5])
new_cleanness['normal'] = fuzz.trimf(new_cleanness.universe, [1, 5, 10])
new_cleanness['clean'] = fuzz.trimf(new_cleanness.universe, [5, 10, 10])

new_price['cheap'] = fuzz.trimf(new_price.universe, [0, 0, 2.5])
new_price['reasonable'] = fuzz.trimf(new_price.universe, [0, 2.5, 5])
new_price['expensive'] = fuzz.trimf(new_price.universe, [2.5, 5, 5])

new_frequency['less'] = fuzz.trimf(new_frequency.universe, [1, 1, 10])
new_frequency['enough'] = fuzz.trapmf(new_frequency.universe, [5, 10, 15, 20])
new_frequency['many'] = fuzz.trapmf(new_frequency.universe, [12, 20, 30, 30])

quality['low'] = fuzz.trimf(quality.universe, [1, 1, 5])
quality['medium'] = fuzz.trimf(quality.universe, [1, 5, 10])
quality['high'] = fuzz.trimf(quality.universe, [5, 10, 10])

# Wang-Mendel Method rules
rule1 = ctrl.Rule(new_cleanness['clean'] & new_price['cheap'] & new_frequency['enough'], quality['medium'])
rule2 = ctrl.Rule(new_cleanness['normal'] & new_price['expensive'] & new_frequency['less'], quality['medium'])
rule3 = ctrl.Rule(new_cleanness['normal'] & new_price['reasonable'] & new_frequency['many'], quality['low'])
rule4 = ctrl.Rule(new_cleanness['normal'] & new_price['cheap'] & new_frequency['less'], quality['low'])
rule5 = ctrl.Rule(new_cleanness['clean'] & new_price['reasonable'] & new_frequency['enough'], quality['medium'])
rule6 = ctrl.Rule(new_cleanness['normal'] & new_price['reasonable'] & new_frequency['less'], quality['low'])
rule7 = ctrl.Rule(new_cleanness['clean'] & new_price['reasonable'] & new_frequency['many'], quality['high'])
rule8 = ctrl.Rule(new_cleanness['clean'] & new_price['cheap'] & new_frequency['many'], quality['high'])
rule9 = ctrl.Rule(new_cleanness['normal'] & new_price['cheap'] & new_frequency['enough'], quality['medium'])

rate_ctrl = ctrl.ControlSystem(
    [rule1, rule2, rule3, rule4, rule5, rule6, rule7, rule8, rule9])

file = pd.read_csv('data/fuzzy_part2_test.csv')

rating = file.rate_class.to_list()
price_list = file.price.to_list()
frequency_list = file.frequency.to_list()
cleanness_list = file.cleanness.to_list()

type_list = []
rating_score = []

for i in range(len(file)):
    rate_system = ctrl.ControlSystemSimulation(rate_ctrl)
    rate_system.input['cleanness'] = cleanness_list[i]
    rate_system.input['price'] = price_list[i]
    rate_system.input['frequency'] = frequency_list[i]

    # defuzzification using centre of gravity"
    try:
        rate_system.compute()
        score = rate_system.output['quality']
    except:
        score = 0
    rating_score.append(score)
    if score < 4:
        type_list.append("low")
    elif score < 8:
        type_list.append("medium")
    else:
        type_list.append("high")

final_file = file.copy().reset_index(drop=True)

final_file['predicted_class'] = pd.Series(type_list)
final_file['predicted_rating'] = pd.Series(rating_score)

count = 0
for i in range(len(file)):
    if type_list[i].strip() != rating[i].strip():
        count += 1

final_file.to_csv('data/result.csv', index=False)
total_input = len(file)
accuracy = (1 - float(count / total_input)) * 100

print("Evaluation: ", accuracy, "%")
