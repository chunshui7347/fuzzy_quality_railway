import numpy as np
import matplotlib.pyplot as plt
import skfuzzy as fuzz
from utils import plot

if __name__ == '__main__':

    # initialize the membership graph
    cleanness = np.arange(1,11)
    price = np.arange(0.00,5.01,0.01)
    frequency = np.arange(1,31)
    quality = np.arange(1,11)

    dirty=fuzz.trimf(cleanness, [1,1,5])
    normal=fuzz.trimf(cleanness, [1,5,10])
    clean=fuzz.trimf(cleanness, [5,10,10])

    cheap=fuzz.trimf(price, [0,0,2.5])
    reasonable=fuzz.trimf(price, [0,2.5,5])
    expensive=fuzz.trimf(price, [2.5,5,5])

    less=fuzz.trimf(frequency, [1,1,10])
    enough=fuzz.trapmf(frequency, [5,10,15,20])
    many=fuzz.trapmf(frequency, [12,20,30,30])

    low=fuzz.trimf(quality, [1,1,5])
    medium=fuzz.trimf(quality, [1,5,10])
    high=fuzz.trimf(quality, [5,10,10])

    # graph plotting
    plot.plot_graph(cleanness,dirty,normal,clean, 'cleanness', 'membership function')
    plot.plot_graph(price, cheap, reasonable, expensive, 'price', 'membership function')
    plot.plot_graph(frequency, less, enough, many, 'frequency', 'membership function')
    plot.plot_graph(quality, low, medium, high, 'quality', 'membership function')

    # user input for testing
    test_expensive = float(input('What are the price?'))
    test_frequency = 60/int(input('What are the frequency?'))
    test_cleanness = int(input('What are the cleanness?'))

    # Rule 1: If the frequency is less the quality is low
    rule_1_less = fuzz.interp_membership(frequency, less, test_frequency)
    fire_rule1 = rule_1_less

    # Rule 2: If the cleanness is dirty the quality is low
    rule_2_dirty = fuzz.interp_membership(cleanness, dirty, test_cleanness)
    fire_rule2 = rule_2_dirty

    # Rule 3: If frequency is enough, cleanness is clean, then quality is medium
    rule_3_medium = fuzz.interp_membership(frequency, enough, test_frequency)
    rule_3_good = fuzz.interp_membership(cleanness, clean, test_cleanness)
    fire_rule3 = min(rule_3_medium, rule_3_good)

    # Rule 4: If frequency is many, pricing is expensive, then quality is medium
    rule_4_many = fuzz.interp_membership(frequency, many, test_frequency)
    rule_4_expensive = fuzz.interp_membership(price, expensive, test_expensive)
    fire_rule4 = min(rule_4_many, rule_4_expensive)

    # Rule 5: If frequency is many, cleanness is clean, price is cheap, then quality is high
    rule_5_many = fuzz.interp_membership(frequency, many, test_frequency)
    rule_5_expensive = fuzz.interp_membership(price, cheap, test_expensive)
    rule_5_clean = fuzz.interp_membership(cleanness, clean, test_cleanness)
    fire_rule5 = min(rule_5_many, rule_5_expensive,rule_5_clean)

    # Rule 6: If frequency is less, cleanness is normal, price is expensive, then quality is poor
    rule_6_less = fuzz.interp_membership(frequency, less, test_expensive)
    rule_6_expensive = fuzz.interp_membership(price, expensive, test_expensive)
    rule_6_clean = fuzz.interp_membership(cleanness, normal, test_cleanness)
    fire_rule6 = min(rule_6_less, rule_6_expensive,rule_6_clean)

    rule1_clip = np.fmin(fire_rule1, low)
    rule2_clip = np.fmin(fire_rule2, low)
    rule3_clip = np.fmin(fire_rule3, medium)
    rule4_clip = np.fmin(fire_rule4, medium)
    rule5_clip = np.fmin(fire_rule5, high)
    rule6_clip = np.fmin(fire_rule6, medium)

    # Aggregate the rules #
    temp1 = np.fmax(rule1_clip, rule2_clip)
    temp2 = np.fmax(temp1, rule3_clip)
    temp3 = np.fmax(temp2, rule4_clip)
    temp4 = np.fmax(temp3, rule5_clip)
    fuzzy_output = np.fmax(temp4, rule6_clip)

    # Defuzzification #
    quality_predict = fuzz.defuzz(quality, fuzzy_output, 'centroid')
    print ('the quality is', quality_predict)

    quality_activation = fuzz.interp_membership(quality, fuzzy_output, quality_predict)
    quality0 = np.zeros_like(quality)
    fig, ax0 = plt.subplots(figsize=(8, 3))
    ax0.plot(quality, low, 'b', linewidth=1.5, linestyle='--', )
    ax0.plot(quality, medium, 'g', linewidth=1.5, linestyle='--')
    ax0.plot(quality, high, 'r', linewidth=1.5, linestyle='--')
    ax0.fill_between(quality, quality0, fuzzy_output, facecolor='Orange', alpha=0.5)
    ax0.plot([quality_predict, quality_predict], [0, quality_activation], 'k', linewidth=2.5, alpha=0.9)
    ax0.get_xaxis().tick_bottom()
    ax0.get_yaxis().tick_left()
    ax0.set_xlim([min(quality),max(quality)])
    ax0.set_ylim([0,1])
    plt.xlabel('Quality')
    plt.ylabel('membership degree')
    plt.title('Quality problem')
    plt.show()

