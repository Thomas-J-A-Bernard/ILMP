import numpy as np

cosmo_param = {
    'Be_stds': {
        'KNSTD07': 1.0000,
        'KNSTD': 0.9042,
        'NIST_Certified': 1.0425,
        'LLNL31000': 0.8761,
        'LLNL10000': 0.9042,
        'LLNL3000': 0.8644,
        'LLNL1000': 0.9313,
        'LLNL300': 0.8562,
        'NIST30000': 0.9313,
        'NIST30200': 0.9251,
        'NIST30300': 0.9221,
        'NIST30600': 0.9130,
        'NIST27900': 1.0000,
        'S555': 0.9124,
        'S2007': 0.9124,
        'BEST433': 1.0000,
        'S555N': 1.0000,
        'S2007N': 1.0000
    },
    'Al_stds': {
        'KNSTD': 1.0000,
        'ZAL94': 0.9134,
        'AL09': 0.9134,
        'ZAL94N': 1.0000,
        'SMAL11': 1.0210,
        'Z92_0222': 1.0000
    },
    'lambda36Cl': 2.3028e-6,
    'lambda26Al': 9.83e-7,
    'lambda10Be': 4.998e-7,
    'Natoms10Be': 2.006e22,
    'Natoms26Al': 1.003e22,
    'sigma190_10Be': 0.094e-27/1.106,
    'sigma190_26Al': 1.41e-27,
    'delk_neg10Be': (0.704 * 0.1828 * 0.0003)/1.106,
    'k_neg10Be': ((0.704 * 0.1828)/1.106)*1.89e-3,
    'delk_neg26Al': 0.296 * 0.6559 * 0.002,
    'k_neg26Al': 0.296 * 0.6559 * 12.1e-3,
    'Ps10Be': {
        'de': 3.69,
        'du': 3.70,
        'li': 4.06,
        'lm': 4.00,
        'st': 4.01,
        'sf': 4.09,
        'sa': 3.92,
    },
    'Ps26Al': {
        'de': 26.3,
        'du': 27.3,
        'li': 28.7,
        'lm': 27.9,
        'st': 27.9,
        'sf': 28.6,
        'sa': 28.5,
    },
    'sigmaPs10Be': {
        'de': 0.58,
        'du': 0.58,
        'li': 0.57,
        'lm': 0.32,
        'st': 0.33,
        'sf': 0.35,
        'sa': 0.31,
    },
    'sigmaPs26Al': {
        'de': 4.3,
        'du': 4.4,
        'li': 4.6,
        'lm': 2.7,
        'st': 2.8,
        'sf': 3.3,
        'sa': 3.1,
    },
    'delsigma190_10Be': 0.252e-30,
    'delsigma190_26Al': 4.03e-30,
}

if __name__ == '__main__':
    
    with open("myfile.txt", 'w') as f: 
        for key, value in cosmo_param.items(): 
            f.write('%s: %s\n' % (key, value))