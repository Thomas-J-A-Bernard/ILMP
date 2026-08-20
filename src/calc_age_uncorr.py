import numpy as np

def CalculateHeliumAgeUncorrected(mineral, He, He_SD, U238, U235, U_SD, Th232, Th_SD, Sm147, Sm_SD):
    
    # Constants
    lambda238 = 1.55125E-10
    lambda235 = 9.84500E-10
    lambda232 = 4.94750E-11
    lambda147 = 6.53900E-12
    
    uncor_age = np.zeros(1000)
    U238_rand = np.random.normal(U238, U238 * U_SD / 100, 1000)
    U235_rand = np.random.normal(U235, U235 * U_SD / 100, 1000)
    Th232_rand = np.random.normal(Th232, Th232 * Th_SD / 100, 1000)
    Sm147_rand = np.random.normal(Sm147, Sm147 * Sm_SD / 100, 1000)
    He_rand = np.random.normal(He, He_SD, 1000)
    
    if mineral == 'apatite':
        
        production = 8*lambda238*U238_rand + 7*lambda235*U235_rand + 6*lambda232*Th232_rand + 1*lambda147*Sm147_rand
        lambda_weight = (8*lambda238**2*U238_rand + 7*lambda235**2*U235_rand + 6*lambda232**2*Th232_rand + 1*lambda147**2*Sm147_rand)/production
        uncor_age = 1/lambda_weight*np.log(lambda_weight/production*He_rand + 1)/1e6
        uncor_age_mean = np.mean(uncor_age)
        uncor_age_SD = np.std(uncor_age)
        
    elif mineral == 'zircon':
        
        production = 8*lambda238*U238_rand + 7*lambda235*U235_rand + 6*lambda232*Th232_rand
        lambda_weight = (8*lambda238**2*U238_rand + 7*lambda235**2*U235_rand + 6*lambda232**2*Th232_rand)/production
        uncor_age = 1/lambda_weight*np.log(lambda_weight/production*He_rand + 1)/1e6
        uncor_age_mean = np.mean(uncor_age)
        uncor_age_SD = np.std(uncor_age)
        
    else:
        
        uncor_age_mean = 0
        uncor_age_SD = 0
        
    return uncor_age_mean, uncor_age_SD

if __name__ == '__main__':
    
    mean_age, sd_age = CalculateHeliumAgeUncorrected('apatite', 10, 0, 10, 10, 0, 10, 0, 10, 0)