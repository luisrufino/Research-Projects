import click
from click.utils import echo
import numpy as np
import math as ma
from click.decorators import option, version_option
from click.termui import prompt
import os
import sys
import contextlib
from numba import jit, njit, vectorize, objmode
import time
import multiprocessing as mp

## version 1.1
#np.set_printoptions(threshold=sys.maxsize)


fileout = []  # fileout either true or false
printbeta = []  # printbeta either true or false
FILE_PATH = 'C:/Users/Luis/Desktop/DM/MADHAT-master'  # file path to MADHAT-master
with open(FILE_PATH + '/Input/flags.dat') as flags:
    # With open is reading the flags file

    # Grabing the string vlaue for fileout
    lines = flags.read().splitlines()
    file_out = lines[8].replace("\t", " ")
    file_out = lines[8].split()
    fileout = file_out[1]

    # Grabing the string value for printbeta
    print_beta = lines[9].replace("\t", " ")
    print_beta = lines[9].split()
    printbeta = print_beta[1]


gal = 58

## Version 1.1 Numba implementation 
@click.group()
@click.version_option(version='1.1', prog_name='MADHAT')
def main():
    '''
    \b
    ************************************************************
    *                                                          *
    *   MADHAT: Model-Agnostic Dark Halo Analysis Tool (v1.0)  *
    *                                                          *
    ************************************************************
    \b
    OPTIONS:
    A: python3 MADHATpy.py a -target [dwarfset.dat] -beta [beta] -dm [model.in]
    B: python3 MADHATpy.py b -target [dwarfset.dat] -beta [beta] -mass [mass] -enrg [integrated spectrum]
    C: python3 MADHATpy.py c -target [dwarfset.dat] -beta [beta]

    '''
    pass

# This point on this is an example on how to add multiple comands

# add_user
# @main.command()
# @click.option('--name','-fn', prompt=True) ## fn stands for first name, and it's how you call the function
# @click.option('--name2','-ln', prompt=True)
# @click.option('--name3','-3rn', prompt=True)
# def add_user(name,name2,name3):
#     '''
#     Adding a user
#     '''
#     print(f'{name},{name2},{name3}')
# ## Lets try to add another function


# @main.command()
# @click.option('--beta','-beta', type=int) ## --beta = command parameter, -beta name of the command you are calling.
# def b(beta):
#     click.echo(beta)
@njit(fastmath = True)
def dist(Ndm, Nbound):
    # PMF of the Possion distribution
    prob = np.exp(Ndm * np.log(Nbound) - Nbound - ma.lgamma(Ndm + 1))
    return prob

dat_file = []
obs_data = []
PMF = []

@njit(fastmath = True, nogil=True)
def process(beta, INPUT, mass, energy, J_use=1, J_error=1, intype=1):
    cross_cons = 8 * np.pi * mass * mass / energy  # a constant for the cross section
    # INPUT coresponds to the dm target model ie. dwarfset.dat
    with objmode(dat_file = "float64[:,:]"):
        dat_file = np.loadtxt(INPUT)
    # dat_file = dwarfset.dat ex, set1.dat
    d_col = dat_file.shape

    if d_col[1] == 2:
        J_use = 1
        J_error = 0
    if d_col[1] >= 3:
        J_use = 1
        J_error = 1

    dwarf_list = [int(i) for i in dat_file[:,0]]
    dwarf_count = len(dat_file)
    dwarf_data = dat_file

    NOBS = FILE_PATH + '/PMFdata/Fermi_Pass8R3_239557417_585481831/NOBS.dat'

    with objmode(obs_data = "float64[:,:]"):
        obs_data = np.loadtxt(NOBS, usecols=(0, 1, 2))

    with objmode(PMF = "float64[:,:]"):    
        PMF = np.loadtxt(FILE_PATH + '/PMFdata/Fermi_Pass8R3_239557417_585481831/pmf.dat')
    # print(PMF[:,0])
    N_obs = 0
    for i in range(len(obs_data)):
        for j in range(len(dwarf_list)):
            if obs_data[i][0] == dwarf_list[j]:
                # Summing over all OBserved events ## code transelation Nobs = Nobs + obs_data[dwarf_list[i] - 1][1];
                N_obs = N_obs + obs_data[i][1]
    N_obs = int(N_obs)
    #print(N_obs)
    zeros_matx = np.zeros(((abs((N_obs + 2) - len(PMF))), (gal + 1)))
    PMF = np.concatenate((PMF,zeros_matx))

    ####################################
    # Calculating P1
    ####################################

    N_bdg = 0

    #P1 = []
    if dwarf_count == 1:
        # for i in range(N_obs + 1):
        #     P1.append(PMF[i][dwarf_list[0]]) ## Nobs + 1 length
        P1 = PMF[:,dwarf_list[0]]

    if dwarf_count != 1:
        P1 = np.zeros((N_obs + 2),dtype='float64')
        I = np.zeros((N_obs + 2),dtype='float64') #nobs + 2 length
        X = np.zeros((N_obs + 2),dtype='float64') #nobs + 2 length
        temp = 0
        I = PMF[:,dwarf_list[0]]
        # for i in range(N_obs + 1):
        #     #I[i] = PMF[i][dwarf_list[0]] ## The background galaxy
        #     I = PMF[:,dwarf_list[0]]
  

        for k in range(1,dwarf_count - 1):
            for j in range(N_obs + 2):
                for i in range(j + 1):
                    temp = (PMF[i][dwarf_list[k]] * I[j - i]) ## Equation 6 paper 1802.03826, the product 
                    X[j] = X[j] + temp ## Stacking the drawrf and computing the sum from equation 6
            for j in range(N_obs + 1):
                I[j] = X[j]
            for j in range(N_obs + 1):
                X[j] = 0
        for n in range(N_obs + 1):
            for k in range(n + 1):
                X[k] = PMF[n - k][dwarf_list[dwarf_count-1]] * I[k] ## Equation 8 paper 1802.03826, the product -LR
            for k in range(n + 1):
                P1[n] = P1[n] + X[k] ## Equation 8 paper 1802.03826, computing the sum -LR
                ## P1 the total expected photon distribution -LR
            N_bdg += 1

    P1 = P1[:-1]
    ###############################
    #    Convolution finsihed     #
    ###############################
    N_bdg = 0
    JAT = 0.0  # Variable for J factor times Aeff*Tobs
    JP = 0.0  # Variable for J factor times Aeff*Tobs + error
    JM = 0.0  # Variable for J factor times Aeff*Tobs -error
    SJAT = 0.0  # sum of J factors times Aeff*Tobs
    SJATP = 0.0  # +error sum of J factors times Aeff*Tobs
    SJATM = 0.0  # -error sum of J factors times Aeff*Tobs
    J = 0.0  # Jfactor
    if (J_use == 1):
        for i in range(dwarf_count):
            J = pow(10, dwarf_data[i][1])  # POW removes the log base 10
            # J multiplied by (Aeff * Tobs)
            JAT = J * obs_data[dwarf_list[i] - 1][2]
            SJAT = SJAT + JAT
            if (J_error == 1):
                # Undoing the log base 10 in +error (+dJ)
                JP = pow(10, dwarf_data[i][1] + dwarf_data[i][2])
                # Mulitplying JP with (Aeff * Tobs)
                JP = JP * obs_data[dwarf_list[i] - 1][2]
                SJATP = SJATP + JP  # summing all positive errors
                # Moving on to repate the process with the negative errors
                # Undoing the log base 10 in -error (-dJ)
                JM = pow(10, dwarf_data[i][1] - dwarf_data[i][3])
                # Mulitplying JM with (Aeff * Tobs)
                JM = JM * obs_data[dwarf_list[i] - 1][2]
                SJATM = SJATM + JM  # sum of al errors
    ### printing the hearder
    
    if (J_use == 1 and J_error == 1 and cross_cons != 0 and mass == 0):
        print(f"#Beta      Nbound        PhiPP(cm^3 s^-1 GeV^-2)   +dPhiPP             -dPhiPP            sigv(cm^3 s^-1)        +dsigv             -dsigv")
    if (J_use == 1 and J_error == 0 and cross_cons != 0 and mass == 0):
        print(f"#Beta      Nbound        PhiPP(cm^3 s^-1 GeV^-2)   sigv(cm^3 s^-1)")
    if (J_use == 1 and J_error == 1 and cross_cons == 0 and mass == 0):
        print(f"#Beta      Nbound        PhiPP(cm^3 s^-1 GeV^-2)   +dPhiPP             -dPhiPP")
    if (J_use == 1 and J_error == 0 and cross_cons == 0 and mass == 0):
        print(f"#Beta      Nbound        PhiPP(cm^3 s^-1 GeV^-2)")
    if (J_use == 0 and mass == 0):  # No J factors No J factor errors No DM mass
        print(f"#Beta      Nbound")
    if (J_use == 1 and J_error == 1 and cross_cons != 0 and mass != 0):
        print(f"#Mass(GeV)   Spectrum       Beta      Nbound        PhiPP(cm^3 s^-1 GeV^-2)   +dPhiPP             -dPhiPP            sigv(cm^3 s^-1)        +dsigv             -dsigv")
    if (J_use == 1 and J_error == 0 and cross_cons != 0 and mass != 0):
        print(f"#Mass(GeV)   Spectrum       Beta      Nbound        PhiPP(cm^3 s^-1 GeV^-2)   sigv(cm^3 s^-1)")
    if (J_use == 1 and J_error == 1 and cross_cons == 0 and mass != 0):
        print(f"#Mass(GeV)   Spectrum       Beta      Nbound        PhiPP(cm^3 s^-1 GeV^-2)   +dPhiPP             -dPhiPP")
    if (J_use == 1 and J_error == 0 and cross_cons == 0 and mass != 0):
        print(f"#Mass(GeV)   Spectrum       Beta      Nbound        PhiPP(cm^3 s^-1 GeV^-2)")
    if (J_use == 0 and mass != 0):  # No J factors No J factor errors No DM mass
        print(f"#Mass(GeV)      Spectrum       Beta      Nbound       ")


    # Has J factors Has J factor errors Has DM mass
    ##########################
    # Final summation loop
    ##########################
    B = 0  # Beta
    Pf = 0 # The complement of Beta
    for Nbound in range (1,N_obs + 1):
        if B < beta:
            N_bdg = N_obs ## Setting Ndm=Nobs
            Pf = 0 # Zeroing the placeholder for the sum
            for i in range(N_obs + 1): # Summing over Nbgd+Ndm<NObs
                P3 = 1
                PP = 0
                if P1[N_bdg] > 0:
                    for Ndm in range(i + 1): # Increasing NDM up till N_obs + 2
                        if P3 > 0 or PP == 0:
                            Nb = Nbound
                            P3 = dist(Ndm,Nbound) # Using the Possion distribution with all the Ndm, and Nbound values
                            Pf = Pf + (P1[N_bdg] * P3) # suming the completment of the beta
                            PP = PP + P3 ## Summing the Possion distribution
                N_bdg -= 1 ## Going down the list of P1[N_bdg]
            B = 1 - Pf
        Nbound = Nb
    

        if (Nbound == 1 and B - beta > 0):  
            print("Error: The value of beta you have entered is too small. Please choose a larger value of beta.\n")
    Nbound = Nb
    
    for Nbound in range(1,Nbound):
        if (printbeta == 'true' and intype != 3):
            PHI = Nbound  # Variable for PHI
            PHIP = Nbound  # Variable for PHI + error
            PHIM = Nbound  # Variable for PHI - error
            PHI = PHI/SJAT  # PHI = Nbound/JAT S stands for sum
            CS = PHI * cross_cons  # Cross section
            if J_error == 1:
                PHIP = PHIP/SJATP
                PHIM = PHIM/SJATM
                CSP = PHIP * cross_cons  # Cross section + error   
                CSM = PHIM * cross_cons  # Cross section - error
                PHIM = PHIM - PHI
                PHIP = -PHIP + PHI
                CSM = CSM - CS
                CSP = -CSP + CS
            with objmode():

                if (J_use == 1 and J_error == 1 and cross_cons != 0 and mass == 0):
                    #print(f"#Beta      Nbound        PhiPP(cm^3 s^-1 GeV^-2)   +dPhiPP             -dPhiPP            sigv(cm^3 s^-1)        +dsigv             -dsigv") ## notes so I can line up with the header
                    print("{:.4f}".format(B),"\t",      "{:.4f}".format(Nbound),"\t",        "{:.5E}",format(PHI),"\t",   "{:.5E}",format(PHIP),"\t",             "{:.5E}",format(PHIM),"\t",            "{:.5E}".format(CS),        "{:.5E}".format(CSP),"\t",             "{:.5E}".format(CSM))
                # Has J factors Has J factor errors Has DM mass
                if (J_use == 1 and J_error == 0 and cross_cons != 0 and mass == 0):
                    #print(f"#Beta      Nbound        PhiPP(cm^3 s^-1 GeV^-2)   sigv(cm^3 s^-1)")
                    print("{:.4f}".format(B) ,"\t",     "{:.4f}".format(Nbound)  ,"\t",      "{:.5E}".format(PHI) ,"\t",  "{:.5E}".format(CS))
                # Has J factors Has J factor errors No DM mass
                if (J_use == 1 and J_error == 1 and cross_cons == 0 and mass == 0):
                    #print(f"#Beta      Nbound        PhiPP(cm^3 s^-1 GeV^-2)   +dPhiPP             -dPhiPP")
                    print("{:.4f}".format(B),"\t",      "{:.4f}".format(Nbound),"\t",        "{:.5E}".format(PHI),"\t",   "{:.5E}".format(PHIM),"\t",             "{:.5E}".format(PHIP))
                # Has J factors No J factor errors No DM mass
                if (J_use == 1 and J_error == 0 and cross_cons == 0 and mass == 0):
                    #print(f"#Beta      Nbound        PhiPP(cm^3 s^-1 GeV^-2)")
                    print("{:.4f}".format(B),"\t",      "{:.4f}".format(Nbound),"\t",        "{:.5E}".format(PHI))
                if (J_use == 0 and mass == 0):  # No J factors No J factor errors No DM mass
                    #print(f"#Beta      Nbound")
                    print("{:.4f}".format(B),"\t",      "{:.4f}".format(Nbound))
                # Has J factors Has J factor errors Has DM mass
                if (J_use == 1 and J_error == 1 and cross_cons != 0 and mass != 0):
                    #print(f"#Mass(GeV)   Spectrum       Beta      Nbound        PhiPP(cm^3 s^-1 GeV^-2)   +dPhiPP             -dPhiPP            sigv(cm^3 s^-1)        +dsigv             -dsigv")
                    print("{:.4f}".format(mass),"\t",   "{:.4f}".format(energy),"\t",       "{:.4f}".format(B),"\t",     "{:.4f}".format(Nbound),"\t",        "{:.5E}".format(PHI),"\t",   "{:.5E}".format(PHIM),"\t",             "{:.5E}".format(PHIP),"\t",            "{:.5E}".format(CS),"\t",        "{:.5E}".format(CSP),"\t",             "{:.5E}".format(CSM))
                # Has J factors Has J factor errors Has DM mass
                if (J_use == 1 and J_error == 0 and cross_cons != 0 and mass != 0):
                    #print(f"#Mass(GeV)   Spectrum       Beta      Nbound        PhiPP(cm^3 s^-1 GeV^-2)   sigv(cm^3 s^-1)")
                    print("{:.4f}".format(mass),"\t",   "{:.4f}".format(energy),"\t",       "{:.4f}".format(B),"\t",      "{:.4f}".format(Nbound),"\t",        "{:.5E}".format(PHI),"\t",   "{:.5E}".format(CS))
                # Has J factors Has J factor errors No DM mass
                if (J_use == 1 and J_error == 1 and cross_cons == 0 and mass != 0):
                    #print(f"#Mass(GeV)   Spectrum       Beta      Nbound        PhiPP(cm^3 s^-1 GeV^-2)   +dPhiPP             -dPhiPP")
                    print("{:.4f}".format(mass),"\t",   "{:.4f}".format(energy),"\t",       "{:.4f}".format(B),"\t",      "{:.4f}".format(Nbound),"\t",        "{:.5E}".format(PHI),"\t",   "{:.5E}".format(PHIM),"\t",             "{:.5E}".format(PHIP))
                # Has J factors No J factor errors No DM mass
                if (J_use == 1 and J_error == 0 and cross_cons == 0 and mass != 0):
                    #print(f"#Mass(GeV)   Spectrum       Beta      Nbound        PhiPP(cm^3 s^-1 GeV^-2)")
                    print("{:.4f}".format(mass),"\t",   "{:.4f}".format(energy),"\t",       "{:.4f}".format(B),"\t",      "{:.4f}".format(Nbound),"\t",        "{:.5E}".format(PHI))
                if (J_use == 0 and mass != 0):  # No J factors No J factor errors No DM mass
                    #print(f"#Mass(GeV)      Spectrum       Beta      Nbound       ")
                    print("{:.4f}".format(mass),"\t",       "{:.4f}".format(energy),"\t",        "{:.4f}".format(B),"\t",       "{:.4f}".format(Nbound))
        Nbound = Nb
    # Final calculations
    Nbound = Nbound - 1 # Decrementing back one to find the bound
    B = 0
    # Loop counter that does the same as final summation but goes down decimals in precision
    for cut in range(1, 10):  # cut is the decimal point
        if (cut > 1):
            Nbound = Nbound + pow(10, -cut - 1)
        for acc in range(9):  # loops over .1 to .9
            if (B < beta):
                N_bdg = N_obs
                Pf = 0
                for i in range (N_obs + 1):
                    P3 = 1
                    PP = 0
                    if P1[N_bdg] > 0: 
                        for Ndm in range(i + 1): # Increasing NDM up till N_obs + 2
                            if P3 > 0 or PP == 0:
                                P3 = dist(Ndm, Nbound) # Using the Possion distribution with all the Ndm, and Nbound values
                                Pf = Pf + (P1[N_bdg] * P3) # suming the completment of the beta
                                PP = PP + P3 ## Summing the Possion distribution
                    N_bdg -= 1 ## Going down the list of P1[N_bdg]
                B = 1 - Pf
                Nbound = Nbound + pow(10,-cut) # raising -cut^10 from 1 to 10

    N_bdg = N_obs  # Setting N_bgd to N_obs
    Pf = 0  # Place holder for the sum
    for i in range(N_obs + 1):
        P3 = 1
        PP = 0
        if P1[N_bdg] > 0:
            for Ndm in range(i + 1):
                if (P3 > 0 or PP):
                    P3 = dist(Ndm, Nbound) # using the Possion distribution with all the Ndm, and Nbound values
                    # Summing over the probability
                    Pf = Pf + (P1[N_bdg] * P3) # suming the completment of the beta
                    PP = PP + P3 ## Summing the Possion distribution
        N_bdg -= 1
    B = 1 - Pf  # Taking the complement of the sum

    if (printbeta == "false" or intype == 3):
        PHI = Nbound  # Establishing PHI with a Nbound as a variable to solve for PHI
        PHIP = Nbound  # Establishing PHI with a Nbound as a variable to solve for PHI + error
        PHIM = Nbound  # Establishing PHI with a Nbound as a variable to solve for PHI - error
        PHI = PHI/SJAT  # PHI = Nbound/JAT S stands for sum
        CS = PHI * cross_cons  # Cross section
        if J_error == 1:
            PHIP = PHIP/SJATM
            PHIM = PHIM/SJATP
            CSP = PHIP * cross_cons  # Cross section + error
            CSM = PHIM * cross_cons  # Cross section - error
            PHIP = PHIP - PHI
            PHIM = -PHIM + PHI
            CSP = -1*(CS - CSP)
            CSM = -CSM + CS
        with objmode():

            if (J_use == 1 and J_error == 1 and cross_cons != 0 and mass == 0):
                #print(f"#Beta      Nbound        PhiPP(cm^3 s^-1 GeV^-2)   +dPhiPP             -dPhiPP            sigv(cm^3 s^-1)        +dsigv             -dsigv")
                #print(f"{B:.4f}      {Nbound:.4f}        {PHI:.5E}   {PHIP:.5E}             {PHIM:.5E}            {CS:.5E}        {CSP:.5E}             {CSM:.5E}")
                print("{:.4f}".format(B),"\t",      "{:.4f}".format(Nbound),"\t",        "{:.5E}",format(PHI),"\t",   "{:.5E}",format(PHIP),"\t",             "{:.5E}",format(PHIM),"\t",            "{:.5E}".format(CS),        "{:.5E}".format(CSP),"\t",             "{:.5E}".format(CSM))
            # Has J factors Has J factor errors Has DM mass
            if (J_use == 1 and J_error == 0 and cross_cons != 0 and mass == 0):
                #print(f"#Beta      Nbound        PhiPP(cm^3 s^-1 GeV^-2)   sigv(cm^3 s^-1)")
                #print(f"{B:.4f}      {Nbound:.4f}        {PHI:.5E}   {CS:.5E}")
                print("{:.4f}".format(B) ,"\t",     "{:.4f}".format(Nbound)  ,"\t",      "{:.5E}".format(PHI) ,"\t",  "{:.5E}".format(CS))
            # Has J factors Has J factor errors No DM mass

            if (J_use == 1 and J_error == 1 and cross_cons == 0 and mass == 0):
                #print(f"#Beta      Nbound        PhiPP(cm^3 s^-1 GeV^-2)   +dPhiPP             -dPhiPP")
                #print(f"{B:.4f}      {Nbound:.4f}        {PHI:.5E}   {PHIP:.5E}             {PHIM:.5E}")
                print("{:.4f}".format(B),"\t",      "{:.4f}".format(Nbound),"\t",        "{:.5E}".format(PHI),"\t",   "{:.5E}".format(PHIP),"\t",             "{:.5E}".format(PHIM))
            # Has J factors No J factor errors No DM mass
            if (J_use == 1 and J_error == 0 and cross_cons == 0 and mass == 0):
                #print(f"#Beta      Nbound        PhiPP(cm^3 s^-1 GeV^-2)")
                #print(f"{B:.4f}      {Nbound:.4f}        {PHI:.5E}")
                print("{:.4f}".format(B),"\t",      "{:.4f}".format(Nbound),"\t",        "{:.5E}".format(PHI))
            if (J_use == 0 and mass == 0):  # No J factors No J factor errors No DM mass
                #print(f"#Beta      Nbound")
                #print(f"{B:.4f}      {Nbound:.4f}")
                print("{:.4f}".format(B),"\t",      "{:.4f}".format(Nbound))
            # Has J factors Has J factor errors Has DM mass
            if (J_use == 1 and J_error == 1 and cross_cons != 0 and mass != 0):
                #print(f"#Mass(GeV)   Spectrum       Beta      Nbound        PhiPP(cm^3 s^-1 GeV^-2)   +dPhiPP             -dPhiPP            sigv(cm^3 s^-1)        +dsigv             -dsigv")
                #print(f"{mass:.4f}   {energy:.4f}       {B:.4f}      {Nbound:.4f}        {PHI:.5E}   {PHIP:.5E}             {PHIM:.5E}            {CS:.5E}        {CSP:.5E}             {CSM:.5E}")
                print("{:.4f}".format(mass),"\t",   "{:.4f}".format(energy),"\t",       "{:.4f}".format(B),"\t",     "{:.4f}".format(Nbound),"\t",        "{:.5E}".format(PHI),"\t",   "{:.5E}".format(PHIM),"\t",             "{:.5E}".format(PHIP),"\t",            "{:.5E}".format(CS),"\t",        "{:.5E}".format(CSP),"\t",             "{:.5E}".format(CSM))
            # Has J factors Has J factor errors Has DM mass
            if (J_use == 1 and J_error == 0 and cross_cons != 0 and mass != 0):
                #print(f"#Mass(GeV)   Spectrum       Beta      Nbound        PhiPP(cm^3 s^-1 GeV^-2)   sigv(cm^3 s^-1)")
                #print(f"{mass:.4f}   {energy:.4f}       {B:.4f}      {Nbound:.4f}        {PHI:.5E}   {CS:.5E}")
                print("{:.4f}".format(mass),"\t",   "{:.4f}".format(energy),"\t",       "{:.4f}".format(B),"\t",      "{:.4f}".format(Nbound),"\t",        "{:.5E}".format(PHI),"\t",   "{:.5E}".format(CS))
            # Has J factors Has J factor errors No DM mass
            if (J_use == 1 and J_error == 1 and cross_cons == 0 and mass != 0):
                #print(f"#Mass(GeV)   Spectrum       Beta      Nbound        PhiPP(cm^3 s^-1 GeV^-2)   +dPhiPP             -dPhiPP")
                #print(f"{mass:.4f}   {energy:.4f}       {B:.4f}      {Nbound:.4f}        {PHI:.5E}   {PHIP:.5E}             {PHIM:.5E}")
                print("{:.4f}".format(mass),"\t",   "{:.4f}".format(energy),"\t",       "{:.4f}".format(B),"\t",      "{:.4f}".format(Nbound),"\t",        "{:.5E}".format(PHI),"\t",   "{:.5E}".format(PHIP),"\t",             "{:.5E}".format(PHIM))
            # Has J factors No J factor errors No DM mass
            if (J_use == 1 and J_error == 0 and cross_cons == 0 and mass != 0):
                #print(f"#Mass(GeV)   Spectrum       Beta      Nbound        PhiPP(cm^3 s^-1 GeV^-2)")
                #print(f"{mass:.4f}   {energy:.4f}       {B:.4f}      {Nbound:.4f}        {PHI:.5E}")
                print("{:.4f}".format(mass),"\t",   "{:.4f}".format(energy),"\t",       "{:.4f}".format(B),"\t",      "{:.4f}".format(Nbound),"\t",        "{:.5E}".format(PHI))
            if (J_use == 0 and mass != 0):  # No J factors No J factor errors No DM mass
                #print(f"#Mass(GeV)      Spectrum       Beta      Nbound       ")
                #print(f"{mass:.4f}      {energy:.4f}       {B:.4f}      {Nbound:.4f}       ")
                print("{:.4f}".format(mass),"\t",       "{:.4f}".format(energy),"\t",        "{:.4f}".format(B),"\t",       "{:.4f}".format(Nbound))

    print('\n')
    Beta = B
    return Nbound

#########
##
# This is a duplicate of the main process if Nbound is already found
# which is the bulk of the Calculation. It exist for looping output
##
#########


dat_file = []
obs_data = []
PMF = []

@njit(fastmath = True, nogil=True)
def output(Nbound, B, INPUT, mass, energy, J_use=1, J_error=1, intype=1):
    cross_cons = 8 * np.pi * mass * mass / energy  # a constant for the cross section
    # INPUT coresponds to the dm target model ie. dwarfset.dat
    with objmode(dat_file = "float64[:,:]"):
        dat_file = np.loadtxt(INPUT)

    # dat_file = dwarfset.dat ex, set1.dat
    d_col = dat_file.shape
    if d_col[1] == 1:
        J_use = 0
        J_error = 0
    if d_col[1] >= 2:
        J_use = 1
        J_error = 1
    dwarf_list = [int(i) for i in dat_file[:,0]]

    dwarf_count = len(dat_file)
    dwarf_data = dat_file
    for i in range(dwarf_count):

        if dwarf_data[i][2] < 0:
            print("Error: Invalid +dJ for dwarfs in .dat file.\n")

        if dwarf_data[i][3] < 0:
            print("Error: Invalid +dJ for dwarfs in .dat file.\n")
            break

    NOBS = FILE_PATH + '/PMFdata/Fermi_Pass8R3_239557417_585481831/NOBS.dat'
    with objmode(obs_data = "float64[:,:]"):
        obs_data = np.loadtxt(NOBS)

    with objmode(PMF = "float64[:,:]"):    
        PMF = np.loadtxt(FILE_PATH + '/PMFdata/Fermi_Pass8R3_239557417_585481831/pmf.dat')    

    N_obs = 0
    for i in range(len(obs_data)):
        for j in range(len(dwarf_list)):
            if obs_data[i][0] == dwarf_list[j]:
                # Summing over all OBserved events ## code transelation Nobs = Nobs + obs_data[dwarf_list[i] - 1][1];
                N_obs = N_obs + obs_data[i][1]
    N_obs = int(N_obs)
    if N_obs > len(PMF):
        N_obs = int(len(PMF))
    else:
        N_obs
    dwarf_list = np.array(dwarf_list)
    # Prepping J factors for Calculations
    JAT = 0.0  # Variable for J factor times Aeff*Tobs
    JP = 0.0  # Variable for J factor times Aeff*Tobs + error
    JM = 0.0  # Variable for J factor times Aeff*Tobs -error
    SJAT = 0.0  # sum of J factors times Aeff*Tobs
    SJATP = 0.0  # +error sum of J factors times Aeff*Tobs
    SJATM = 0.0  # -error sum of J factors times Aeff*Tobs
    J = 0.0  # Jfactor
    if (J_use == 1):
        for i in range(dwarf_count):
            J = pow(10, dwarf_data[i][1])  # POW removes the log base 10

            # J multiplied by (Aeff * Tobs)
            JAT = J * obs_data[dwarf_list[i] - 1][2]
            SJAT = SJAT + JAT
            if J_error == 1:
                # Undoing the log base 10 in +error (+dJ)
                JP = pow(10, dwarf_data[i][1] + dwarf_data[i][2])
                # Mulitplying JP with (Aeff * Tobs)
                JP = JP * obs_data[dwarf_list[i] - 1][2]
                SJATP = SJATP + JP  # summing all positive errors
                # Moving on to repate the process with the negative errors
                # Undoing the log base 10 in -error (-dJ)
                JM = pow(10, dwarf_data[i][1] - dwarf_data[i][3])
                # Mulitplying JM with (Aeff * Tobs)
                JM = JM * obs_data[dwarf_list[i] - 1][2]
                SJATM = SJATM + JM  # sum of al errors
        PHI = Nbound  # Establishing PHI with a Nbound as a variable to solve for PHI
        PHIP = Nbound  # Establishing PHI with a Nbound as a variable to solve for PHI + error
        PHIM = Nbound  # Establishing PHI with a Nbound as a variable to solve for PHI - error
        CS = 0.0  # A variable for CS
        CSP = 0.0  # A variable for CS + error
        CSM = 0.0  # A variable for CS - error
        PHI = PHI/SJAT
        CS = PHI * cross_cons  # Cross section
        if J_error == 1:
            # Calculating PHI = Nbound/JAT along with PHI +/-
            # PHI = Nbound/JAT S stands for sum
            PHIP = PHIP/SJATM
            PHIM = PHIM/SJATP
            CSP = PHIP * cross_cons  # Cross section + error
            CSM = PHIM * cross_cons  # Cross section - error
            PHIP = PHIP - PHI
            PHIM = -PHIM + PHI
            CSP = -1*(CS - CSP)
            CSM = -CSM + CS
        with objmode():
        ## .4f 4 decimal place, .5E 5 decimal places 
        # Has J factors Has J factor errors Has DM mass
            if (J_use == 1 and J_error == 1 and cross_cons != 0 and mass == 0):
                #print(f"#Beta      Nbound        PhiPP(cm^3 s^-1 GeV^-2)   +dPhiPP             -dPhiPP            sigv(cm^3 s^-1)        +dsigv             -dsigv")
                #print(f"{B:.4f}      {Nbound:.4f}        {PHI:.5E}   {PHIP:.5E}             {PHIM:.5E}            {CS:.5E}        {CSP:.5E}             {CSM:.5E}")
                print("{:.4f}".format(B),"\t",      "{:.4f}".format(Nbound),"\t",        "{:.5E}",format(PHI),"\t",   "{:.5E}",format(PHIP),"\t",             "{:.5E}",format(PHIM),"\t",            "{:.5E}".format(CS),        "{:.5E}".format(CSP),"\t",             "{:.5E}".format(CSM))
            # Has J factors Has J factor errors Has DM mass
            if (J_use == 1 and J_error == 0 and cross_cons != 0 and mass == 0):
                #print(f"#Beta      Nbound        PhiPP(cm^3 s^-1 GeV^-2)   sigv(cm^3 s^-1)")
                #print(f"{B:.4f}      {Nbound:.4f}        {PHI:.5E}   {CS:.5E}")
                print("{:.4f}".format(B) ,"\t",     "{:.4f}".format(Nbound)  ,"\t",      "{:.5E}".format(PHI) ,"\t",  "{:.5E}".format(CS))
            # Has J factors Has J factor errors No DM mass
            if (J_use == 1 and J_error == 1 and cross_cons == 0 and mass == 0):
                #print(f"#Beta      Nbound        PhiPP(cm^3 s^-1 GeV^-2)   +dPhiPP             -dPhiPP")
                #print(f"{B:.4f}      {Nbound:.4f}        {PHI:.5E}   {PHIP:.5E}             {PHIM:.5E}")
                print("{:.4f}".format(B),"\t",      "{:.4f}".format(Nbound),"\t",        "{:.5E}".format(PHI),"\t",   "{:.5E}".format(PHIP),"\t",             "{:.5E}".format(PHIM))
            # Has J factors No J factor errors No DM mass
            if (J_use == 1 and J_error == 0 and cross_cons == 0 and mass == 0):
                #print(f"#Beta      Nbound        PhiPP(cm^3 s^-1 GeV^-2)")
                #print(f"{B:.4f}      {Nbound:.4f}        {PHI:.5E}")
                print("{:.4f}".format(B),"\t",      "{:.4f}".format(Nbound),"\t",        "{:.5E}".format(PHI))
            if (J_use == 0 and mass == 0):  # No J factors No J factor errors No DM mass
                #print(f"#Beta      Nbound")
                #print(f"{B:.4f}      {Nbound:.4f}")
                print("{:.4f}".format(B),"\t",      "{:.4f}".format(Nbound))
            # Has J factors Has J factor errors Has DM mass
            if (J_use == 1 and J_error == 1 and cross_cons != 0 and mass != 0):
                #print(f"#Mass(GeV)   Spectrum       Beta      Nbound        PhiPP(cm^3 s^-1 GeV^-2)   +dPhiPP             -dPhiPP            sigv(cm^3 s^-1)        +dsigv             -dsigv")
                #print(f"{mass:.4f}   {energy:.4f}       {B:.4f}      {Nbound:.4f}        {PHI:.5E}   {PHIP:.5E}             {PHIM:.5E}            {CS:.5E}        {CSP:.5E}             {CSM:.5E}")
                print("{:.4f}".format(mass),"\t",   "{:.4f}".format(energy),"\t",       "{:.4f}".format(B),"\t",     "{:.4f}".format(Nbound),"\t",        "{:.5E}".format(PHI),"\t",   "{:.5E}".format(PHIP),"\t",             "{:.5E}".format(PHIM),"\t",            "{:.5E}".format(CS),"\t",        "{:.5E}".format(CSP),"\t",             "{:.5E}".format(CSM))
            # Has J factors Has J factor errors Has DM mass
            if (J_use == 1 and J_error == 0 and cross_cons != 0 and mass != 0):
                #print(f"#Mass(GeV)   Spectrum       Beta      Nbound        PhiPP(cm^3 s^-1 GeV^-2)   sigv(cm^3 s^-1)")
                #print(f"{mass:.4f}   {energy:.4f}       {B:.4f}      {Nbound:.4f}        {PHI:.5E}   {CS:.5E}")
                print("{:.4f}".format(mass),"\t",   "{:.4f}".format(energy),"\t",       "{:.4f}".format(B),"\t",      "{:.4f}".format(Nbound),"\t",        "{:.5E}".format(PHI),"\t",        "{:.5E}".format(CS))
            # Has J factors Has J factor errors No DM mass
            if (J_use == 1 and J_error == 1 and cross_cons == 0 and mass != 0):
                #print(f"#Mass(GeV)   Spectrum       Beta      Nbound        PhiPP(cm^3 s^-1 GeV^-2)   +dPhiPP             -dPhiPP")
                #print(f"{mass:.4f}   {energy:.4f}       {B:.4f}      {Nbound}        {PHI:.5E}   {PHIP:.5E}             {PHIM:.5E}")
                print("{:.4f}".format(mass),"\t",   "{:.4f}".format(energy),"\t",       "{:.4f}".format(B),"\t",      "{:.4f}".format(Nbound),"\t",        "{:.5E}".format(PHI),"\t",   "{:.5E}".format(PHIP),"\t",             "{:.5E}".format(PHIM))
            # Has J factors No J factor errors No DM mass
            if (J_use == 1 and J_error == 0 and cross_cons == 0 and mass != 0):
                #print(f"#Mass(GeV)   Spectrum       Beta      Nbound        PhiPP(cm^3 s^-1 GeV^-2)")
                #print(f"{mass:.4f}   {energy:.4f}       {B:.4f}      {Nbound:.4f}        {PHI:.5E}")
                print("{:.4f}".format(mass),"\t",   "{:.4f}".format(energy),"\t",       "{:.4f}".format(B),"\t",      "{:.4f}".format(Nbound),"\t",        "{:.5E}".format(PHI))
            if (J_use == 0 and mass != 0):  # No J factors No J factor errors No DM mass
                #print(f"#Mass(GeV)      Spectrum       Beta      Nbound       ")
                #print(f"{mass:.4f}      {energy:.4f}       {B:.4f}      {Nbound:.4f}       ")
                print("{:.4f}".format(mass),"\t",   "{:.4f}".format(energy),"\t",       "{:.4f}".format(B),"\t",      "{:.4f}".format(Nbound))

    return 1

start = time.time()
@main.command()
@click.option('--beta', '-beta', type=float, help="A number between 0 and 1 that specifies the confidence level (EX: '-beta 0.95' for 95% CL)")
@click.option('--target', '-target', type=str, help="Dwarfset.dat INPUT file, EX: '-DAT set1.dat' \nA file containing the parameters for the dwarfs you'd like to analyze and must be located in the directory")
@click.option('--dm', '-dm', type=str, help=" [model.in] is a file containing a list of dark matter masses and integrated photon spectra as described below and must be located in the directory Input.\nMADHAT will read [model.in] and calculate output for each line until it reaches the end of the [model.in] file.")

def A(beta, target, dm):  # Option A from Madhat Wiki
    '''
    \b
    This option requires three arguments to run:
    -target [dwarfset.dat] -beta [beta] -dm [model.in]

    '''
    dmset = FILE_PATH + '/Input/' + target  # dwarfset.dat
    dmmass = FILE_PATH + '/Input/' + dm  # dmmodel.in
    mass_data = np.loadtxt(dmmass, usecols=0)  # Mass data from DM~~.in files
    # Spectrum data from DM~~.in files
    spec_data = np.loadtxt(dmmass, usecols=1)
    dat_file = np.loadtxt(dmset) ## INPUT = dwarfset.dat
    d_col = dat_file.shape

    if d_col[1] == 2:
        J_use = 1
        J_error = 0
    if d_col[1] >= 3:
        J_use = 1
        J_error = 1

    if(printbeta == "true"):
        print('Warning: The Output Will Not be a table of Nbound(beta)')
    # if not MASS:
    #     click.echo('Error: Failed to load dark matter parameters.\n')
    # THe link below will be for creating the output directory
    # https://djangocentral.com/check-if-a-directory-exists-if-not-create-it/#:~:text='''Check%20if%20directory%20exists,t%20exist%2C%20then%20create%20it.

    # Checking if the directory 'Output' exist
    CHECK_FOLDER = os.path.isdir('Output')
    # If directory 'Output' does not exist it creates a directory named 'Output'
    if not CHECK_FOLDER:
        os.makedirs('Output')
        os.chdir('Output')
    else:
        print(os.getcwd(), "Output folder exists.")
        os.chdir('Output')
    print("Working ...\nFile will be created shortly")

    print(os.getcwd())
    name = target.split('.')[0]  # Name of the dwarfset file ## type=string
    model = dm.split('.')[0]  # Name of the DM model file ## type=string

    # print(type(name))
    # print(model)
    out_file = model+"_"+name+"_"+str(beta)+".out"  # Creating the output file
    # print(out_file)

    

    head_count = 0
    data_count = 0
    header = ''
    data = ''
    # grabbing the header and it's contexts
    with open(dmset, 'r') as r:  # reading the dwarfset.dat file

        for i in r:
            line = i.split('\t')
            # print(line)
            if not line[0].isdigit():
                head_count += 1  # number of lines before numbers.
            if line[0].isdigit():
                data_count += 1

    with open(dmset, 'r') as r:  # reading the dwarfset.dat file
        for i in range(head_count):
            line = r.readline()
            header = header + line  # grabbing the header of any dwarfset.dat
        for i in range(data_count+1):
            line = r.readline()
            data = data + "#" + line
    # this exist to remove the extra line and # in the strings data and header
    data = "\n", data[0:len(data)-1]
    # this exist to remove the extra line and # in the strings data and header
    header = header[0:len(header)-1]

    fermi_header = "\n#####################################################################################################################################################################\n# Fermi_Pass8R3_239557417_585481831\n###################################################################################OUTPUT############################################################################\n"

    with open(out_file, 'w') as out:
        out.writelines(header)
        out.writelines(data)
        out.writelines(fermi_header)

        mass = mass_data[0]
        energy = spec_data[0]
        cross_cons = 8 * np.pi * mass * mass / energy
        Nbound = process(beta, dmset, mass, energy,
                         J_use=J_use, J_error=J_error, intype=3)
     
        if (J_use == 1 and J_error == 1 and cross_cons != 0 and mass == 0):
            out.writelines(f"#Beta      Nbound        PhiPP(cm^3 s^-1 GeV^-2)   +dPhiPP             -dPhiPP            sigv(cm^3 s^-1)        +dsigv             -dsigv\n")
        if (J_use == 1 and J_error == 0 and cross_cons != 0 and mass == 0):
            out.writelines(f"#Beta      Nbound        PhiPP(cm^3 s^-1 GeV^-2)   sigv(cm^3 s^-1)\n")
        if (J_use == 1 and J_error == 1 and cross_cons == 0 and mass == 0):
            out.writelines(f"#Beta      Nbound        PhiPP(cm^3 s^-1 GeV^-2)   +dPhiPP             -dPhiPP\n")
        if (J_use == 1 and J_error == 0 and cross_cons == 0 and mass == 0):
            out.writelines(f"#Beta      Nbound        PhiPP(cm^3 s^-1 GeV^-2)\n")
        if (J_use == 0 and mass == 0):  # No J factors No J factor errors No DM mass
            out.writelines(f"#Beta      Nbound\n")
        if (J_use == 1 and J_error == 1 and cross_cons != 0 and mass != 0):
            out.writelines(f"#Mass(GeV)   Spectrum       Beta      Nbound        PhiPP(cm^3 s^-1 GeV^-2)   +dPhiPP             -dPhiPP            sigv(cm^3 s^-1)        +dsigv             -dsigv\n")
        if (J_use == 1 and J_error == 0 and cross_cons != 0 and mass != 0):
            out.writelines(f"#Mass(GeV)   Spectrum       Beta      Nbound        PhiPP(cm^3 s^-1 GeV^-2)   sigv(cm^3 s^-1)\n")
        if (J_use == 1 and J_error == 1 and cross_cons == 0 and mass != 0):
            out.writelines(f"#Mass(GeV)   Spectrum       Beta      Nbound        PhiPP(cm^3 s^-1 GeV^-2)   +dPhiPP             -dPhiPP\n")
        if (J_use == 1 and J_error == 0 and cross_cons == 0 and mass != 0):
            out.writelines(f"#Mass(GeV)   Spectrum       Beta      Nbound        PhiPP(cm^3 s^-1 GeV^-2)\n")
        if (J_use == 0 and mass != 0):  # No J factors No J factor errors No DM mass
            out.writelines(f"#Mass(GeV)      Spectrum       Beta      Nbound       \n")
  
        for mass, spec in zip(mass_data, spec_data):
            # mass_data[mass] ## Changing the mass, This variabile will be called in the fucntion below.
            mass = mass
            energy = spec  # spec_data[spec]
            with contextlib.redirect_stdout(out): ## this contextlib.redirect_stdout allows me to write a function to a file
                output(Nbound, beta, dmset, mass,
                      spec, J_use=J_use, J_error=J_error, intype=3)

    print('\n')
    end = time.time()
    return 0, print(print("Elapsed (with compilation) = %s" % (end - start)))


@main.command()
@click.option('--target', '-target', type=str)
@click.option('--beta', '-beta', type=float)
@click.option('--mass', '-mass', type=float)
@click.option('--enrg', '-enrg', type=float)
def B(target, beta, mass, enrg):
    '''
    \b
    This option requires three arguments to run:
    -target [dwarfset.dat] -beta [beta] -mass [mass] -enrg [integrated spectrum]

    '''

    dmset = FILE_PATH + '/Input/' + target  # dwarfset.dat
    # INPUT = dwarfset.dat ## Target is our input, dwarfset.dat
    #cross_cons = 8 * np.pi * mass * mass / enrg  # a constant for the cross section
    
    name = target.split('.')[0]  # Name of the dwarfset file ## type=strin

    out_file = name+"_"+str(beta)+".out"
    head_count = 0
    data_count = 0
    header = ''
    data = ''
    with open(dmset, 'r') as r:
        for i in r:
            line = i.split('\t')
            # print(line)
            if not line[0].isdigit():
                head_count += 1  # number of lines before numbers.
            if line[0].isdigit():
                data_count += 1

    with open(dmset, 'r') as r:
        for i in range(head_count):
            line = r.readline()
            header = header + line  # grabbing the header of any dwarfset.dat
        for i in range(data_count+1):
            line = r.readline()
            data = data + "#" + line
    # this exist to remove the extra line and # in the strings data and header
    data = data[0:len(data)-1]
    # this exist to remove the extra line and # in the strings data and header
    header = header[0:len(header)-1]

    fermi_header = "\n#####################################################################################################################################################################\n# Fermi_Pass8R3_239557417_585481831\n###################################################################################OUTPUT############################################################################\n"

    # out_file = name+"_"+str(beta)+".out" ## Creating the output file. Option B does not create an output file

    # Checking if the directory 'Output' exist
    CHECK_FOLDER = os.path.isdir('Output')
    # If directory 'Output' does not exist it creates a directory named 'Output'
    if(fileout == 'true'):
        if not CHECK_FOLDER:
            os.makedirs('Output')
            os.chdir('Output')
        else:
            print(os.getcwd(), "Output folder exists.")
            os.chdir('Output')
            print(f"Working... File output {os.getcwd()}\{out_file}")
        with open(out_file, 'w') as out:
            out.writelines(header)
            out.writelines(data)
            out.writelines(fermi_header)
            with contextlib.redirect_stdout(out): ## this contextlib.redirect_stdout allows me to write a function to a file
                process(beta, dmset, mass, enrg, J_use=1, J_error=1,intype=1)


    # print(header)
    # print(data)
    # print(fermi_header)
    # process(beta, dmset, mass, enrg, J_use=1, J_error=1, intype=1)

    d_col = np.loadtxt(dmset).shape
    if d_col[1] == 1:
        J_use = 0
        J_error = 0
    if d_col[1] == 2:
        J_use = 1
        J_error = 0
    # Running the function
    #process(beta, dmset, mass, enrg, J_use=1, J_error=1)
    if(fileout == 'false'):
        print('Working ...')
        print('Output will be created shortly.')
        process(beta, dmset, mass,enrg, J_use=1, J_error=1, intype=3)
    return 0


@main.command()
@click.option('--target', '-target', type=str)
@click.option('--beta', '-beta', type=float)
def C(target, beta):
    '''
    \b
    This option requires three arguments to run:
    -target [dwarfset.dat] -beta [beta]

    '''
    dmset = FILE_PATH + '/Input/' + target
    mass = 0
    energy = 1
    cross_cons = 8 * np.pi * mass * mass / energy
    fermi_header = "\n#####################################################################################################################################################################\n# Fermi_Pass8R3_239557417_585481831\n###################################################################################OUTPUT############################################################################\n"
    name = target.split('.')[0]  # Name of the dwarfset file ## type=string
    # Creating the output file dwarfset name with beta value
    out_file = name+"_"+str(beta)+".out"

    head_count = 0
    data_count = 0
    header = ''
    data = ''
    dat_file = np.loadtxt(dmset)
    # dat_file = dwarfset.dat ex, set1.dat
    d_col = dat_file.shape

    if d_col[1] == 2:
        J_use = 1
        J_error = 0
    if d_col[1] >= 3:
        J_use = 1
        J_error = 1
    with open(dmset, 'r') as r:
        for i in r:
            line = i.split('\t')
            # print(line)
            if not line[0].isdigit():
                head_count += 1  # number of lines before numbers.
            if line[0].isdigit():
                data_count += 1

    with open(dmset, 'r') as r:
        for i in range(head_count):
            line = r.readline()
            header = header + line  # grabbing the header of any dwarfset.dat
        for i in range(data_count+1):
            line = r.readline()
            data = data + "#" + line

    # Checking if the directory 'Output' exist
    CHECK_FOLDER = os.path.isdir('Output')
    # If directory 'Output' does not exist it creates a directory named 'Output'
    if(fileout == 'true'):
        if not CHECK_FOLDER:
            os.makedirs('Output')
            os.chdir('Output')
        else:
            print(os.getcwd(), "Output folder exists.")
            os.chdir('Output')
        with open(out_file, 'w') as out:
            out.writelines(header)
            out.writelines(data)
            out.writelines(fermi_header)
            with contextlib.redirect_stdout(out): ## this contextlib.redirect_stdout allows me to write a function to a file
                process(beta, dmset, mass=0, energy=1, J_use=J_use, J_error=J_error, intype=1)
    if(fileout == 'false'):
        print('Working ...')
        print('Output will be created shortly.')
        process(beta, dmset, mass=0, energy=1, J_use=J_use, J_error=J_error, intype=3)
    #print(fermi_header)

    # Running the main function
    #process(beta, dmset, mass=0, energy=1, J_use=1, J_error=1, intype=1)
    return 0  # The end of the program.


if __name__ == '__main__':
    main()
    
    