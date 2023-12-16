import numpy as np
import matplotlib.pyplot as plt

from linalg_interp import spline_function

def main():

    #loading the datasets into the program
    water = np.loadtxt("/Users/adriansmac/Documents/School/Fall2023/GOPH419/Lab2/GOPH419Lab2Smit/GOPH419Lab2Smit/h2o_density_vs_temp.txt")
    air = np.loadtxt("/Users/adriansmac/Documents/School/Fall2023/GOPH419/Lab2/GOPH419Lab2Smit/GOPH419Lab2Smit/air_density.txt")

    s1_water = spline_function(water[:, 0], water[:, 1], order = 1) #compute first order spline function
    s2_water = spline_function(water[:, 0], water[:, 1], order = 2) #compute seecond order spline function
    s3_water = spline_function(water[:, 0], water[:, 1], order = 3) #compute third order spline function
    
    s1_air = spline_function(air[:, 0], air[:, 1], order = 1) #compute first order spline function for the air data
    s2_air = spline_function(air[:, 0], air[:, 1], order = 2) #compute second order spline function for the air data
    s3_air = spline_function(air[:, 0], air[:, 1], order = 3) #compute third order spline function for the air data
 
    Tw = np.linspace(water[0, 0] ,water[-1, 0], 100) #initialize T vector for water data
    Ta = np.linspace(air[0, 0], air[-1, 0], 100) #initialize T vector for air data

    water1 = np.array([s1_water(T) for T in Tw]) #first order spline interpolation for water data with T vector
    water2 = np.array([s2_water(T) for T in Tw]) #second order spline interpolation for water data with T vector
    water3 = np.array([s3_water(T) for T in Tw]) #third order spline interpolation for water data with T vector

    air1 = np.array([s1_air(T) for T in Ta]) #first order spline interpolation for water data with T vector
    air2 = np.array([s2_air(T) for T in Ta]) #second order spline interpolation for water data with T vector
    air3 = np.array([s3_air(T) for T in Ta]) #third order spline interpolation for water data with T vector

    #plotting figures and setting figure parameters for first, second, and third order water graphs
    plt.figure(figsize=(12, 12))
    plt.subplot(3, 2, 1)
    plt.title("Interpolated water density vs tempature")
    plt.plot(water[:, 0], water[:, 1], linestyle = "none", marker = "x", markersize = 7, markeredgecolor = "red", label = "Water Data")
    plt.plot(Tw, water1, linestyle = "dashed", color = "black", linewidth = 2, label = "Linear Interpolation")
    plt.legend()
    plt.ylabel("Density (kg/m^3)")
    plt.xlabel("Temperature (*C)")

    plt.subplot(3,2,3)
    plt.plot(water[:, 0], water[:, 1], linestyle = "none", marker = "x", markersize = 7, markeredgecolor = "red", label = "Water Data")
    plt.plot(Tw, water2, linestyle = "dashed", color = "black", linewidth = 2, label = "Quadratic Interpolation")
    plt.legend()
    plt.ylabel("Density (kg/m^3)")
    plt.xlabel("Temperature (*C)")

    plt.subplot(3,2,5)
    plt.plot(water[:, 0], water[:, 1], linestyle = "none", marker = "x", markersize = 7, markeredgecolor = "red", label = "Water Data")
    plt.plot(Tw, water3, linestyle = "dashed", color = "black", linewidth = 2, label = "Cubic Interpolation")
    plt.legend()
    plt.ylabel("Density (kg/m^3)")
    plt.xlabel("Temperature (*C)")

    #plotting figures and setting figure parameters for first, second, and third order air graphs
    plt.subplot(3, 2, 2) #plotting first order 
    plt.title("Interpolated air density vs tempature")
    plt.plot(air[:, 0], air[:, 1], linestyle = "none", marker = "x", markersize = 7, markeredgecolor = "red", label = "Air Data")
    plt.plot(Ta, air1, linestyle = "dashed", color = "black", linewidth = 2, label = "Linear Interpolation")
    plt.legend()
    plt.ylabel("Density (kg/m^3)")
    plt.xlabel("Temperature (*C)")

    plt.subplot(3, 2, 4)
    plt.plot(air[:, 0], air[:, 1], linestyle = "none", marker = "x", markersize = 7, markeredgecolor = "red", label = "Air Data")
    plt.plot(Ta, air2, linestyle = "dashed", color = "black", linewidth = 2, label = "Quadratic Interpolation")
    plt.legend()
    plt.ylabel("Density (kg/m^3)")
    plt.xlabel("Temperature (*C)")

    plt.subplot(3, 2, 6)
    plt.plot(air[:, 0], air[:, 1], linestyle = "none", marker = "x", markersize = 7, markeredgecolor = "red", label = "Air Data")
    plt.plot(Ta, air3, linestyle = "dashed", color = "black", linewidth = 2, label = "Cubic Interpolation")
    plt.legend()
    plt.ylabel("Density (kg/m^3)")
    plt.xlabel("Temperature (*C)")
    plt.savefig("Figures/densities.png") 

if __name__ == "__main__":
    main()