

import numpy as np
import matplotlib.pyplot as plt
import scipy
from scipy import signal
import copy
import csv
import os


from tkinter import *
from tkinter import ttk,messagebox
from tkinter import filedialog as fd
#from tkinter import messagebox
import ast

import scipy.optimize

import tkinter as tk
from tkinter import filedialog

from datetime import datetime

def sum_of_gauss(x,y,n,params):

    if n != len(params):
        print('Number of parameters sets must equal number of gaussians')
        return -1
    
    gauss = np.zeros(len(x))
    p0_out = []
    for i in range(0,n):
        p0,cov = scipy.optimize.curve_fit(gauss_fit,x,y,params[i])
        fit = gauss_fit(x,*p0)
        gauss += fit
        p0_out.append(p0)
    return gauss,p0_out

def gauss_fit(x,A0,A1,A2,A3):

    out = A0 + A1 * np.exp(-(x-A2)**2/(2*A3**2))

    return out

plt.close('all')
 
 # ------------------------- folder select -----------------#
root = tk.Tk() # Create a Tkinter root window
root.withdraw()  # Hide the root window
folder_path = filedialog.askdirectory(title="Select a folder with data") # Open a folder selection dialog
print(f"Selected folder: {folder_path}")
direc = folder_path

#--------------------- Plot folder name and number --------------#

name = os.path.basename(folder_path)

# Get the current date in the desired format (e.g., YYYY-MM-DD)
current_date = datetime.now().strftime('%m_%d_%Y')
folder_name = f"{name}_Plots_{current_date}"

#folder_name = name + '_Plots'

iii = 0 #plot number for save 
base_save_folder = folder_path # where folder will be created
save_folder = os.path.join(base_save_folder, folder_name)# folder name path



if not os.path.exists(save_folder): # Check if the folder already exists
    os.makedirs(save_folder)      # Create the folder if it doesn't exist
    print(f'Folder created: {save_folder}')
else:
    print(f'Folder already exists: {save_folder}')

def idl_boxcar(y,pts,*args,**kwargs):

    n_dimensions=y.ndims
    dimensions=y.shape

    real_width = pts
    half_width = real_width/2.

    temp_width = 1.

    output_array = np.zeros([dimensions])

    


def boxcar_con(y,pts):
    '''This is much faster and replicates the IDL smooth'''
    out = np.convolve(y,np.ones(pts)/pts,mode='same')
    out[0:int(pts/2)] = y[0:int(pts/2)]
    out[-int(pts/2):] = y[-int(pts/2):]

    return out


def boxcar_2d(y,pts):
    '''2d boxcar smoothing function'''
    out = signal.convolve2d(y,np.ones([pts,pts])/(pts**2),mode='same')

    out[0:int(pts/2)] = y[0:int(pts/2)]
    out[-int(pts/2):] = y[-int(pts/2):]
    out[:,0:int(pts/2)] = y[:,0:int(pts/2)]
    out[:,-int(pts/2):] = y[:,-int(pts/2):]

    return out





def read_fizeau_input_v2():

    file_default = direc
    dirName = os.path.basename(folder_path)
    #else:   
    #    dirName = file_default

    fle_print = open(direc+'defaults.txt','w')
    fle_print.write(file_default)
    fle_print.close()

    #First line is comment describing data set
    #ensure correct types for input variables

    comment = ' ' 
    nfTot = 0
    nf_start = 0
    nf_analyze = 0
    x = 0
    y = 0
    rad = 0
    sm = 1
    subtraction = 'N'

##-------------------------  Open a file----------------------#
    root = tk.Tk()
    root.withdraw()  # Hide the root window

    file_path = filedialog.askopenfilename(title="Select the settings file")
    print(f"Selected file: {file_path}")

    filein = file_path
  
    fle = open(filein,'r')
    comment = fle.readline()
    descript = comment
    nfTot = fle.readline()
    nframesTot = int(nfTot.split()[0])
    i = np.arange(nframesTot)
    nf_start = fle.readline()
    nf_analyze = fle.readline()
    nframes_start = int(nf_start.split()[0])
    nframes_analyze = int(nf_analyze.split()[0])
    x = fle.readline()
    y = fle.readline()
    x_center = int(x.split()[0])
    y_center = int(y.split()[0])
    rad = fle.readline()
    sm = fle.readline()
    radius = int(rad.split()[0])
    smoothing = int(sm.split()[0])
    subtraction = fle.readline()
    sub = subtraction.split()[0]

    #temp = []#np.zeros(2*nframesTot)
    wv = []
    pw = []
    blank = fle.readline()
    for line in fle.readlines():
        temp = line
        wv.append(float(temp.split()[0]))
        pw.append(float(temp.split()[1]))


    wavelength = np.asarray(wv)
    power = np.asarray(pw)

    fle.close()

    out = {'dirName':dirName,\
            'descript':descript,\
            'nframesTot':nframesTot,\
            'wavelength':wavelength,\
            'power':power,\
            'nframes_start':nframes_start,\
            'nframes_analyze':nframes_analyze,\
            'x_center':x_center,\
            'y_center':y_center,\
            'radius':radius,\
            'sm':smoothing,\
            'sub':sub}

    return out


def wavetuningfit(x,d,A):

    bx = A[0]+A[1]*np.cos(((2*np.pi)/x)*2*2.24*(d+A[2]))

    return bx

def unwrap_phase(data):

    decision = data.ndim

    if decision > 2:
        print('Array dimensions higher than 2 not supported')
        return -1

    jumps = data - np.roll(data,1)
    jumps[0] = 0.0
    j1 = (jumps >= 0.9*np.pi).astype(int)
    j2 = (jumps <= -0.9*np.pi).astype(int)
    jumps = j1 -j2
    jumpindex = np.where(jumps != 0)[0]
    if len(jumpindex) < 1:
        jumpindex = np.array([])
    result = data

    if len(jumpindex) != 0:
        for count in range(0,len(jumpindex)):
            result[jumpindex[count]:] = result[jumpindex[count]:] - 2*np.pi*jumps[jumpindex[count]]

    return result

#def Fizeau2D_fft_v9():

    #Fred has some stuff here defining common block and shared variable
    #Python doesn't use common blocks.  Can probably do something with
    #a class definition here.

n_unwrap = 17
radius = 0
x_center = 0
y_center = 0

##--------------------------- dir folder path---------------#
dir_new  = folder_path +'/'

rfi = read_fizeau_input_v2()

orient = 7
plan = 0
quiet = 0

for key_out in rfi.keys():
    print(key_out+':'+str(rfi[key_out]))

power_scale = 2.00/rfi['power']

start_wavelength = rfi['nframes_start']
lam_da = np.zeros([rfi['nframes_analyze']])
k = np.zeros([rfi['nframes_analyze']])
power_scale_factor = np.zeros([rfi['nframes_analyze']])

d = 1.0e6

ny_pixel = 2560#1392
nx_pixel = 2160#1040
pixel = np.zeros([ny_pixel,nx_pixel,rfi['nframes_analyze']])
pixel_raw = np.zeros([ny_pixel,nx_pixel,rfi['nframes_analyze']])
pix_sub = np.zeros([ny_pixel,nx_pixel])
pix = np.zeros([ny_pixel,nx_pixel,rfi['nframes_analyze']])
results = np.zeros([ny_pixel,nx_pixel,2])
#thickness = np.zeros([ny_pixel,nx_pixel])
#new_thickness = np.zeros([ny_pixel,nx_pixel])
phase_new = np.zeros([ny_pixel,nx_pixel])
phase_new1 = np.zeros([ny_pixel,nx_pixel])
#pixel_num = np.arange(2560,dtype=int)
pixel_num = np.arange(2160,dtype=int)
fle = []
    
for j in range(0,rfi['nframes_analyze']):
    print(j)
    #if j == rfi['nframes_analyze']-1:
    #    fle.append('105_Volt fringes.tif')
    #else:
    #    jj = 10*j
    #    fle.append(str(jj)+'_Volt fringes.tif')
    #j_get = rfi['nframes_start']+j
    #if j_get < 10:
    #    fle.append('00'+str(j_get)+'.tif')
    #else:
    #    fle.append('0'+str(j_get)+'.tif')
    try:   
        fle.append(str(j)+'_Volt fringes.tif')
        lam_da[j] = rfi['wavelength'][start_wavelength-1+j]
        power_scale_factor[j] = power_scale[start_wavelength-1+j]
        k[j] = 1./lam_da[j]
    except:
        pass

pix_out = np.zeros([ny_pixel,nx_pixel,16])
k_out = np.zeros([16])
min_k = np.min(k)
max_k = np.max(k)
delta_k = (max_k-min_k)/15.0
for j in range(0,16):
    k_out[j] = min_k+j*delta_k

if rfi['sub'] == 'Y':
    pix_sub = boxcar_2d(plt.imread(dir_new+'UW2_Background_Before.tif'),rfi['sm'])
else:
    pix_sub = np.zeros([ny_pixel,nx_pixel])

#pix_sub = 665.
#sm = 1.
for j in range(0,rfi['nframes_analyze']):
    pixel_raw[:,:,j] = boxcar_2d(plt.imread(dir_new+fle[j])-pix_sub,rfi['sm'])
    pixel[:,:,j] = power_scale_factor[j] * boxcar_2d(plt.imread(dir_new+fle[j])-pix_sub,rfi['sm'])

    
    """
    fig,ax = plt.subplots(1,1)
    ax.set_title(fle[j])

    pos = ax.imshow(pixel[:,:,j],cmap='nipy_spectral')
    pos.set_clim(0,np.max(pixel[:,:,j]))
    ax.invert_yaxis()
    fig.colorbar(pos,ax=ax)
    """

#Find boundary of region

col_start = np.zeros([ny_pixel],dtype=int)
col_stop = np.zeros([ny_pixel],dtype=int)
row_start = np.zeros([nx_pixel],dtype=int)
row_stop = np.zeros([nx_pixel],dtype=int)
rowstart = int(rfi['y_center'] - rfi['radius'])

print('Rowstart %0.d, %1.d, %2.d' %(rowstart, rfi['y_center'], rfi['radius']))
rowend = int(rfi['y_center']+rfi['radius'])
for j in range(rowstart,rfi['y_center']+1):
    col_start[j] = np.floor(rfi['x_center'] - np.sqrt(rfi['radius']**2 - (rfi['y_center']-j)**2))
    #print(j, col_start[j],np.sqrt(rfi['radius']**2 - (rfi['y_center']-j)**2))
    col_stop[j] = np.floor(rfi['x_center'] + np.sqrt(rfi['radius']**2 - (rfi['y_center']-j)**2))
    col_start[rowend+rowstart-j] = col_start[j]
    col_stop[rowend+rowstart-j] = col_stop[j]

x_array = np.zeros([2*(rowend-rowstart+1)])
y_array = np.zeros([2*(rowend-rowstart+1)])

for i in range(rowstart,rowend+1):
    x_array[(i-rowstart)*2] = col_start[i]
    x_array[2*(i-rowstart)+1] = col_stop[i]

    y_array[(i-rowstart)*2] = i
    y_array[2*(i-rowstart)+1] = i



#Show the image
fig,ax = plt.subplots(1,1)
ax.set_title(rfi['dirName'] + ' image')
pos = ax.imshow(pixel[:,:,0],cmap='nipy_spectral')
ax.invert_yaxis()
fig.colorbar(pos,ax=ax,cmap ='nipy_spectral')
ax.plot(x_array,y_array,'w,')

# Save the plot to the specified folder with a unique filename
save_path = os.path.join(save_folder, f'Plot_{iii+1}_Image.png')
plt.savefig(save_path, bbox_inches='tight', pad_inches=0)
iii = iii + 1 

for j in range(rfi['x_center']-rfi['radius'],rfi['x_center']+1):
    row_start[j] = np.floor(rfi['y_center'] - np.sqrt(rfi['radius']**2-(rfi['x_center']-j)**2))
    row_stop[j] = np.floor(rfi['y_center'] + np.sqrt(rfi['radius']**2-(rfi['x_center']-j)**2))
    row_start[rfi['x_center']*2-j] = row_start[j]
    row_stop[rfi['x_center']*2-j] = row_stop[j]

#End of boundary

min_t = 1.0e7

plt.figure()
plt.title('Fit')
for i in range(rowstart,rowend+1):
    for j in range(int(col_start[i]),int(col_stop[i]+1)):
        a_min = np.min(pixel[i,j])
        a_max = np.max(pixel[i,j])
        A0 = (a_min+a_max)/2.
        A1 = (a_max-a_min)/2.
        pix[i,j] = (pixel[i,j]-A0)/A1

for i in range(rowstart,rowend+1):
    for j in range(int(col_start[i]),int(col_stop[i]+1)):
        f = scipy.interpolate.interp1d(k,pix[i,j])
        pix_out[i,j] = f(k_out)
    
# --- interactive mode on ---
plt.ion()

# --- make a dedicated folder for fit plots ---
fit_folder = os.path.join(save_folder, "Fit_Plots")
os.makedirs(fit_folder, exist_ok=True)

# --- your existing processing code ---
plt.figure()
plt.title('Fit')

for i in range(rowstart, rowend + 1):
    for j in range(int(col_start[i]), int(col_stop[i] + 1)):
        a_min = np.min(pixel[i, j])
        a_max = np.max(pixel[i, j])
        A0 = (a_min + a_max) / 2.
        A1 = (a_max - a_min) / 2.
        pix[i, j] = (pixel[i, j] - A0) / A1

for i in range(rowstart, rowend + 1):
    for j in range(int(col_start[i]), int(col_stop[i] + 1)):
        f = scipy.interpolate.interp1d(k, pix[i, j])
        pix_out[i, j] = f(k_out)

# --- plotting loop ---
iii = 0
for i in range(rowstart, rowend + 1):
    plt.clf()
    plt.plot(k_out, pix_out[i, rfi['x_center']], label='Interpolated')
    plt.plot(k, pix[i, rfi['x_center']], 'r', label='Original')
    plt.ticklabel_format(axis='x', style='sci')
    plt.legend()
    plt.draw()
    plt.pause(0.2)

    # save each plot in the new folder
    save_path = os.path.join(fit_folder, f'Fit_{iii + 1}.png')
    plt.savefig(save_path, bbox_inches='tight', pad_inches=0)
    iii += 1

# --- wrap up ---
plt.ioff()
plt.close()

iii = iii + 1 

r = 1/pix_out.shape[2]*np.fft.fft(pix_out,axis=2)

plt.figure()
plt.title('Phase')
#plt.figure()
phase = np.arctan2(np.imag(r),np.real(r))
phase_2d = np.zeros([ny_pixel,nx_pixel])
phase_2d = phase[:,:,1]# + 1 #this is a cheat to remove the offset in the phase 7/27/24
plt.plot(phase_2d[rowstart:rowend+1,rfi['x_center']])

# Save the plot to the specified folder with a unique filename
save_path = os.path.join(save_folder, f'Plot_{iii+1}_Phase.png')
plt.savefig(save_path, bbox_inches='tight', pad_inches=0)
iii = iii + 1 
    
fig = plt.figure()
ax = fig.subplots(1,1)
ax.imshow(phase_2d,cmap='nipy_spectral')
ax.invert_yaxis()
ax.set_title(rfi['dirName'] +' Phase')

# Save the plot to the specified folder with a unique filename
save_path = os.path.join(save_folder, f'Plot_{iii+1}_Wrapped_Phase.png')
plt.savefig(save_path, bbox_inches='tight', pad_inches=0)
iii = iii + 1 


#Unwrap the phase
print('unwrap the phase')
######################################################
for l in range(0,n_unwrap):
    #phase_new = copy.copy(phase_2d)
    phase_new = phase_2d
    dat = np.zeros([2560])
    new_data = np.zeros([2560])
    dat[rfi['y_center']:rowend+1] = phase_2d[rfi['y_center']:rowend+1,rfi['x_center']]
    new_data[rfi['y_center']:rowend+1] = unwrap_phase(dat[rfi['y_center']:rowend+1])
    phase_new[rfi['y_center']:rowend+1,rfi['x_center']] = new_data[rfi['y_center']:rowend+1]

        
    for i in range(rfi['y_center'],rfi['y_center']+rfi['radius']+1):
        dat[rfi['x_center']:col_stop[i]+1] = phase_new[i,rfi['x_center']:col_stop[i]+1]
        new_data[rfi['x_center']:col_stop[i]+1] = unwrap_phase(dat[rfi['x_center']:col_stop[i]+1])
        phase_new[i,rfi['x_center']:col_stop[i]+1] = new_data[rfi['x_center']:col_stop[i]+1]

        
    for i in range(rfi['y_center'],rfi['y_center']+rfi['radius']+1):
        dat[rfi['x_center']:int(col_start[i]-1):-1] = phase_new[i,rfi['x_center']:int(col_start[i]-1):-1]
        new_data[rfi['x_center']:int(col_start[i]-1):-1] = unwrap_phase(dat[rfi['x_center']:int(col_start[i]-1):-1])
        phase_new[i,rfi['x_center']:int(col_start[i]-1):-1] = new_data[rfi['x_center']:int(col_start[i]-1):-1]

    phase_2d = phase_new

for l in range(0,n_unwrap):

    dat[rfi['y_center']:rowstart-1:-1] = phase_2d[rfi['y_center']:rowstart-1:-1,rfi['x_center']]
    new_data[rfi['y_center']:rowstart-1:-1] = unwrap_phase(dat[rfi['y_center']:rowstart-1:-1])
    phase_new[rfi['y_center']:rowstart-1:-1,rfi['x_center']] = new_data[rfi['y_center']:rowstart-1:-1]

        
    for i in range(rfi['y_center'],rfi['y_center']-rfi['radius'],-1):
        dat[rfi['x_center']:int(col_stop[i]+1)] = phase_new[i,rfi['x_center']:int(col_stop[i]+1)]
        new_data[rfi['x_center']:int(col_stop[i]+1)] = unwrap_phase(dat[rfi['x_center']:int(col_stop[i]+1)])
        phase_new[i,rfi['x_center']:int(col_stop[i]+1)] = new_data[rfi['x_center']:int(col_stop[i]+1)]
        
        
    for i in range(rfi['y_center'],rfi['y_center']-rfi['radius'],-1):
        dat[rfi['x_center']:int(col_start[i]-1):-1] = phase_new[i,rfi['x_center']:int(col_start[i]-1):-1]
        new_data[rfi['x_center']:int(col_start[i]-1):-1] = unwrap_phase(dat[rfi['x_center']:int(col_start[i]-1):-1])
        phase_new[i,rfi['x_center']:int(col_start[i]-1):-1] = new_data[rfi['x_center']:int(col_start[i]-1):-1]

    phase_2d = phase_new


offset = phase_new[rfi['y_center'],rfi['x_center']]
offset = np.min(phase_new)


for i in range(rowstart,rowend+1):
    for j in range(int(col_start[i]),int(col_stop[i]+1)):
        phase_new1[i,j] = phase_new[i,j]-offset

for i in range(rfi['x_center']-rfi['radius'],rfi['x_center']+1):
    phase_new1[int(row_start[i]),i] = 0.0
    phase_new1[int(row_stop[i]),i] = 0.0

for i in range(rfi['x_center'],rfi['x_center']+rfi['radius']+1):
    phase_new1[int(row_start[i]),i] = 0.0
    phase_new1[int(row_stop[i]),i] = 0.0            


# -------------------- plots in Pixels ---------------#
thickness = phase_new1 * lam_da[int(np.floor(rfi['nframes_analyze']/2))]/(4*np.pi*2.24)


"""
fig,ax = plt.subplots(1,1)
ax.set_title(rfi['dirName']+' Phase with offset')
pos = ax.imshow(phase_new1,cmap='nipy_spectral')
ax.invert_yaxis()
fig.colorbar(pos,ax=ax,label='Phase (radians)')
ax.plot(x_array,y_array,'w,')

# Save the plot to the specified folder with a unique filename
save_path = os.path.join(save_folder, f'Plot_{iii+1}_Phase_Offsett.png')
plt.savefig(save_path, bbox_inches='tight', pad_inches=0)
iii = iii + 1 
thickness = phase_new1 * lam_da[int(np.floor(rfi['nframes_analyze']/2))]/(4*np.pi*2.24)

fig,ax = plt.subplots(1,1)
ax.set_title(rfi['dirName']+' Thickness')
pos = ax.imshow(thickness,cmap='nipy_spectral')
ax.invert_yaxis()
fig.colorbar(pos,ax=ax,label='Thickness (nm)')

# Save the plot to the specified folder with a unique filename
save_path = os.path.join(save_folder, f'Plot_{iii+1}_Thickness.png')
plt.savefig(save_path, bbox_inches='tight', pad_inches=0)
iii = iii + 1 
 

plt.figure()
plt.plot(phase_new1[:,rfi['x_center']]*lam_da[int(np.floor(rfi['nframes_analyze']/2))]/(4*np.pi*2.24))
plt.title(rfi['dirName']+' vertical thickness')


# Save the plot to the specified folder with a unique filename
save_path = os.path.join(save_folder, f'Plot_{iii+1}_Vertical_Thickness.png')
plt.savefig(save_path, bbox_inches='tight', pad_inches=0)
iii = iii + 1 

plt.figure()
plt.plot(phase_new1[rfi['y_center']]*lam_da[int(np.floor(rfi['nframes_analyze']/2))]/(4*np.pi*2.24))
plt.title(rfi['dirName']+' horizontal thickness')


# Save the plot to the specified folder with a unique filename
save_path = os.path.join(save_folder, f'Plot_{iii+1}_Horizontal_Thickness.png')
plt.savefig(save_path, bbox_inches='tight', pad_inches=0)
iii = iii + 1 
"""


pix_to_mm = 16.85
x_cal = np.arange(2560)/pix_to_mm
y_cal = np.arange(2160)/pix_to_mm


#write output file

#dir2 = '/Users/mgalante/Documents/Nova/polishing/data/test/' + 'thickness_output_light_machine_test_v3.csv'

#with open(dir2,'w') as csvfile:
#    out = csv.writer(csvfile,delimiter=',')
#    out.writerows(thickness)

#return thickness

"""
button_run_code = ttk.Button(window,text='Read Fizeau Data',command=Fizeau2D_fft_v9)
button_run_code.pack(ipadx=5,pady=15)

window.mainloop()
"""



#-------------------------------------- Pixel to mm -------------------------#
pixel_radius = int(rfi['radius'])   # pixels
mask_radius = 98/2       # mm
# Define the conversion factor from pixels to millimeters
conversion_factor_x = (pixel_radius / mask_radius) 
conversion_factor_y = ((pixel_radius+2) / mask_radius) + 0

# Create a new axis for millimeters by multiplying the pixel positions
x_pixels = np.arange(thickness.shape[1])
y_pixels = np.arange(thickness.shape[0])

x_mm = x_pixels / conversion_factor_x
y_mm = y_pixels / (conversion_factor_y)

#int(rfi['y_center'] - rfi['radius'])
custom_center_pixel_x = int(rfi['x_center'])  # Set the center point on the x-axis (horizontal)
custom_center_pixel_y = int(rfi['y_center'])  # Set the center point on the y-axis (vertical)

# Create new x and y axes for millimeters, centered at the custom points
x_pixels = np.arange(thickness.shape[1])  # X-axis in pixels
y_pixels = np.arange(thickness.shape[0])  # Y-axis in pixels
x_shifted_mm = (x_pixels - custom_center_pixel_x) / conversion_factor_x  # Shift x-axis
y_shifted_mm = -(y_pixels - custom_center_pixel_y) / conversion_factor_y  # Shift y-axis

#--------------------- min value on plot------------------------#
array = thickness
center = (int(rfi['x_center']) , int(rfi['y_center']) )
radius = int(rfi['radius']) *0.98 #92mm radius mask is 98mm
rows, cols = array.shape

# Create an empty list to store non-zero values within the radius
values_within_radius = []
min_value = np.inf  # Start with a large number
min_location = None  # Location will be (row, col)

# Iterate through the array
for i in range(rows):
    for j in range(cols):
        # Calculate the Euclidean distance from the center point
        distance = np.sqrt((i - center[0])**2 + (j - center[1])**2)
        
        # Check if the distance is within the specified radius and value is non-zero
        if distance <= radius and array[i, j] != 0:
            # If this value is the smallest found so far, update min_value and min_location
            if array[i, j] < min_value:
                min_value = array[i, j]
                min_location = (i, j)
        
        # Check if the distance is within the specified radius
        if distance <= radius and array[i, j] != 0:
            # Append the non-zero value to the list
            values_within_radius.append(array[i, j])

# Find the minimum non-zero value within the radius if values exist
if values_within_radius:
    min_value = min(values_within_radius)
    print("Minimum within the radius:", min_value)
    print("Minimum location:", min_location)
    max_value = max(values_within_radius)
    TTV = max_value - min_value
    print("The TTV:", TTV)
else:
    print("No non-zero values found within the radius.")




#--------------------- thickness plot no min subtracted ------------#
""""
fig, ax = plt.subplots(1, 1)
ax.set_title(f"{rfi['dirName']}  Thickness Map in mm")

pos = ax.imshow(thickness, cmap='nipy_spectral', # Display the thickness map with shifted x and y axes
                extent=[x_shifted_mm[0], x_shifted_mm[-1], y_shifted_mm[-1], y_shifted_mm[0]])  # Flip y-axis

# Set the range for the x and y axes
x_range_min, x_range_max = -50, 50  # Define the range for x-axis (in mm)
y_range_min, y_range_max = -50, 50  # Define the range for y-axis (in mm)
ax.set_xlim(x_range_min, x_range_max)  # Set x-axis range
ax.set_ylim(y_range_min, y_range_max)  # Set y-axis range
ax.set_xlabel('Position X (mm)')
ax.set_ylabel('Position Y (mm)')
fig.colorbar(pos, ax=ax, label='Thickness (nm)')
plt.show()

# Save the plot to the specified folder with a unique filename
save_path = os.path.join(save_folder, f'Plot_{iii+1}_mm_Thickness.png')
plt.savefig(save_path, bbox_inches='tight', pad_inches=0)
iii = iii + 1 
"""

x_values = phase_new1[rfi['y_center']] * lam_da[int(np.floor(rfi['nframes_analyze']/2))] / (4 * np.pi * 2.24)
min_value_x = min([x for x in np.abs(x_values) if x != 0])
#new_x = x_values - min_value_x 

y_values = phase_new1[:,rfi['x_center']]*lam_da[int(np.floor(rfi['nframes_analyze']/2))]/(4*np.pi*2.24)
#y_values_range = y_values[custom_center_pixel_x - (pixel_radius-15):custom_center_pixel_x + (pixel_radius-15)]
min_value_y = min([x for x in np.abs(y_values) if x != 0])

min_total  = min(min_value_x, min_value_y)
TTV = max_value - min_total

new_x = np.where(x_values != 0, x_values - min_total, x_values)
max_value_x = np.max(np.abs(new_x))
new_y = np.where(y_values != 0, y_values - min_total, y_values)
max_value_y = np.max(np.abs(new_y))

plt.figure() # Plot the horizontal thickness with centered x-axis
plt.plot(x_shifted_mm, new_x)
plt.title(f"{rfi['dirName']} horizontal thickness\n98mm Clear Aperture\nMax = {max_value_x:.2f}nm")

range_min, range_max = -46.5, 46.5  # Define the range for x-axis
plt.xlim(range_min, range_max)   # Set x-axis range using plt.xlim()
plt.xlabel('Position (mm)')  # X-axis in millimeters
plt.ylabel('Thickness (nm)')  # Y-axis in thickness units

# Save the plot to the specified folder with a unique filename
save_path = os.path.join(save_folder, f'Plot_{iii+1}_mm_Horizontal_Thickness.png')
plt.savefig(save_path, bbox_inches='tight', pad_inches=0)
plt.show()

iii = iii + 1



plt.figure() # Plot the vertical thickness with centered y-axis
plt.plot(y_shifted_mm, new_y)
plt.title(f"{rfi['dirName']} vertical thickness\n98mm Clear Aperture\nMax = {max_value_y:.2f}nm")

plt.xlim(range_min, range_max)   # Set x-axis range using plt.xlim()
plt.xlabel('Position (mm)')  # X-axis in millimeters
plt.ylabel('Thickness (nm)')  # Y-axis in thickness units

# Save the plot to the specified folder with a unique filename
save_path = os.path.join(save_folder, f'Plot_{iii+1}_mm_Vertical_Thickness.png')
plt.savefig(save_path, bbox_inches='tight', pad_inches=0)
plt.show()

iii = iii + 1



#--------------------------------Excel of cross section data --------------------------#

import pandas as pd
import os

# Create DataFrames for each dataset
df_horizontal = pd.DataFrame({
    'X Position (mm)': x_shifted_mm,
    'Horizontal Thickness (nm)': x_values
})

df_vertical = pd.DataFrame({
    'Y Position (mm)': y_shifted_mm,
    'Vertical Thickness (nm)': y_values
})

# Define the Excel file path
excel_path = os.path.join(save_folder, f"Thickness_Data_{rfi['dirName']}.xlsx")

# Write both DataFrames to separate sheets in the same Excel file
with pd.ExcelWriter(excel_path, engine='openpyxl') as writer:
    df_horizontal.to_excel(writer, sheet_name='Horizontal Thickness', index=False)
    df_vertical.to_excel(writer, sheet_name='Vertical Thickness', index=False)


#--------------------------------TTV thickness map --------------------------#
#   min from one cross section is subtracted off the total map 
new_thickness  = np.where(thickness != 0, thickness - min_total, thickness)
fig, ax = plt.subplots(1, 1)
ax.set_title(f"{rfi['dirName']}  Thickness Map in mm\nMax TTV = {TTV:.2f}nm")

pos = ax.imshow(new_thickness, cmap='nipy_spectral', # Display the thickness map with shifted x and y axes
                extent=[x_shifted_mm[0], x_shifted_mm[-1], y_shifted_mm[-1], y_shifted_mm[0]])  # Flip y-axis

# Set the range for the x and y axes
x_range_min, x_range_max = -50, 50  # Define the range for x-axis (in mm)
y_range_min, y_range_max = -50, 50  # Define the range for y-axis (in mm)
ax.set_xlim(x_range_min, x_range_max)  # Set x-axis range
ax.set_ylim(y_range_min, y_range_max)  # Set y-axis range
ax.set_xlabel('Position X (mm)')
ax.set_ylabel('Position Y (mm)')
fig.colorbar(pos, ax=ax, label='Thickness (nm)')

# Save the plot to the specified folder with a unique filename
save_path = os.path.join(save_folder, f'Plot_{iii+1}_mm_TTV.png')
plt.savefig(save_path, bbox_inches='tight', pad_inches=0)
plt.show()

iii = iii + 1 


#-----------------------Total Thickness---------------------#
#this is to calc the center thickness of the wafer. This function will take 
# the images and plot the center intensity. 
# then a cos function will be fit to the varation in intencity and this
# is related to the wafer thickness 
##   the index used is 2.3. 
import matplotlib.ticker as ticker
from scipy.stats import linregress
from scipy.optimize import curve_fit

# Define a cosine function to fit
def cosine_func(x, A, B, C, D):
    """
    A: Amplitude
    B: Frequency
    C: Phase shift
    D: Vertical shift
    """
    return A * np.cos(B * x + C) + D


def plot_center_pixel_intensity(frames_array, center_x , center_y, wavelengths, iii, save_folder):
    center_intensities = []
    patch_size = 2

    for idx, frame in enumerate(frames_array):
        # Calculate the average intensity of the center patch_size^2 pixels
        center_patch = frame[center_y - patch_size:center_y + patch_size, center_x - patch_size:center_x + patch_size]
        center_intensity = np.mean(center_patch)
        center_intensities.append(center_intensity)

    # Initial guesses for the parameters (Amplitude, Frequency, Phase shift, Vertical shift)
    initial_guess = [50, 100, 0, 100]
    # Fit the data to the cosine function
    params, params_covariance = curve_fit(cosine_func, wavelengths, center_intensities, p0=initial_guess)
    # Generate the best-fit cosine curve
    fit_curve = cosine_func(wavelengths, *params)

    A=params[0] 
    B=params[1] 
    C=params[2]
    D=params[3]

    # Print the explicit function
    print(f"The best-fit function is: f(x) = {A:.2f} * cos({B:.2f} * x + {C:.2f}) + {D:.2f}")

    # Generate an extended range of wavelengths
    lamb_1 = np.linspace(400, 800, 200)  # Extended range
    # Generate the best-fit cosine curve over the extended range
    fit = A * np.cos(B * lamb_1 + C) + D

    # Fit the data to the cosine function
    n = 2.24  # Refractive index
    thickness = B / (4 * np.pi * n*3)
    print(f"Calculated Thickness: {thickness:.4f} mm")

    # Plot the center pixel intensities vs Wavelength
    plt.figure(figsize=(18, 4))
    plt.plot(wavelengths, center_intensities, marker='o', linestyle='', markersize=5)
    plt.plot(wavelengths, fit_curve, color='red', label=f'Cosine Fit: A*cos(Bx + C) + D\nA={params[0]:.4f}, B={params[1]:.4f}, C={params[2]:.4f}, D={params[3]:.4f}')
    plt.title(f"Average Intensity of Center Pixels vs Wavelength\nCalculated Thickness: {thickness:.4f} mm\nf(x) = {A:.2f} * cos({B:.2f} * x + {C:.2f}) + {D:.2f}")
    plt.xlabel('Wavelength (nm)') 
    plt.gca().xaxis.set_major_formatter(ticker.FormatStrFormatter('%.4f'))
    plt.ylabel('Intensity')
    plt.grid(True)
    plt.tight_layout()  # Adjust plot parameters for better layout
    plt.show()

    # Save the plot to the specified folder with a unique filename
    #save_path = os.path.join(save_folder, f'Plot_{iii+1}_Center_Intensity.png')
    #plt.savefig(save_path, bbox_inches='tight', pad_inches=0)
    #iii = iii + 1
    return center_intensities, fit_curve


def print_intensity_and_wavelength(frames_array, center_x, center_y, wavelengths):
    """Prints the average intensity and corresponding wavelength for each image."""
    patch_size = 2  # Size of the patch around the center
    print("\nImage Intensity and Wavelengths:")
    print("Index\tWavelength (nm)\tAverage Intensity")
    print("-" * 40)
    
    for idx, frame in enumerate(frames_array):
        # Calculate the average intensity of the center patch
        center_patch = frame[center_y - patch_size:center_y + patch_size, center_x - patch_size:center_x + patch_size]
        center_intensity = np.mean(center_patch)
        
        # Print the wavelength and corresponding intensity
        print(f"{idx}\t{wavelengths[idx]:.4f}\t\t{center_intensity:.2f}")


'''
####  ----------   Reselect Images for thickness calc
import cv2
def load_images(folder_path):
    # Load all .tif images from a folder, excluding files named "ref"
    images = []
    filenames = []
    for filename in sorted(os.listdir(folder_path)):
        if filename.endswith(".tif") and "ref" not in filename.lower():
            filepath = os.path.join(folder_path, filename)
            img = cv2.imread(filepath, cv2.IMREAD_GRAYSCALE)
            images.append(img)
            filenames.append(filename)
    return images, filenames

selected_folder = direc
wavelength = rfi['wavelength']
nframes_start = rfi['nframes_start']
nframes_analyze = rfi['nframes_analyze']
wavelength_subset = wavelength[nframes_start:nframes_start + nframes_analyze]
images, filenames = load_images(selected_folder)
images_subset = images[nframes_start:nframes_start + nframes_analyze]


print_intensity_and_wavelength(images_subset, rfi['x_center'], rfi['y_center'], wavelength_subset)
center_intensities, fit_curve = plot_center_pixel_intensity(images_subset, rfi['x_center'], rfi['y_center'], wavelength_subset, iii, save_folder)
'''

