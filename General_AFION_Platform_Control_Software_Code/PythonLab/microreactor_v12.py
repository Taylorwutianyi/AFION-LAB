# Import required libraries
import numpy as np
import matplotlib.pyplot as plt
import time, os
import sys, traceback
from scipy.signal import savgol_filter
from scipy import optimize
from numpy import trapz

# Import modules from the custom package
from PythonLab.reset_instru import *
import PythonLab.pylab.instruments as ins
from PythonLab.pylab.instruments import pv_const

# Define constants
ul = 1e-6
OSCI_VOL = 0.2 #oscillation volume inside reactor
THRESH_WIDTH = 21
THRESH_INTENSITY = 0.005
START_WL1 = 400 # Starting wavelength 1
START_WL2 = 480 # Starting wavelength 2
WL_RANGE1 = 600 # Spectrometer wavelength range 1 (CCS200 spectrometer, 200-1000nm)
WL_RANGE2 = 520 # Spectrometer wavelength range 2
FULL_STORKE_TIME = 70  # Time for full pump stroke (adjust as needed)

# Initialize instruments
spectrometer = ins.ThorlabsCCS(r'USB0::0x1313::0x8081::M00548012::RAW') # Initialize Thorlabs spectrometer
ard = ins.ard('COM13') # Initialize Arduino
p = ins.PumpPSD('ASRL14::INSTR', 0) # Initialize primary pump
op = ins.PumpPSD('ASRL15::INSTR', 0) # Initialize oscillation pump

# Initialize primary pump and oscillation pump settings
p.init_syringe(init_valve = 1, syringe_volume = 100 * ul, psdtype = 'psd8')
op.init_syringe(init_valve = 7, syringe_volume = 2500 * ul, psdtype = 'psd6')
op.set_velocity(op.get_max_steps() * 2 // FULL_STORKE_TIME) # Set oscillation pump velocity
time.sleep(2) # Wait for 2 seconds

# Draw fluid using oscillation pump
op.draw(volume=2400 * ul, velocity = 1200)

# Define a Gaussian function
def gaussian(x, height, center, sigma, offset):
	return height*np.exp(-(x - center)**2/(2*sigma**2)) + offset

# Define a Lorentzian function
def lorentzian(x,height,center,gamma,offset):
	return height*gamma**2/((x-center)**2+gamma**2) + offset

# Define a function to calculate the inscribed error for fitting
def inscribed_error(p, offset, height, x,  y):
	res = []
	g = gaussian(x, height, p[1], p[0], offset)
	for i in range(len(x)):
		if (y[i] >= g[i]):
			res.append(y[i]-g[i])
		else:
			res.append(5*(y[i]-g[i]))
	return (np.asarray(res))

# Define a function for Gaussian-Lorentzian fitting
def glfit(wl,abs,peaklist,area):
	fittedpeaklist = []
	shifted_abs = abs

	# multi-peak
	for peaki in peaklist:
		peak_center = wl[peaki[0]]
		peak_width = max(peaki[2] - peaki[0], peaki[0] - peaki[1])

		peak_height = np.around(shifted_abs[peaki[0]],decimals=5)
		peak_height_rela = np.around(shifted_abs[peaki[0]] - shifted_abs[peaki[1]],decimals=5)

		x = [peak_width, peak_center]
		wll = max(peaki[0] - peak_width, 0)
		wlr = min(peaki[0] + peak_width, 800)
		try:

			res = optimize.least_squares(inscribed_error,x,args=(0, peak_height, wl[wll:wlr], shifted_abs[wll:wlr]),bounds=([0,peak_center - 2],[np.inf,peak_center + 2]))
			sigma = np.around(res.x[0],decimals=2)
			peak_center = np.around(res.x[1],decimals=0)
			std_fit = np.around(res.cost / peak_height_rela ** 2,decimals=5)

		except TypeError:
			print("unable to fit\n")
			sigma = x[0]
			peak_center = x[1]
			std_fit = np.around(0.5 * np.sum(inscribed_error(x, 0, peak_height, wl[wll:wlr], shifted_abs[wll:wlr]) ** 2) / peak_height_rela ** 2,decimals=5)

		plt.plot(wl[wll:wlr], gaussian(wl[wll:wlr], peak_height, peak_center, sigma, 0),'k')

		optim = [peak_height, peak_center, sigma * 2.35, std_fit, area]

		fittedpeaklist.append(optim)

	return (fittedpeaklist)

# Define a function to extract peak information
def peakinfo(wl,abs):
	peaks_info = []
	maxinten = 0
	abs = np.nan_to_num(abs)

	# Find local maxima and minima
	minmax = [0] * len(abs)
	for i in range(1, len(abs) - 1):

		if (abs[i - 1] >= abs[i - 2] and abs[i] >= abs[i - 1] and abs[i] > abs[i + 1]):
			minmax[i] = 1

		if (abs[i] < abs[i-1] and abs[i]< abs[i+1]):
			minmax[i] = -1

	# non-fit method
	i = 0
	while (i < len(abs)):
		# print (maxinten)
		if (minmax[i] == 1):
			maxintenl = maxintenr = 0
			l = int(i - THRESH_WIDTH / 2)
			while (l > 0):
				if (minmax[l] == -1):
					maxintenl = max(maxintenl, abs[i] - abs[l])
				if (abs[l] > abs[i] or (minmax[l] == -1 and abs[i] - abs[l] > THRESH_INTENSITY)):
					break
				l -= 1
			if (maxintenl == 0 and l >= 0):
				maxintenl = abs[i] - abs[l]
			r = int(i + THRESH_WIDTH / 2)
			while (r < len(abs) - 1):
				if (minmax[r] == -1):
					maxintenr = max(maxintenr, abs[i] - abs[r])
				if (abs[r] > abs[i] or (minmax[r] == -1 and abs[i] - abs[r] > THRESH_INTENSITY)):
					break
				r += 1
			if (maxintenr == 0 and r < len(abs)):
				maxintenr = abs[i] - abs[r]
			maxinten = max(maxinten, min(maxintenl, maxintenr))
			if (l >= 0 and r < len(abs) and abs[i] - abs[l] > THRESH_INTENSITY and abs[i] - abs[r] > THRESH_INTENSITY):
				# print(wl[i],wl[l],wl[r])
				# center = wl[i]
				height = abs[i] - min(abs[l], abs[r])
				j = int(i - THRESH_WIDTH / 2)
				while (j > 0):
					if (np.abs(abs[j] - min(abs[l], abs[r])) < height / 10):
						lh = wl[i] - wl[j]
						break
					j -= 1
				k = int(i + THRESH_WIDTH / 2)
				while (k < len(abs) - 1):
					if (np.abs(abs[k] - min(abs[l], abs[r])) < height / 10):
						rh = wl[k] - wl[i]
						break
					k += 1
				if (j >= 0 and k < len(abs)):
					peaks_info.append([i,j,k])
					i += int(THRESH_WIDTH / 2)
		i += 1
#
	maxinten = np.around(maxinten, decimals=5)
	return (peaks_info, maxinten)

# Function to measure the spectrum
def specm():
	wl, _ = spectrometer.get_wavelength_data()

	spec400480,_ = spectrometer.measure(30, 0.053)
	time.sleep(0.1)
	spec4801000, _ = spectrometer.measure(30, 0.0022)
	time.sleep(0.1)
	wl,spec = binspec(wl,spec400480,spec4801000)

	plt.plot(wl, spec)
	plt.xlim((400, 1000))
	plt.show()

	wl = np.around(wl, decimals=0)
	return (wl,spec)

# Function to perform spectra measurements
def absm(bgspec, bg_dark):

	wl, spec = specm()

	while 1:
		if np.any(spec >= 0.8):
			break
		else:
			p.draw_and_dispense(1, 7, 10*ul,velocity=1000)
			wl, spec = specm()

	raw = -np.log10(spec/bgspec) # A=-log(T)=-log(I/I0)
	wl, abs = connectspec(wl,raw)
################################################ windows update bug LinAlgError("SVD did not converge in Linear Least Squares")
	try:
		abs = savgol_filter(abs,99,2) # Apply Savitzky-Golay filter for smoothing
	except:
		abs = savgol_filter(abs,99,2)
###############################################

	abs = np.around(abs,decimals=5)
	startwl1 = np.where(wl == START_WL1)[0][0]
	startwl2 = np.where(wl == START_WL2)[0][0]

	return (wl,raw,abs,startwl1,startwl2)

# Function to connect and process spectrum points
def connectspec(wl,spec):
	#connect missing wl
	newwl = []
	newspec = []
	for i in range(len(wl) - 1):
		for j in range(wl[i + 1] - wl[i]):
			newwl.append(wl[i] + j)
			newspec.append(spec[i]+(spec[i+1]-spec[i])/(wl[i + 1] - wl[i])*j)
			# newspec.append(spec[i])

	newwl = np.asarray(newwl)
	newspec = np.asarray(newspec)

	#shift break points
	i480 = np.where(newwl == 480)[0][0]
	shift480 = 0

	newspec[:i480] += shift480

	return (newwl, newspec)

# Function to combine spectra with smaller bins
def binspec(wl,spec1,spec2):
	avgspec = []
	avgwl = []
	wl_int = int(wl[0])
	avg = [0,0]
	i = 0
	while (i < len(wl)):
		if (int(wl[i]) == wl_int):
			avg[1] += 1
			if (wl_int < 480):
				avg[0] += spec1[i]
			else:
				avg[0] += spec2[i]
			i += 1
		else:
			avgspec.append(avg[0] / avg[1])
			avgwl.append(wl_int)
			avg = [0,0]
			wl_int = int(wl[i])
	avgspec.append(avg[0] / avg[1])
	avgwl.append(wl_int)
	avgwl = np.asarray(avgwl)
	avgspec = np.asarray(avgspec)
	return (avgwl,avgspec)

# Function to calculate peak area
def peakarea(wl, abs):
	abs = np.nan_to_num(abs)
	wl_i = np.where(wl == 650)[0][0]
	wl_f = np.where(wl == 1000)[0][0]
	area = trapz (abs[wl_i:wl_f], dx = 1) #650 to 1000
	return area

# perform wash steps
def wash(cycles):
	for i in range (cycles):

		p.draw_and_dispense(8, 1, 100*ul,velocity=1000) #water
		p.draw_and_dispense(7, 1, 150*ul,velocity=1000) #gas

		p.draw_and_dispense(8, 1, 100*ul,velocity=1000) #water
		p.draw_and_dispense(7, 1, 150*ul,velocity=1000) #gas


		p.draw_and_dispense(8, 1, 100*ul,velocity=1000) #water
		p.draw_and_dispense(7, 1, 400*ul,velocity=1000) #gas

		p.set_valve(8)

		op.draw(volume = 80 * ul, velocity = 500)
		time.sleep(2)
		op.dispense(volume=2480 * ul, velocity = 1000)
		time.sleep(2)

		op.draw(volume = 2400 * ul, velocity = 1200)

	print ("washing done")

# perform background
def bg():
	for i in range(3):
		p.draw_and_dispense(8, 1, 100*ul,velocity=800) #water
		p.draw_and_dispense(7, 1, 600*ul,velocity=800)
		p.set_valve(1)
		time.sleep(3)

		op.dispense(volume=100 * ul,velocity=500) #140

		time.sleep(3)

		wl,bgspec = specm()

		op.dispense(volume=2300 * ul, velocity = 1000)
		time.sleep(2)
		op.draw(volume = 2400 * ul, velocity = 1200)

		if np.any(bgspec >= 0.8):
			np.savetxt('wl.txt',wl)
			np.savetxt('bg.txt',bgspec)
			print ("bg done")
			break

		else:
			print ("bg drift - Redo")


def countdown(t):
	while t:
		mins, secs = divmod(t, 60)
		timer = '{:02d}:{:02d}'.format(mins, secs)
		print(timer, end="\r")
		time.sleep(1)
		t -= 1

# conducting reactions
def react(n,v1,v2,v3,v4,v5,height,rtime,cycle_time,color,rep):

	time.sleep(1)

	if rep == 0 or rep == 2:
		ard.moveto(height)
		print("move to " + str(height) + " cm")

		try:
			f = open('current_exp/height.txt')
			n1 = len(f.readlines()) + 1
			f.close()
		except FileNotFoundError:
			n1 = 1
	#
		file = open('current_exp/height.txt','a')
		file.write(str(n1) + '\t' + str(height) + '\t')
		file.close()

	rtime *= 60
	bgspec = np.loadtxt('bg.txt',dtype=float)
	bg_dark = np.loadtxt('bg_dark.txt',dtype=float)

	p.draw_and_dispense(4, 1, v5*ul,wait=0.2,velocity=600) #acetone
	p.draw_and_dispense(3, 1, v1*ul,wait=0.2,velocity=600) #HAuCl4
	p.draw_and_dispense(2, 1, v3*ul,wait=0.2,velocity=600) #PVP / CTAB
	p.draw_and_dispense(6, 1, v4*ul,wait=0.2,velocity=600) #I-2959
	p.draw_and_dispense(8, 1, ((100 - v1 - v2 - v3 - v4 - v5)/2) *ul,velocity=600) #water
	p.draw_and_dispense(5, 1, v2*ul,wait=0.2,velocity=600) #AgNO3
	p.draw_and_dispense(8, 1, ((100 - v1 - v2 - v3 - v4 - v5)/2) *ul,velocity=600) #water
	p.draw_and_dispense(7, 1, 150*ul,wait=0.5,velocity=600) #gas to separate
	p.draw_and_dispense(8, 1, 100*ul,wait=0.5,velocity=600) #water soln to wash valve
	p.draw_and_dispense(7, 1, 450*ul,wait=0.5,velocity=600) #gas

	p.set_valve(8)
	print ("chemicals loaded")

	time.sleep(3)

	op.dispense(volume=550 * ul,velocity=500)

	cycle = int(rtime / cycle_time)
	FULL_STORKE_TIME = 12.5 * cycle_time # 1 cycle is 200uL in 1s; the total volume is 2500 uL; Therefore, full stroke time is 2500 / 200 = 12.5 for 1 HZ.
	op.set_velocity(op.get_max_steps() * 2 // int(FULL_STORKE_TIME))
	time.sleep(1)

	ard.turnon()
	print("UV on")


	cycle = math.ceil(rtime / cycle_time)

	time_left = rtime % cycle_time
	print("time left " + str(time_left) + " s")
	distance_left = time_left * (2 * OSCI_VOL / cycle_time) # time_left * speed of flow

	for i in range(cycle):
		op.dispense(volume = OSCI_VOL * 1000 * ul)
		op.draw(volume = OSCI_VOL * 1000 * ul)
		print(i,'/',cycle, end="\r")
	if distance_left >= OSCI_VOL:
		op.dispense(volume = OSCI_VOL * 1000 * ul)
		op.draw(volume = ((distance_left - OSCI_VOL) * 1000 * ul))
		distance_togo = (2* OSCI_VOL - distance_left)
		print("distance left " + str(distance_left) + " ul")
	else:
		op.dispense(volume=distance_left * 1000 * ul)
		distance_togo = distance_left
		print("distance left " + str(distance_left) + " ul")


	ard.turnoff()
	print("UV off")
	time.sleep(1)

	op.draw(volume=(550) * ul,velocity=500) #
	time.sleep(1)


	if (color is not None):
		#1 slug

		wl, raw, abs, startwl1, startwl2 = absm(bgspec,bg_dark)
		absneg = np.all(abs[startwl1:startwl1 + WL_RANGE1] > -0.03)
		abs = abs - np.min(abs[startwl1: startwl1 + WL_RANGE1])
		inten_400 = np.around(abs[startwl1], decimals=5)
		area_ratio = peakarea(wl, abs)
		peaklist, maxinten = peakinfo(wl[startwl2:startwl2 + WL_RANGE2],abs[startwl2:startwl2 + WL_RANGE2])
		fittedpeaklist = glfit(wl[startwl2:startwl2 + WL_RANGE2],abs[startwl2:startwl2 + WL_RANGE2],peaklist,area_ratio,inten_400)


		plt.plot(wl[startwl1:startwl1 + WL_RANGE1],abs[startwl1:startwl1 + WL_RANGE1],color=color)
		plt.scatter(wl[startwl1:startwl1 + WL_RANGE1],raw[startwl1:startwl1 + WL_RANGE1],s=10,c='r')
		plt.xlim((START_WL1,START_WL1 + WL_RANGE1))
		# plt.ylim((-0.025,0.3))
		ax = plt.gca()
		for i,peaki in enumerate(fittedpeaklist):
			plt.text(1,1 - 0.1 * i,str(fittedpeaklist[i]),horizontalalignment='right',verticalalignment='top',transform = ax.transAxes)
		# plt.text(1,1 - 0.1 * len(fittedpeaklist),str(maxinten),horizontalalignment='right',verticalalignment='top',transform = ax.transAxes)
		plt.savefig('current_exp/' + str(n) + 'rep' + str(rep) + '.png')
		plt.show()

		plt.clf()
		np.savetxt('current_exp/' + str(n) + 'rep' + str(rep) + 'smo.txt',abs)
		np.savetxt('current_exp/' + str(n) + 'rep' + str(rep) + 'raw.txt',raw)
		np.savetxt('current_exp/wl.txt', wl)
		abs800 = True



	time.sleep(1)

	if rep == 1 or rep == 2:
		ard.moveback(height)
		print("move back to HOME = 9.9 cm")
		time.sleep(1)

		file = open('current_exp/height.txt','a')
		file.write(str(9.9) + '\n')
		file.close()
	# exit(0)

	op.dispense(volume=(2400) * ul, velocity = 800)
	time.sleep(1)
	op.draw(volume = 2400 * ul, velocity = 1200)

	return (fittedpeaklist, maxinten, abs800, absneg, area_ratio, inten_400)



def mainpeak(fittedpeaklist):
	if (fittedpeaklist == []):
		return ([[None,None,None,None,None,None]])
	else:
		sorted_fittedpeaklist = sorted(fittedpeaklist, key=lambda inten : inten[0], reverse=True)
		return(sorted_fittedpeaklist)


def perform_exp(n,v1,v2,v3,v4,v5,height,rtime,cycle_time,i):
	for j in range(3):
		try:
			wash(2)
			if bg() == -1:
				exit(3)
			else:
				fittedpeaklist, maxinten, abs800, absneg, area_ratio, inten_400 = react(n,v1,v2,v3,v4,v5,height,rtime,cycle_time,'b',i)
				wash(1)
				if absneg and abs800:
					break
				print ("spectrum drifted")
		except:
			# exit(0)
			traceback.print_exc()
			ard.turnoff()
			# reset_pumps(p,op)

			with open('current_exp/height.txt','r') as f:
				lines = f.read().splitlines()
				last_line = lines[-1]
				last_num = last_line[-1:-5:-1]
				if last_num[-4] == "\t":
					last_num = last_num.replace("\t", "")

				last_move = last_num[::-1]
				print ("last move " + last_move)
			if last_move == str(height):
				ard.moveback(height)
				print("move back to HOME = 9.9 cm")
			else:
				ard.moveback(9.9)
				print("at HOME = 9.9 cm")

			wash(2)
			exit(3)

	return (fittedpeaklist, maxinten, abs800, absneg, area_ratio, inten_400)


def running(n,v1,v2,v3,v4,v5,height,rtime,cycle_time):#rtime is in minute

	# reset_pumps(p,op)
	res_list = []

	for i in range(2):
		print("repeat" + str(i))

		fittedpeaklist, maxinten, abs800, absneg, area_ratio, inten_400 = perform_exp(n,v1,v2,v3,v4,v5,height,rtime,cycle_time,i)

		# countdown(900)

		res = mainpeak(fittedpeaklist)
		if (abs800 and absneg and res[0][1] is not None and res[0][2] > 0):
			res_list.append(res)


	# no results for two reps
	if res_list == []:
		num_peak = 0
		area_ratio = 0
		d_wl = 0
		FWHM = 800
		I400 = 0
		inten_ratio = 0
		std = np.array([0,0,0,0,0,0])

	# both reps have results
	elif len(res_list) == 2:
		# both reps have 2 or more peaks
		if len(res_list[0]) >= 2 and len(res_list[1]) >= 2:
			rep1 = sorted([res_list[0][0], res_list[0][1]], key=lambda wl : wl[1], reverse=False)
			rep2 = sorted([res_list[1][0], res_list[1][1]], key=lambda wl : wl[1], reverse=False)

			res_list = np.asarray([rep1, rep2])
			res_mean = np.mean(res_list, axis=0)

			I400 = np.around(res_mean[0][5], decimals=3)
			inten_ratio = np.around(res_mean[1][0]/res_mean[1][5], decimals=2)
			area_ratio = np.around(res_mean[0][4], decimals=2)
			d_wl = np.around(res_mean[1][1], decimals=2)
			FWHM = np.around(res_mean[1][2], decimals=2)
			std = np.std(res_list, axis=0)
			num_peak = 2


		# both reps have 1 peak
		elif len(res_list[0]) == 1 and len(res_list[1]) == 1:
			res_list = np.asarray([res_list[0], res_list[1]])
			res_mean = np.mean(res_list, axis=0)

			I400 = np.around(res_mean[0][5], decimals=3)
			inten_ratio = np.around(res_mean[0][0]/res_mean[0][5], decimals=2)
			area_ratio = np.around(res_mean[0][4], decimals=2)
			d_wl = np.around(res_mean[0][1], decimals=2)
			FWHM = np.around(res_mean[0][2], decimals=2)
			std = np.std(res_list, axis=0)
			num_peak = 1

		#elif len(res_list[0]) < 2 or len(res_list[1]) < 2:
		else:
			fittedpeaklist, maxinten, abs800, absneg, area_ratio, inten_400 = perform_exp(n,v1,v2,v3,v4,v5,height,rtime,cycle_time,2)

			res = mainpeak(fittedpeaklist)
			if (abs800 and absneg and res[0][1] is not None and res[0][2] > 0):
				res_list.append(res)

			# 3rd rep is N/A
			if fittedpeaklist == []:
				num_peak = 0
				area_ratio = 0
				d_wl = 0
				FWHM = 800
				I400 = 0
				inten_ratio = 0
				std = np.array([0,0,0,0,0,0])

			# 3rd rep has two peaks
			elif len(fittedpeaklist) >= 2:
				rep3 = sorted([res_list[2][0], res_list[2][1]], key=lambda wl : wl[1], reverse=False)
				num_peak = 2

				if len(res_list[1]) >= 2:
					rep2 = sorted([res_list[1][0], res_list[1][1]], key=lambda wl : wl[1], reverse=False)

					res_list = np.asarray([rep2, rep3])
					res_mean = np.mean(res_list, axis=0)

					I400 = np.around(res_mean[0][5], decimals=3)
					inten_ratio = np.around(res_mean[1][0]/res_mean[1][5], decimals=2)
					area_ratio = np.around(res_mean[0][4], decimals=2)
					d_wl = np.around(res_mean[1][1], decimals=2)
					FWHM = np.around(res_mean[1][2], decimals=2)
					std = np.std(res_list, axis=0)

				else:
					rep1 = sorted([res_list[0][0], res_list[0][1]], key=lambda wl : wl[1], reverse=False)

					res_list = np.asarray([rep1, rep3])
					res_mean = np.mean(res_list, axis=0)

					I400 = np.around(res_mean[0][5], decimals=3)
					inten_ratio = np.around(res_mean[1][0]/res_mean[1][5], decimals=2)
					area_ratio = np.around(res_mean[0][4], decimals=2)
					d_wl = np.around(res_mean[1][1], decimals=2)
					FWHM = np.around(res_mean[1][2], decimals=2)
					std = np.std(res_list, axis=0)

			# 3rd rep has one peak
			else:
				num_peak = 1

				if len(res_list[0]) < 2:
					res_list = np.asarray([res_list[0], res_list[2]])
					res_mean = np.mean(res_list, axis=0)

					I400 = np.around(res_mean[0][5], decimals=3)
					inten_ratio = np.around(res_mean[0][0]/res_mean[0][5], decimals=2)
					area_ratio = np.around(res_mean[0][4], decimals=5)
					d_wl = np.around(res_mean[0][1], decimals=2)
					FWHM = np.around(res_mean[0][2], decimals=2)
					std = np.std(res_list, axis=0)

				else:
					res_list = np.asarray([res_list[1], res_list[2]])
					res_mean = np.mean(res_list, axis=0)

					I400 = np.around(res_mean[0][5], decimals=3)
					inten_ratio = np.around(res_mean[0][0]/res_mean[0][5], decimals=2)
					area_ratio = np.around(res_mean[0][4], decimals=2)
					d_wl = np.around(res_mean[0][1], decimals=2)
					FWHM = np.around(res_mean[0][2], decimals=2)
					std = np.std(res_list, axis=0)

	# one of rep test has results
	else:
		fittedpeaklist, maxinten, abs800, absneg, area_ratio, inten_400 = perform_exp(n,v1,v2,v3,v4,v5,height,rtime,cycle_time,2)

		res = mainpeak(fittedpeaklist)
		if (abs800 and absneg and res[0][1] is not None and res[0][2] > 0):
			res_list.append(res)

		if fittedpeaklist == []:
			num_peak = 0
			area_ratio = 0
			d_wl = 0
			FWHM = 800
			I400 = 0
			inten_ratio = 0
			std = np.array([0,0,0,0,0,0])

		# 3rd exp has one peak and one of the previous exps has one peak:
		elif len(fittedpeaklist) == 1 and len(res_list[0]) == 1:
			num_peak = 1
			res_list = np.asarray([res_list[0], res_list[1]])
			res_mean = np.mean(res_list, axis=0)

			I400 = np.around(res_mean[0][5], decimals=3)
			inten_ratio = np.around(res_mean[0][0]/res_mean[0][5], decimals=2)
			area_ratio = np.around(res_mean[0][4], decimals=2)
			d_wl = np.around(res_mean[0][1], decimals=2)
			FWHM = np.around(res_mean[0][2], decimals=2)
			std = np.std(res_list, axis=0)

		# 3rd exp has more than 2 peaks and one of the previous exps are the same:
		elif len(fittedpeaklist) >= 2 and len(res_list[0]) >= 2:
			num_peak = 2
			rep1 = sorted([res_list[0][0], res_list[0][1]], key=lambda wl : wl[1], reverse=False)
			rep3 = sorted([res_list[1][0], res_list[1][1]], key=lambda wl : wl[1], reverse=False)

			res_list = np.asarray([rep1, rep3])
			res_mean = np.mean(res_list, axis=0)

			I400 = np.around(res_mean[0][5], decimals=3)
			inten_ratio = np.around(res_mean[1][0]/res_mean[1][5], decimals=2)
			area_ratio = np.around(res_mean[0][4], decimals=2)
			d_wl = np.around(res_mean[1][1], decimals=2)
			FWHM = np.around(res_mean[1][2], decimals=2)
			std = np.std(res_list, axis=0)

		else:
			num_peak = 0
			area_ratio = 0
			d_wl = 0
			FWHM = 800
			I400 = 0
			inten_ratio = 0
			std = np.array([0,0,0,0,0,0])




	file = open('current_exp/param.txt','a')
	file.write(str(n) + '\t' + str(v1) + '\t' + str(v2) + '\t' + str(v3) + '\t' + \
				str(v4) + '\t' + str(v5) + '\t' + str(height) + '\t' + str(rtime) + '\t' + \
				str(cycle_time) + '\t' + str(area_ratio) + '\t' + str(num_peak) + '\t' + str(d_wl) + '\t' + \
				str(FWHM) + '\t' + str(I400) + '\t' + str(inten_ratio) + '\t' + str(std.tolist()) + '\n')
	file.close()



	return (area_ratio, num_peak, d_wl, FWHM, I400, inten_ratio, std)
