from IPython.core.display import HTML
import matplotlib.pyplot as plt
import pandas as pnd
import numpy as np
import pyabf
from matplotlib.backends.backend_pdf import PdfPages
import utils.plotting as traces

### Stuff used on different methods:
def find_nearest(array, value):
    '''Returns index of value in 1D numpy array that is the closest to the specified value.'''
    array = np.asarray(array)
    idx = (np.abs(array - value)).argmin()
    return array[idx]

def get_mean_traces(abf,i1=int,i2=int):
    '''Gets mean trace of specified cell from channel 0 traces in ABF file'''
    all_X = []
    all_Y = []
    for i in abf.sweepList:
        abf.setSweep(i, channel=0) #voltage or current, depends on the mode. 
        time = abf.sweepX[i1:i2] - abf.sweepX[i1]
        trace = abf.sweepY[i1:i2]
        all_X.append(time)
        all_Y.append(trace)
    mean_X = np.mean(all_X,axis=0)
    mean_Y = np.mean(all_Y,axis=0)
    return mean_X, mean_Y

### Resting membrane potential test - Adrian 2020
def get_rest_Vm(file,genotype,datapoint,drug,region):
    '''Makes one-row sized dataframe for resting membrane potential.
    
    arguments:
    file = str, name of abf file
    genotype = str, genotype of cell
    datapoint = int, used for indexing overall 'n' of experiment
    drug = str, label that denotes whether drug is present (should be 'Yes' or 'No')
    '''
    print(file, 'RMP')
    abf = pyabf.ABF(file)
    sweep = 0 
    abf.setSweep(sweep,channel=0) #voltage
    resting_Vm = np.mean(abf.sweepY)
    data = {'RMP_mV':[resting_Vm],'datapoint':[datapoint],'genotype':[genotype],'drug_condition':[drug],
            'hippocampal_region':[region],'abf_filename':[file]}
    RMP = pnd.DataFrame(data)
    return RMP  

def get_RMP_df(exp_table,genotype,datapoint,drug_label,region):
    '''Obtains Resting Membrane Potential. It is expected to produce a dataframe with a single row, but more rows or 
    values may exist per cell, so we use this format for consistency across all functions that get measures from abfs. 
    If no RMP experiment is present, returns empty dataframe.
    
    arguments:
    exp_table = dataframe, table of abf files with corresponding experiment type and drug condition
    genotype = str, genotype of cell
    datapoint = int, used for indexing overall 'n' of experiment
    all_csv_files = list of csv files with associated metadata, such as sweep epochs
    drug_label = str, label used in type_dic items to determine if drug was present during recording 
    region = str, label that denotes region of hippocampus that was recorded/stimulated
    '''    
    all_RMP_onecell = pnd.DataFrame()
    for i, row in exp_table.iterrows():
        if 'RMP' in row.Experiment:
            file = row.File
            if drug_label in row.Experiment:
                drug = 'Yes'
            else:
                drug = 'No'
            RMP = get_rest_Vm(file,genotype,datapoint,drug_label,region)
            all_RMP_onecell = all_RMP_onecell.append(RMP, ignore_index=False)       
    return all_RMP_onecell

### Firing Frequency test - Adrian 2020
import utils.expinfo as expinfo
from scipy import signal
def get_Hz(file,genotype,datapoint,drug,region):
    '''Makes one DataFrame where each row contains measurements of each sweep in the abf file: firing frequency,
    peak voltages and timepoints. 
    
    arguments:
    file = str, name of abf file
    genotype = str, genotype of cell
    datapoint = int, used for indexing overall 'n' of experiment
    drug = str, label that denotes whether drug is present (should be 'Yes' or 'No')
    region = str, label that denotes region of hippocampus that was recorded/stimulated
    '''
    print(file,'Hz')
    abf = pyabf.ABF(file)
    Hz_fig = plt.figure(figsize=(8,5))
    title_text = 'Firing frequency test with ' + drug + ' drug added, '+file
    plt.title(title_text)
    cm = plt.get_cmap("viridis") #Make colormap to distinguish sweeps
    colors = [cm(x/abf.sweepCount) for x in abf.sweepList]
    freqs_allsweeps = pnd.DataFrame()
    step_start,step_end = expinfo.get_I_step_interval(abf) #get 1-sec time window of traces
    if not step_start: # if step_start is []
        step_start, step_end = 1625, 21625 #hard coded fix in case there's files where abf.sweepEpochs isn't correct
    for i in abf.sweepList:
        abf.setSweep(i, channel=0) #membrane potential
        I_step = np.mean(abf.sweepC[step_start:step_end]) #command waveform for clamp current, the real current wobbles
        trace = abf.sweepY[step_start:step_end]
        time = abf.sweepX[step_start:step_end]
        peaks = signal.find_peaks(trace,height=20) # use +20 mV as threshold for spikes, probably okay to do
        Hz = len(peaks[0]) #number of APs in one-second
        peak_Vms = []
        timepoints = []
        peak_inds = []
        for peak in peaks[0]:
            j = peak + step_start
            Vm = abf.sweepY[j]
            timepoint = j/abf.dataRate
            peak_Vms.append(Vm)
            timepoints.append(timepoint)
            peak_inds.append(j)

        plt.plot(time,trace,color=colors[i])
        plt.scatter(timepoints,peak_Vms,color='r',marker='*')
        if len(timepoints)>= 4:
            ISI_ratio = (timepoints[-1] - timepoints[-2])/(timepoints[1] - timepoints[0])
        else:
            ISI_ratio = 0
        data = {'Hz':[Hz],'clamp_current_pA':[I_step],'ISI_ratio':[ISI_ratio],'raw_trace_mV':[abf.sweepY],
                'time_s':[abf.sweepX],'abf_filename':[file],'datapoint':[datapoint],
                'genotype':[genotype],'drug_condition':[drug],'hippocampal_region':[region]}
        
        freqs = pnd.DataFrame(data)
        '''
        freqs['peak_Vm'] = freqs['peak_Vm'].astype('object')
        freqs['timepoints'] = freqs['timepoints'].astype('object')
        freqs.at[0,'peak_Vm'] = peak_Vms
        freqs.at[0,'timepoints'] = timepoints 
        '''
        freqs_allsweeps = freqs_allsweeps.append(freqs, ignore_index=False)
        
    return freqs_allsweeps, Hz_fig

def get_Hz_df(exp_table,Hzplt_path,genotype,datapoint,drug_label,region):
    '''Data summary for current-voltage experiment, where firing frequency vs clamp current is saved
    along with timepoints for potentially getting measure of 'burstiness'. Also returns figure(s) with measured
    values as vlines overlayed on plots of raw traces in PDF format. If no current-voltage experiment present, 
    returns empty dataframe and PDF.
    
    arguments:
    cell = str, name of cell as it appears in corresponding folder
    exp_table = dataframe, table of abf files with corresponding experiment type and drug condition
    genotype = str, genotype of cell
    datapoint = int, used for indexing overall 'n' of experiment
    all_csv_files = list of csv files with associated metadata, such as sweep epochs
    drug_label = str, label used in type_dic items to determine if drug was present during recording
    region = str, label that denotes region of hippocampus that was recorded/stimulated
    '''
    frequency_df = pnd.DataFrame()
    pdf = PdfPages(Hzplt_path)
    for i, row in exp_table.iterrows():
        if 'IV' in row.Experiment:
            file = row.File
            if drug_label in row.Experiment:
                drug = 'Yes'
            else:
                drug = 'No'
            freq, Hz_fig = get_Hz(file,genotype,datapoint,drug,region)
            frequency_df = frequency_df.append(freq, ignore_index=False)
            pdf.savefig(Hz_fig)
    pdf.close()
    return frequency_df

### Capacitance test - Adrian 2020
from scipy import integrate
def get_capacitance(file,genotype,datapoint,drug,region):
    ''' Creates one-row DataFrame for measurement of capacitance. Returns list of capacitances per sweep, mean 
    capacitance, Also returns reporter figure that only shows traces and window for steady-state current. 
    
    arguments:
    file = str, name of abf file
    genotype = str, genotype of cell
    datapoint = int, used for indexing overall 'n' of experiment
    drug = str, label that denotes whether drug is present (should be 'Yes' or 'No')
    region = str, label that denotes region of hippocampus that was recorded/stimulated
    '''
    print(file,' capacitance')
    abf = pyabf.ABF(file)
    cm = plt.get_cmap("cividis") #Make colormap to distinguish sweeps
    colors = [cm(x/abf.sweepCount) for x in abf.sweepList]
    caps = []
    cap_fig = plt.figure(figsize=(12,5))
    title_text = 'Capacitance V-clamp test with ' +drug+ ' drug present, ' +file
    plt.title(title_text)
    plt.xlabel('time (s)')
    plt.ylabel('Current (pA)')
    for i in abf.sweepList:
        abf.setSweep(i,channel=0)
        raw_str = str(abf.sweepEpochs)
        step_start, step_end = expinfo.get_V_step_interval(raw_str)
        steady_win = step_end - int(0.020*abf.dataRate)
        steady_I = np.mean(abf.sweepY[steady_win:step_end])
        norm_trace = abf.sweepY[step_start:step_end] - steady_I
        time = abf.sweepX[step_start:step_end]
        plt.plot(time,abf.sweepY[step_start:step_end],color=colors[i])
        AOC = integrate.simps(norm_trace,time,dx=abf.dataSecPerPoint) #dx is distance between samples in time!
        delta = np.mean(abf.sweepC[steady_win])/1000#just to pick out the voltage step, should be something like -5
        C = AOC/delta # picofarads / volts
        caps.append(C)
    plt.axvline(abf.sweepX[steady_win],linestyle=':',c='k')
    mean_cap = np.mean(caps)
    data = {'capaticances_by_sweep_pF':[caps],'mean_pF':[mean_cap],'genotype':[genotype],'drug_present?':[drug],
            'hippocampal_region':[region],'abf_filename':[file]}
    cap_df = pnd.DataFrame(data)
    return cap_df, cap_fig

def get_capacitance_df(exp_table,Capplt_path,genotype,datapoint,drug_label,region):
    '''Data summary for Capacitance experiment, where individual capacitances per sweep are saved as well as reporter
    figure. Still a work in progress, need to find a way to plot area under the curve (charge).
    
    arguments:
    cell = str, name of cell as it appears in corresponding folder
    exp_table = dataframe, table of abf files with corresponding experiment type and drug condition
    genotype = str, genotype of cell
    datapoint = int, used for indexing overall 'n' of experiment
    all_csv_files = list of csv files with associated metadata, such as sweep epochs
    drug_label = str, label used in type_dic items to determine if drug was present during recording
    region = str, label that denotes region of hippocampus that was recorded/stimulated
    '''
    pdf = PdfPages(Capplt_path)
    cap_df = pnd.DataFrame()
    for i, row in exp_table.iterrows():
        if 'capacitance' in row.Experiment:
            file = row.File
            if drug_label in row.Experiment:
                drug = 'Yes'
            else:
                drug = 'No'
            pre_cap, cap_fig = get_capacitance(file,genotype,datapoint,drug,region)
            cap_df = cap_df.append(pre_cap, ignore_index=False)
            pdf.savefig(cap_fig)
    pdf.close()
    return cap_df


###  Membrane Resistance (Rm) test - Adrian 2020
def get_Rm(file,genotype,datapoint,cond,region):
    
    print(file,' Rm')
    abf = pyabf.ABF(file)
    step_start, step_end = expinfo.get_I_step_interval(abf)
    if not step_start: # if step_start is []
        step_start, step_end = 1625, 21625 #hard coded fix in case there's files where abf.sweepEpochs isn't correct
    voltage = []
    current = []
    abs_frames = step_end - step_start #total number of 'frames' in the selected chunk of the traces
    rm_3quarters = abs_frames*0.25 # Get 75% the length of abs_frames
    filt_start = int(step_end - rm_3quarters) #Make new starting point that skips the first 25% of the step
    for i in abf.sweepList:
        abf.setSweep(i, channel=0) # voltage
        Vm = np.mean(abf.sweepY[filt_start:step_end])
        voltage.append(Vm)
        abf.setSweep(i, channel=1) #current
        I = np.mean(abf.sweepY[filt_start:step_end])
        current.append(I)
    slope, intercept = np.polyfit(current,voltage, 1) #V=IR, so R=V/I, and since m=(delta_y/deltax), y=V,x=I
    MOhms = slope*1000 #to get Mega Ohms
    
    data = {'Rm_MOhms':[MOhms],'datapoint':[datapoint],'genotype':[genotype],'drug_present?':[cond],
           'hippocampal_region':[region],'abf_filename':[file]}
    Rm = pnd.DataFrame(data)
    return Rm

def get_Rm_df(exp_table,genotype,datapoint,drug_label,region):
    '''Obtains Membrane Resistance. It is expected to produce a dataframe with a single row, but more rows or values may exist
    per cell, so we use this format for consistency across all functions that get measures from abfs.
    
    arguments:
    cell = str, name of cell as it appears in corresponding folder
    exp_table = dataframe, table of abf files with corresponding experiment type and drug condition
    genotype = str, genotype of cell
    datapoint = int, used for indexing overall 'n' of experiment
    all_csv_files = list of csv files with associated metadata, such as sweep epochs
    drug_label = str, label used in type_dic items to determine if drug was present during recording
    region = str, label that denotes region of hippocampus that was recorded/stimulated
    '''
    all_Rm_onecell = pnd.DataFrame()
    for i, row in exp_table.iterrows():
        if 'Rm' in row.Experiment:
            file = row.File
            if drug_label in row.Experiment:
                drug = 'Yes'
            else:
                drug = 'No'
            Rm = get_Rm(file,genotype,datapoint,drug,region)
            all_Rm_onecell = all_Rm_onecell.append(Rm, ignore_index=False)  
    return all_Rm_onecell

### Membrane time constant or tau test - Adrian 2020
def get_tau_step_interval(abf):
    '''Parses through raw text to obtain mV step interval in membrane test. Assumes that there is only a single 
    voltage step in voltage clamp experiments.
    
    arguments:
    abf = class, abf file loaded with pyabf.ABF()
    '''
    step_start, step_end = [],[]
    abf.setSweep(0,channel=0)
    raw_str = str(abf.sweepEpochs)
    parsed_str = raw_str.split(', ')
    parsed_total = (len(parsed_str))
    for step in parsed_str:
        if 'Step 0.00' not in step:    
            values = step[step.find("[")+1:step.find("]")]
            step_points = (values.split(':'))
            step_start = int(step_points[0])
            step_end = int(step_points[1]) 
    return step_start, step_end

def monoExp1d(Xs, tau):
    return np.exp(-Xs/tau)

def monoExp1dErr(Xs, data, tau, plotTitle=False):
    """return the summed difference (error) of the data from the real curve."""    
    fitted=monoExp1d(Xs,tau)
    err=np.sum(fitted-data)    
    '''
    if plotTitle:
        plt.plot(Xs,data,'.',label="data");
        plt.plot(Xs,fitted,'k--',label="tau: %.03f ms"%(tau*1e3));
        plt.legend()
        plt.title(plotTitle)
        plt.show()    
    '''
    return err
def bestTau(Xs, data, tau=.1, step=.1):
    """Returns time constant of the data in the same time units as the Xs (list of time points) provided."""
    errs=[np.inf]
    while(len(errs))<50:
        #find plotting 
        assert len(Xs)==len(data)
        normed=data/data[0]
        tau=np.max((0.000001,tau))
        if len(errs)%5==0:
            errs.append(monoExp1dErr(Xs,normed,tau,plotTitle="iteration #%d"%len(errs)))
        else:
            errs.append(monoExp1dErr(Xs,normed,tau))
        if np.abs(errs[-1])<0.001:
            return tau
        if (errs[-1]>0 and errs[-2]<0) or (errs[-1]<0 and errs[-2]>0):
            step/=2
        if errs[-1]<0:
            tau+=step
        elif errs[-1]>0:
            tau-=step
    return tau

def get_tau(file,genotype,datapoint,drug,region):
    '''Makes one-row sized dataframe for tau measurement. Returned tau value is taken from all of the average of
    all normalized traces, prone to change. Alternative methods include: getting tau value for each trace and getting
    tau value from average trace where normalization of axes is done post-hoc. 
    
    arguments:
    file = str, name of abf file
    genotype = str, genotype of cell
    datapoint = int, used for indexing overall 'n' of experiment
    drug = str, label that denotes whether drug is present (should be 'Yes' or 'No')
    region = str, label that denotes region of hippocampus that was recorded/stimulated
    '''
    print(file,' tau')
    abf = pyabf.ABF(file)
    tau_fig = plt.figure(figsize=(10,7))
    cm = plt.get_cmap("plasma") #Make colormap to distinguish sweeps
    colors = [cm(x/abf.sweepCount) for x in abf.sweepList]
    step_start, step_end = get_tau_step_interval(abf)
    if not step_start: # if step_start is []
        step_start, step_end = 1625, 21625 #hard coded fix in case there's files where abf.sweepEpochs isn't correct
    clamp_pA = abf.sweepC[step_start]
    cutoff = int(0.3*abf.dataRate) + step_start
    time,trace = get_mean_traces(abf,i1=step_start,i2=cutoff)
    limit = np.mean(trace[int(abf.dataRate*0.25):])
    trace = trace - limit
    start_mV = trace[0]
    end_mV = np.mean(trace[int(abf.dataRate*0.25):])
    tau_mV = (start_mV - end_mV)*(1 - 0.632) #*(1-1/e), can use np.exp(1) later maybe
    nearest = find_nearest(trace,tau_mV)
    manual_tau = np.mean(time[np.where(trace == nearest)])
    fit_tau = bestTau(time,-trace)
    fit = monoExp1d(time,fit_tau)*trace[0]
    plt.plot(time,trace,color='k')
    plt.plot(time,fit,color='g',alpha=0.5)
    plt.axvline(fit_tau,color='r')
    plt.axvline(manual_tau,color='b')
    one_trial = {'fit_tau_sec':fit_tau,'manual_tau_sec':manual_tau,'clamp_pA':clamp_pA,'datapoint':[datapoint],
                 'genotype':[genotype],'drug_present?':[drug],'hippocampal_region':[region],'abf_filename':[file]} 
    df = pnd.DataFrame(one_trial)
    title_text = 'Tau (='+str(fit_tau*1000)+'ms) test with ' + drug + ' drug added, '+file
    plt.title(title_text)
    return df, tau_fig

def get_tau_df(exp_table,tauplt_path,genotype,datapoint,drug_label,region):
    '''Data summary for Tau experiment, where a current step is applied repetitively to yield a voltage trace with
    a curve that can be fitted with a single exponential (tau). Returns dataframe of one row and mean tau value 
    and reporter figure. 
    
    arguments:
    cell = str, name of cell as it appears in corresponding folder
    exp_table = dataframe, table of abf files with corresponding experiment type and drug condition
    genotype = str, genotype of cell
    datapoint = int, used for indexing overall 'n' of experiment
    all_csv_files = list of csv files with associated metadata, such as sweep epochs
    drug_label = str, label used in type_dic items to determine if drug was present during recording
    region = str, label that denotes region of hippocampus that was recorded/stimulated
    '''
    
    pdf = PdfPages(tauplt_path)
    tau_df = pnd.DataFrame()
    for i, row in exp_table.iterrows():
        if 'tau' in row.Experiment:
            file = row.File
            if drug_label in row.Experiment:
                drug = 'Yes'
            else:
                drug = 'No'
            tau_iter, tau_fig = get_tau(file,genotype,datapoint,drug,region)
            tau_df = tau_df.append(tau_iter,ignore_index=True)
            pdf.savefig(tau_fig)
    pdf.close()
    return tau_df

### Ih currents measured in voltage clamp - Adrian 2020
import math
def round_decimals_up(number:float, decimals:int=2):
    """
    Returns a value rounded up to a specific number of decimal places.
    """
    if not isinstance(decimals, int):
        raise TypeError("decimal places must be an integer")
    elif decimals < 0:
        raise ValueError("decimal places has to be 0 or more")
    elif decimals == 0:
        return math.ceil(number)

    factor = 10 ** decimals
    return math.ceil(number * factor) / factor

def get_IhVc(file,genotype,datapoint,drug,region):
    '''Gets dataframe Ih current Voltage-Clamp setting experiment. Dataframe contains voltage step and peak Ih 
    currents.
    
    arguments:
    file = str, name of abf file
    genotype = str, genotype of cell
    datapoint = int, used for indexing overall 'n' of experiment
    drug = str, label that denotes whether drug is present (should be 'Yes' or 'No')
    region = str, label that denotes region of hippocampus that was recorded/stimulated
    '''
    abf = pyabf.ABF(file)
    Ihdf_allsweeps = pnd.DataFrame()
    print(file,' IhVc')
    #Very manual and feels very very wrong: some files have a buggy abf.sweepC, so I'm inputting the steps manually
    steps = [-50.0, -60.0, -70.0, -80.0, -90.0, -100.0, -110.0] #REMOVE WHEN TESTING NEW DATA
    
    
    raw_str = str(abf.sweepEpochs)
    bl_start = int(0.150*abf.dataRate)
    bl_end = int(0.200*abf.dataRate) #50 ms window right before mV step
    step_end = int(5.225*abf.dataRate) # hard coded asf
    step_start = int(step_end - 0.050*abf.dataRate) # index for 50 ms before the end of the voltage step
    df_trace_start = int(5*abf.dataRate) # used to make trace that goes into dataframe (for representative figs)
    df_trace_end = int(5.5*abf.dataRate)
    rep_start = int(step_end + 0.0155*abf.dataRate) # first index of trace for reporter figure, plotted as black line 
    trace_end = int(rep_start + 0.300*abf.dataRate) # last index of trace for dataframe, 300 ms long, 
    Ih_start = int(rep_start + 0.030*abf.dataRate) #plotted as red line 
    Ih_end = int(Ih_start + 0.020*abf.dataRate) #next red line
    IhVc_fig = plt.figure(figsize=(8,5))
    title_text = 'Ih-Vc test with ' + drug + ' drug present, '+file
    plt.title(title_text)
    cm = plt.get_cmap("viridis") #Make colormap to distinguish sweeps
    colors = [cm(x/abf.sweepCount) for x in abf.sweepList]
    
    for i in abf.sweepList:
        abf.setSweep(i, channel=1) #membrane potential
        V = np.mean(abf.sweepY[bl_start:bl_end]) #baseline voltage from one sweep
        bl_voltage = round(V,0)# baseline voltage that should resemble -50.0, -40.0, 
        abf.setSweep(i, channel=0) # current
        bl_pA = np.mean(abf.sweepY[bl_start:bl_end]) # baseline needed to get steady-state current, to then get tail current
        steady_pA = np.mean(abf.sweepY[step_start:step_end]) - bl_pA
        '''
        magnitude = abf.sweepY[rep_start],abf.sweepY[Ih_start] #not keeping this idea since it introduces variance
        if magnitude[1] < magnitude[0]: # if trace is going down
            Ih = np.min(abf.sweepY[Ih_start:Ih_end])# returns negative peak
            Ih_ind = np.argmin(abf.sweepY[Ih_start:Ih_end]) + Ih_start
        elif magnitude[1] > magnitude[0]: # if trace is going up
            Ih = np.max(abf.sweepY[Ih_start:Ih_end])
            Ih_ind = np.argmax(abf.sweepY[Ih_start:Ih_end]) + Ih_start
        '''
        Ih = abf.sweepY[Ih_start] #Make the chosen value of the tail currents simply the amplitude 30 ms after onset
        abs_Ih = abs(Ih - bl_pA) #ABSOLUTE MAGNITUDE OF PEAK WITH SUBTRACTED IS USED TO QUANTIFY IH EFFECT ON TAIL CURRENT
        df_trace = abf.sweepY[df_trace_start:df_trace_end] #used for representative figure made from table
        df_time = abf.sweepX[df_trace_start:df_trace_end]
        inset = abf.sweepY[rep_start:Ih_end] #last 300 ms of 'trace', just a zoom-in tbh
        inset_time = abf.sweepX[rep_start:Ih_end]

        command_mV = abf.sweepC[df_trace_start:df_trace_end] + bl_voltage #trace of command waveform (for rep. figures)
        step_mV = abf.sweepC[step_start] + bl_voltage # holding voltage command waveform - baseline
        data = {'step_mV':[steps[i]],'steady_pA':[steady_pA],'abs_peak_Ih_pA':[abs_Ih],'datapoint':[datapoint],
                'genotype':[genotype],'drug_present?':[drug],'hippocampal_region':[region],'abf_filename':[file],
                'big_trace_pA':[df_trace],'big_trace_time_s':[df_time],'inset_mV':[inset],
                'inset_time_s':[inset_time],'command_mV':[command_mV]} #IN 'step_mV' entry, replace with step_mV later!!!!
        Ihdf= pnd.DataFrame(data)
        Ihdf_allsweeps = Ihdf_allsweeps.append(Ihdf, ignore_index=False)
        peak_range_t = Ih_start/abf.dataRate
        plt.axvline(peak_range_t,c='red') #Show where amplitude is measured
        #plt.scatter(abf.sweepX[Ih_ind],Ih,color='r',marker='*')
        plt.plot(inset_time,inset,color=colors[i]) #plotted basline-subtracted trace btw
        plt.ylabel(abf.sweepLabelY)
        plt.xlabel(abf.sweepLabelX)
        plt.ylim(-130,40)
    return Ihdf_allsweeps, IhVc_fig

def get_IhVc_df(exp_table,Ihplt_path,genotype,datapoint,drug_label,region):
    '''Data summary for Ih -V-clamp experiment, where Ih current amplitude peaks are saved along with
    voltage steps and steady-state currents
    
    arguments:
    cell = str, name of cell as it appears in corresponding folder
    exp_table = dataframe, table of abf files with corresponding experiment type and drug condition
    genotype = str, genotype of cell
    datapoint = int, used for indexing overall 'n' of experiment
    all_csv_files = list of csv files with associated metadata, such as sweep epochs
    drug_label = str, label used in type_dic items to determine if drug was present during recording
    region = str, label that denotes region of hippocampus that was recorded/stimulated
    '''
    pdf = PdfPages(Ihplt_path)
    Ihdf = pnd.DataFrame()
    for i, row in exp_table.iterrows():
        if 'Ih-Vc' in row.Experiment:
            file = row.File
            if drug_label in row.Experiment:
                drug = 'Yes'
            else:
                drug = 'No'
            Ih, IhVc_fig = get_IhVc(file,genotype,datapoint,drug,region)
            Ihdf = Ihdf.append(Ih, ignore_index=False)
            pdf.savefig(IhVc_fig)
    pdf.close()
    return Ihdf

### Sag test - Adrian 2020
def get_Ih_sag(file,genotype,datapoint,drug,region):
    '''Gets Ih sag dataframe, where sag ratios are obtained as follows: sag = (steady - baseline)/(peak - baseline)
    Outputs are not averaged and are returned as a list in a single cell of a dataframe. Also returns reporter figure.
    
    arguments:
    file = str, name of abf file
    genotype = str, genotype of cell
    datapoint = int, used for indexing overall 'n' of experiment
    drug = str, label that denotes whether drug is present (should be 'Yes' or 'No')
    region = str, label that denotes region of hippocampus that was recorded/stimulated
    '''
    print(file,' sag')
    abf = pyabf.ABF(file)
    sag_start, sag_end, bl_start, bl_end = expinfo.get_sag_step_intervals(abf)
    sag_window = int(sag_start + 0.3*abf.dataRate) #seconds to look for sag peak, 300 ms window
    peak_mVs = []
    steady_mVs = []
    clamp_pAs = []
    sags = []
    
    steady_start = int(sag_end - 0.050*abf.dataRate) #50 milliseconds near the end of trace
    
    cm = plt.get_cmap("viridis") #Make colormap to distinguish sweeps
    colors = [cm(x/abf.sweepCount) for x in abf.sweepList]
    Sagfig = plt.figure(figsize=(12,8))
    title_text = 'Sag test with ' +drug+' drug added, '+ file 
    plt.title(title_text)
    
    for i in abf.sweepList:
        abf.setSweep(i,channel=0) #voltage
        baseline = np.mean(abf.sweepY[bl_start:bl_end])
        dataX = abf.sweepX[sag_start:sag_end] 
        dataY = abf.sweepY[sag_start:sag_end] - baseline
        plt.plot(dataX, dataY, alpha=.5, color=colors[i])
        peak = np.min(abf.sweepY[sag_start:sag_window]) - baseline
        peak_mVs.append(peak)
        peak_ind = np.argmin(abf.sweepY[sag_start:sag_window]) + sag_start
        plt.scatter(abf.sweepX[peak_ind],peak,color='k',marker='+')
        steady = np.mean(abf.sweepY[steady_start:sag_end]) - baseline # mean voltage at steady-state
        steady_mVs.append(steady)
        sag = steady/peak
        sags.append(sag)
        
    mean_peak_mV = np.mean(peak_mVs)
    mean_steady_mV = np.mean(steady_mVs)
    mean_sag = mean_steady_mV/mean_peak_mV
    mean_sag_ratio = np.mean(sags)
    avgX,avgY = get_mean_traces(abf,i1=int(sag_start-50),i2=int(sag_end+50)) #mean traces of time and voltage for rep fig
    command_pA = abf.sweepC[int(sag_start-50):int(sag_end+50)]
    
    data = {'peak_mV':0,'sag_ratios':0,'mean_sag_ratio':[mean_sag_ratio],'avg_time_s':[avgY],'avg_trace_mV':[avgX],
            'command_pA':[command_pA],'datapoint':[datapoint],'genotype':[genotype],'drug_present?':[drug],
            'hippocampal_region':[region],'abf_filename':[file]}
    Sagdf= pnd.DataFrame(data)
    Sagdf['sag_ratios'] = Sagdf['sag_ratios'].astype('object')
    Sagdf.at[0,'sag_ratios'] = sags
    
   
    plt.ylabel(abf.sweepLabelY)
    plt.xlabel(abf.sweepLabelX)
    plt.axvline(x=abf.sweepX[steady_start], ymin=0.0,ymax=1.0,linestyle='--',linewidth=0.5, color='r')
    plt.axvline(x=abf.sweepX[sag_window],ymin=0.0,ymax=1.0,linestyle=':',linewidth=0.5,color='k')
    plt.ylim(-50,0)
    return Sagdf, Sagfig

def get_Ih_sag_df(exp_table,Sagplt_path,genotype,datapoint,drug_label,region):
    '''Data summary for Ih sag I-clamp experiment, where current is dropped and "sag" response amplitude and then 
    ratios are determined. Also returns figure with reported measurements of peak sag amplitude and steady-state 
    voltage.
    
    arguments:
    cell = str, name of cell as it appears in corresponding folder
    type_dic = dic, dictionary of abf files with corresponding experiment type and drug condition
    genotype = str, genotype of cell
    datapoint = int, used for indexing overall 'n' of experiment
    all_csv_files = list of csv files with associated metadata, such as sweep epochs
    drug_label = str, label used in type_dic items to determine if drug was present during recording
    region = str, label that denotes region of hippocampus that was recorded/stimulated
    '''
    Sagdf = pnd.DataFrame()
    pdf = PdfPages(Sagplt_path)
    for i, row in exp_table.iterrows():
        if 'sag' in row.Experiment:
            file = row.File
            if drug_label in row.Experiment:
                drug = 'Yes'
            else:
                drug = 'No'
            sag, sagfig = get_Ih_sag(file,genotype,datapoint,drug,region)
            pdf.savefig(sagfig)
            Sagdf = Sagdf.append(sag, ignore_index=False)
    pdf.close()
    return Sagdf

### I-clamp Chirp protocol - Adrian 2020
from numpy import ceil, log2
from scipy import fft

def get_chirp_fft(file,genotype,datapoint,drug,region):
    ''' Creates one-row DataFrame with fft outputs of voltage, current and impedance of oscillatory stimulus. Returns
    frequency at highest impedance value. Also returns figure with reported values.
    arguments:
    file = str, name of abf file
    genotype = str, genotype of cell
    datapoint = int, used for indexing overall 'n' of experiment
    drug = str, label that denotes whether drug is present (should be 'Yes' or 'No')
    region = str, label that denotes region of hippocampus that was recorded/stimulated
    '''
    print(file,' Chirp')
    abf = pyabf.ABF(file)
    raw_str = str(abf.sweepEpochs)
    step_start, step_end = expinfo.get_chirp_step_intervals(raw_str)
    trace_size = step_end - step_start
    Inorms = np.empty([trace_size,abf.sweepCount])
    Vnorms = np.empty([trace_size,3])
    
    for i in abf.sweepList:
        abf.setSweep(i, channel=0) #voltage
        V = abf.sweepY[step_start:step_end]
        Vnorm = V - np.mean(abf.sweepY[:step_start]) #maybe don't need to normalize to zero
        abf.setSweep(i, channel=1) #current
        I = abf.sweepY[step_start:step_end] #injected current waveform! do not use the recorded current its messy!
        Inorm = I - np.mean(abf.sweepY[:step_start])
        fft_sample = int(2**ceil(log2((len(Vnorm)))))
        n = int((fft_sample/2)+1) #chunk to sample from fft output
        Inorms[:,i] = Inorm
        Vnorms[:,i] = Vnorm
    
    PVs = np.empty([n,len(abf.sweepList)])
    PIs = np.empty([n,len(abf.sweepList)])
    Imps = np.empty([n,len(abf.sweepList)])     
    x = fft_sample/2
    xf = 1/abf.dataSecPerPoint*np.arange(0,x+1)/fft_sample

    for i in abf.sweepList:
        fftV = fft(Vnorms[:,i],fft_sample)
        fftI = fft(Inorms[:,i],fft_sample) 
        P = abs(fftV/fft_sample)
        PV = P[:n]
        PVs[:,i] = PV
        P = abs(fftI/fft_sample)
        PI = P[:n]
        PIs[:,i] = PI  
        Imp = PV/PI*100  #NEED TO TAKE A LOOK AT CONVERSION, NUMBER IS RIGHT, ORDER IS STILL VERY WRONG
        Imps[:,i] = Imp
            
    mean_PV = np.mean(PVs,axis=1)
    mean_PI = np.mean(PIs,axis=1)
    mean_Imp = np.mean(Imps,axis=1)
   
    near_1Hz = find_nearest(xf, 1.0) #only real frequencies injected here are from 1 to 20 Hz
    near_20Hz = find_nearest(xf, 20.0)
    search = np.where(xf == near_1Hz) 
    startoff = int(search[0])
    search = np.where(xf == near_20Hz)
    cutoff = int(search[0])
    peak_Imp = np.max(mean_Imp[startoff:cutoff]) 
    peak_ind = np.argmax(mean_Imp[startoff:cutoff])
    peak_Hz = xf[peak_ind+startoff]
    
    chirp_fig, axs = plt.subplots(3,1,figsize=(7,8),sharey=False, sharex=True)
    axs[0].plot(xf[startoff:cutoff],PVs[:,1][startoff:cutoff],color='r')
    axs[1].plot(xf[startoff:cutoff],PIs[:,1][startoff:cutoff],color='g')
    axs[2].plot(xf[startoff:cutoff],Imps[:,1][startoff:cutoff],color='b')
    axs[2].set_xlim(0.5,20)
    axs[2].set_ylim(0,70)
    axs[0].set_ylabel('Voltage (mV)')
    axs[1].set_ylabel('Current (pA)')
    axs[2].set_xlabel('Frequency (Hz)')
    axs[2].set_ylabel('Impedance (MOhms)')
    axs[2].axvline(peak_Hz,linestyle='--')
    title_text = 'Chirp test from 1-20 Hz, 1 Hz/s with ' + drug + ' drug added, '+file 
    plt.suptitle(title_text)
    
    data = {'fft_mV':0,'fft_pA':0,'Impedance_MOhm':0,'Peak_Hz':[peak_Hz],'Peak_Impedance_MOhms':[peak_Imp],
            'genotype':[genotype],'datapoint':[datapoint],'drug_present?':[drug],'hippocampal_region':[region],
            'abf_filename':[file]}
    Chirpdf= pnd.DataFrame(data)
    Chirpdf['fft_mV'] = Chirpdf['fft_mV'].astype('object')
    Chirpdf['fft_pA'] = Chirpdf['fft_pA'].astype('object')
    Chirpdf['Impedance_MOhm'] = Chirpdf['Impedance_MOhm'].astype('object')
    Chirpdf.at[0,'fft_mV'] = PVs
    Chirpdf.at[0,'fft_pA'] = PIs
    Chirpdf.at[0,'Impedance_MOhm'] = Imps
    
    return Chirpdf, chirp_fig
def get_chirp_df(exp_table,ZAPplt_path,genotype,datapoint,drug_label,region):
    '''Data summary for IC chirp experiment, where Fourier Transforms of injected current and voltage are obtained,
    which can be used to plot cell Impedance from a linearly increasing, oscillatory current injection. Peak impedance,
    corresponding to most resonant frequency, is also reported.
    
    arguments:
    exp_table = dataframe, table of abf files with corresponding experiment type and drug condition
    genotype = str, genotype of cell
    datapoint = int, used for indexing overall 'n' of experiment
    all_csv_files = list of csv files with associated metadata, such as sweep epochs
    drug_label = str, label used in type_dic items to determine if drug was present during recording
    region = str, label that denotes region of hippocampus that was recorded/stimulated
    '''
    pdf = PdfPages(ZAPplt_path)
    chirpdf = pnd.DataFrame()
    for i, row in exp_table.iterrows():
        if 'Ic-chirp' in row.Experiment:
            file = row.File
            if drug_label in row.Experiment:
                drug = 'Yes'
            else:
                drug = 'No'
            chirp, zap_fig = get_chirp_fft(file,genotype,datapoint,drug,region)
            chirpdf = chirpdf.append(chirp, ignore_index=False)
            pdf.savefig(zap_fig)
    pdf.close()
    return chirpdf

### rheobase test - Adrian 2020
def get_rheobase(file,genotype,datapoint,drug,region):
    ''' Creates one-row DataFrame for measurement of rheobase. Returns all of the current at which the cell spikes, 
    such that the real value for rheobase would be the minimum in this list. Also returns spike number, peaks and 
    timepoints.
    
    arguments:
    file = str, name of abf file
    genotype = str, genotype of cell
    datapoint = int, used for indexing overall 'n' of experiment
    drug = str, label that denotes whether drug is present (should be 'Yes' or 'No')
    '''
    print(file,' rheobase')
    abf = pyabf.ABF(file)    
    raw_str = str(abf.sweepEpochs)
    step_start, step_end = expinfo.get_rheobase_step_intervals(raw_str)
    rheobases = []
    spikes = []
    spike_inds = []
    for i in abf.sweepList:
        abf.setSweep(i, channel=0) # voltage
        sp_inds = signal.find_peaks(abf.sweepY[step_start:step_end],height=0) # get spikes that pass 0 mV
        if sp_inds[0].size > 0: #if the list is not empty
            spike = sp_inds[1].get('peak_heights') # get peak heights from dictionary part of class variable
            spikes.extend(spike)
            spike_ind = sp_inds[0] + step_start # add start index because find_peaks gets indices relative to input range
            spike_inds.extend(spike_ind)
            abf.setSweep(i, channel=1) # current
            baseline = np.mean(abf.sweepY[:(step_start-1)])
            rheobase = np.mean(abf.sweepY[step_start:step_end]) - baseline #mean injected current during epoch
            rheobases.append(rheobase)
    data = {'rheobase_pA':[rheobases[0]],'peak_times_s':0,'peaks_mV':0,'datapoint':[datapoint],'genotype':[genotype],
           'drug_present?':[drug],'hippocampal_region':[region],'abf_filename':[file]}
    
    peak_times = [i/abf.dataRate for i in spike_inds]
    rhe = pnd.DataFrame(data)
    rhe['rheobase_pA'] = rhe['rheobase_pA'].astype('object')
    rhe['peak_times_s'] = rhe['peak_times_s'].astype('object')
    rhe['peaks_mV'] = rhe['peaks_mV'].astype('object')
    rhe.at[0,'peak_times_s'] = peak_times
    rhe.at[0,'peaks_mV'] = spikes
    return rhe

def get_rheobase_df(exp_table,genotype,datapoint,drug_label,region):
    '''Data summary for rheobase experiment, where current steps of increasing amplitude are injected until the cell 
    spikes. The minimum current injected to evoke a spike is saved as rheobase value. Also summarizes number of 
    spikes per step.
    
    arguments:
    cell = str, name of cell as it appears in corresponding folder
    exp_table = dataframe, table of abf files with corresponding experiment type and drug condition
    genotype = str, genotype of cell
    datapoint = int, used for indexing overall 'n' of experiment
    all_csv_files = list of csv files with associated metadata, such as sweep epochs
    drug_label = str, label used in type_dic items to determine if drug was present during recording
    region = str, label that denotes region of hippocampus that was recorded/stimulated
    '''
    rhedf = pnd.DataFrame()
    rheo_table = exp_table[exp_table['Experiment'].str.contains("rheobase")]
    for i, row in rheo_table.iterrows():
        if 'rheobase' in row.Experiment:
            file = row.File
            if drug_label in row.Experiment:
                drug = 'yes'
            else:
                drug = 'no'           
            rhe_onetest = get_rheobase(file,genotype,datapoint,drug,region)
            rhedf = rhedf.append(rhe_onetest,ignore_index=False)
    return rhedf

### Action Potential waveform test - Adrian 2020
def get_AP_profile(file,genotype,datapoint,drug,region):
    ''' Creates one-row DataFrame for measurement of action potential properties. Returns peak amplitude, height,
    half-width and threshold potential. Also returns reporter figure where the measured values are plotted. 
    
    arguments:
    file = str, name of abf file
    genotype = str, genotype of cell
    datapoint = int, used for indexing overall 'n' of experiment
    drug = str, label that denotes whether drug is present (should be 'Yes' or 'No')
    region = str, label that denotes region of hippocampus that was recorded/stimulated
    '''
    print(file,' SS-stim')
    abf = pyabf.ABF(file)
    cm = plt.get_cmap("cividis") #Make colormap to distinguish sweeps
    colors = [cm(x/abf.sweepCount) for x in abf.sweepList] 
    start = int(0.165*abf.dataRate) # just for plotting
    cutoff = int(0.17*abf.dataRate) # 100% hard coded!!!! Need to find a better method
    end = int(0.195*abf.dataRate) # also for plotting
    peaks = []
    peak_times = []
    peak_mVs = []
    thresholds = []
    half_widths = []
    peak_heights = []
    AP_fig = plt.figure(figsize=(15,10))
    title_text = 'Single-spike stimulus test with ' +drug+ ' drug present, '+file
    plt.title(title_text)
    for i in abf.sweepList:
        
        abf.setSweep(i,channel=0) #voltage
        time = abf.sweepX[start:end]
        trace = abf.sweepY[start:end]
        plt.plot(time, trace,color=colors[i])
        peak = signal.find_peaks(abf.sweepY,height=20) # consider an AP any trace where the voltage crosses +20
        if peak[0].size != 0: 
                
            p = peak[0][0]
            peaks.append(p)
            timepoint = p/abf.dataRate
            peak_times.append(timepoint)
            mV = peak[1]['peak_heights'][0]
            peak_mVs.append(mV)
            baseline = np.mean(abf.sweepY[:int((0.1*abf.dataRate))]) # move this around
            height = mV - baseline
            peak_heights.append(height)
            trace_win = abf.sweepY[cutoff:p] #chunk of trace that we care about
            time_win = (abf.sweepX[cutoff:p] - abf.sweepX[cutoff])*1000# time axis in milliseconds
            dVdt = np.diff(trace_win)/np.diff(time_win) #first deriv of voltage / by first deriv of time
            max_deriv = np.argmax(dVdt) #find the max value in dVdt trace to make linear fit
            idx = (np.abs(dVdt[:max_deriv] - 20.000)).argmin() #timepoint dVdt was closest to 20 in REAl trace
            fit = np.polyfit(time_win[idx:max_deriv],dVdt[idx:max_deriv],1) #run linear fit on linear part of dVdt (rise)
            fit_time = np.linspace(time_win[idx-1],time_win[max_deriv],num=5000)             
            fit_dV = fit[0]*fit_time + fit[1] # y = mx + b
            idx2 = (np.abs(fit_dV - 20.000)).argmin() #timepoint that dVdt was way closer to 20 mV/ms in fit
            if dVdt[idx] < 20.0: #if the nearest value is lower than 20.0, draw the line to the next point
                tiny_fit_time = np.linspace(time_win[idx],time_win[idx+1],num=1000)
                tiny_fit = np.polyfit(time_win[idx:idx+1],trace_win[idx:idx+1],1)
            else: #if the nearest value in the dVdt trace is higher than 20.0, draw the line to the previous point
                tiny_fit_time = np.linspace(time_win[idx-1],time_win[idx],num=1000)
                tiny_fit = np.polyfit(time_win[idx-1:idx],trace_win[idx-1:idx],1)
            tiny_fit_mV = tiny_fit[0]*tiny_fit_time + tiny_fit[1]
            thr_idx = (np.abs(tiny_fit_time - fit_time[idx2])).argmin()
            threshold = tiny_fit_mV[thr_idx] #get exact threshold at timepoint based on drawn line
            thresholds.append(threshold)
            
            half = (height)/2 + baseline
            first = (np.abs(abf.sweepY[cutoff:p]-half)).argmin() + cutoff
            second = (np.abs(abf.sweepY[p:]-half)).argmin() + p
            half_width = int(second - first)/abf.dataRate * 1000
            half_widths.append(half_width)
            plt.hlines(half,abf.sweepX[first],abf.sweepX[second],color='r',LineStyle=':')
            plt.scatter(abf.sweepX[first],half,color='g',marker='x')
            plt.scatter(abf.sweepX[second],half,color='g',marker='x')
            plt.scatter(abf.sweepX[p],mV,color='b',marker='x')
            plt.scatter(abf.sweepX[idx+cutoff],threshold,color='k',marker='+')
            plt.plot((tiny_fit_time/1000) + abf.sweepX[cutoff],tiny_fit_mV,linewidth=4.0,color='g')
    plt.xlabel(abf.sweepLabelX)
    plt.ylabel(abf.sweepLabelY)
    plt.ylim(-65,42)
    plt.grid(True)
    
    mean_thr = np.mean(thresholds)
    mean_peak = np.mean(peak_heights)
    mean_HW = np.mean(half_widths)
    mean_peakmV = np.mean(peak_mVs)
    data = {'mean_threshold_mV':[mean_thr],'mean_half-width_ms':[mean_HW],'mean_peak_mV':[mean_peakmV],
            'mean_peak_height_mV':[mean_peak],'genotype':[genotype],'drug_present?':[drug],
            'hippocampal_region':[region],'abf_filename':[file]}
    APdf= pnd.DataFrame(data)
   
    return APdf, AP_fig

def get_SS_stim_df(exp_table,SSplt_path,genotype,datapoint,drug_label,region):                 
    '''Data summary for Single-Spike stimulus experiment, where current is injected into the cell toelicit a single spike, of       which the features [AP height, amplitude, threshold, half-width] are obtained. Returns a
    reporter figure with all of these values plotted onto each trace and dataframe with row=1
    
    arguments:
    cell = str, name of cell as it appears in corresponding folder
    type_dic = dic, dictionary of abf files with corresponding experiment type and drug condition
    genotype = str, genotype of cell
    datapoint = int, used for indexing overall 'n' of experiment
    all_csv_files = list of csv files with associated metadata, such as sweep epochs
    drug_label = str, label used in type_dic items to determine if drug was present during recording
    region = str, label that denotes region of hippocampus that was recorded/stimulated
    '''
    SS_stim_onecell = pnd.DataFrame()
    pdf = PdfPages(SSplt_path)
    for i, row in exp_table.iterrows():
        if 'SS' in row.Experiment:
            file = row.File
            if drug_label in row.Experiment:
                drug = 'yes'
            else:
                drug = 'no'
            APdf,AP_fig = get_AP_profile(file,genotype,datapoint,drug,region)
            SS_stim_onecell = SS_stim_onecell.append(APdf,ignore_index=False)
            pdf.savefig(AP_fig)
    pdf.close()
    return SS_stim_onecell   

### Presynaptic stimulus protocols (CA3 or EC inputs) - Adrian 2020
def get_stim_profile(stim_int,stim_type,file,genotype,datapoint,drug,region):
    '''Data summary for presynaptic stimulus experiment, where presynaptic axons are stimulated at different intensities.           Measures AP profile when the cell spikes or EPSP magnitudes if it does not. Returns a reporter figure with all of these         values plotted onto each trace and dataframe with row=1. If two pathways are stimulated in order, all variables with '1'       correspond to first response, and '2' with the second response. The identity of the stimulated axons is saved in the df as     a list of strings.
    
    arguments:
    cell = str, name of cell as it appears in corresponding folder
    type_dic = dic, dictionary of abf files with corresponding experiment type and drug condition
    genotype = str, genotype of cell
    datapoint = int, used for indexing overall 'n' of experiment
    all_csv_files = list of csv files with associated metadata, such as sweep epochs
    drug_label = str, label used in type_dic items to determine if drug was present during recording
    region = str, label that denotes region of hippocampus that was recorded/stimulated
    '''
    print(file,'presynaptic stimulus')
    abf = pyabf.ABF(file)
    cm = plt.get_cmap("cividis") #Make colormap to distinguish sweeps
    colors = [cm(x/abf.sweepCount) for x in abf.sweepList]
    stim_interval = int(4.950*abf.dataRate) #seconds between stimulus onsets when 2 are given in one sweep
    artf_ind_single = int(0.17*abf.dataRate) #timepoint if the stimulus is just one stimtrode, not two in sequence
    artf_ind1 = int(0.275*abf.dataRate) #hard-coded! timepoint where the stimulus is given (UNSOLVED ISSUE)
    artf_ind2 = artf_ind1 + stim_interval #there's not a perfect 5 seconds between stimuli so i adjusted
    artf_skip = int(0.005*abf.dataRate) #milliseconds to skip to not accidentally save artifact peak
    post_window_skip = int(0.05*abf.dataRate) # 500, how many frames the window is for getting EPSC peak (50 ms)
    post_window1 = artf_ind1 + post_window_skip
    post_window2 = artf_ind2 + post_window_skip
    post_window_single = artf_ind_single + post_window_skip
    if 'SC' in stim_type and 'EC' in stim_type: #consider order of these in string of metadata, the first should be first 
        stimtrodes = ['SC', 'EC']#if new stimuli places are added this won't hold up well, hard-coded for now
    elif 'SC' in stim_type:
        stimtrodes = ['SC']
    elif 'EC' in stim_type:
        stimtrodes = ['EC']
    else:
        stimtrodes = ['?']
    
    i1, i2 = 3000, 7000 #range of trace to plot first stim
    i3, i4 = i1+stim_interval,i2+stim_interval #range of trace to plot second stim
    i5, i6 = 2000, 5000 #NOT SURE ABOUT THESE YET ADJUST LATER, range of trace to plot single stim

    if len(stimtrodes) == 2:
        stim_fig, axs = plt.subplots(2,1,figsize=(10,5),sharey=False, sharex=False)
        #title1 = 'SC stim at' + str(stim_int) + 'microAmps'
        #axs[1].set_title(title1)
        title = 'SC then EC stim at ' + str(stim_int) + ' microAmps, '+file
        axs[0].set_title(title)
        axs[1].set_ylim(-65,-55)
    else:
        stim_fig = plt.figure(figsize=(8,5))
        title = str(stimtrodes[0])+' stim only at '+str(stim_int)+' microAmps, '+file
        plt.title(title)  

    peak_mVs1=[]
    peak_mVs2=[]
    AP_heights1=[]
    AP_heights2=[]
    peak_sweeps1=[]
    peak_sweeps2=[]
    PSPeak_heights1=[]
    PSPeak_heights2=[]
    PSPeak_sweeps1=[]
    PSPeak_sweeps2=[]
    for j in abf.sweepList: 
        abf.setSweep(j, channel=0) #voltage
        if len(stimtrodes) == 2: # if 2 stimuli in trace, make indices and axes for both separately
            bl_start1 = int(artf_ind1 - 0.05*abf.dataRate)  
            bl_start2 = bl_start1 + stim_interval
            baseline1 = np.mean(abf.sweepY[bl_start1:artf_ind1]) #get baseline right before stimulus
            baseline2 = np.mean(abf.sweepY[bl_start2:artf_ind2])
            time1 = abf.sweepX[i1:i2]
            trace1 = abf.sweepY[i1:i2]
            time2 = abf.sweepX[i3:i4]
            trace2 = abf.sweepY[i3:i4] 
            peak1 = signal.find_peaks(abf.sweepY[:len(abf.sweepY)//2],height=0) #find spikes in first stim
            peak2 = signal.find_peaks(abf.sweepY[len(abf.sweepY)//2:],height=0) #find spikes in second stim
            axs[0].axvline(abf.sweepX[artf_ind1],color='k',Linestyle='--')
            axs[1].axvline(abf.sweepX[artf_ind2],color='k',Linestyle='--')
            axs[0].plot(time1,trace1,color=colors[j])
            axs[1].plot(time2,trace2,color=colors[j])
            
            if peak1[0].size !=0: ### if the cell spiked at that sweep, get action potential peak height and voltage
                peak_sweeps1.append(j) #save sweep number
                p1 = peak1[0][0] #peak voltage index
                value1 = peak1[1] 
                mV1 = (value1['peak_heights'][0]) #get peak voltage
                peak_mVs1.append(mV1) #save peak voltage
                AP_height1 = mV1 - baseline1 #get peak height
                AP_heights1.append(AP_height1) #save peak height
                axs[0].scatter(abf.sweepX[p1],mV1,color='b',marker='x') #plot peak height
            else: ### if the cell DID NOT spike at that sweep, get PSP peak height and voltage
                PSPeak_sweeps1.append(j) #save sweep number
                PSPeak1 = max(abf.sweepY[artf_ind1+artf_skip:post_window1]) #find PSP peak, may be at noise level
                PSP_height = PSPeak1 - baseline1 #get PSP peak height
                PSPeak_heights1.append(PSP_height) #save height
                PSP_window = artf_ind1+artf_skip
                PSPeak_ind1 = int(np.argmax(abf.sweepY[PSP_window:post_window1])+PSP_window) #get index of PSP peak     
                axs[0].scatter(abf.sweepX[PSPeak_ind1],PSPeak1,color='r',marker='x') #plot PSP peak
                axs[0].axvline(abf.sweepX[artf_ind1+artf_skip],color='g',Linestyle=':')
            if peak2[0].size !=0: #if and else are the same but for second stimulus
                peak_sweeps2.append(j) #save sweep number
                p2 = peak2[0][0]
                value2 = peak2[1] 
                mV2 = value2['peak_heights'][0] #get peak voltage
                peak_mVs2.append(mV2) #save peak voltage
                AP_height2 = mV2 - baseline2 #get peak height
                AP_heights2.append(AP_height2)#save peak height
                axs[1].scatter(abf.sweepX[p2],mV2,color='b',marker='x') #plot peak onto plot of this sweep
            else:
                PSPeak_sweeps2.append(j) #save sweep number
                PSPeak2 = max(abf.sweepY[artf_ind2+artf_skip:post_window2]) #get PSP peak voltage 
                PSP_height = PSPeak2 - baseline2 #get PSP peak height
                PSPeak_heights2.append(PSP_height) #save PSP peak height
                PSP_window = artf_ind2+artf_skip
                PSPeak_ind2 = int(np.argmax(abf.sweepY[PSP_window:post_window2])+PSP_window) #get PSP peak index    
                axs[1].scatter(abf.sweepX[PSPeak_ind2],PSPeak2,color='r',marker='x') # plot PSP peak
                axs[1].axvline(abf.sweepX[artf_ind2+artf_skip],color='g',Linestyle=':')

        else: #if 1 stimuli in trace, just add values to version of variables with "1" instead of "2"
            bl_start = int(artf_ind_single - 0.05*abf.dataRate) 
            baseline = np.mean(abf.sweepY[bl_start:artf_ind_single]) #get baseline voltage
            plt.axvline(abf.sweepX[artf_ind_single],color='k',Linestyle='--') #vline the timepoint of stimulus onset
            time = abf.sweepX[i5:i6] #made with indices corresponding to SINGLE STIM protocol, different than 2 stim
            trace = abf.sweepY[i5:i6]
            plt.plot(time,trace,color=colors[j]) #plot traces for reporter figure
            #add comprehensive figure title that mentions conditions of that cell
            peak = signal.find_peaks(abf.sweepY,height=0)
            if peak[0].size != 0: #if the cell spiked even once, use SS stim analysis method.
                peak_sweeps1.append(j) #to keep track of at which sweep was there a spike
                p1stim = peak[0][0] #get AP peak index
                value1stim = peak[1]['peak_heights'][0] #get AP voltage peak
                onestim_mV1 = value1stim
                peak_mVs1.append(onestim_mV1) #save voltage peak
                AP_height1 = onestim_mV1 - baseline #get peak height to baseline
                AP_heights1.append(AP_height1) #save height
                plt.scatter(abf.sweepX[p1stim],onestim_mV1,color='b',marker='x') #plot AP peak onto sweep
            else:
                PSPeak_sweeps1.append(j) #save sweep number
                PSPeak1 = max(abf.sweepY[artf_ind_single+artf_skip:post_window_single]) #get PSP peak voltage
                PSP_h1stim = PSPeak1 - baseline #get peak height
                PSPeak_heights1.append(PSP_h1stim) #save peak height
                PSP_window = artf_ind_single+artf_skip
                PSPeak_ind1 = int(np.argmax(abf.sweepY[PSP_window:post_window_single])+PSP_window)      
                plt.scatter(abf.sweepX[PSPeak_ind1],PSPeak1,color='r',marker='x') #plot PSP peak onto sweep
            plt.axvline(abf.sweepX[artf_ind_single+artf_skip],color='g',Linestyle=':')
    
   
    mean_AP1 = np.mean(peak_mVs1)
    
    mean_AP2 = np.mean(peak_mVs2)
    mean_AP_h1 = np.mean(AP_heights1)
    mean_AP_h2 = np.mean(AP_heights2)
    mean_PSP_h1 = np.mean(PSPeak_heights1)
    mean_PSP_h2 = np.mean(PSPeak_heights2)
    data = {'peak_sweeps_1':[len(peak_sweeps1)],'AP_peak_mV_1':[mean_AP1],'AP_height_mV_1':[mean_AP_h1],
            'PSP_peak_height_mV_1':[mean_PSP_h1],'PSP_peak_sweeps_1':[len(PSPeak_sweeps1)],
            'peak_sweeps_2':[len(peak_sweeps2)],'AP_peak_mV_2':[mean_AP2],'APpeak_height_mV_2':[mean_AP_h2],
            'mean_PSP_peak_height_mV_2':[mean_PSP_h2],'PSP_peak_sweeps_2':[len(PSPeak_sweeps2)],
            'stim_location':[stimtrodes],'stim_intensity_micro_A':[stim_int],'genotype':[genotype],
            'datapoint':[datapoint],'drug_present?':[drug],'sweep_total':[abf.sweepCount],
            'hippocampal_region':[region],'abf_filename':[file]}
    
    #peak_mVs1=peak_mVs2=AP_heights1=AP_heights2=peak_sweeps1=peak_sweeps2=[]
    #PSPeak_heights1=PSPeak_heights2=PSPeak_sweeps1=PSPeak_sweeps2=[]
    stim_df = pnd.DataFrame(data)

    #see if stimtrode works fine cause we may not even have to switch the type of the df cells
    #make note in the function description that spike probability can be obtained from len of peak_sweeps
    return stim_fig, stim_df

def get_stim_profile_df(exp_table,stimplt_path,genotype,datapoint,drug_label,region):    
    stim_data = exp_table[exp_table.Experiment.str.contains("stim")] # keep stimulus entries on table
    stim_data = stim_data[~stim_data.Experiment.str.contains("SS")] #ignore SS stim there's a method for this
    
    stim_profile_onecell = pnd.DataFrame()
    pdf = PdfPages(stimplt_path)
    for i, row in stim_data.iterrows():
        stim_type = row.Experiment
        if drug_label in stim_type:
            drug = 'yes'
        else:
            drug = 'no'    
        stim_int = row.notes
        file = row.File
        stim_fig, stim_df = get_stim_profile(stim_int,stim_type,file,genotype,datapoint,drug,region)
        stim_profile_onecell = stim_profile_onecell.append(stim_df,ignore_index=False)
        pdf.savefig(stim_fig)
    pdf.close()
    return stim_profile_onecell
