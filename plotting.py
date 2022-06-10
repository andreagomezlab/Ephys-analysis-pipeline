from IPython.core.display import HTML
import matplotlib.pyplot as plt
import numpy as np

def plot_2ch(abf):
    '''Plots voltage and current traces from all sweeps of one abf file. Assumes there is only 1 set of these per
    file (one cell recorded in that file). Mainly used for troubleshooting/inspecting data, not for pipeline.
    
    arguments:
    abf = class 'pyabf.abf.ABF', loaded with pyabf.ABF('filename')
    '''
    cm = plt.get_cmap("viridis") #Make colormap to distinguish sweeps
    colors = [cm(x/abf.sweepCount) for x in abf.sweepList]
    frame_axis = np.arange(abf.sweepPointCount)
    f,axs = plt.subplots(2,1,figsize=(10,5),sharey=False, sharex=True)
    for i in abf.channelList:
        for j in abf.sweepList:
            abf.setSweep(j, channel=i)
            ax = axs[i]
            dataX = abf.sweepX
            dataY = abf.sweepY
            ax.plot(dataX, dataY, alpha=.5, color=colors[j])
            ax.set_ylabel(abf.sweepLabelY)
            ax.set_xlabel(abf.sweepLabelX)
            
def plot_2ch_sliced(abf,i1,i2):
    '''Plots voltage and current traces with specified indices from all sweeps in, to get only a slice of the trace.
    Assumes there is only one set of traces per file. Mainly used for troubleshooting/inspecting data.
    
    arguments:
    abf = class 'pyabf.abf.ABF', loaded with pyabf.ABF('filename')
    i1, i2 = int, indices to slice the traces with
    '''
    cm = plt.get_cmap("viridis") #Make colormap to distinguish sweeps
    colors = [cm(x/abf.sweepCount) for x in abf.sweepList]
    frame_axis = np.arange(abf.sweepPointCount)
    f,axs = plt.subplots(2,1,figsize=(10,5),sharey=False, sharex=True)
    for i in abf.channelList:
        for j in abf.sweepList:
            abf.setSweep(j, channel=i)
            ax = axs[i]
            dataX = abf.sweepX[i1:i2]
            dataY = abf.sweepY[i1:i2]
            ax.plot(dataX, dataY, alpha=.5, color=colors[j])
            ax.set_ylabel(abf.sweepLabelY)
            ax.set_xlabel(abf.sweepLabelX)

def plot_2ch_command(abf):
    '''Plots voltage and current traces from all sweeps of one abf file. Assumes there is only 1 set of these per
    file (one cell recorded in that file). Mainly used for troubleshooting/inspecting data, not for pipeline.
    
    arguments:
    abf = class 'pyabf.abf.ABF', loaded with pyabf.ABF('filename')
    '''
    cm = plt.get_cmap("viridis") #Make colormap to distinguish sweeps
    colors = [cm(x/abf.sweepCount) for x in abf.sweepList]
    frame_axis = np.arange(abf.sweepPointCount)
    f,axs = plt.subplots(2,1,figsize=(10,5),sharey=False, sharex=True)
    for j in abf.sweepList:
        abf.setSweep(j, channel=0)
        ax = axs[0]
        dataX = abf.sweepX
        dataY = abf.sweepY
        ax.plot(dataX, dataY, alpha=.5, color=colors[j])
        ax.set_ylabel(abf.sweepLabelY)
        ax = axs[1]
        dataY = abf.sweepC
        ax.plot(dataX, dataY, alpha=.5, color=colors[j])
        abf.setSweep(j, channel=1)
        ax.set_ylabel(abf.sweepLabelY)
        ax.set_xlabel(abf.sweepLabelX)
        
def plot_Ih_Vc(abf,i1,i2):
    '''Used to plot all the sweeps from an Ih-Vc test on one cell'''
    cm = plt.get_cmap("viridis") #Make colormap to distinguish sweeps
    colors = [cm(x/abf.sweepCount) for x in abf.sweepList]
    f,axs = plt.subplots(2,1,figsize=(8,5),sharey=False, sharex=True,)
    for i in abf.channelList:
        for j in abf.sweepList:
            abf.setSweep(j, channel=i)
            ax = axs[i]
            dataX = abf.sweepX[i1:i2]
            dataY = abf.sweepY[i1:i2]
            ax.plot(dataX, dataY, alpha=.5,color=colors[j])
            ax.set_ylabel(abf.sweepLabelY)
            ax.set_xlabel(abf.sweepLabelX)
            
def plot_sliced_with_vline(abf,i1,i2,points):
    '''Plots slice of all traces in channel 0, along with vertical line indices that are specified as '''
    cm = plt.get_cmap("viridis") #Make colormap to distinguish sweeps
    colors = [cm(x/abf.sweepCount) for x in abf.sweepList]
    fig = plt.figure(figsize=(12,5))
    for j in abf.sweepList:
        abf.setSweep(j, channel=0)
        for point in points:
            plt.axvline(x=abf.sweepX[point],color='blue',ymin =0.0, ymax=1.0,linestyle='--',linewidth=0.5)
        dataX = abf.sweepX[i1:i2]
        dataY = abf.sweepY[i1:i2]
        plt.plot(dataX, dataY, alpha=.5,color=colors[j])
        plt.ylabel(abf.sweepLabelY)
        plt.xlabel(abf.sweepLabelX)
        plt.ylim(-100,100)
    return fig