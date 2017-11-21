# Author: Martin Zeller
# find calls in reasonably clean microphone recordings
# results will hopefully reflect singing of a single P. littoralis male
# results may be plotted and/or saved to pickle
# for queries over 100 seconds (e.g. whole recordings), expect considerable runtime
# only really works for 2015-07-29-aa

# note some seemingly unneccesary "if True:" - statements. In my editor of choice (TextMate), they simply allow me to conveniently wrap away code.

if True: ## load packages
    import numpy as np
    import matplotlib.pyplot as plt
    import thunderfish.dataloader as dl
    from scipy import signal
    from scipy.io.wavfile import write
    import sounddevice as sd
    import time
    from numba import jit, int32, float64
    import pickle
    import glob
    from mpl_toolkits.mplot3d import Axes3D
    import gc
    gc.enable()
if True: ## function definitions
    # some of these functions are numba optimised by including the @jit decorator.
    # i dont think these optimisations are necessary, since they don't optimize the most time critical step (convolving the envelope)
    # but i wanted to try this feature out anyway
    # if your distribution doesn't support numba (i.e. it is not anaconda), decorators may be removed
    
    @jit(float64[:](float64,int32,float64))
    def mygauss(dt,steps,sigma): # produce gaus array centered on array middle
        time = np.arange(-dt*steps,dt*(steps+1),dt)
        mygauss = np.exp(-time**2/(2*sigma**2))
        mygauss = mygauss/np.sum(mygauss)
        return mygauss
    
    @jit(int32[:](float64[:],float64))
    def findpeaksJit(x,thresh=0.): # find local maxima over certain threshold
        peaks = np.zeros(len(x),dtype=np.int32)
        ind = range(1,len(x)-1)
        c = 0
        for i in ind:
            diffFwd = x[i+1]-x[i]
            diffBwd = x[i]-x[i-1]
            if diffFwd < 0 and diffBwd > 0 and x[i] > thresh:
                peaks[c] = i
                c += 1
        return peaks[:c]
    
    def loadRaw(filename,readStart,readEnd,verbose=False): # load raw file
        with dl.open_data(filename,0.0,60.0) as data:
            Fs = data.samplerate
            dt = 1 / Fs
            readout = data[round(readStart/dt):round(readEnd/dt)]
            timeAx = np.arange(0,len(readout)*dt,dt)
            timeAx = timeAx[:len(readout)]
            if verbose:
                print('Read from '+filename)
                print(readStart,'to',readEnd)
                print('Total recording:',len(data)*dt)
                print('Sampling rate:',Fs)
            return  readout, timeAx, Fs
    
    def bpfilterTrace(trace,nyq,fLow=7000,fHigh=10000,butterOrder=3): # butterworth bp filter on trace
        low = fLow/nyq
        high = fHigh/nyq
        b, a = signal.butter(butterOrder, [low,high],btype='bandpass')
        filtered = signal.filtfilt(b,a,trace)
        return filtered
        
    def findCalls(envelope,timeAx,callThreshold): # find peaks in envelope, that correspond to syllables. segment them to calls
        allPeakInds = findpeaksJit(envelope,thresh = callThreshold)
        allPeakInds = np.append(0,allPeakInds)
        interpeak = np.diff(timeAx[allPeakInds])
        interpeak = np.append(timeAx[allPeakInds[0]]-timeAx[0],interpeak)
        callStartInds = np.array([i for i,v in enumerate(interpeak) if v > 0.5]) # suprathreshold interpeak indicates call start
        callEndInds = allPeakInds[callStartInds[1:]-1] # call starts are proceeded by call ends
        callEndInds = np.append(callEndInds,allPeakInds[-1]) # define a last call end
        callStartInds = allPeakInds[callStartInds]
        callStartInds = callStartInds.astype(int)
        callEndInds = callEndInds.astype(int)
        return callStartInds, callEndInds
    
    def readCalls(filename,callThreshold,callScanThresh,callTimeThresh,callPeakThresh): # routine to read out single trace, is verbose
        if True: # setup
            print('Read file ...')
            readout, timeAx, Fs = loadRaw(filename,read_start,read_end,verbose=True)
            dt = 1/Fs
            nyq = 0.5*Fs
            print('Filter audio ...')
            readoutFiltered = bpfilterTrace(readout,nyq)
    
            del readout
            gc.collect()
            print('Square trace ...')
            readoutFiltered = readoutFiltered**2
            print('Convolve for envelope')
            ## calculate pseudo-envelope:
            envelope = signal.fftconvolve(readoutFiltered,mygauss(1/Fs,3000,0.005),mode='same') 
            del readoutFiltered
            gc.collect()
            envelope = np.sqrt(envelope)
            # normalize envelope:
            envelope = envelope - np.min(envelope)
            envelope = envelope/np.max(envelope)
    
        print('Find calls ...')
        callStartInds, callEndInds = findCalls(envelope,timeAx,callThreshold)
    
        callStart = timeAx[callStartInds]
        callEnd = timeAx[callEndInds]
    
        # find call peaks:
        callPeak = np.zeros(len(callStartInds))
        for j in range(len(callStartInds)):
            if callStartInds[j] != callEndInds[j]:
                callPeak[j] = np.amax(envelope[callStartInds[j]:callEndInds[j]])
            else:
                callPeak[j] = envelope[callStartInds[j]]
    

        # post process putative calls
        # an elimination process
        eliminate = np.ones(len(callStart), dtype=bool)
        for j in range(len(callPeak)):
            inds = np.arange(callStartInds[j],callEndInds[j])
            syllables = findpeaksJit(envelope[inds],0)
            if len(syllables)<=1:
                eliminate[j] = False # eliminate call
            else:
                envPlot = envelope[inds[syllables]]
                envPlot = envPlot - np.min(envPlot)
                envPlot = envPlot/np.max(envPlot)
            
                timePlot = timeAx[inds[syllables]]
            
                c = np.argmax(envPlot) # counter for envelope point to be checked, begins at maximum of putative call
                q = True # loop break criterion, becomes true if under thresh count becomes greater than tree
                a = 0 # regeneration of under thresh count
                penalty = 0 # under thresh cozbt
                while q and c > 0:
                    c -= 1
                    q = envPlot[c] > callScanThresh
                    if q:
                        a +=1
                        if a > 3:
                            penalty = 0
                    else:
                        a = 0
                        penalty +=1
                        if penalty < 3:
                            q = True
            
                callStartInds[j] = inds[syllables[c]]
                callStart[j] = timeAx[callStartInds[j]]
            
                d = len(envPlot)
                q = True
                # from end of call, find first suprathreshold point for to determine call end
                while q:
                    d -= 1
                    q = envPlot[d] < callScanThresh
                
                callEndInds[j] = inds[syllables[d]]
                callEnd[j] = timeAx[callEndInds[j]]
                
                # eliminate calls too short
                if callEnd[j] - callStart[j] < callTimeThresh:
                    eliminate[j] = False
            
                # elimiante calls too weak
                if callPeak[j] < callPeakThresh:
                    eliminate[j] = False
                
                if False: # plot to control results of elimination process
                    f, axarr = plt.subplots(2, sharex=True)
                    axarr[0].plot(timePlot,envPlot)
                    axarr[0].plot(timeAx[inds[syllables]],timeAx[inds[syllables]]*0 + 0.25)
                    axarr[0].plot(timeAx[inds[syllables[c]]],envPlot[c],'o')
                    axarr[1].plot(timeAx[callStartInds[j]-30000:callEndInds[j]+30000],envelope[callStartInds[j]-30000:callEndInds[j]+30000])
                    plt.show()

        callStart = callStart[eliminate]
        callEnd = callEnd[eliminate]
        callPeak = callPeak[eliminate]
    
        print('Done.')
        return timeAx,envelope,callStart,callEnd,callPeak
if True: ## Parameters        
    dopickle = False
    doplot = True

    read_start = 0 # in seconds
    read_end = 800 # in seconds

    filenames = sorted(glob.glob('./2015/recordings/2015-07-28-aa/trace-?.raw'))

    # some parameters
    # i liked these for recording 2015-07-29-aa:
    callThresholds = np.ones(len(filenames))*0.1

    callThresholds[-1] = 0.2

    #study code to learn what these do
    callScanThresh = np.ones(len(filenames)) * 0.25
    callTimeThresh = np.ones(len(filenames)) * 0.4
    callPeakThresh = np.ones(len(filenames)) * 0.25




## start analys:
print('Files found: ')
print(filenames)
cols = ['blue','red','orange','green','brown','blue','red','blue','red','purple']

if doplot:
    f, axarr = plt.subplots(len(filenames),2, sharex='col',sharey='col')

## Analysis Loop:
for i,filename in enumerate(filenames):
    
    print('Trace' + str(i+1))
    
    time,envelope,start,end,peak = readCalls(filename,callThresholds[i],callScanThresh[i],callTimeThresh[i],callPeakThresh[i])
    doCall = False
    if dopickle:
        results = {'Time': time, 'Envelope': envelope, 'CallStart': start, 'CallEnd': end, 'CallPeak': peak,'Threshold':callThresholds[i],'File':filename}
        pickleName = 'Analysis_complete'+'_goodtraces_'+str(i+1)+'.p'
        pickle.dump(results, open( pickleName, 'wb' ) )
        print('Pickle dumped to:',pickleName)
    
    if doplot:
        plotsig = signal.fftconvolve(np.sqrt(envelope),mygauss(1/35000,3000,1),mode='same')
        plotsig = signal.decimate(signal.decimate(envelope,13,zero_phase=True),13,zero_phase=True)
        timeAx = np.arange(0,len(plotsig))
        timeAx = timeAx * (1/40000)*13**2
        timeAx = timeAx[:len(plotsig)]
        axarr[i,0].plot(timeAx,plotsig,cols[i],label='Trace'+str(i+1))
        axarr[i,0].plot([timeAx[0],timeAx[-1]],[callThresholds[i],callThresholds[i]],linewidth=0.5,color='black')
        for j in range(len(start)):
            axarr[i,0].plot([start[j],end[j]],[.5,.5],'black')
        axarr[i,1].hist(peak,normed=True)
            
    print('')

if doplot:
    plt.show()
