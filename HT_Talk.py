# -*- coding: utf-8 -*-
"""
Created on Wed Oct 07 01:38:19 2015

@author: Lucky
"""

import os
import subprocess
import numpy as np
import pandas as pd
import scipy as sp
import glob
import ntpath
import shutil
import wave
import matplotlib.pyplot as plt
from scipy.io.wavfile import read 
import csv
from scipy.fftpack import rfft
from scipy.fftpack import fft
from scipy.fftpack.realtransforms import dct 
from sklearn.externals import joblib
import math

eps = 0.00000001

filename='/HackTheTalk/EMOTIONCLASSIFIER.joblib.pkl'
clf=joblib.load(filename) 

def convertMP3ToWav():
    dirName="/HackTheTalk"
    Fs=48000
    nC=1
    types=(dirName+os.sep+'*.mp3',)
    filesToProcess=[]
    for files in types:
        filesToProcess.extend(glob.glob(files))
    for f in filesToProcess:
        wavFileName=f.replace(".mp3",".wav")
        command = "avconv -i \"" + f + "\" -ar " +str(Fs) + " -ac " + str(nC) + " \"" + wavFileName + "\""; 
        os.system(command)

def FilterBanks(fs, nfft):

    # filter bank params:
    lowfreq = 133.33
    linsc = 200/3.
    logsc = 1.0711703
    numLinFiltTotal = 13
    numLogFilt = 27

    if fs < 8000:
        nlogfil = 5

    # Total number of filters
    nFiltTotal = numLinFiltTotal + numLogFilt

    # Compute frequency points of the triangle:
    freqs = np.zeros(nFiltTotal+2)
    freqs[:numLinFiltTotal] = lowfreq + np.arange(numLinFiltTotal) * linsc
    freqs[numLinFiltTotal:] = freqs[numLinFiltTotal-1] * logsc ** np.arange(1, numLogFilt + 3)
    heights = 2./(freqs[2:] - freqs[0:-2])

    # Compute filterbank coeff (in fft domain, in bins)
    fbank = np.zeros((nFiltTotal, nfft))
    nfreqs = np.arange(nfft) / (1. * nfft) * fs

    for i in range(nFiltTotal):
        lowTrFreq = freqs[i]
        cenTrFreq = freqs[i+1]
        highTrFreq = freqs[i+2]

        lid = np.arange(np.floor(lowTrFreq * nfft / fs) + 1, np.floor(cenTrFreq * nfft / fs) + 1, dtype=np.int)
        lslope = heights[i] / (cenTrFreq - lowTrFreq)
        rid = np.arange(np.floor(cenTrFreq * nfft / fs) + 1, np.floor(highTrFreq * nfft / fs) + 1, dtype=np.int)
        rslope = heights[i] / (highTrFreq - cenTrFreq)
        fbank[i][lid] = lslope * (nfreqs[lid] - lowTrFreq)
        fbank[i][rid] = rslope * (highTrFreq - nfreqs[rid])

    return fbank, freqs


def MFCC(X, fbank, nceps):
    mspec = np.log10(np.dot(X, fbank.T)+eps)
    ceps = dct(mspec, type=2, norm='ortho', axis=-1)[:nceps]
    return ceps   
    
def FeatureExtraction(signal, Fs, Win, Step):
    Win = int(Win)
    Step = int(Step)

    # Signal normalization
    signal = np.double(signal)

    signal = signal / (2.0 ** 15)
    DC = signal.mean()
    MAX = (np.abs(signal)).max()
    signal = (signal - DC) / MAX

    N = len(signal)                                # total number of samples
    curPos = int(Fs)
    countFrames = 0
    nFFT = Win / 2
    [fbank, freqs] = FilterBanks(Fs, nFFT)
    nceps = 13
    FinalFeature=[]
    F=[]
    c_pprev=[]
    i=0
    while i < 13:
        c_pprev.append(0)
        i=i+1
    c_prev=[]
    i=0
    while i < 13:
        c_prev.append(0)
        i=i+1
    c_cur=[]
    i=0
    while i < 13:
        c_cur.append(0)
        i=i+1
    c_aft=[]
    i=0
    while i < 13:
        c_aft.append(0)
        i=i+1
    delta1b=[]
    i=0
    while i < 13:
        delta1b.append(0)
        i=i+1
    delta1a=[]
    i=0
    while i < 13:
        delta1a.append(0)
        i=i+1
    #...................................
    delta2a=[]
    i=0
    while i < 13:
        delta2a.append(0)
        i=i+1
    delta2b=[]
    i=0
    while i < 13:
        delta2b.append(0)
        i=i+1
    FinalFeature=[]
    i=0
    while i < 39:
        FinalFeature.append(0)
        i=i+1
    count1=0
    count2=0
    count3=0
    count4=0
    while curPos + Win - 1 < N:                        # for each short-term window until the end of signal
        countFrames += 1
        x = signal[curPos:curPos+Win]                    # get current window
        curPos = curPos + Step                           # update window position
        X = abs(fft(x))                                  # get fft magnitude
        X = X[0:nFFT]                                    # normalize fft
        X = X / len(X)
        CEP=MFCC(X, fbank, nceps)
        i=0
        while i < 13:
            c_pprev[i]=c_prev[i]
            i=i+1
        #..........................
        i=0
        while i < 13:
            c_prev[i]=c_cur[i]
            i=i+1
        #................................
        i=0
        while i < 13:
            c_cur[i]=c_aft[i]
            i=i+1
        #.................................
        i=0
        while i < 13:
            c_aft[i]=CEP[i]
            i=i+1
         
        #..................................
       
        #print('frame no'+str(countFrames))
        if countFrames > 3:
            i=0
            while i < 13:
                delta1b[i]=(c_cur[i]-c_prev[i])/2
                i=i+1
            i=0
            while i < 13:
                delta1a[i]=(c_aft[i]-c_cur[i])/2
                i=i+1
            i=0
            while i < 13:
                delta2b[i]=(c_aft[i]-2*c_cur[i]+c_prev[i])/4
                i=i+1
            i=0
            while i < 13:
                delta2a[i]=(c_aft[i]-2*c_cur[i]+c_prev[i])/4
                i=i+1
            i=0
            while i < 13:
                FinalFeature[i]=c_cur[i]
                i=i+1
            while i < 26:
                FinalFeature[i]=delta1a[i-13]
                i=i+1
            while i < 39:
                FinalFeature[i]=delta2a[i-26]
                i=i+1
            arr=clf.predict(FinalFeature)
            if arr[0]==1.0:
                count1=count1+1
            if arr[0]==2.0:
                count2=count2+1
            if arr[0]==3.0:
                count3=count3+1
            if arr[0]==4.0:
                count4=count4+1
    s=count1+count2+count3+count4 
    cnt=0
    if (count1 >= 0.25*s):
        cnt=count1
        ss='Angry'
    else:
        ss='Neutral'
    if (count2 >= 0.25*s) and ( count2 > cnt):
        cnt=count2
        ss='Happy'
    if (count3 >= 0.25*s) and ( count3 > cnt):
        ss='Unhappy'
    print(ss)
def readAudioFile(path):
    [Fs,x]=read(path)
    return [Fs,x]
            
def classify():
    Fs=48000
    nC=1
    source='/HackTheTalk'
    types=(source+os.sep+'*.wav',)
    filesToProcess=[]
    #print(filesToProcess)
    for files in types:
        filesToProcess.extend(glob.glob(files))
    print(filesToProcess)
    i=1
    for f in filesToProcess:
        [F,x]=readAudioFile(f)
        x=np.double(x)
        nFFT=(Fs/2)
        Win=Fs*0.25
        Step=Fs*0.1
        FeatureExtraction(x, Fs, Win, Step)
        i=i+1

def main():
     convertMP3ToWav()
     classify()
    
main()