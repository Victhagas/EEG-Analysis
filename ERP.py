
import numpy as np
import matplotlib.pyplot as plt
import mne
from scipy import signal, stats
import pipeline_functions as pf
%matplotlib qt5

# Loading datasets
dados_eeg = []
# vol = [i for i in range(12) if i != 0] # Grupo controle
vol = [i for i in range(12)]

for i in vol:
    if i < 9:
        data_folder = 'E:\Projetos\FisCog\Controle\S0' + str(i+1) + '\Sem re-referenciar'
        voluntario = '\S0' + str(i+1) + '_10-epo'
    else:
        data_folder = 'E:\Projetos\FisCog\Controle\S'  + str(i+1) + '\Sem re-referenciar'
        voluntario = '\S' + str(i + 1) + '_10-epo'
    print(data_folder+voluntario)
    dados_epocas = mne.read_epochs(data_folder + voluntario + '.fif')
    dados_eeg.append(dados_epocas.get_data(picks='eeg'))

# Calculating signal mean
voluntarios = len(dados_eeg)
dados_erp_vol = np.zeros((voluntarios, 30, 6001))
for indice in range(voluntarios):
    dados_erp_vol[indice,:,:] = np.mean(dados_eeg[indice], axis=0)
fs = dados_epocas.info['sfreq']
ch_name = dados_epocas.ch_names
t = np.arange(-2, 8, 1/fs)

# Paradigm
# Motor Execution 2200-2400ms (positive peak)
# Motor Execution 2400-2600ms (negative peak)

max = []
latencias = []
interval1 = list(range(2200, 2401))
interval2 = list(range(2400, 2601))
for i in range(len(vol)):
    valor_max = np.max(dados_erp_vol[i,3,2400:2600]) # max
    max.append(valor_max)
    latencias_max = np.where(dados_erp_vol[i,3,2400:2600] == np.max(dados_erp_vol[i,3,2400:2600])) # onde neste intervalo está a latência correspondente ao max
    latencias.append(latencias_max)
    print(len(latencias))

# Plotting
xlim = (-2, 8)
ylim = (-8e-6, 8e-6)
ind_plots = [19, 5, 24, 3, 4, 20, 25, 17, 18, 22, 23, 8]
ch = 3
fig, axes = plt.subplots(nrows=4, ncols=3, sharex=True, sharey=True)
plt.suptitle([ch_name[ch] + ' ERP - Movimento Braço Esquerdo'])
fig.subplots_adjust(wspace=0.08)
major_ticks_top = np.linspace(-2, 8, 11)
minor_ticks_top = np.linspace(-2, 8, 21)
axes = axes.flatten()
for ind, ind_plots in enumerate(vol):
    axes[ind_plots].plot(t, dados_erp_vol[ind, ch, :-1], linewidth=0.8, color='k')
    axes[ind_plots].set_ylim(ylim)
    axes[ind_plots].set_xlim(xlim)
    axes[ind_plots].set_xticks(major_ticks_top)
    axes[ind_plots].set_xticks(minor_ticks_top, minor=True)
    axes[ind_plots].grid(which="major", alpha=0.6)
    axes[ind_plots].grid(which="minor", alpha=0.3)
    axes[ind_plots].set(ylabel=str(vol[ind]+1))
    axes[ind_plots].axvline(x=2, linewidth=1, color='b', linestyle="--")
    axes[ind_plots].axvline(x=0, linewidth=1, color='g', linestyle="--")
plt.show()

ch_names = ['F3','F4','Fz','C3','C4','Cz','P3','P4','Pz','F7','F8','T3','T4','T5','T6','Fpz','Oz', 'FC1', 'FC3', 'C1','C5', 'CP5', 'FC2', 'FC4','C2','C6', 'CP6', 'CPz', 'CP1', 'CP2']
ch_types = ['eeg', 'eeg', 'eeg', 'eeg', 'eeg', 'eeg','eeg', 'eeg', 'eeg','eeg', 'eeg', 'eeg','eeg', 'eeg', 'eeg','eeg', 'eeg', 'eeg','eeg', 'eeg', 'eeg','eeg', 'eeg', 'eeg','eeg', 'eeg', 'eeg','eeg','eeg','eeg']
sfreq = 600
info = mne.create_info(ch_names=ch_names,ch_types=ch_types,sfreq=sfreq)
info.set_montage('standard_1020')

sujeito = 0
dados_erp_plot = dados_erp_vol[sujeito,:,:]

# -2000 ~ 8000ms (real-time signal) -> 0 ~ 10000ms (array signal)
# (array) planning interval -> 2000 - 3999 / execution interval -> 4000 - 10000
# baseline interval -> 0-2000ms
# interesting intervals -> 1900 - 2400 // 3900 - 4400  

evoked = mne.EvokedArray(dados_erp_plot[:,0:10000], info=info, nave=len(dados_epocas))
layout = mne.channels.find_layout(dados_epocas.info)
evoked.plot_topomap(times='interactive', scalings=None, ch_type='eeg',sphere=None, image_interp='cubic', time_unit='ms')
