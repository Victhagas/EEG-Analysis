import numpy as np
import matplotlib.pyplot as plt
import mne
from scipy import signal, stats
import pipeline_functions as pf

### Carregando o dataset
data_folder = r"C:\Users\pipin\FisCog"
edf = '\Maria_Cecilia_04042022.edf'
dados = mne.io.read_raw_edf(data_folder+edf, preload=True)

# Definir o que é TRIGGER e o que é EMG e remover canais extras
dados.set_channel_types({'TRIGGER[DC1]': 'stim', 'EMG-0': 'emg', 'EMG-1': 'emg'})
dados.drop_channels(['POSITION Posição', 'DC2'])

# Plotar o mapa com as cabecinhas
ch_names_montage = {'EEG FP1-AA': 'Fp1','EEG FP2-AA': 'Fp2', 'EEG F3-AA': 'F3','EEG F4-AA': 'F4','EEG FZ-AA': 'Fz','EEG C3-AA': 'C3','EEG C4-AA': 'C4',
                'EEG CZ-AA': 'Cz', 'EEG P3-AA': 'P3', 'EEG P4-AA': 'P4', 'EEG PZ-AA': 'Pz', 'EEG F7-AA': 'F7', 'EEG F8-AA': 'F8', 'EEG T3-AA': 'T3',
                'EEG T4-AA': 'T4', 'EEG T5-AA': 'T5', 'EEG T6-AA': 'T6', 'EEG FPZ-AA': 'Fpz', 'EEG OZ-AA': 'Oz', 'EEG A2-AA': 'A2', 'EEG A1-AA': 'A1',
                'EEG FC1-AA': 'FC1', 'EEG FC3-AA': 'FC3', 'EEG C1-AA': 'C1', 'EEG C5-AA': 'C5', 'EEG CP5-AA': 'CP5', 'EEG FC2-AA': 'FC2', 'EEG FC4-AA': 'FC4',
                'EEG C2-AA': 'C2', 'EEG C6-AA': 'C6', 'EEG CP6-AA': 'CP6', 'EEG CPZ-AA': 'CPz', 'EEG CP1-AA': 'CP1', 'EEG CP2-AA': 'CP2'}
mne.rename_channels(dados.info, ch_names_montage)
dados = pf.my_montage(dados, dados.ch_names)
dados2 = dados.copy()
emg = dados.get_data(['EMG-0', 'EMG-1'])
dados2.drop_channels(['TRIGGER[DC1]', 'EMG-0', 'EMG-1', 'A1', 'A2'])
info2 = dados2.info
dados2 = dados2.get_data()

#Filtrar com o filtro do Eric
#my_plot_fft(dados2/1e-6, 1000, 'Dados Não-filtrados')
dados2 = pf.my_notch(dados2, 500, 60, 15) #Para plotar dados não filtrados, basta comentar esta linha
dados2 = pf.my_notch(dados2, 500, 120, 15)
dados2 = pf.my_notch(dados2, 500, 180, 15)
#my_plot_fft(dados2/1e-6, 1000, 'Dados Filtrados de 1-55 Hz')
dados2 = pf.my_filter(dados2, 500, 2, 1, 45)  #Para plotar dados não filtrados, basta comentar esta linha
#my_plot_fft(dados2/1e-6, 1000, 'Dados Filtrados de 1-55 Hz + Filtro Notch')
#plt.show()

#Plot no tempo para ver a diferença com e sem os filtros
dados_filtrados = mne.io.RawArray(dados2, info2)
#dados.plot(title='Dados não filtrados') #plotar os dados não filtrados
#dados_filtrados.plot(title='Dados filtrados') #plotar os dados filtrados
plt.show()

emg_info = mne.create_info(['EMG-0', 'EMG-1'], dados.info['sfreq'], ['emg', 'emg'])
emg_raw = mne.io.RawArray(emg, emg_info)
dados_filtrados.add_channels([emg_raw], force_update_info=True)

# Criando um novo canal de TRIGGER
a = dados.get_data()
pos,_ = signal.find_peaks(a[19, :], height=1, distance=5*500)
pos2 = np.zeros([1, len(a[19, :])])
pos2[0, pos] = 1


#Incorporando o novo canal de TRIGGER
info_eog = mne.create_info(['EOG'], dados_filtrados.info['sfreq'], ['eog']) # Adicionando um canal de EOG (média de Fp1 e Fp2)
temp_eog = dados_filtrados.get_data(['Fp1', 'Fp2'])
eog = np.zeros([1, len(temp_eog[0, :])])
eog[:, :] = (temp_eog[0, :] + temp_eog[1, :]) / 2
w1 = 10 / (500 / 2)  # Normalize the frequency
b1, a1 = signal.butter(2, w1, btype='low')
eog_filtrado = signal.filtfilt(b1, a1, eog)
eog_raw = mne.io.RawArray(eog_filtrado, info_eog)
info2 = mne.create_info(['STI'], dados_filtrados.info['sfreq'], ['stim'])
stim_raw = mne.io.RawArray(pos2, info2)
dados_filtrados.add_channels([stim_raw, eog_raw], force_update_info=True)
eventos = mne.find_events(dados_filtrados, stim_channel='STI')

#Carregando o código de eventos
with open(r'C:\Users\pipin\FisCog\maria_cecilia_04042022.txt','r') as f:
    eventos_txt = f.read().splitlines()
eventos_int = list(map(int, eventos_txt))
eventos_array = np.array(eventos_int)
eventos_teste = eventos
eventos_teste[:, 2] = eventos_array
dados_filtrados.plot(title='Dados filtrados e TRIGGER', events=eventos)
plt.show()
# Re-referenciar os dados para a média
# dados_average = dados_filtrados.copy()
# average = ['fp1', 'fp2', 'F3', 'F4','Fz', 'C3', 'C4', 'Cz', 'P3', 'P4','Pz', 'F7', 'F8', 'T3',
#                 'T4', 'T5', 'T6', 'fpz', 'Oz', 'FC1', 'FC3','C1', 'C5', 'CP5', 'FC2', 'FC4',
#                 'C2', 'C6', 'CP6', 'CPz', 'CP1', 'CP2']
# mne.set_eeg_reference(dados_average, ref_channels=average, copy=False, projection=False, ch_type='eeg')
# #dados_average.plot(title='Dados re-referenciados para a média')