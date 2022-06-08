import scipy 
import matplotlib.pyplot as plt
import mne
import scipy.signal
import numpy as np
from matplotlib import pyplot
from scipy import stats
from scipy import signal
from scipy.fft import fft
from scipy import linalg as LA
import math
from matplotlib.ticker import MultipleLocator, FormatStrFormatter

#Função para montagem de eletrodos - plotar as cabecinhas
def my_montage(raw, ch_name):
    # Montage channel choice
    montage = mne.channels.make_standard_montage('standard_1020')
    ind = [i for (i, channel) in enumerate(montage.ch_names) if channel in ch_name]
    montage_new = montage.copy()
    # Keep only the desired channels
    montage_new.ch_names = [montage.ch_names[x] for x in ind]
    kept_channel_info = [montage.dig[x + 3] for x in ind]
    # Keep the first three rows as they are the fiducial points information
    montage_new.dig = montage.dig[0:3] + kept_channel_info
    raw.set_montage(montage_new)
    return raw

#Função do filtro notch
def my_notch(x, fs, freq, q):
    b1, a1 = signal.iirnotch(freq, q, fs)
    x_out  = signal.lfilter(b1, a1, x)
    return x_out

#Função de filtragem
def my_filter(x, fs, order, freq1, freq2):
    w1 = freq1 / (fs / 2)  # Normalize the frequency
    b1, a1 = signal.butter(order, w1, btype='high')
    x_aux = signal.filtfilt(b1, a1, x)
    w2 = freq2 / (fs / 2)
    b2, a2 = signal.butter(order, w2, btype='low')
    x_out = signal.filtfilt(b2, a2, x_aux)
    return x_out

#Plota a função densidade espectral por meio do Periodograma de Welch
def my_plot_psd(x, fs, title):
    plt.figure(figsize=(10, 4))
    f, Pxx_den = signal.welch(x, fs, nperseg=2048)
    print(Pxx_den.shape)
    plt.plot(f, Pxx_den.mean(axis=0))
    plt.yscale('log')
    plt.xlim((0, 280))
    #plt.ylim((10e-12, 10e2))
    plt.grid(which='both', axis='both')
    plt.ylabel('PSD [V²/Hz] (dB)')
    plt.xlabel('Frequency [Hz]')
    plt.title(title)

def my_plot_fft(x, fs, title): # acrescente um ch_name para mostrar o nome de canais individuais
    x = x.transpose()
    n = np.size(x, 0)
    dt = 1 / fs
    yf = fft(x, axis=0)
    xf = np.linspace(0.0, 1.0 / (2.0 * dt), n // 2)
    plt.figure(figsize=(10, 4))
    plt.plot(xf, 2.0 / n * np.abs(yf[0:n // 2])) #pode acrescentar um label=ch_name para mostrar as labels
    plt.xlim((-1, 150))
    #plt.legend(loc='best')
    plt.title(title)
    plt.xlabel('Frequency [Hz]')
    plt.ylabel('Single-Sided Amplitude Spectrum |X|')
    plt.grid()
    plt.show()

def my_sobi(X):
   N = np.size(X,0)
   m = np.size(X, 1)
   X_mean = X.mean(axis=0)
   X -= X_mean

   # Pre-whiten the data based directly on SVD
   u, d, v = LA.svd(X, full_matrices=False, check_finite=False)
   d = np.diag(d) #d -> matriz diagonal
   Q = np.dot(LA.pinv(d), v)
   X1 = np.dot(Q, X.T)
   p = min(100, math.ceil(N / 3))
   pm = p * m
   Rxp = np.zeros((m, m))
   M = np.zeros((m, pm), dtype=complex)

   # Estimate the correlation matrices
   k = 0
   for u in range(0, pm - 1, m):
       k += 1
       Rxp = np.dot(X1[:, k:N], X1[:, 0:N - k].T) / (N - k)
       M[:, u:u + m] = LA.norm(Rxp, 'fro') * Rxp

   # Perform joint diagonalization
   epsil = 1 / math.sqrt(N) / 100
   encore = 1
   V = np.eye(m, dtype=complex)
   g = np.zeros((m, p))

   while encore:
       encore = 0
       for p_ind in range(0, m - 1):
           for q_ind in range(p_ind + 1, m):
               g = [M[p_ind, p_ind:pm + 1:m] - M[q_ind, q_ind:pm + 1:m],
                    M[p_ind, q_ind:pm + 1:m] + M[q_ind, p_ind:pm + 1:m],
                    1j * (M[q_ind, p_ind:pm + 1:m] - M[p_ind, q_ind:pm + 1:m])]
               g = np.array(g)
               z = np.real(np.dot(g, g.T))
               w, vr = LA.eig(z, left=False, right=True)
               K = np.argsort(abs(w))
               temp_ang = vr[:, K[2]]
               angles = np.sign(temp_ang[0]) * temp_ang
               c = np.sqrt(0.5 + angles[0] / 2)
               sr = 0.5 * (angles[1] - 1j * angles[2]) / c
               sc = np.conj(sr)
               oui = np.abs(sr) > epsil
               encore = encore | oui

               if oui:
                   temp_M = np.copy(M)
                   colp = temp_M[:, p_ind:pm + 1:m]
                   colq = temp_M[:, q_ind:pm + 1:m]
                   M[:, p_ind:pm + 1:m] = c * colp + sr * colq
                   M[:, q_ind:pm + 1:m] = c * colq - sc * colp

                   temp_M2 = np.copy(M)
                   rowp = temp_M2[p_ind, :]
                   rowq = temp_M2[q_ind, :]
                   M[p_ind, :] = c * rowp + sc * rowq
                   M[q_ind, :] = c * rowq - sr * rowp

                   temp_V = np.copy(V)
                   V[:, p_ind] = c * temp_V[:, p_ind] + sr * temp_V[:, q_ind]
                   V[:, q_ind] = c * temp_V[:, q_ind] - sc * temp_V[:, p_ind]

   # Estimate the mixing matrix
   H = np.dot(LA.pinv(Q), V)

   # Estimated source activities
   Source = np.dot(V.T, X1)
   return H, Source


def my_ica(raw):
    eeg_ica = raw.get_data(picks='eeg')
    eog = raw.get_data(picks='eog')

    dp = np.std(eeg_ica, 1)
    eeg_ica = eeg_ica.T
    eeg_ica = np.divide(eeg_ica, dp)

    H, S = my_sobi(eeg_ica)
    H = H.real
    S = S.real

    corrp = np.zeros((2, np.size(S, 0)))
    for ind in range(np.size(S, 0)):
        teste = S[ind, :]
        teste = teste.astype(float)
        teste = teste.flatten()
        eog = eog.T
        eog = eog.flatten()
        corrp[:, ind] = stats.pearsonr(teste, eog)

    print(corrp[0, :])
    componente = np.nanargmax(np.absolute(corrp[0, :]))
    print(componente)
    H0 = np.copy(H)
    H0[:, componente] = np.zeros((np.size(H, 0)))
    eeg_recon = H0 @ S
    eeg_recon = eeg_recon * dp[:, None]

    return eeg_recon, S, corrp


def time_plot(x, fs, ylim, title):

   t = np.arange(0, np.size(x, 1) / fs, 1 / fs)
   a = np.size(x, 0)
   b = 1
   xlim = [0, 20]
   #ylim = 1e-2
   fig, axes = plt.subplots(a, b, sharex=True, sharey=True, figsize=(10, 8))
   plt.title(title)
   for ind, ax in enumerate(axes.flatten()):
       ax.plot(t, x[ind, :].real)
       #ax.set(ylabel=ch_name[ind])
       ax.set_xlim(xlim)
       ax.set_ylim([-ylim, ylim])
       plt.grid()
   plt.show()


def my_time_comparison(x, x_f, fs):
    t = np.arange(0, len(x) / fs, 1 / fs)
    plt.figure(figsize=(10, 8))
    ax1 = plt.subplot(211)
    plt.plot(t, x)
    plt.setp(ax1.get_xticklabels(), visible=False)
    plt.grid()
    plt.ylabel('Não Filtrado')
    plt.xlim((0, 10))
    ax2 = plt.subplot(212, sharex=ax1, sharey=ax1)
    plt.plot(t, x_f)
    plt.grid()
    plt.ylabel('Filtrado')
    #plt.ylabel('Filtrado + Notch')
    plt.xlabel('Time [s]')
    plt.show()


def comp_plot(x, x1, fs, ylim ,ch_name, title):

    t = np.arange(0, np.size(x, 1) / fs, 1 / fs)
    a = np.size(x, 0)
    b = 1

    xlim = [0, 20]
    fig, axes = plt.subplots(a, b, sharex=True, sharey=True, figsize=(10, 8))
    fig.suptitle(title)
    for ind, ax in enumerate(axes.flatten()):
        ax.plot(t, x[ind, :].real, 'r-', linewidth=1)
        ax.plot(t, x1[ind, :], 'k', linewidth=1)
        ax.set(ylabel=ch_name[ind])
        ax.set_xlim(xlim)
        ax.set_ylim([-ylim, ylim])
        ax.grid()
    plt.show()


def ica_plot(x, eog, fs, IC_name, corrcoef, title, escala):
     t = np.arange(0, np.size(x, 1) / fs, 1 / fs)
     a = np.size(x, 0)
     xlim = [0, 20]
     ylim = [-200, 200]
     fig, axes = plt.subplots(nrows=a + 1, sharex=True, sharey=True, figsize=(10, 8))
     fig.suptitle(title)
     for ind, ax in enumerate(axes.flatten()):
         if ind == a:
             ax.plot(t, eog*escala*50, 'r', linewidth=1)
             ax.set(ylabel='eog')
             ax.set_xlim(xlim)
             ax.set_ylim(ylim)
             ax.yaxis.set_ticklabels([])
             ax.grid()
         else:
            ax.plot(t, x[ind, :]*escala, 'k', linewidth=1)
            aux = str(IC_name[ind]) + '\n (' + ('%.2f' % corrcoef[ind]) +')'
            ax.set(ylabel=aux)
            ax.set_ylim(ylim)
            ax.yaxis.set_ticklabels([])
            ax.grid()
     plt.show()

def rejeitaartefato(epoch_eeg, epoch_base, m_dp=3, m_percent=5, m_percent_total=10, amp=150e-6):

    discarded_m_percent = np.zeros(np.shape(epoch_eeg[:, :, 0]), dtype=bool)
    discarded_m_percent_total = np.zeros(np.shape(epoch_eeg[:, :, 0]), dtype=bool)
    discarded_amp = np.zeros(np.shape(epoch_eeg[:, :, 0]), dtype=bool)
    signal_discarded = []
    signal_out = []
    epoch_discarded = []

    dp_base = np.std(epoch_eeg[epoch_base, :, :], 2)
    dp_mean = np.mean(dp_base, 0)
    threshold = m_dp * dp_mean
    window_percent = np.ceil(np.size(epoch_eeg, 2) * (m_percent * 0.01))
    window_percent_total = np.ceil(np.size(epoch_eeg, 2) * (m_percent_total * 0.01))
    #print(window_percent)
    #print(window_percent_total)

    for ind_epoch in range(np.size(epoch_eeg, 0)):
        for ind_ch in range(np.size(epoch_eeg, 1)):
            sinal_teste = epoch_eeg[ind_epoch, ind_ch, :]
            xpos = threshold[ind_ch] < sinal_teste
            xneg = -threshold[ind_ch] > sinal_teste

            dif_neg = np.diff(np.where(np.concatenate(([xneg[0]], xneg[:-1] != xneg[1:], [True])))[0])[::2]
            dif_pos = np.diff(np.where(np.concatenate(([xpos[0]], xpos[:-1] != xpos[1:], [True])))[0])[::2]
            total_neg = np.sum(xneg)
            total_pos = np.sum(xpos)

            discarded_m_percent[ind_epoch, ind_ch] = any(dif_neg > window_percent) or any(dif_pos > window_percent)
            discarded_m_percent_total[ind_epoch, ind_ch] = (total_neg > window_percent_total) or (
                        total_pos > window_percent_total)
            discarded_amp[ind_epoch, ind_ch] = any(sinal_teste > amp) or any(sinal_teste < -amp)

        if (discarded_m_percent[ind_epoch, :].any() or
                discarded_m_percent_total[ind_epoch, :].any() or
                discarded_amp[ind_epoch, :].any()):
            signal_discarded.append(epoch_eeg[ind_epoch, :, :])
            epoch_discarded.append(ind_epoch)
        else:
            signal_out.append(epoch_eeg[ind_epoch, :, :])

    return np.array(signal_out), np.array(signal_discarded), np.array(epoch_discarded)


def ICA_comp_plot(raw, S, corrp, maxch=9):

    eog = raw.get_data(picks='eog').squeeze()
    fs = raw.info['sfreq']

    indplot = -(-np.size(S, 0) // maxch)
    IC_name = np.arange(0, np.size(S, 0))

    for auxplot in range(indplot):

        if auxplot < indplot:
            S_plot = S[auxplot * maxch:(auxplot + 1) * maxch, :]
            corrcoef = corrp[0, auxplot * maxch:(auxplot + 1) * maxch]
            IC = IC_name[auxplot * maxch:(auxplot + 1) * maxch]

        else:
            S_plot = S[auxplot * maxch:, :]
            corrcoef = corrp[0, auxplot * maxch:]
            IC = IC_name[auxplot * maxch:, :]

        ica_plot(S_plot, eog, fs, IC, corrcoef, 'ICA ' + str(auxplot), 2e4)


def recon_plot(raw, eeg_recon, maxch=9):
    eeg = raw.get_data(picks='eeg')
    fs = raw.info['sfreq']
    ch_names = raw.info['ch_names']
    indplot = -(-np.size(eeg, 0) // maxch)
    for auxplot in range(indplot):

        if auxplot < indplot:
            eeg_recon_plot = eeg_recon[auxplot * maxch:(auxplot + 1) * maxch, :]
            eeg_plot = eeg[auxplot * maxch:(auxplot + 1) * maxch, :]
            ch = ch_names[auxplot * maxch:(auxplot + 1) * maxch]

        else:
            eeg_recon_plot = eeg_recon[auxplot * maxch:, :]
            eeg_plot = eeg[0, auxplot * maxch:]
            ch = ch_names[auxplot * maxch:, :]

        comp_plot(eeg_recon_plot, eeg_plot, fs, 1e-4, ch, 'Comparacao ' + str(auxplot))

# Função para processar o sinal de EMG
def emg_filter(emg, fs):
    emg1 = np.squeeze(emg[:, 0, :])
    emg2 = np.squeeze(emg[:, 1, :])
    f1 = 20
    f2 = 200
    order = 2
    emg1_filt = my_filter(emg1, fs, order, f1, f2)
    emg2_filt = my_filter(emg2, fs, order, f1, f2)
    emg1_filt = (emg1_filt) ** 2 # Transformar para energia
    emg2_filt = (emg2_filt) ** 2
    emg1_filt = np.mean(emg1_filt, 0)
    emg2_filt = np.mean(emg2_filt, 0)
    emg1_norm = (emg1_filt - np.min(emg1_filt)) / (np.max(emg1_filt) - np.min(emg1_filt))
    emg2_norm = (emg2_filt - np.min(emg2_filt)) / (np.max(emg2_filt) - np.min(emg2_filt))
    emg_norm = np.vstack((emg1_norm, emg2_norm))
    return emg_norm

def strided_app(a, L, S ):  # Window len = L, Stride len/stepsize = S
    nrows = ((a.size-L)//S)+1
    n = a.strides[0]
    return np.lib.stride_tricks.as_strided(a, shape=(nrows, L), strides=(S*n, n))

def erds(eeg, t1, t2, freq1, freq2, fs):
    # Constantes
    ## Filtros
    order = 4
    ## Janelamento ERDS
    w_size = 0.5
    noverlap = 0.5
    ref = 1
    ## MA
    w = 3

    vol2 = len(eeg)
    n_ch = np.size(eeg[0], 1)
    n = np.size(eeg[0], 2)

    # ERP
    erp_vol = np.zeros((vol2, n_ch, n))
    for ind in range(vol2):
        erp_vol[ind, :, :] = np.mean(eeg[ind], axis=0)
    erp = np.mean(erp_vol, axis=0)

    # ERDS - Subtração do ERP, Filtragem por Banda
    eeg_filt = []
    for ind in range(vol2):
        eeg_epocas_filt = []
        eeg_epocas_vol = eeg[ind]
        for ind_epoca in range(np.size(eeg_epocas_vol, 0)):
            sinal1 = eeg_epocas_vol[ind_epoca, :, :] - erp[:, :]  # -ERP
            sinal_aux1 = my_filter(sinal1, fs, order, freq1, freq2)
            eeg_epocas_filt.append(sinal_aux1)
        eeg_epocas_filt = np.array(eeg_epocas_filt)
        eeg_filt.append(eeg_epocas_filt)

    # ERDS
    ERDS_vol = []
    ERDS_vol_N = []
    for ind in range(vol2):
        ERDS_ind = [] #janela
        ERDS_ind_N = [] #amostra
        eeg_epocas_vol = eeg_filt[ind]
        for ind_ch in range(n_ch):
            eeg_ch = eeg_epocas_vol[:, ind_ch, :-1]
            eeg_pwr = eeg_ch ** 2
            P_0 = np.mean(eeg_pwr, 0)  # Media das épocas
            P_w = strided_app(P_0, int(w_size * fs), int(noverlap * w_size * fs))  # Janelamento
            R_erds = np.mean(P_0[0:int(ref * fs)])  # Referencia
            P_erds = np.mean(P_w, 1)  # Média de cada janela
            ERDS_ch = ((P_erds - R_erds) / R_erds) * 100
            ERDS_ch_N = ((P_0 - R_erds) / R_erds) * 100
            ERDS_ind.append(ERDS_ch) # ERDS com janelamento
            ERDS_ind_N.append(ERDS_ch_N) # ERDS sem janelamento

        ERDS_vol.append(ERDS_ind)
        ERDS_vol_N.append(ERDS_ind_N)

    ERDS_avg = np.mean(np.array(ERDS_vol), axis=0)  # Média dos voluntários com janeamento
    ERDS_avg_N = np.mean(np.array(ERDS_vol_N), axis=0) # Média dos voluntários sem janelamento

    # ERDS - Filtragem (Moving Average)
    ERDS_MA = []
    ERDS_N_MA = []
    for ind_ch in range(n_ch):
        ERDS_MA.append(np.convolve(ERDS_avg[ind_ch], np.ones(w) / w, 'valid'))

    t_erds = strided_app(np.linspace(t1, t2, np.size(P_0)), int(w_size * fs), int(noverlap * w_size * fs))
    t_w = np.convolve(t_erds[:, 0], np.ones(w) / w, 'valid')

    for ind_ch in range(n_ch):
        ERDS_N_MA.append(np.convolve(ERDS_avg_N[ind_ch], np.ones(w) / w, 'valid'))

    t_N = np.linspace(t1, t2, np.size(P_0))

    return ERDS_MA, t_w, erp, ERDS_vol, t_erds[:, 0], erp_vol, ERDS_avg, ERDS_avg_N, ERDS_vol_N, ERDS_N_MA, t_N

def plotersp(erds_1, erds_2, t_erds, y1, title, line_labels, ch_name):
    fig, axes = plt.subplots(nrows=4, ncols=5, sharey=True, sharex=True, figsize=(12, 8))
    axes = axes.flatten()
    ylim = [-y1, y1]
    ch_v = [0, 5, 10, 1, 6, 11, 16, 4, 9, 14, 3, 8, 13, 18, 2, 7, 12]
    ind = 0
    #fig.subplots_adjust(wspace=0.1, hspace=0.5)
    #minor_ticks_top = np.linspace(t1, t2, 2*(abs(t1) + t2) + 1)
    #major_ticks_top = np.linspace(t1, t2, abs(t1) + t2 + 1)
    majorLocator = MultipleLocator(2)
    majorFormatter = FormatStrFormatter('%d')
    minorLocator = MultipleLocator(1)

    for ind_ch in ch_v:
        plt.suptitle(title)
        l1 = axes[ind_ch].plot(t_erds, erds_1[ind])
        l2 = axes[ind_ch].plot(t_erds, erds_2[ind])
        #axes[ind_ch].set_xticks(major_ticks_top)
        #axes[ind_ch].set_xticks(minor_ticks_top, minor=True)
        axes[ind_ch].xaxis.set_major_locator(majorLocator)
        axes[ind_ch].xaxis.set_major_formatter(majorFormatter)
        axes[ind_ch].xaxis.set_minor_locator(minorLocator)
        axes[ind_ch].grid(which="major", alpha=0.6)
        axes[ind_ch].grid(which="minor", alpha=0.3)
        axes[ind_ch].axvline(x=0, linewidth=1, color='r', linestyle="--")
        axes[ind_ch].axvline(x=2.5, linewidth=1, color='g', linestyle="--")
        axes[ind_ch].set_title(ch_name[ind])
        axes[ind_ch].set_ylim(ylim)
        ind += 1

    fig.delaxes(axes[15])
    fig.delaxes(axes[17])
    fig.delaxes(axes[19])

    fig.tight_layout()
    fig.legend([l1, l2],  # The line objects
               labels=line_labels,  # The labels for each line
               loc="upper right",  # Position of legend
               #borderaxespad=0.1,  # Small spacing around legend box
               # title="Legend Title"  # Title for the legend
               )
    plt.show()

def topographicmap(erds, tmin, tmax, t0, title, ch_name, vmin, vmax): # função antiga, para usar com mais eletrodos

    t_atividade = np.where((np.round(t0,2) >= tmin) & (np.round(t0,2) <= tmax))

    ch_row = [0, 0, 0, 1, 1,  # Coordenadas exemplo (0,0) = ch1
              1, 2, 2, 2, 0,
              0, 1, 1, 2, 2,
              3]
    ch_col = [1, 3, 2, 1, 3,
              2, 1, 3, 2, 0,
              4, 0, 4, 0, 4,
              2]

    Zvector = np.zeros(len(erds))
    for ind in range(len(erds)):
        Zind = erds[ind, t_atividade]
        # Zvector[ind] = max(Zind.min(), Zind.max(), key=abs)
        Zvector[ind] = np.mean(Zind)

    Zmatrix = np.zeros((4, 5))
    for ind in range(len(Zvector)):
        Zmatrix[ch_row[ind], ch_col[ind]] = Zvector[ind]
    Zmatrix[Zmatrix == 0] = np.nan

    fig, (ax) = plt.subplots(1, 1)
    plt.title(title)
    Z = np.ma.masked_where(np.isnan(Zmatrix), Zmatrix)
    c = ax.pcolormesh(Z, shading='flat', edgecolors='w', linewidths=10, vmin=vmin, vmax=vmax)
    plt.gca().invert_yaxis()

    for ind in range(len(ch_name)):
        if not (Z.mask[ch_row[ind], ch_col[ind]]):
            plt.text(ch_col[ind] + 0.5, ch_row[ind] + 0.5, ch_name[ind],
                     horizontalalignment='center',
                     verticalalignment='center',
                     color='w', fontsize='14'
                     )

    fig.colorbar(c)
    plt.axis('off')
    plt.show()

def topographicmap2(erds, tmin, tmax, t0, title, ch_name, vmin, vmax): #função nova, c3, c4, p3, p4, f3, f4

    t_atividade = np.where((np.round(t0,2) >= tmin) & (np.round(t0,2) <= tmax))

    ch_v = [0, 2, 1,
            3, 5, 4,
            6, 8, 7,
            9]

    ch_row = [0, 0, 0,
              1, 1, 1,
              2, 2, 2,
              3]

    ch_col = [0, 1, 2,
              0, 1, 2,
              0, 1, 2,
              1]

    Zvector = np.zeros(len(ch_v))
    ind = 0
    for ind_ch in ch_v:
        Zind = erds[ind_ch, t_atividade]
        # Zvector[ind] = max(Zind.min(), Zind.max(), key=abs)
        Zvector[ind] = np.mean(Zind)
        ind += 1

    Zmatrix = np.zeros((4, 3))
    for ind in range(len(Zvector)):
        Zmatrix[ch_row[ind], ch_col[ind]] = Zvector[ind]
    Zmatrix[Zmatrix == 0] = np.nan

    fig, (ax) = plt.subplots(1, 1)
    plt.title(title)
    Z = np.ma.masked_where(np.isnan(Zmatrix), Zmatrix)
    c = ax.pcolormesh(Z, shading='flat', edgecolors='w', linewidths=10, vmin=vmin, vmax=vmax)
    plt.gca().invert_yaxis()

    for ind in range(len(ch_v)):
        if not (Z.mask[ch_row[ind], ch_col[ind]]):
            plt.text(ch_col[ind] + 0.5, ch_row[ind] + 0.5, ch_name[ch_v[ind]],
                     horizontalalignment='center',
                     verticalalignment='center',
                     color='w', fontsize='14'
                     )

    fig.colorbar(c)
    plt.axis('off')
    plt.show()

def erdstest(erds,tmin,tmax,t,ch):
    t_atividade = np.where((np.round(t,2) >= tmin) & (np.round(t,2) <= tmax))
    erds = np.array(erds)
    aux_atividade = np.squeeze(erds[:, :, t_atividade])

    erds_mean =[]
    erds_max = []
    for ind in range(len(ch)):
        aux = np.squeeze(aux_atividade[:, ch[ind], :])
        erds_mean.append(np.mean(aux,1))
        erds_max_vol = []
        for ind2 in range(len(aux_atividade)):
            erds_max_vol.append(max(aux[ind2, :].min(), aux[ind2, :].max(), key=abs))
        erds_max.append(np.array(erds_max_vol))
    return np.array(erds_mean), np.array(erds_max)

def analisestest(erds_mean, voluntarios, ch_name, ch_box, title0):
    fig, ax = plt.subplots(1, figsize=(10,6))
    fig.suptitle('ERDS Mean - ' + title0)
    ax.boxplot(np.transpose(erds_mean[:,voluntarios]), labels=[ch_name[ind] for ind in ch_box])
    ax.set_ylabel('Amplitude [uV]')
    for ind in range(3):
        ch = [ind * 2, ind * 2 + 1]
        a = stats.wilcoxon(erds_mean[ch[0], voluntarios], erds_mean[ch[1], voluntarios], alternative='two-sided')
        x1, x2 = ch[0] + 1, ch[1] + 1
        y, h, col = np.max([erds_mean[ch[0], voluntarios], erds_mean[ch[1], voluntarios]]) + 2, 2, 'k'
        plt.plot([x1, x1, x2, x2], [y, y + h, y + h, y], lw=1.5, c=col)
        plt.text((x1 + x2) * .5, y + h, "{:.4f}".format(a[1]), ha='center', va='bottom', color=col)
    plt.show()

def analisesplot(erds_mean, ch, ch_name, ch_box, title0):
    erds_teste = np.transpose(erds_mean[ch, :])
    aux_teste = erds_mean[ch[0]] - erds_mean[ch[1]]
    MV_testes = np.column_stack((erds_teste, aux_teste))
    label_ch = [ch_name[ind] for ind in ch_box[ch]]
    label_ch.append(ch_name[ch_box[ch[0]]] + ' - ' + ch_name[ch_box[ch[1]]])

    fig, ax = plt.subplots(2, figsize=(10, 6))
    fig.suptitle('ERDS Mean - ' + title0)
    ax[0].boxplot(MV_testes, labels=label_ch)
    ax[0].set_ylabel('Amplitude [uV]')
    ax[1].plot(erds_mean[ch[0]], 'xr', label=ch_name[ch_box[ch[0]]])
    ax[1].plot(erds_mean[ch[1]], 'xb', label=ch_name[ch_box[ch[1]]])
    ax[1].set_ylabel(ch_name[ch_box[ch[0]]] + ' - ' + ch_name[ch_box[ch[1]]])
    ax[1].legend()
    plt.show()

def erdsvol(eeg, t1, t2, freq1, freq2, fs):
    # Mudança no ERDS_vol para plot (plot precisa do filtro MA)
    # Para estatistica não precisa do MA

    # Constantes
    ## Filtros
    order = 4
    ## Janelamento ERDS
    w_size = 0.5
    noverlap = 0.5
    ref = 1
    ## MA
    w = 3

    vol2 = len(eeg)
    n_ch = np.size(eeg[0], 1)
    n = np.size(eeg[0], 2)

    # ERP
    erp_vol = np.zeros((vol2, n_ch, n))
    for ind in range(vol2):
        erp_vol[ind, :, :] = np.mean(eeg[ind], axis=0)
    erp = np.mean(erp_vol, axis=0)

    # ERDS - Subtração do ERP, Filtragem por Banda
    eeg_filt = []
    for ind in range(vol2):
        eeg_epocas_filt = []
        eeg_epocas_vol = eeg[ind]
        for ind_epoca in range(np.size(eeg_epocas_vol, 0)):
            sinal1 = eeg_epocas_vol[ind_epoca, :, :] - erp[:, :]  # -ERP
            sinal_aux1 = my_filter(sinal1, fs, order, freq1, freq2)
            eeg_epocas_filt.append(sinal_aux1)
        eeg_epocas_filt = np.array(eeg_epocas_filt)
        eeg_filt.append(eeg_epocas_filt)

    # ERDS
    ERDS_vol = []
    ERDS_vol_N = []
    for ind in range(vol2):
        ERDS_ind = []  # janela
        ERDS_ind_N = []  # amostra
        eeg_epocas_vol = eeg_filt[ind]
        for ind_ch in range(n_ch):
            eeg_ch = eeg_epocas_vol[:, ind_ch, :-1]
            eeg_pwr = eeg_ch ** 2
            P_0 = np.mean(eeg_pwr, 0)  # Media das épocas
            P_w = strided_app(P_0, int(w_size * fs), int(noverlap * w_size * fs))  # Janelamento
            R_erds = np.mean(P_0[0:int(ref * fs)])  # Referencia
            P_erds = np.mean(P_w, 1)  # Média de cada janela
            ERDS_ch = ((P_erds - R_erds) / R_erds) * 100
            ERDS_ch_N = ((P_0 - R_erds) / R_erds) * 100
            ERDS_ind.append(ERDS_ch)
            ERDS_ind_N.append(ERDS_ch_N)

        ERDS_vol.append(ERDS_ind)
        ERDS_vol_N.append(ERDS_ind_N)

    ERDS_avg = np.mean(np.array(ERDS_vol), axis=0)  # Média dos voluntários

    # ERDS - Filtragem (Moving Average)
    ERDS_MA = []

    for ind_ch in range(n_ch):
        ERDS_MA.append(np.convolve(ERDS_avg[ind_ch], np.ones(w) / w, 'same'))

    ERDS_vol_MA = []

    for ind_v in range(vol2):
        ERDS_vol_MA_ch = []
        ERDS_vol_aux = ERDS_vol[ind_v]
        for ind_ch in range(n_ch):

            ERDS_vol_MA_ch.append(np.convolve(ERDS_vol_aux[ind_ch], np.ones(w) / w, 'same'))

        ERDS_vol_MA.append(ERDS_vol_MA_ch)
    t_erds = strided_app(np.linspace(t1, t2, np.size(P_0)), int(w_size * fs), int(noverlap * w_size * fs))
    # t_w = np.convolve(t_erds[:, 0], np.ones(w) / w, 'valid')

    return ERDS_MA, erp, ERDS_vol, t_erds[:, 0], erp_vol, ERDS_vol_MA

def plotersp34(erds, t_erds, y1, title, ch_name):
    ch_3 = [0, 3, 6]
    ch_4 = [1, 4, 7]

    fig, axes = plt.subplots(nrows=3, ncols=1, sharey=True, sharex=True, figsize=(6, 8))
    plt.suptitle(title)
    axes = axes.flatten()
    ylim = [-y1, y1]

    majorLocator_x = MultipleLocator(2)
    majorFormatter_x = FormatStrFormatter('%d')
    minorLocator_x = MultipleLocator(1)
    majorLocator_y = MultipleLocator(25)
    majorFormatter_y = FormatStrFormatter('%d')


    ind_plot = 0
    for ind_ch in range(len(ch_3)):
        l1 = axes[ind_plot].plot(t_erds, erds[ch_3[ind_ch]], label=ch_name[ch_3[ind_ch]])
        l2 = axes[ind_plot].plot(t_erds, erds[ch_4[ind_ch]], label=ch_name[ch_4[ind_ch]])
        axes[ind_plot].xaxis.set_major_locator(majorLocator_x)
        axes[ind_plot].xaxis.set_major_formatter(majorFormatter_x)
        axes[ind_plot].xaxis.set_minor_locator(minorLocator_x)
        axes[ind_plot].yaxis.set_major_locator(majorLocator_y)
        axes[ind_plot].yaxis.set_major_formatter(majorFormatter_y)
        axes[ind_plot].grid(which="major", alpha=0.6)
        axes[ind_plot].grid(which="minor", alpha=0.3)
        axes[ind_plot].axvline(x=0, linewidth=1, color='y', linestyle="--")
        axes[ind_plot].axvline(x=2, linewidth=1, color='g', linestyle="--")
        name = ch_name[ch_3[ind_ch]] + ' - ' + ch_name[ch_4[ind_ch]]
        axes[ind_plot].set_title(name)
        axes[ind_plot].set_ylim(ylim)
        axes[ind_plot].legend(loc="upper left")
        axes[ind_plot].set_ylabel('% ERD/ERS')
        ind_plot += 1

    fig.tight_layout()
    plt.show()

def ploterspanalise(erds, erds_vol, t_erds, y1, title, ch_name, ch_box, voluntarios, c):
    tini_1 = 0
    tfim_1 = 4.5
    tini_2 = 6
    tfim_2 = 7.5
    janela = 0.5
    tmin_1 = np.arange(tini_1, tfim_1, janela)
    tmax_1 = np.arange(tini_1 + janela, tfim_1 + janela, janela)
    tmin_2 = np.arange(tini_2, tfim_2, janela)
    tmax_2 = np.arange(tini_2 + janela, tfim_2 + janela, janela)
    tmin_v = np.concatenate((tmin_1, tmin_2), axis=0)
    tmax_v = np.concatenate((tmax_1, tmax_2), axis=0)
    alpha = 0.05

    fig, axes = plt.subplots(nrows=3, ncols=1, sharey=True, sharex=True, figsize=(6, 8))
    plt.suptitle(title)
    axes = axes.flatten()
    ylim = [-y1, y1]

    majorLocator_x = MultipleLocator(1)
    majorFormatter_x = FormatStrFormatter('%d')
    minorLocator_x = MultipleLocator(0.5)
    majorLocator_y = MultipleLocator(25)
    majorFormatter_y = FormatStrFormatter('%d')

    ind_plot = 0
    for ind_ch in range(int(len(ch_box) / 2)):
        l1 = axes[ind_plot].plot(t_erds, erds[ch_box[ind_ch * 2]], label=ch_name[ch_box[ind_ch * 2]])
        l2 = axes[ind_plot].plot(t_erds, erds[ch_box[ind_ch * 2 + 1]], label=ch_name[ch_box[ind_ch * 2 + 1]])
        axes[ind_plot].xaxis.set_major_locator(majorLocator_x)
        axes[ind_plot].xaxis.set_major_formatter(majorFormatter_x)
        axes[ind_plot].xaxis.set_minor_locator(minorLocator_x)
        axes[ind_plot].yaxis.set_major_locator(majorLocator_y)
        axes[ind_plot].yaxis.set_major_formatter(majorFormatter_y)
        axes[ind_plot].grid(which="major", alpha=0.6)
        axes[ind_plot].grid(which="minor", alpha=0.3)
        axes[ind_plot].axvline(x=0, linewidth=1, color='y', linestyle="--")
        axes[ind_plot].axvline(x=2, linewidth=1, color='g', linestyle="--")
        name = ch_name[ch_box[ind_ch * 2]] + ' - ' + ch_name[ch_box[ind_ch * 2 + 1]]
        axes[ind_plot].set_title(name)
        axes[ind_plot].set_ylim(ylim)
        axes[ind_plot].legend(loc="upper left")
        axes[ind_plot].set_ylabel('% ERD/ERS')
        ind_plot += 1

    col_t = []

    for ind_t in range(len(tmin_v)):
        erds_mean, _ = erdstest(erds_vol, tmin_v[ind_t], tmax_v[ind_t], t_erds, ch_box)
        row_ch = []

        for ind in range(int(len(ch_box) / 2)):
            ch = [ind * 2, ind * 2 + 1]
            a = stats.wilcoxon(erds_mean[ch[0], voluntarios], erds_mean[ch[1], voluntarios], alternative='two-sided')
            print(ch)
            print([tmin_v[ind_t], tmax_v[ind_t]])
            print(a[1])
            row_ch.append(a[1])
            if a[1] <= alpha:
                axes[ind].axvspan(tmin_v[ind_t], tmax_v[ind_t], color=c)
        col_t.append(row_ch)
    table = np.array(col_t).transpose()

    fig.tight_layout()
    plt.show()
    return table, tmin_v

def ploterspanalise2(erds, erds_vol, t_erds, y1, title, ch_name, ch_box, voluntarios, c):
    tini_1 = 0
    tfim_1 = 4.5
    tini_2 = 6
    tfim_2 = 7.5
    janela = 0.5
    tmin_1 = np.arange(tini_1, tfim_1, janela)
    tmax_1 = np.arange(tini_1 + janela, tfim_1 + janela, janela)
    tmin_2 = np.arange(tini_2, tfim_2, janela)
    tmax_2 = np.arange(tini_2 + janela, tfim_2 + janela, janela)
    tmin_v = np.concatenate((tmin_1, tmin_2), axis=0)
    tmax_v = np.concatenate((tmax_1, tmax_2), axis=0)
    alpha = 0.05

    fig, axes = plt.subplots(nrows=3, ncols=1, sharey=True, sharex=True, figsize=(6, 8))
    plt.suptitle(title)
    axes = axes.flatten()
    ylim = [-y1, y1]

    majorLocator_x = MultipleLocator(1)
    majorFormatter_x = FormatStrFormatter('%d')
    minorLocator_x = MultipleLocator(0.5)
    majorLocator_y = MultipleLocator(25)
    majorFormatter_y = FormatStrFormatter('%d')

    ind_plot = 0
    for ind_ch in range(int(len(ch_box) / 2)):
        l1 = axes[ind_plot].plot(t_erds, erds[ch_box[ind_ch * 2]], label=ch_name[ch_box[ind_ch * 2]])
        l2 = axes[ind_plot].plot(t_erds, erds[ch_box[ind_ch * 2 + 1]], label=ch_name[ch_box[ind_ch * 2 + 1]])
        axes[ind_plot].xaxis.set_major_locator(majorLocator_x)
        axes[ind_plot].xaxis.set_major_formatter(majorFormatter_x)
        axes[ind_plot].xaxis.set_minor_locator(minorLocator_x)
        axes[ind_plot].yaxis.set_major_locator(majorLocator_y)
        axes[ind_plot].yaxis.set_major_formatter(majorFormatter_y)
        axes[ind_plot].grid(which="major", alpha=0.6)
        axes[ind_plot].grid(which="minor", alpha=0.3)
        axes[ind_plot].axvline(x=0, linewidth=1, color='y', linestyle="--")
        axes[ind_plot].axvline(x=2, linewidth=1, color='g', linestyle="--")
        name = ch_name[ch_box[ind_ch * 2]] + ' - ' + ch_name[ch_box[ind_ch * 2 + 1]]
        axes[ind_plot].set_title(name)
        axes[ind_plot].set_ylim(ylim)
        axes[ind_plot].legend(loc="upper left")
        axes[ind_plot].set_ylabel('% ERD/ERS')
        ind_plot += 1

    col_t = []

    for ind_t in range(len(tmin_v)):
        erds_mean, _ = erdstest(erds_vol, tmin_v[ind_t], tmax_v[ind_t], t_erds, ch_box)
        row_ch = []

        for ind in range(int(len(ch_box) / 2)):
            ch = [ind * 2, ind * 2 + 1]
            # a = stats.wilcoxon(erds_mean[ch[0], voluntarios], erds_mean[ch[1], voluntarios], alternative='two-sided')
            print(ch)
            print([tmin_v[ind_t], tmax_v[ind_t]])
            #print(a[1])
            #row_ch.append(a[1])
            #if a[1] <= alpha:
            #   axes[ind].axvspan(tmin_v[ind_t], tmax_v[ind_t], color=c)
        #col_t.append(row_ch)
    table = np.array(col_t).transpose()

    fig.tight_layout()
    plt.show()
    return table, tmin_v

def ploterspMVIMG(MV_erds, MV_erds_vol, IM_erds, IM_erds_vol, t_erds, y1, title, ch_name, ch_box, c):
    tini_1 = 0
    tfim_1 = 4.5
    tini_2 = 6
    tfim_2 = 7.5
    janela = 0.5
    tmin_1 = np.arange(tini_1, tfim_1, janela)
    tmax_1 = np.arange(tini_1 + janela, tfim_1 + janela, janela)
    tmin_2 = np.arange(tini_2, tfim_2, janela)
    tmax_2 = np.arange(tini_2 + janela, tfim_2 + janela, janela)
    tmin_v = np.concatenate((tmin_1, tmin_2), axis=0)
    tmax_v = np.concatenate((tmax_1, tmax_2), axis=0)
    alpha = 0.05

    fig, axes = plt.subplots(nrows=3, ncols=1, sharey=True, sharex=True, figsize=(6, 8))
    plt.suptitle(title)
    axes = axes.flatten()
    ylim = [-y1, y1]

    majorLocator_x = MultipleLocator(1)
    majorFormatter_x = FormatStrFormatter('%d')
    minorLocator_x = MultipleLocator(0.5)
    majorLocator_y = MultipleLocator(25)
    majorFormatter_y = FormatStrFormatter('%d')

    ind_plot = 0
    for ind_ch in range(int(len(ch_box))):
        l1 = axes[ind_plot].plot(t_erds, MV_erds[ch_box[ind_ch]], label='MV')
        l2 = axes[ind_plot].plot(t_erds, IM_erds[ch_box[ind_ch]], label='MI')
        axes[ind_plot].xaxis.set_major_locator(majorLocator_x)
        axes[ind_plot].xaxis.set_major_formatter(majorFormatter_x)
        axes[ind_plot].xaxis.set_minor_locator(minorLocator_x)
        axes[ind_plot].yaxis.set_major_locator(majorLocator_y)
        axes[ind_plot].yaxis.set_major_formatter(majorFormatter_y)
        axes[ind_plot].grid(which="major", alpha=0.6)
        axes[ind_plot].grid(which="minor", alpha=0.3)
        axes[ind_plot].axvline(x=0, linewidth=1, color='r', linestyle="--")
        axes[ind_plot].axvline(x=2.5, linewidth=1, color='g', linestyle="--")
        axes[ind_plot].set_title(ch_name[ch_box[ind_ch]])
        axes[ind_plot].set_ylim(ylim)
        axes[ind_plot].legend(loc="upper left")
        axes[ind_plot].set_ylabel('% ERD/ERS')
        ind_plot += 1

    col_t = []

    for ind_t in range(len(tmin_v)):
        MV_erds_mean, _ = erdstest(MV_erds_vol, tmin_v[ind_t], tmax_v[ind_t], t_erds, ch_box)
        IM_erds_mean, _ = erdstest(IM_erds_vol, tmin_v[ind_t], tmax_v[ind_t], t_erds, ch_box)
        row_ch = []
        for ind in range(int(len(ch_box))):
            a = stats.wilcoxon(MV_erds_mean[ind, :], IM_erds_mean[ind, :], alternative='two-sided')
            row_ch.append(a[1])
            if a[1] <= alpha:
                axes[ind].axvspan(tmin_v[ind_t], tmax_v[ind_t], color=c)

        col_t.append(row_ch)

    table = np.array(col_t).transpose()

    fig.tight_layout()
    plt.show()

    return table, tmin_v