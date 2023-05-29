
import mne
import numpy as np
import scipy.stats
from scipy.stats import wilcoxon
import matplotlib.pyplot as plt
import pipeline_functions as pf
%matplotlib qt5

# Loading raw dataset  
path = "E:\Projetos\FisCog\Arthur Moreira_14032023\Arthur_Moreira.edf"
dados_raw, info = pf.abre_preprocessa_reamostra(path)

layout = mne.channels.find_layout(dados_raw.info)

condicoes = {'esq': {'controle_wait': {'data_folder': 'E:\Projetos\FisCog\Controle', 'voluntarios': [i+1 for i in range(12)], 'epoca': '10-epo'},
                     'controle_exe':  {'data_folder': 'E:\Projetos\FisCog\Controle', 'voluntarios': [i+1 for i in range(12)], 'epoca': '11-epo'},
                     'imaginacao_exe': {'data_folder': 'E:\Projetos\FisCog\Imaginacao', 'voluntarios': [i+1 for i in range(12)], 'epoca': '11-epo'},
                     'imaginacao_img': {'data_folder': 'E:\Projetos\FisCog\Imaginacao', 'voluntarios': [i+1 for i in range(12)], 'epoca': '10-epo'}},
            
             'dir': {'controle_wait': {'data_folder': 'E:\Projetos\FisCog\Controle', 'voluntarios': [i+1 for i in range(12)], 'epoca': '20-epo'},
                     'controle_exe':  {'data_folder': 'E:\Projetos\FisCog\Controle', 'voluntarios': [i+1 for i in range(12)], 'epoca': '21-epo'},
                     'imaginacao_exe': {'data_folder': 'E:\Projetos\FisCog\Imaginacao', 'voluntarios': [i+1 for i in range(12)], 'epoca': '21-epo'},
                     'imaginacao_img': {'data_folder': 'E:\Projetos\FisCog\Imaginacao', 'voluntarios': [i+1 for i in range(12)], 'epoca': '20-epo'}}}

condicoes_epocas = {}

def gerar_erp(condicoes):
    dados_erp = {}
    dados_erp_mean = {}
    
    for movimento, condicoes_dir in condicoes.items():
        for condicao, info in condicoes_dir.items():
            dados_epocas = []
            for i in info['voluntarios']:
                data_folder = info['data_folder'] + (f'\S0{i}' if i <= 9 else f'\S{i}') + '\Sem re-referenciar'
                voluntario = (f'\S0{i}' if i <= 9 else f'\S{i}') + '_' + info['epoca']

                print(data_folder + voluntario)

                epocas = mne.read_epochs(data_folder + voluntario + '.fif')
                dados_epocas.append(epocas.get_data(picks='eeg'))

            condicoes_epocas[condicao] = epocas
            voluntarios = len(dados_epocas)
            dados_erp[condicao] = np.zeros((voluntarios, 30, 6001))

            for indice in range(voluntarios):
                dados_erp[condicao][indice,:,:] = np.mean(dados_epocas[indice], axis=0)

            dados_erp_mean[condicao] = np.mean(dados_erp[condicao], axis=0)

        dados_erp[f'{movimento}_controle_wait'], dados_erp_mean[f'{movimento}_controle_wait'] = dados_erp['controle_wait'], dados_erp_mean['controle_wait']
        dados_erp[f'{movimento}_controle_exe'], dados_erp_mean[f'{movimento}_controle_exe'] = dados_erp['controle_exe'], dados_erp_mean['controle_exe']
        dados_erp[f'{movimento}_imaginacao_exe'], dados_erp_mean[f'{movimento}_imaginacao_exe'] = dados_erp['imaginacao_exe'], dados_erp_mean['imaginacao_exe']
        dados_erp[f'{movimento}_imaginacao_img'], dados_erp_mean[f'{movimento}_imaginacao_img'] = dados_erp['imaginacao_img'], dados_erp_mean['imaginacao_img']
       
    return (dados_erp['esq_controle_wait'], dados_erp_mean['esq_controle_wait'], 
            dados_erp['esq_controle_exe'],  dados_erp_mean['esq_controle_exe'], 
            dados_erp['esq_imaginacao_exe'], dados_erp_mean['esq_imaginacao_exe'], 
            dados_erp['esq_imaginacao_img'], dados_erp_mean['esq_imaginacao_img'],
            dados_erp['dir_controle_wait'], dados_erp_mean['dir_controle_wait'], 
            dados_erp['dir_controle_exe'],  dados_erp_mean['dir_controle_exe'], 
            dados_erp['dir_imaginacao_exe'], dados_erp_mean['dir_imaginacao_exe'], 
            dados_erp['dir_imaginacao_img'], dados_erp_mean['dir_imaginacao_img'])

dados_erp = {}
dados_erp_mean = {}

(dados_erp['esq_controle_wait'], dados_erp_mean['esq_controle_wait'], dados_erp['esq_controle_exe'], 
 dados_erp_mean['esq_controle_exe'], dados_erp['esq_imaginacao_exe'], dados_erp_mean['esq_imaginacao_exe'], 
 dados_erp['esq_imaginacao_img'], dados_erp_mean['esq_imaginacao_img'],dados_erp['dir_controle_wait'], 
 dados_erp_mean['dir_controle_wait'], dados_erp['dir_controle_exe'], dados_erp_mean['dir_controle_exe'], 
 dados_erp['dir_imaginacao_exe'], dados_erp_mean['dir_imaginacao_exe'], dados_erp['dir_imaginacao_img'], 
 dados_erp_mean['dir_imaginacao_img']) = gerar_erp(condicoes)

# epocas_controle_exe = condicoes_epocas['controle_exe']
# epocas_controle_wait = condicoes_epocas['controle_wait']
# epocas_imaginacao_exe = condicoes_epocas['imaginacao_exe']
# epocas_imaginacao_img = condicoes_epocas['imaginacao_img']

# If you want to access a single subject, use these variables
subject = 0
dados_erp_controle_exe_subject_esq = dados_erp['esq_controle_exe'][subject,:,:]
dados_erp_controle_wait_subject_esq = dados_erp['esq_controle_wait'][subject,:,:]
dados_erp_imaginacao_exe_subject_esq = dados_erp['esq_imaginacao_exe'][subject,:,:]
dados_erp_imaginacao_img_subject_esq = dados_erp['esq_imaginacao_img'][subject,:,:]

dados_erp_controle_exe_subject_dir = dados_erp['dir_controle_exe'][subject,:,:]
dados_erp_controle_wait_subject_dir = dados_erp['dir_controle_wait'][subject,:,:]
dados_erp_imaginacao_exe_subject_dir = dados_erp['dir_imaginacao_exe'][subject,:,:]
dados_erp_imaginacao_img_subject_dir = dados_erp['dir_imaginacao_img'][subject,:,:]


condicoes = {}

t1 = [:,1350:1800]
t2 = [:,2400:3000]

condicoes['conexe_esq_1'] = dados_erp_mean['esq_controle_exe'][t1]
condicoes['conexe_esq_2'] = dados_erp_mean['esq_controle_exe'][t2]
condicoes['conwai_esq_1'] = dados_erp_mean['esq_controle_wait'][t1]
condicoes['conwai_esq_2'] = dados_erp_mean['esq_controle_wait'][t2]
condicoes['imgexe_esq_1'] = dados_erp_mean['esq_imaginacao_exe'][t1]
condicoes['imgexe_esq_2'] = dados_erp_mean['esq_imaginacao_exe'][t2]
condicoes['imgimg_esq_1'] = dados_erp_mean['esq_imaginacao_img'][t1]
condicoes['imgimg_esq_2'] = dados_erp_mean['esq_imaginacao_img'][t2]

condicoes['conexe_dir_1'] = dados_erp_mean['dir_controle_exe'][t1]
condicoes['conexe_dir_2'] = dados_erp_mean['dir_controle_exe'][t2]
condicoes['conwai_dir_1'] = dados_erp_mean['dir_controle_wait'][t1]
condicoes['conwai_dir_2'] = dados_erp_mean['dir_controle_wait'][t2]
condicoes['imgexe_dir_1'] = dados_erp_mean['dir_imaginacao_exe'][t1]
condicoes['imgexe_dir_2'] = dados_erp_mean['dir_imaginacao_exe'][t2]
condicoes['imgimg_dir_1'] = dados_erp_mean['dir_imaginacao_img'][t1]
condicoes['imgimg_dir_2'] = dados_erp_mean['dir_imaginacao_img'][t2]

pontos = {}

for condicao, dados in condicoes.items(): 
    pontos[condicao+'_pos_i1'] = (np.argmax(np.max(dados,axis=0))+1350)//600
    pontos[condicao+'_neg_i1'] = (np.argmin(np.min(dados,axis=0))+1350)//600
    pontos[condicao+'_pos_i2'] = (np.argmax(np.max(dados,axis=0))+2400)//600
    pontos[condicao+'_neg_i2'] = (np.argmin(np.min(dados,axis=0))+2400)//600

# Evoked objects creation (Intervals in samples)
evoked_controle_exe_esq = mne.EvokedArray(dados_erp_mean['esq_controle_exe'][:,:3000], info=info, comment='Wait', 
tmin=0, baseline=(0,2))
evoked_controle_wait_esq = mne.EvokedArray(dados_erp_mean['esq_controle_wait'][:,:3000], info=info, comment='Execução', 
tmin=0, baseline=(0,2))
evoked_imaginacao_exe_esq = mne.EvokedArray(dados_erp_mean['esq_imaginacao_exe'][:,:3000], info=info, comment='Imaginação', 
tmin=0, baseline=(0,2)) 
evoked_imaginacao_img_esq = mne.EvokedArray(dados_erp_mean['esq_imaginacao_img'][:,:3000], info=info, comment='Imaginação', 
tmin=0, baseline=(0,2)) 

evoked_controle_exe_dir = mne.EvokedArray(dados_erp_mean['dir_controle_exe'][:,:3000], info=info, comment='Wait', 
tmin=0, baseline=(0,2))
evoked_controle_wait_dir = mne.EvokedArray(dados_erp_mean['dir_controle_wait'][:,:3000], info=info, comment='Execução', 
tmin=0, baseline=(0,2))
evoked_imaginacao_exe_dir = mne.EvokedArray(dados_erp_mean['dir_imaginacao_exe'][:,:3000], info=info, comment='Imaginação', 
tmin=0, baseline=(0,2)) 
evoked_imaginacao_img_dir = mne.EvokedArray(dados_erp_mean['dir_imaginacao_img'][:,:3000], info=info, comment='Imaginação', 
tmin=0, baseline=(0,2)) 

evokeds = dict(Controle_Exe_Esq=evoked_controle_exe_esq, Controle_Wait_Esq=evoked_controle_wait_esq, 
               Imaginação_Exe_Esq=evoked_imaginacao_exe_esq, Imaginação_Img_Esq=evoked_imaginacao_img_esq, 
               Controle_Exe_Dir=evoked_controle_exe_dir, Controle_Wait_Dir=evoked_controle_wait_dir, 
               Imaginação_Exe_Dir=evoked_imaginacao_exe_dir, Imaginação_Img_Dir=evoked_imaginacao_img_dir)

# Left
evoked_controle_exe_esq.plot_topomap(times=[1.5, pontos['conexe_esq_1_pos_i1'], pontos['conexe_esq_1_neg_i1'], pontos['conexe_esq_2_pos_i2'], pontos['conexe_esq_2_neg_i2']], 
                                 average=0.1, scalings=None, ch_type='eeg',
                                 sphere=None, image_interp='cubic', time_unit='s', time_format="%0.2f s" , vlim=(-3,3))
evoked_controle_wait_esq.plot_topomap(times=[1.5, pontos['conwai_esq_1_pos_i1'], pontos['conwai_esq_1_neg_i1'], pontos['conwai_esq_2_pos_i2'], pontos['conwai_esq_2_neg_i2']], 
                                  average=0.1, scalings=None, ch_type='eeg',
                                  sphere=None, image_interp='cubic', time_unit='s', time_format="%0.2f s", vlim=(-3,3))
evoked_imaginacao_exe_esq.plot_topomap(times=[1.5, pontos['imgexe_esq_1_pos_i1'], pontos['imgexe_esq_1_neg_i1'], pontos['imgexe_esq_2_pos_i2'], pontos['imgexe_esq_2_neg_i2']], 
                                   average=0.1, scalings=None, ch_type='eeg',
                                   sphere=None, image_interp='cubic', time_unit='s', time_format="%0.2f s", vlim=(-3,3))
evoked_imaginacao_img_esq.plot_topomap(times=[1.5, pontos['imgimg_esq_1_pos_i1'], pontos['imgimg_esq_1_neg_i1'], pontos['imgimg_esq_2_pos_i2'], pontos['imgimg_esq_2_neg_i2']], 
                                   average=0.1, scalings=None, ch_type='eeg',
                                   sphere=None, image_interp='cubic', time_unit='s', time_format="%0.2f s", vlim=(-3,3))

# Right
evoked_controle_exe_dir.plot_topomap(times=[1.5, pontos['conexe_dir_1_pos_i1'], pontos['conexe_dir_1_neg_i1'], pontos['conexe_dir_2_pos_i2'], pontos['conexe_dir_2_neg_i2']], 
                                 average=0.1, scalings=None, ch_type='eeg',
                                 sphere=None, image_interp='cubic', time_unit='s', time_format="%0.2f s", vlim=(-3,3))
evoked_controle_wait_dir.plot_topomap(times=[1.5, pontos['conwai_dir_1_pos_i1'], pontos['conwai_dir_1_neg_i1'], pontos['conwai_dir_2_pos_i2'], pontos['conwai_dir_2_neg_i2']], 
                                  average=0.1, scalings=None, ch_type='eeg',
                                  sphere=None, image_interp='cubic', time_unit='s', time_format="%0.2f s", vlim=(-3,3))
evoked_imaginacao_exe_dir.plot_topomap(times=[1.5, pontos['imgexe_dir_1_pos_i1'], pontos['imgexe_dir_1_neg_i1'], pontos['imgexe_dir_2_pos_i2'], pontos['imgexe_dir_2_neg_i2']], 
                                   average=0.1, scalings=None, ch_type='eeg',
                                   sphere=None, image_interp='cubic', time_unit='s', time_format="%0.2f s", vlim=(-3,3))
evoked_imaginacao_img_dir.plot_topomap(times=[1.5, pontos['imgimg_dir_1_pos_i1'], pontos['imgimg_dir_1_neg_i1'], pontos['imgimg_dir_2_pos_i2'], 4.448], 
                                   average=0.1, scalings=None, ch_type='eeg',
                                   sphere=None, image_interp='cubic', time_unit='s', time_format="%0.2f s", vlim=(-3,3))

evoked_controle_exe_esq.plot_joint(times=[1.5, pontos['conexe_esq_1_pos_i1'], pontos['conexe_esq_1_neg_i1'], pontos['conexe_esq_2_pos_i2'], pontos['conexe_esq_2_neg_i2']],
                                    title='ERP: Execução do Movimento - Esquerdo (Controle)')
evoked_controle_wait_esq.plot_joint(times=[1.5, pontos['conwai_esq_1_pos_i1'], pontos['conwai_esq_1_neg_i1'], pontos['conwai_esq_2_pos_i2'], pontos['conwai_esq_2_neg_i2']],
                                    title='ERP: Ausência do Movimento - Esquerdo (Controle)')
evoked_imaginacao_exe_esq.plot_joint(times=[1.5, pontos['imgexe_esq_1_pos_i1'], pontos['imgexe_esq_1_neg_i1'], pontos['imgexe_esq_2_pos_i2'], pontos['imgexe_esq_2_neg_i2']], 
                                    title='ERP: Execução do Movimento - Esquerdo (Imaginação)')
evoked_imaginacao_img_esq.plot_joint(times=[1.5, pontos['imgimg_esq_1_pos_i1'], pontos['imgimg_esq_1_neg_i1'], pontos['imgimg_esq_2_pos_i2'], pontos['imgimg_esq_2_neg_i2']], 
                                    title='ERP: Imaginação do Movimento - Esquerdo (Imaginação)')

evoked_controle_exe_dir.plot_joint(times=[1.5, pontos['conexe_dir_1_pos_i1'], pontos['conexe_dir_1_neg_i1'], pontos['conexe_dir_2_pos_i2'], pontos['conexe_dir_2_neg_i2']], 
                                    title='ERP: Execução do Movimento - Direito (Controle)')
evoked_controle_wait_dir.plot_joint(times=[1.5, pontos['conwai_dir_1_pos_i1'], pontos['conwai_dir_1_neg_i1'], pontos['conwai_dir_2_pos_i2'], pontos['conwai_dir_2_neg_i2']], 
                                    title='ERP: Ausencia do Movimento - Direito (Controle)')
evoked_imaginacao_exe_dir.plot_joint(times=[1.5, pontos['imgexe_dir_1_pos_i1'], pontos['imgexe_dir_1_neg_i1'], pontos['imgexe_dir_2_pos_i2'], pontos['imgexe_dir_2_neg_i2']], 
                                    title='ERP: Execução do Movimento - Direito (Imaginação)')
evoked_imaginacao_img_dir.plot_joint(times=[1.5, pontos['imgimg_dir_1_pos_i1'], pontos['imgimg_dir_1_neg_i1'], pontos['imgimg_dir_2_pos_i2'], 4.448], 
                                    title='ERP: Imaginação do Movimento - Direito (Imaginação)')

mne.viz.plot_compare_evokeds(evokeds, picks='eeg', combine='mean')
mne.viz.plot_evoked_topo(evokeds,layout=layout, title='Event Related Potentials') # Multi-ERP plot (interactive)





condicoes = {'esq': {'controle_wait': {'data_folder': 'E:\Projetos\FisCog\Controle', 'voluntarios': [i+1 for i in range(12)], 'epoca': '10-epo'},
                     'controle_exe':  {'data_folder': 'E:\Projetos\FisCog\Controle', 'voluntarios': [i+1 for i in range(12)], 'epoca': '11-epo'},
                     'imaginacao_exe': {'data_folder': 'E:\Projetos\FisCog\Imaginacao', 'voluntarios': [i+1 for i in range(12)], 'epoca': '11-epo'},
                     'imaginacao_img': {'data_folder': 'E:\Projetos\FisCog\Imaginacao', 'voluntarios': [i+1 for i in range(12)], 'epoca': '10-epo'}},
            
             'dir': {'controle_wait': {'data_folder': 'E:\Projetos\FisCog\Controle', 'voluntarios': [i+1 for i in range(12)], 'epoca': '20-epo'},
                     'controle_exe':  {'data_folder': 'E:\Projetos\FisCog\Controle', 'voluntarios': [i+1 for i in range(12)], 'epoca': '21-epo'},
                     'imaginacao_exe': {'data_folder': 'E:\Projetos\FisCog\Imaginacao', 'voluntarios': [i+1 for i in range(12)], 'epoca': '21-epo'},
                     'imaginacao_img': {'data_folder': 'E:\Projetos\FisCog\Imaginacao', 'voluntarios': [i+1 for i in range(12)], 'epoca': '20-epo'}}}












# pontos[condicoecas] = {}

# def gerar_erp(condicoes):
#     dados_erp = {}
#     dados_erp_mean = {}
    
#     for movimento, condicoes_dir in condicoes.items():
#         for condicao, info in condicoes_dir.items():
#             dados_epocas = []
#             for i in info['voluntarios']:
#                 data_folder = info['data_folder'] + (f'\S0{i}' if i <= 9 else f'\S{i}') + '\Sem re-referenciar'
#                 voluntario = (f'\S0{i}' if i <= 9 else f'\S{i}') + '_' + info['epoca']

#                 print(data_folder + voluntario)

#                 epocas = mne.read_epochs(data_folder + voluntario + '.fif')
#                 dados_epocas.append(epocas.get_data(picks='eeg'))

#             condicoes_epocas[condicao] = epocas
#             voluntarios = len(dados_epocas)
#             dados_erp[condicao] = np.zeros((voluntarios, 30, 6001))

#             for indice in range(voluntarios):
#                 dados_erp[condicao][indice,:,:] = np.mean(dados_epocas[indice], axis=0)

#             dados_erp_mean[condicao] = np.mean(dados_erp[condicao], axis=0)

#         dados_erp[f'{movimento}_controle_wait'], dados_erp_mean[f'{movimento}_controle_wait'] = dados_erp['controle_wait'], dados_erp_mean['controle_wait']
#         dados_erp[f'{movimento}_controle_exe'], dados_erp_mean[f'{movimento}_controle_exe'] = dados_erp['controle_exe'], dados_erp_mean['controle_exe']
#         dados_erp[f'{movimento}_imaginacao_exe'], dados_erp_mean[f'{movimento}_imaginacao_exe'] = dados_erp['imaginacao_exe'], dados_erp_mean['imaginacao_exe']
#         dados_erp[f'{movimento}_imaginacao_img'], dados_erp_mean[f'{movimento}_imaginacao_img'] = dados_erp['imaginacao_img'], dados_erp_mean['imaginacao_img']
       
#     return (dados_erp['esq_controle_wait'], dados_erp_mean['esq_controle_wait'], 
#             dados_erp['esq_controle_exe'], dados_erp_mean['esq_controle_exe'], 
#             dados_erp['esq_imaginacao_exe'], dados_erp_mean['esq_imaginacao_exe'], 
#             dados_erp['esq_imaginacao_img'], dados_erp_mean['esq_imaginacao_img'],
#             dados_erp['dir_controle_wait'], dados_erp_mean['dir_controle_wait'], 
#             dados_erp['dir_controle_exe'], dados_erp_mean['dir_controle_exe'], 
#             dados_erp['dir_imaginacao_exe'], dados_erp_mean['dir_imaginacao_exe'], 
#             dados_erp['dir_imaginacao_img'], dados_erp_mean['dir_imaginacao_img'])

# dados_erp = {}
# dados_erp_mean = {}

# dados_erp['esq_controle_wait'], dados_erp_mean['esq_controle_wait'], dados_erp['esq_controle_exe'], dados_erp_mean['esq_controle_exe'], dados_erp['esq_imaginacao_exe'], dados_erp_mean['esq_imaginacao_exe'], dados_erp['esq_imaginacao_img'], dados_erp_mean['esq_imaginacao_img'],dados_erp['dir_controle_wait'], dados_erp_mean['dir_controle_wait'], dados_erp['dir_controle_exe'], dados_erp_mean['dir_controle_exe'], dados_erp['dir_imaginacao_exe'], dados_erp_mean['dir_imaginacao_exe'], dados_erp['dir_imaginacao_img'], dados_erp_mean['dir_imaginacao_img'] = gerar_erp(condicoes)