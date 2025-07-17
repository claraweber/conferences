"""
May 2025
Clara Weber MD / MPI CBS

Analysis code for CNS 2025 abstract
With data kindly shared by 
Cabalo et al. 2025 (MICA-PNI)
Markello et al. 2022 (neuromaps)
Vaishnavi et al. 2010 (CBF and CBV maps from PET)
Satterthwaite et al. 2014 (CBF map from ASL)
Laumann et al. 2021 (pediatric stroke case)
"""
# %%
import numpy as np
import pandas as pd
import nibabel as nib
import os
import seaborn as sbn
import matplotlib.pyplot as plt
from brainspace.plotting import plot_hemispheres
from brainspace.datasets import load_conte69, load_parcellation
from brainspace.utils.parcellation import reduce_by_labels, map_to_labels
from brainspace.mesh.mesh_creation import build_polydata
from brainspace.null_models import SpinPermutations
from neuromaps.datasets import fetch_annotation, fetch_fslr
from neuromaps.images import load_data
from neuromaps.transforms import mni152_to_fslr
from neuromaps.resampling import resample_images
from scipy.stats import spearmanr, ttest_ind
import babybrains_cfw as bb


datadir = '/'

#%%
# load surfaces and parcellation
s400 = load_parcellation('schaefer', 400, join = True)
surf_lh, surf_rh = load_conte69()

# %%
# load all maps from subject folders, aggregate into dataframe
data = []
ids = []
ns = []

for i in range(1,11):
    for count, n in enumerate(['curv', 'midthickness_ADC', 'midthickness_FA', 'midthickness_T1map', 'thickness', 'white_ADC', 'white_FA', 'white_T1map']):
        try:
            lh_map = nib.load(f'{datadir}sub-PNC00{i}_ses-01_hemi-L_surf-fsLR-32k_label-{n}.func.gii').agg_data()
            rh_map = nib.load(f'{datadir}sub-PNC00{i}_ses-01_hemi-R_surf-fsLR-32k_label-{n}.func.gii').agg_data()
        except:
            lh_map = nib.load(f'{datadir}sub-PNC0{i}_ses-01_hemi-L_surf-fsLR-32k_label-{n}.func.gii').agg_data()
            rh_map = nib.load(f'{datadir}sub-PNC0{i}_ses-01_hemi-R_surf-fsLR-32k_label-{n}.func.gii').agg_data()
        both_map = np.concatenate((lh_map, rh_map))

        data.append(both_map)
        ids.append(i)
        ns.append(n)

df = pd.DataFrame(data)
df['ID'] = ids
df['metric'] = ns
df.to_csv(f'{datadir}overview_maps.csv')

# %%
# determine modality-specific mean values, plot onto cortical surface
labels = ['curv', 'midthickness_ADC', 'midthickness_FA', 'midthickness_T1map', 'thickness', 'white_ADC', 'white_FA', 'white_T1map']

curv = np.array(df[df['metric']=='curv'])[:,:64984]
curv_flat = np.asarray(np.mean(curv, axis = 0), dtype ='f')
curv_flat[s400 == 0] = np.nan

midthicknessADC = np.array(df[df['metric']=='midthickness_ADC'])[:,:64984]
midthicknessADC_flat = np.asarray(np.mean(midthicknessADC, axis = 0), dtype ='f')
midthicknessADC_flat[s400 == 0] = np.nan

midthicknessFA = np.array(df[df['metric']=='midthickness_FA'])[:,:64984]
midthicknessFA_flat = np.asarray(np.mean(midthicknessFA, axis = 0), dtype ='f')
midthicknessFA_flat[s400 == 0] = np.nan

midthicknessT1 = np.array(df[df['metric']=='midthickness_T1map'])[:,:64984]
midthicknessT1_flat = np.asarray(np.mean(midthicknessT1, axis = 0), dtype ='f')
midthicknessT1_flat[s400 == 0] = np.nan

thickness = np.array(df[df['metric']=='thickness'])[:,:64984]
thickness_flat = np.asarray(np.mean(thickness, axis = 0), dtype ='f')
thickness_flat[s400 == 0] = np.nan

whiteADC = np.array(df[df['metric']=='white_ADC'])[:,:64984]
whiteADC_flat = np.asarray(np.mean(whiteADC, axis = 0), dtype ='f')
whiteADC_flat[s400 == 0] = np.nan

whiteFA = np.array(df[df['metric']=='white_FA'])[:,:64984]
whiteFA_flat = np.asarray(np.mean(whiteFA, axis = 0), dtype ='f')
whiteFA_flat[s400 == 0] = np.nan

whiteT1 = np.array(df[df['metric']=='white_T1map'])[:,:64984]
whiteT1_flat = np.asarray(np.mean(whiteT1, axis = 0), dtype ='f')
whiteT1_flat[s400 == 0] = np.nan

array = [curv_flat, midthicknessADC_flat, midthicknessFA_flat, midthicknessT1_flat, thickness_flat, whiteADC_flat, whiteFA_flat, whiteT1_flat]

plot_hemispheres(surf_lh, surf_rh, array, 
size = (900,1300),zoom = 1.4, label_text = labels, 
cmap = ['RdYlGn', 'binary', 'Reds', 'Purples', 'viridis', 'binary', 'Reds', 'Purples'],
color_range = [(-0.2,0.1), (0.0004,0.0007), (0.1,0.4), (1300, 2200),(1.5,3.5),(0.0004,0.0007),(0.1,0.4), (1300,2200)],
embed_nb = True, color_bar = True, nan_color = (.7, .7, .7, 1))

# %%
# load neuromaps
# load margulies gradient in 32k resolution for downsampling
fslr_32k = fetch_annotation(source = 'margulies2016', desc = 'fcgradient01', space = 'fsLR', den = '32k')
margulies = load_data(fslr_32k)
margulies[s400 == 0] = np.nan # mask median wall
plot_hemispheres(surf_lh, surf_rh, margulies,
                 cmap = 'RdYlGn', size =(1200,300), zoom = 1.2, 
                 embed_nb=True, color_bar=True, nan_color = (.7,.7,.7,1))

cbf_map = fetch_annotation(source = 'raichle', desc = 'cbf', space = 'fsLR', den = '164k')
cbf_vector = resample_images(src = cbf_map, trg = fslr_32k, src_space = 'fsLR', trg_space = 'fsLR',  resampling = 'downsample_only')
cbf_vec = load_data(cbf_vector[0])
cbf_vec[s400 == 0] = np.nan
plot_hemispheres(surf_lh, surf_rh, cbf_vec,
                 cmap = 'viridis', size =(1200,300), zoom = 1.2, 
                 embed_nb=True, color_bar=True, nan_color = (.7,.7,.7,1))

cbv_map = fetch_annotation(source = 'raichle', desc = 'cbv', space = 'fsLR', den = '164k')
cbv_vector = resample_images(src = cbv_map, trg = fslr_32k, src_space = 'fsLR', trg_space = 'fsLR',  resampling = 'downsample_only')
cbv_vec = load_data(cbv_vector[0])
cbv_vec[s400 == 0] = np.nan 
plot_hemispheres(surf_lh, surf_rh, cbv_vec,
                 cmap = 'viridis', size =(1200,300), zoom = 1.2, 
                 embed_nb=True,  color_bar=True, nan_color = (.7,.7,.7,1))

cbv_ted = fetch_annotation(source = 'satterthwaite2014', desc = 'meancbf', space = 'MNI152', res = '1mm')
data_cbv_ted = mni152_to_fslr(cbv_ted)
cbvted_vec = load_data(data_cbv_ted)
surf_lh, surf_rh = load_conte69()
cbvted_vec[s400 ==0 ] = np.nan
plot_hemispheres(surf_lh, surf_rh, cbvted_vec, 
                cmap = 'magma', size =(1200,300), zoom = 1.2, 
                embed_nb=True, color_bar=True, nan_color = (.7,.7,.7,1))

# %%
# Plot stats
df_plot = pd.DataFrame({'FG1': reduce_by_labels(margulies, s400),
'CBF_ted': reduce_by_labels(cbvted_vec, s400),
'CBV': reduce_by_labels(cbv_vec, s400),
'CBF': reduce_by_labels(cbf_vec,s400)})

for i, n in enumerate(['curv', 'midthickness_ADC', 'midthickness_FA', 'midthickness_T1map', 'thickness', 'white_ADC', 'white_FA', 'white_T1map']):
    all_metric = np.array(df[df['metric']==n])[:,:64984]
    mean_metric = reduce_by_labels(np.mean(all_metric, axis = 0), s400)
    df_plot[f'z_{n}'] = (mean_metric-np.mean(mean_metric))/np.std(all_metric)
    df_plot[f'mean_{n}'] = mean_metric
    for subj in range(10):
        case = reduce_by_labels(np.array(df[(df['metric'] == n) & (df['ID'] == subj+1)])[0,:64984], s400)
        df_plot[f'{n}_{subj+1}'] = case
        case_z = (case - np.mean(mean_metric))/np.std(all_metric)
        df_plot[f'z_{n}_{subj+1}'] = case_z

# %%
# Plot values on cardiovascular axes
x_vars = ['CBF', 'CBV', 'CBF_ted']
y_vars = ['thickness', 'midthickness_ADC', 'midthickness_FA', 'midthickness_T1map', 'white_ADC', 'white_FA', 'white_T1map']

mode = 'mean' # or 'z' for z-scored metrics

fig, ax = plt.subplots(7,3, figsize = (12,15))

for col, x_var in enumerate(x_vars):
    for row, y_var in enumerate(y_vars):
        sbn.scatterplot(data=df_plot, x=f'{mode}_{x_var}', y=y_var, ax=ax[row, col], s=5)
        sbn.regplot(data=df_plot, x=f'{mode}_{x_var}', y=y_var, ax=ax[row, col], scatter=False, order=2)

fig.savefig(f'{datadir}meanmetrics_cardiovascularaxes.png')
plt.close()

# %%
# plot separate lines for each individual 
fig, ax = plt.subplots(len(y_bases), len(x_vars), figsize=(14, 18), sharex='col')
subject_colors = plt.cm.cubehelix(np.linspace(0, 1, 14))

for col, x_var in enumerate(x_vars):
    for row, y_base in enumerate(y_bases):
        for subj in range(1, 11):
            y_indiv_col = f'{y_base}_{subj}' 
            color = subject_colors[subj - 1]
            sbn.scatterplot(data=df_plot, x=x_var, y=y_indiv_col,ax=ax[row, col], s=5, alpha=0.7, color=color)
            sbn.regplot(data=df_plot, x=x_var, y=y_indiv_col,ax=ax[row, col], scatter=False, order=2, color=color)

        ax[row, col].spines['right'].set_visible(False)
        ax[row, col].spines['top'].set_visible(False)

fig.tight_layout(rect=[0, 0.03, 1, 0.98])  
fig.savefig(f'{datadir}individualmetrics_cardiovascularaxes.png')
plt.close()

# %%
# handedness

# populate original df with data for handedness and sex - forgot this in the original dataframe, oops
df['handedness'] = df['ID'].replace({1:'R', 2:'R', 3:'R', 4:'R', 5:'L', 6:'R', 7:'R', 8:'L', 9:'R', 10:'R'})
df['sex'] = df['ID'].replace({1:'F', 2:'M', 3:'M', 4:'F', 5:'F', 6:'F', 7:'F', 8:'F', 9:'F', 10:'F'})

df_plot_handedness = pd.DataFrame({'CBF_ted': reduce_by_labels(cbvted_vec, s400),
'CBV': reduce_by_labels(cbv_vec, s400),
'CBF': reduce_by_labels(cbf_vec,s400)})

for i, n in enumerate(['curv', 'midthickness_ADC', 'midthickness_FA', 'midthickness_T1map', 'thickness', 'white_ADC', 'white_FA', 'white_T1map']):
    l_all_metric = np.array(df[(df['metric']==n) & (df['handedness'] == 'L')])[:,:64984]
    r_all_metric = np.array(df[(df['metric']==n) & (df['handedness'] == 'R')])[:,:64984]
    l_mean_metric = reduce_by_labels(np.mean(l_all_metric, axis = 0), s400)
    r_mean_metric = reduce_by_labels(np.mean(r_all_metric, axis = 0), s400)
    df_plot_handedness[f'z_{n}_l'] = (l_mean_metric-np.mean(l_mean_metric))/np.std(l_all_metric)
    df_plot_handedness[f'z_{n}_r'] = (r_mean_metric-np.mean(r_mean_metric))/np.std(r_all_metric)
    df_plot_handedness[f'mean_{n}_l'] = l_mean_metric
    df_plot_handedness[f'mean_{n}_r'] = r_mean_metric

# %%
fig, ax = plt.subplots(len(y_bases), len(x_vars), figsize=(14, 18), sharex='col')

for col, x_var in enumerate(x_vars):
    for row, y_base in enumerate(y_bases):
        for side in ['l', 'r']:
            y_indiv_col = f'z_{y_base}_{side}' 
            if side == 'l':
                color = 'red'
            elif side == 'r':
                color = 'blue'
            sbn.scatterplot(data=df_plot_handedness, x=x_var, y=y_indiv_col, ax=ax[row, col], s=5, alpha=0.7, color=color)
            sbn.regplot(data=df_plot_handedness, x=x_var, y=y_indiv_col, ax=ax[row, col], scatter=False, order=2, color=color)

fig.tight_layout(rect=[0, 0.03, 1, 0.98])  
fig.savefig(f'{datadir}handedness.png')
plt.close()

# %%
# sex analysis

df_plot_sex = pd.DataFrame({'CT': reduce_by_labels(midthicknessT1_flat, s400),
'CBF_ted': reduce_by_labels(cbvted_vec, s400),
'CBV': reduce_by_labels(cbv_vec, s400),
'CBF': reduce_by_labels(cbf_vec,s400)})

for i, n in enumerate(['curv', 'midthickness_ADC', 'midthickness_FA', 'midthickness_T1map', 'thickness', 'white_ADC', 'white_FA', 'white_T1map']):
    f_all_metric = np.array(df[(df['metric']==n) & (df['sex'] == 'F')])[:,:64984]
    m_all_metric = np.array(df[(df['metric']==n) & (df['sex'] == 'M')])[:,:64984]
    f_mean_metric = reduce_by_labels(np.mean(f_all_metric, axis = 0), s400)
    m_mean_metric = reduce_by_labels(np.mean(m_all_metric, axis = 0), s400)
    df_plot_sex[f'z_{n}_f'] = (f_mean_metric-np.mean(f_mean_metric))/np.std(f_all_metric)
    df_plot_sex[f'z_{n}_m'] = (m_mean_metric-np.mean(m_mean_metric))/np.std(m_all_metric)
    df_plot_sex[f'mean_{n}_f'] = f_mean_metric
    df_plot_sex[f'mean_{n}_m'] = m_mean_metric

# %%
fig, ax = plt.subplots(len(y_bases), len(x_vars), figsize=(14, 18), sharex='col')

for col, x_var in enumerate(x_vars):
    for row, y_base in enumerate(y_bases):
        for sex in ['f', 'm']:
            y_indiv_col = f'mean_{y_base}_{sex}' 
            if sex == 'f':
                color = 'cornflowerblue'
            elif sex == 'm':
                color = 'grey'
            sbn.scatterplot(data=df_plot_sex, x=x_var, y=y_indiv_col,ax=ax[row, col], s=5, alpha=0.7, color=color)
            sbn.regplot(data=df_plot_sex, x=x_var, y=y_indiv_col,ax=ax[row, col], scatter=False, order=2,color=color, line_kws={'alpha': 0.8, 'lw': 1.5})

fig.tight_layout(rect=[0, 0.03, 1, 0.98])  
fig.savefig(f'{datadir}sex.png')
plt.close()

# %%
# laterality

df_plot_sym = pd.DataFrame({'CBF_ted_l': reduce_by_labels(cbvted_vec, s400)[1:201],
'CBF_ted_r': reduce_by_labels(cbvted_vec, s400)[201:],
'CBV_l': reduce_by_labels(cbv_vec, s400)[1:201],
'CBV_r': reduce_by_labels(cbv_vec, s400)[201:],
'CBF_l': reduce_by_labels(cbf_vec,s400)[1:201],
'CBF_r': reduce_by_labels(cbf_vec,s400)[201:]})

for i, n in enumerate(['curv', 'midthickness_ADC', 'midthickness_FA', 'midthickness_T1map', 'thickness', 'white_ADC', 'white_FA', 'white_T1map']):
    all_metric = np.array(df[df['metric']==n])[:,:64984]
    mean_metric = reduce_by_labels(np.mean(all_metric, axis = 0), s400)
    lh_metric = mean_metric[1:201]
    rh_metric = mean_metric[201:]
    df_plot[f'z_{n}_lh'] = (lh_metric-np.mean(mean_metric[1:201]))/np.std(mean_metric[:201])
    df_plot[f'z_{n}_rh'] = (rh_metric-np.mean(mean_metric[201:]))/np.std(mean_metric[201:])
    df_plot[f'mean_{n}_lh'] = lh_metric
    df_plot[f'mean_{n}_rh'] = rh_metric

# %%
x_vars_l = ['CBF_l', 'CBV_l', 'CBF_ted_l']
x_vars_r = ['CBF_r', 'CBV_r', 'CBF_ted_r']
y_vars = ['mean_thickness', 'mean_midthickness_ADC', 'mean_midthickness_FA', 'mean_midthickness_T1map', 'mean_white_ADC', 'mean_white_FA', 'mean_white_T1map']

fig, ax = plt.subplots(7,3, figsize = (12,18))

for col, (x_var_l, x_var_r) in enumerate(zip(x_vars_l, x_vars_r)):
    for row, y_var in enumerate(y_vars):
        left_data = f'{y_var}_lh'
        right_data = f'{y_var}_rh'
        
        sbn.scatterplot(data=df_plot_sym, x=x_var_l, y=left_data, ax=ax[row, col], s=5, color='palevioletred', label='Left' if row==0 else None, legend = False)
        sbn.regplot(data=df_plot_sym, x=x_var_l, y=left_data, ax=ax[row, col], scatter=False, order=2, color='palevioletred')
        
        sbn.scatterplot(data=df_plot_sym, x=x_var_r, y=right_data, ax=ax[row, col], s=5, color='cornflowerblue', label='Right' if row==0 else None, legend = False)
        sbn.regplot(data=df_plot_sym, x=x_var_r, y=right_data, ax=ax[row, col], scatter=False, order=2, color='cornflowerblue')

        ax[row, col].spines['right'].set_visible(False)
        ax[row, col].spines['top'].set_visible(False)

fig.tight_layout(rect=[0, 0.03, 1, 0.98])  
fig.savefig(f'{datadir}symmetry.png')
plt.close()

# %%
# demographic

data_base = pd.read_csv(f'{datadir}participants.tsv', sep = '\t')
data_base['common'] = np.zeros((10,))

sbn.set_theme(style = 'ticks', rc = {'axes.spines.right':False, 'axes.spines.top':False})
fig, ax = plt.subplots(figsize = (5,2))
sbn.histplot(data_base, x = 'age', multiple = 'stack', alpha = 1, binwidth = 1, color = 'grey', legend = False)
fig.savefig(f'{datadir}age_histogram.png')
plt.close()

# %%
# stroke contextualization

datadir_stroke = '/stroke/'

data_l = nib.load(f'{datadir_stroke}sub-CIMT001_ses-38659_hemi-L_surf-fsLR-32k_label-thickness.func.gii').agg_data()
data_r = nib.load(f'{datadir_stroke}sub-CIMT001_ses-38659_hemi-R_surf-fsLR-32k_label-thickness.func.gii').agg_data()
data_ct = np.concatenate((data_l, data_r))

surf_l = nib.load(f'{datadir_stroke}surf/sub-CIMT001_ses-38659_hemi-L_space-nativepro_surf-fsLR-32k_label-midthickness.surf.gii/')
surf_r = nib.load(f'{datadir_stroke}surf/sub-CIMT001_ses-38659_hemi-R_space-nativepro_surf-fsLR-32k_label-midthickness.surf.gii/')
pl, vl = surf_l.agg_data()
pr, vr = surf_r.agg_data()
surf_sl = build_polydata(pl, vl)
surf_sr = build_polydata(pr, vr)

# %%
data_ct[s400 == 0] = np.nan
data_curv[s400 == 0] = np.nan
plot_hemispheres(surf_sl, surf_sr, data_ct, embed_nb = True, cmap = 'viridis', size = (1200,300), color_range = (1,4), zoom = 1.2, nan_color = (.7, .7,.7,1), color_bar = True)

# %%
plot_hemispheres(surf_sl, surf_sr, [cbf_vec, cbv_vec, cbvted_vec], 
embed_nb = True, cmap = 'magma', size = (1200,600), 
zoom = 1.45, nan_color = (.7, .7,.7,1), color_bar = True)

# %%
# correlate CT stroke and HC

all_ct = np.array(df[df['metric']=='thickness'])[:,:64984]
mean_metric = reduce_by_labels(np.mean(all_ct, axis = 0), s400)


dfs_plot = pd.DataFrame({'FG1': reduce_by_labels(margulies, s400),
    'CBF_ted': reduce_by_labels(cbvted_vec, s400),
'CBV': reduce_by_labels(cbv_vec, s400),
'CBF': reduce_by_labels(cbf_vec,s400),
'CT_stroke': reduce_by_labels(data_ct,s400),
'mean_CT_7T': mean_metric})

# %%
fig, ax = plt.subplots(3, figsize = (3,9))

sbn.scatterplot(dfs_plot, x = 'CBF_ted', y = 'mean_CT_7T', ax = ax[0], s = 10, color = 'grey')
sbn.regplot(dfs_plot, x = 'CBF_ted', y = 'mean_CT_7T', ax = ax[0], scatter = False, color = 'grey', order = 2)
sbn.scatterplot(dfs_plot, x = 'CBF_ted', y = 'CT_stroke', ax = ax[0], s = 10, color = 'crimson')
sbn.regplot(dfs_plot, x = 'CBF_ted', y = 'CT_stroke', ax = ax[0], scatter = False, color = 'crimson', order = 2)

sbn.scatterplot(dfs_plot, x = 'CBF', y = 'mean_CT_7T', ax = ax[1], s = 10, color = 'grey')
sbn.regplot(dfs_plot, x = 'CBF', y = 'mean_CT_7T', ax = ax[1], scatter = False, color = 'grey', order = 2)
sbn.scatterplot(dfs_plot, x = 'CBF', y = 'CT_stroke', ax = ax[1], s = 10, color = 'crimson')
sbn.regplot(dfs_plot, x = 'CBF', y = 'CT_stroke', ax = ax[1], scatter = False, color = 'crimson', order = 2)

sbn.scatterplot(dfs_plot, x = 'CBV', y = 'mean_CT_7T', ax = ax[2], s = 10, color = 'grey')
sbn.regplot(dfs_plot, x = 'CBV', y = 'mean_CT_7T', ax = ax[2], scatter = False, color = 'grey', order = 2)
sbn.scatterplot(dfs_plot, x = 'CBV', y = 'CT_stroke', ax = ax[2], s = 10, color = 'crimson')
sbn.regplot(dfs_plot, x = 'CBV', y = 'CT_stroke', ax = ax[2], scatter = False, color = 'crimson', order = 2)

# %%
fig, ax = plt.subplots(1, 4, figsize = (12,2))

sbn.scatterplot(dfs_plot, x = 'CT_stroke', y = 'mean_CT_7T', hue = 'FG1', palette = 'RdYlGn', s = 10, color = 'grey', ax = ax[0], legend = False)
sbn.scatterplot(dfs_plot, x = 'CT_stroke', y = 'mean_CT_7T', hue = 'CBF', palette = 'magma', s = 10, color = 'grey', ax = ax[1], legend = False)
sbn.scatterplot(dfs_plot, x = 'CT_stroke', y = 'mean_CT_7T', hue = 'CBV', palette = 'magma', s = 10, color = 'grey', ax = ax[2], legend = False)
sbn.scatterplot(dfs_plot, x = 'CT_stroke', y = 'mean_CT_7T', hue = 'CBF_ted', palette = 'magma', s = 10, color = 'grey', ax = ax[3], legend = False)

for i in range(4):
    ax[i].spines['right'].set_visible(False)
    ax[i].spines['top'].set_visible(False)
    ax[i].set_ylabel('')

# %%
# statistical analysis - get spin permutations for future testing
n_rand = 1000
sphere_lh, sphere_rh = load_conte69(as_sphere=True)
sp = SpinPermutations(n_rep=n_rand, random_state=0)
sp.fit(sphere_lh, points_rh=sphere_rh)

# %%
# define wrapper function 

def correlate(feat, embedding, sp, name, n_rand):
    feat = np.nan_to_num(feat, 0)
    feat_rotated = np.hstack(sp.randomize(feat[:int(len(feat)/2)], feat[int(len(feat)/2):]))
    
    r_spin = np.empty(n_rand)

    mask = ~np.isnan(embedding)
    embedding = np.nan_to_num(embedding, 0)

    r_obs, pv_obs = spearmanr(feat[mask], embedding[mask])

    for i, perm in enumerate(feat_rotated):
        mask_rot = mask & ~np.isnan(perm)  # Remove midline
        r_spin[i] = spearmanr(perm[mask_rot], embedding[mask_rot])[0]
    pv_spin = np.mean(np.abs(r_spin) >= np.abs(r_obs))

    return r_obs, pv_obs, pv_spin

# %%
data = []
names = ['CBF', 'CBV', 'CBFTed']
for i, target in enumerate([cbf_vec, cbv_vec, cbvted_vec]):
    rs, ps, ps_spin = correlate(target, np.array(data_ct, dtype = 'f'), sp, 'stroke', n_rand)
    data.append(['stroke', names[i], rs, ps, ps_spin])
    rt, pt, pt_spin = correlate(target, np.array(mean_thickness_flat, dtype = 'f'), sp, 'mean CT', n_rand)
    data.append(['mean ct', names[i], rt, pt, pt_spin])
    ra, pa, pa_spin = correlate(target, np.array(mean_ADC_flat, dtype = 'f'), sp, 'mean ADC', n_rand)
    data.append(['mean ADC', names[i], ra, pa, pa_spin])
    rf, pf, pf_spin = correlate(target, np.array(mean_FA_flat, dtype = 'f'), sp, 'mean FA', n_rand)
    data.append(['mean FA', names[i], rf, pf, pf_spin])
    r1, p1, p1_spin = bb.correlate(target, np.array(mean_T1_flat, dtype = 'f'), sp, 'mean T1', n_rand)
    data.append(['mean T1', names[i], r1, p1, p1_spin])
    pd.DataFrame(data, columns = ['name', 'map', 'r', 'p', 'p_spin']).to_csv('spintestresults.csv')

rs, ps, ps_spin = correlate(np.array(mean_thickness, dtype = 'f'), np.array(data_ct, dtype = 'f'), sp, 'stroke', n_rand)
data.append(['stroke', 'mean CT', rs, ps, ps_spin])
rt, pt, pt_spin = correlate(fslr_32k, np.array(mean_thickness, dtype = 'f'), sp, 'mean CT', n_rand)
data.append(['mean ct', 'margulies', rt, pt, pt_spin])
ra, pa, pa_spin = correlate(fslr_32k, data_ct, sp, 'mean ADC', n_rand)
data.append(['stroke', 'margulies', ra, pa, pa_spin])

pd.DataFrame(data, columns = ['name', 'map', 'r', 'p', 'p_spin']).to_csv('spintestresults.csv')

# %%
# asymmetry analysis using similar df as in plotting
df_plot = pd.DataFrame({'CBF_ted':cbvted_vec,
'CBV': cbv_vec,
'CBF': cbf_vec})

for i, n in enumerate(['curv', 'midthickness_ADC', 'midthickness_FA', 'midthickness_T1map', 'thickness', 'white_ADC', 'white_FA', 'white_T1map']):
    all_metric = np.array(df[df['metric']==n])[:,:64984]
    mean_metric = np.mean(np.array(all_metric, dtype ='f'), axis = 0)
    lh_metric = mean_metric.copy()
    lh_metric[32492:] = np.nan
    rh_metric = mean_metric.copy()
    rh_metric[:32492] = np.nan
    df_plot[f'mean_{n}_lh'] = lh_metric
    df_plot[f'mean_{n}_rh'] = rh_metric

data = []
names = ['CBF', 'CBV', 'CBF_ted']
for i, target in enumerate(names):
    for j, x in enumerate([ 'midthickness_ADC', 'midthickness_FA', 'midthickness_T1map', 'thickness']):
        rl, pl, pl_spin = correlate(np.array(df_plot[names[i]], dtype = 'f'),
        np.array(df_plot[f'mean_{x}_lh'], dtype ='f'), sp, 'stroke', n_rand)
        data.append([f'lh_{x}', names[i], rl, pl, pl_spin])
        rr, pr, pr_spin = correlate(np.array(df_plot[names[i]], dtype ='f'), 
        np.array(df_plot[f'mean_{x}_rh'], dtype = 'f'), sp, 'stroke', n_rand)
        data.append([f'rh_{x}', names[i], rr, pr, pr_spin])
    
        pd.DataFrame(data, columns = ['name', 'map', 'r', 'p', 'p_spin']).to_csv('spintestresults_symm.csv')

# %%
# difference score

thickness = np.array(df[df['metric']=='thickness'])[:,:64984]
mean_thickness = np.mean(thickness, axis = 0)

z_stroke = (data_ct - np.mean(data_ct))/np.std(data_ct)
z_ct = mean_thickness/np.std(thickness)

diff = (np.asarray(z_stroke-z_ct, dtype = 'f')-np.mean(thickness))/np.std(thickness)
diff[s400 ==0] = np.nan

diff_comp = reduce_by_labels(diff, s400)
diff_parc = map_to_labels(diff_comp, s400)
diff_bin = abs(diff_parc)>5

plot_hemispheres(surf_lh, surf_rh, [diff_parc, diff], embed_nb = True, size = (1200,600), zoom = 1.2, color_bar = True, cmap = 'binary_r', color_range = (-5, 0), nan_color = (.7,.7,.7,1))

#%%
# distribution of highly impacted regions on cerebrovascular axes

cbv_vec2 = cbv_vec.copy()
cbfted_vec2 = cbvted_vec.copy()

df_plot = pd.DataFrame({'cbv_400': reduce_by_labels(cbv_vec2, s400),
'cbf_400': reduce_by_labels(cbf_vec, s400),
'cbfted_400': reduce_by_labels(cbfted_vec2, s400),
'diff_bin_400': reduce_by_labels(diff_bin, s400),
'adc_400': reduce_by_labels(midthicknessADC_flat, s400),
'fa_400': reduce_by_labels(midthicknessFA_flat, s400),
't1_400': reduce_by_labels(midthicknessT1_flat, s400),
'diff_bin_400': reduce_by_labels(diff_bin, s400)
})


fig, ax = plt.subplots(1, 3, figsize=(10,4))
sbn.violinplot(df_plot, y = 'cbf_400', hue =diff_bin_400, ax = ax[0], split = True, inner = 'quart', gap = .1, palette = 'binary', legend = False)
sbn.violinplot(df_plot,y = 'cbv_400', hue =diff_bin_400, ax = ax[1], split = True, inner = 'quart', gap = .1, palette = 'binary', legend = False)
sbn.violinplot(df_plot,y = 'cbfted_400', hue =diff_bin_400, ax = ax[2], split = True, inner = 'quart', gap = .1, palette = 'binary', legend = False)

for i in range(3):
    ax[i].spines['right'].set_visible(False)
    ax[i].spines['top'].set_visible(False)
    ax[i].spines['left'].set_visible(False)


# %%
# distribution of microstructural markers in highly impacted regions

fig, ax = plt.subplots(1, 3, figsize=(10,4))
sbn.violinplot(df_plot, y = 'adc_400', hue =diff_bin_400, ax = ax[0], split = True, inner = 'quart', gap = .1, palette = 'binary', legend = False)
sbn.violinplot(df_plot,y = 'fa_400', hue =diff_bin_400, ax = ax[1], split = True, inner = 'quart', gap = .1, palette = 'binary', legend = False)
sbn.violinplot(df_plot,y = 't1_400', hue =diff_bin_400, ax = ax[2], split = True, inner = 'quart', gap = .1, palette = 'binary', legend = False)

for i in range(3):
    ax[i].spines['right'].set_visible(False)
    ax[i].spines['top'].set_visible(False)
    ax[i].spines['left'].set_visible(False)

#%%
# statitiscs
def calculate_group_differences(df, variables, group_var):
    results = {}
    for var in variables:
        group0 = df[df[group_var] == 0][var].dropna().values
        group1 = df[df[group_var] == 1][var].dropna().values
        mean0, mean1 = np.mean(group0), np.mean(group1)
        std0, std1 = np.std(group0, ddof=1), np.std(group1, ddof=1)

        n0, n1 = len(group0), len(group1)
        pooled_std = np.sqrt(((n0-1)*std0**2 + (n1-1)*std1**2) / (n0 + n1 - 2))

        cohen_d = (mean1 - mean0) / pooled_std

        _, p_value = ttest_ind(group0, group1, equal_var=True)
        
        results[var] = (cohen_d, p_value)
    return results


variables = ['cbf_400', 'cbv_400', 'cbfted_400', 'adc_400', 'fa_400', 't1_400']
results = calculate_group_differences(df_plot, variables, 'diff_bin_400')

for var, (cohen_d, p_value) in results.items():
    print(f"{var}: Cohen's d: {cohen_d:.3f}, p-value: {p_value:.4f}")


