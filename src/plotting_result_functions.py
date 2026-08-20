import numpy as np
import pandas as pd
import scipy.io as sio
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
from matplotlib.lines import Line2D
from pathlib import Path
# import cloupy as cl

# get absolute path of main directory
home_dirname = str(Path(__file__).parent.parent.absolute())

def River_Profile_Map_Points_Plot(basin_data, resolution_distance=1000, threshold_distance=10000, save=False, filename='river-profile+map.png'):
    '''
    DESCRIPTION:
        Plot the river profile and map in function of tributaries, lithology and tectonic field
    ----------
    PARAMETERS
    basin_data : dictionnary
        data of the basin. 
    resolution_distance : float, optional
        . The default is 1000.
    threshold_distance : TYPE, optional
        threshold distance of the tributaries in meter. The default is 10000.
    save : Bool, optional
        save plot if true. The default is False.
    filename : TYPE, optional
        filename of the plot. The default is 'river-profile+map.png'.
    -------
    RETURNS
    None.
    '''
    
    # build dataframe to make plotting easier
    array = np.transpose(np.array([basin_data['latitude'], basin_data['longitude'], basin_data['x'], basin_data['initial_elevation'], basin_data['source'], basin_data['lithology'], basin_data['uplift']]))
    name = ['latitude', 'longitude', 'flow_distance', 'elevation', 'source', 'lithology', 'uplift']
    df = pd.DataFrame(data=array, columns=name)
    df = df.astype({'latitude':float, 'longitude':float, 'flow_distance':float, 'elevation':float, 'source':float, 'lithology':str, 'uplift':float})

    # figure of the resampled and thresholded river profile and map
    fig1 = plt.figure(figsize=(10.0,5.0))
    ax1 = fig1.add_subplot(231)
    ax2 = fig1.add_subplot(232)
    ax3 = fig1.add_subplot(233)
    ax4 = fig1.add_subplot(234)
    ax5 = fig1.add_subplot(235)
    ax6 = fig1.add_subplot(236)
    
    source_key = np.unique(df['source'])
    n_skey = np.size(source_key)
    cmap = plt.cm.get_cmap('rainbow', n_skey)
    c = 0
    for i in source_key:
        ax1.plot(np.array(df.loc[df['source'] == i][['flow_distance']]), np.array(df.loc[df['source'] == i][['elevation']]), 'o', ms=0.5, c=cmap(c))
        ax4.plot(np.array(df.loc[df['source'] == i][['latitude']]), np.array(df.loc[df['source'] == i][['longitude']]), 'o', ms=0.5, c=cmap(c))
        c = c + 1
        
    litho_key = np.unique(df['lithology'])
    n_lkey = np.size(litho_key)
    cmap = plt.cm.get_cmap('rainbow', n_lkey)
    c = 0
    for i in litho_key:
        ax2.plot(np.array(df.loc[df['lithology'] == i][['flow_distance']]), np.array(df.loc[df['lithology'] == i][['elevation']]), 'o', ms=0.5, c=cmap(c))
        ax5.plot(np.array(df.loc[df['lithology'] == i][['latitude']]), np.array(df.loc[df['lithology'] == i][['longitude']]), 'o', ms=0.5, c=cmap(c))
        c = c + 1
        
    uplift_key = np.unique(df['uplift'])
    n_ukey = np.size(uplift_key)
    cmap = plt.cm.get_cmap('rainbow', n_ukey)
    c = 0
    for i in uplift_key:
        ax3.plot(np.array(df.loc[df['uplift'] == i][['flow_distance']]), np.array(df.loc[df['uplift'] == i][['elevation']]), 'o', ms=0.5, c=cmap(c))
        ax6.plot(np.array(df.loc[df['uplift'] == i][['latitude']]), np.array(df.loc[df['uplift'] == i][['longitude']]), 'o', ms=0.5, c=cmap(c))
        c = c + 1
    
    ax1.set_xlabel('Flow distance (m)') 
    ax2.set_xlabel('Flow distance (m)')
    ax3.set_xlabel('Flow distance (m)') 
    ax4.set_xlabel('Latitude')
    ax5.set_xlabel('Latitude')
    ax6.set_xlabel('Latitude')
    ax1.set_ylabel('Elevation (m)')
    ax4.set_ylabel('Longitude')
    ax1.set_title('Tributary', fontsize=10)
    ax2.set_title('Lithology', fontsize=10)
    ax3.set_title('Uplift', fontsize=10)
    fig1.suptitle('River profile and map\n(point resampling: ' + str(resolution_distance/1000) + ' km + basin threshold: ' + str(threshold_distance/1000) + ' km)', fontsize=10)
    
    ticks_x = ticker.FuncFormatter(lambda x, pos: '{0:g}'.format(x*1e-3))
    ax1.xaxis.set_major_formatter(ticks_x)
    ax2.xaxis.set_major_formatter(ticks_x)
    ax3.xaxis.set_major_formatter(ticks_x)
    ax4.invert_yaxis()
    ax5.invert_yaxis()
    ax6.invert_yaxis()
    
    fig1.tight_layout()
    if save == True:
        fig1.savefig(home_dirname + '/data/figures/' + filename + '.png', dpi=720)
        fig1.savefig(home_dirname + '/data/figures/' + filename + '.pdf', dpi=720)

def River_Point_Map_Plot(data, parameter, key='initial_elevation', label='elevation (m)', cmap='jet', figsize=(7.125, 4.5), save=False, filename='river-point-map-plot'):
    
    fig = plt.figure(figsize=figsize)
    ax1 = fig.add_subplot(111)
    
    s = ax1.scatter(data['longitude'], data['latitude'], c=data[key], cmap=cmap)
    
    fig.colorbar(s, orientation='vertical', label=label, pad=0.05)
    ax1.set_ylabel('Latitude (°)')
    ax1.set_xlabel('Longitude (°)')
    fig.tight_layout()
    
    if save == True:
        fig.savefig(home_dirname + '/data/figures/boxplot-topo+analytical-dataset.png', dpi=720)
        fig.savefig(home_dirname + '/data/figures/boxplot-topo+analytical-dataset.pdf', dpi=720)
    

def River_Map_Points_Plot(basin_data, save=False, filename='river-map.png'):
    '''
    DESCRIPTION:
        plot the river map in function of the tributaries, lithologies and tectonic fields
    ----------
    PARAMETERS
    basin_data : dictionnary
        data of the basin. 
    save : Bool, optional
        save plot if true. The default is False.
    filename : TYPE, optional
        filename of the plot. The default is 'river-profile+map.png'.
    -------
    RETURNS
    None.
    '''

    # build dataframe to make plotting easier
    array = np.transpose(np.array([basin_data['latitude'], basin_data['longitude'], basin_data['x'], basin_data['initial_elevation'], basin_data['source'], basin_data['lithology'], basin_data['uplift'][-1]]))
    name = ['latitude', 'longitude', 'flow_distance', 'elevation', 'source', 'lithology', 'uplift']
    df = pd.DataFrame(data=array, columns=name)
    df = df.astype({'latitude':float, 'longitude':float, 'flow_distance':float, 'elevation':float, 'source':float, 'lithology':str, 'uplift':float})
    
    # figure of the resampled and thresholded river profile and map
    fig1 = plt.figure(figsize=(7.125, 2.4))
    ax1 = fig1.add_subplot(131)
    ax2 = fig1.add_subplot(132)
    ax3 = fig1.add_subplot(133)
    
    source_key = np.unique(df['source'])
    n_skey = np.size(source_key)
    cmap = plt.cm.get_cmap('rainbow', n_skey)
    c = 0
    for i in source_key:
        ax1.plot(np.array(df.loc[df['source'] == i][['longitude']]), np.array(df.loc[df['source'] == i][['latitude']]), 'o', ms=0.5, c=cmap(c))
        c = c + 1
        
    litho_key = np.unique(df['lithology'])
    n_lkey = np.size(litho_key)
    cmap = plt.cm.get_cmap('rainbow', n_lkey)
    c = 0
    for i in litho_key:
        ax2.plot(np.array(df.loc[df['lithology'] == i][['longitude']]), np.array(df.loc[df['lithology'] == i][['latitude']]), 'o', ms=0.5, c=cmap(c))
        c = c + 1
        
    uplift_key = np.unique(df['uplift'])
    n_ukey = np.size(uplift_key)
    cmap = plt.cm.get_cmap('rainbow', n_ukey)
    c = 0
    for i in uplift_key:
        ax3.plot(np.array(df.loc[df['uplift'] == i][['longitude']]), np.array(df.loc[df['uplift'] == i][['latitude']]), 'o', ms=0.5, c=cmap(c))
        c = c + 1
    
    ax1.set_title('Tributary', fontsize=10)
    ax2.set_title('Lithology', fontsize=10)
    ax3.set_title('Block', fontsize=10)
    
    ax1.axis('equal')
    ax2.axis('equal')
    ax3.axis('equal')
    
    ax1.set_ylabel('Latitude (°)')
    ax1.set_xlabel('Longitude (°)')
    ax2.set_xlabel('Longitude (°)')
    ax3.set_xlabel('Longitude (°)')
    
    loc = ticker.MultipleLocator(base=0.5)
    ax1.xaxis.set_major_locator(loc)
    ax2.xaxis.set_major_locator(loc)
    ax3.xaxis.set_major_locator(loc)
    
    fig1.tight_layout()
    if save:
        fig1.savefig(home_dirname + '/data/figures/' + filename + '.png', dpi=720)
        fig1.savefig(home_dirname + '/data/figures/' + filename + '.pdf', dpi=720)
        
def River_Profile_Points_Plot(basin_data, save=False, filename='river-profile'):
    '''
    DESCRIPTION:
        plot the river profile in function of the tributaries, lithologies and tectonic fields
    ----------
    PARAMETERS
    basin_data : dictionnary
        data of the basin. 
    save : Bool, optional
        save plot if true. The default is False.
    filename : TYPE, optional
        filename of the plot. The default is 'river-profile+map.png'.
    -------
    RETURNS
    None.
    '''
    
    # build dataframe to make plotting easier
    array = np.transpose(np.array([basin_data['latitude'], basin_data['longitude'], basin_data['x'], basin_data['initial_elevation'], basin_data['source'], basin_data['lithology'], basin_data['uplift'][-1]]))
    name = ['latitude', 'longitude', 'flow_distance', 'elevation', 'source', 'lithology', 'uplift']
    df = pd.DataFrame(data=array, columns=name)
    df = df.astype({'latitude':float, 'longitude':float, 'flow_distance':float, 'elevation':float, 'source':float, 'lithology':str, 'uplift':float})

    # figure of the resampled and thresholded river profile and map
    fig1 = plt.figure(figsize=(3.5, 6.0))
    ax1 = fig1.add_subplot(311)
    ax2 = fig1.add_subplot(312)
    ax3 = fig1.add_subplot(313)
    
    source_key = np.unique(df['source'])
    n_skey = np.size(source_key)
    cmap = plt.cm.get_cmap('rainbow', n_skey)
    c = 0
    for i in source_key:
        ax1.plot(np.array(df.loc[df['source'] == i][['flow_distance']]), np.array(df.loc[df['source'] == i][['elevation']]), 'o', ms=0.5, c=cmap(c))
        c = c + 1
        
    litho_key = np.unique(df['lithology'])
    n_lkey = np.size(litho_key)
    cmap = plt.cm.get_cmap('rainbow', n_lkey)
    c = 0
    for i in litho_key:
        ax2.plot(np.array(df.loc[df['lithology'] == i][['flow_distance']]), np.array(df.loc[df['lithology'] == i][['elevation']]), 'o', ms=0.5, c=cmap(c))
        c = c + 1
        
    uplift_key = np.unique(df['uplift'])
    n_ukey = np.size(uplift_key)
    cmap = plt.cm.get_cmap('rainbow', n_ukey)
    c = 0
    for i in uplift_key:
        ax3.plot(np.array(df.loc[df['uplift'] == i][['flow_distance']]), np.array(df.loc[df['uplift'] == i][['elevation']]), 'o', ms=0.5, c=cmap(c))
        c = c + 1
    
    ax3.set_xlabel('Flow distance (km)') 
    ax1.set_ylabel('Elevation (m)')
    ax2.set_ylabel('Elevation (m)')
    ax3.set_ylabel('Elevation (m)')
    ax1.set_title('Tributary', fontsize=10)
    ax2.set_title('Lithology', fontsize=10)
    ax3.set_title('Block', fontsize=10)

    ticks_x = ticker.FuncFormatter(lambda x, pos: '{0:g}'.format(x*1e-3))
    ax1.xaxis.set_major_formatter(ticks_x)
    ax2.xaxis.set_major_formatter(ticks_x)
    ax3.xaxis.set_major_formatter(ticks_x)
    
    fig1.tight_layout()
    if save == True:
        fig1.savefig(home_dirname + '/data/figures/' + filename + '.png', dpi=720)
        fig1.savefig(home_dirname + '/data/figures/' + filename + '.pdf', dpi=720)
        

def River_Profile_Obs_vs_Mod_Plot(basin_data, modelled_elevation, modelled_hillslope=[], save=False, filename='river-profile_obs-vs-pre'):
    '''
    DESCRIPTION:
        Plot the observed river profile versus the modelled river profile and modelled hillslope
    ----------
    PARAMETERS:
    basin_data : dict
        dataset of the basin (observed)
    modelled_elevation : array of float
        modelled river elevation of the basin
    modelled_hillslope : array of float
        modelled hillslope of the basin
    save : bool
        save plot if true
    filename : TYPE, optional
        filename of the plot.
    -------
    RETURNS  
    None.
    '''

    fig = plt.figure(figsize=(7.125, 3.0))
    ax1 = fig.add_subplot(111)
    
    pairs = basin_data['pairs']
    for i in range(len(pairs)):
        x = np.array([basin_data['x'][int(pairs[i,0]-1)], basin_data['x'][int(pairs[i,1]-1)]])
        y = np.array([basin_data['initial_elevation'][int(pairs[i,0]-1)], basin_data['initial_elevation'][int(pairs[i,1]-1)]])
        obs = ax1.plot(x, y, marker='', ls='-', lw=1, c='steelblue', zorder=1)
        
        x = np.array([basin_data['x'][int(pairs[i,0]-1)], basin_data['x'][int(pairs[i,1]-1)]])
        y = np.array([modelled_elevation[int(pairs[i,0]-1)], modelled_elevation[int(pairs[i,1]-1)]])
        pre = ax1.plot(x, y, marker='', ls='-', lw=1, c='darkred', zorder=2)
    
    if modelled_hillslope:
        x_hillslope = np.zeros(np.shape(modelled_hillslope))
        for i in range(np.shape(x_hillslope)[1]):
            x_hillslope[:,i] = basin_data['x'][i]
            
    
        x_hillslope_rm = np.delete(x_hillslope, list(range(0, x_hillslope.shape[1], 2)), axis=1)
        modelled_hillslope_rm = np.delete(modelled_hillslope, list(range(0, modelled_hillslope.shape[1], 2)), axis=1)
        ax1.plot(x_hillslope_rm, modelled_hillslope_rm, marker='', ms=0.5, mfc='lightcoral', mec='lightcoral', ls=':', lw=1, color='lightcoral', zorder=0)

    custom_legend = [Line2D([0], [0], color='steelblue', ls='-', lw=1),
                     Line2D([0], [0], color='darkred', ls='-', lw=1),
                     Line2D([0], [0], color='lightcoral', ls=':', lw=1)]

    ax1.legend(custom_legend, ['Observed elevation', 'Modelled elevation', 'Modelled hillslope'], loc='best', fancybox=False, edgecolor='black')

    ticks_x = ticker.FuncFormatter(lambda x, pos: '{0:g}'.format(x*1e-3))
    ax1.xaxis.set_major_formatter(ticks_x)
    ax1.set_xlabel('Flow distance (km)')
    ax1.set_ylabel('Elevation (m)')

    fig.tight_layout()
    if save:
        fig.savefig(home_dirname + '/data/figures/' + filename + '.png', dpi=720)
        fig.savefig(home_dirname + '/data/figures/' + filename + '.pdf', dpi=720)
        
def Interpolation_Map_Plot(parameterx, parametery, parameterz, country,
                           interpolation_method='cubic', n_levels=50, cbar_nticks=5,
                           ylabel=None, xlabel=None, cbar_title=None, 
                           show_points=True, points_size=1,
                           figsize=(5.5, 7.125), save=None):
    '''
    DESCRIPTION:
        perform an elliptic interpolation and plot it for a specific area of the sphere
    ----------
    PARAMETERS
    parameterx : array of floats
        longitude coordinate
    parametery : array of floats
        latitude coordinate
    parameterz : array of floats
        parameter to interpolate 
    country : string
        country where the interpolation is plotted
    for other parameters refer to the imap.draw function
    -------
    RETURNS
    None.
    '''
    
    dataframe = pd.DataFrame(np.stack((parameterz, parameterx, parametery), axis=0).T).astype('float64')
    
    imap = cl.m_MapInterpolation(country=country, dataframe=dataframe)
    
    fig, zi = imap.draw(interpolation_method=interpolation_method, n_levels=n_levels, cbar_nticks=cbar_nticks, 
                        ylabel=ylabel, xlabel=xlabel, cbar_title=cbar_title, 
                        show_contours=True, show_points=show_points, points_size=points_size,
                        figsize=figsize, save=save)
    
def Misfit_2D_Plot(results, prior, x, y, true_param=[], misfit_max=[], ds1=40, ds2=40, log=True,
                   figsize=(7.125, 7.125),save=False, filename='inverse-modelling_misfit'):
    '''
    DESCRIPTION:
        Plot the misfit for two free parameters based on a random inverse modelling
    ----------
    PARAMETERS:
    misfit : array of float
        misfit based on a random inverse modelling
    datax : array of float
        data to plot on the x axis for the misfit
    datay : array of float
        data to plot on the y axis for the misfit
    true_param : list of float
        if not empty plot the true parameters location
    ds1 : int
        size of the misfit dots
    ds2 : int
        size of the true parameters dot
    xlabel : string
        label of the x axis
    ylabel : string
        label of the y axis
    axis: list of float
        if not empty set the range of the x and y axis
    log : bool
        logarithmic misfit if true
    figsize : tuple of int
        size of the figure (n, n)
    save : bool
        save plot if true
    filename : TYPE, optional
        filename of the plot.
    -------
    RETURNS:
    None.
    '''
    
    misfit = results[1]
    datax = results[0][:,x]
    datay = results[0][:,y]
    xlabel = prior.label[x]
    ylabel = prior.label[y]
    print(datax)
    print(datay)
    print(misfit)
    
    # set misfit scale
    if log: 
        misfit = np.log10(misfit)
    if not misfit_max:
        misfit_max = np.max(misfit)
    
    # sort misfit index 
    order = np.argsort(misfit)[::-1]
    
    # plot the data
    fig = plt.figure(figsize=figsize)
    ax1 = fig.add_subplot(111)
    sc1 = ax1.scatter(datax[order], datay[order], c=misfit[order], vmin=min(misfit), vmax=misfit_max, cmap='jet_r', s=ds1)
    
    fig.colorbar(sc1, label='Misfit', ax=ax1)
    if true_param:
        ax1.plot(true_param[0], true_param[1], marker='o', ms=ds2, mec='black', mfc='white')
    
    ax1.axis([prior.low[x], prior.high[x], prior.low[y], prior.high[y]])
    ax1.set_xlabel(xlabel)
    ax1.set_ylabel(ylabel)
    
    fig.tight_layout()
    
    if save:
        fig.savefig(home_dirname + '/data/figures/' + filename + '.png', dpi=720)
        fig.savefig(home_dirname + '/data/figures/' + filename + '.pdf', dpi=720)
        
        
def SBI_Probability_Multi_Dimension_Plot(samples, prior, num_samples=10000, true_param=[], figsize=(7.125, 7.125), bins=100, save=False, filename='inverse-modelling_sbi-probabibility'):
    '''
    DESCIPTIONS:
        plot the probability to predict data given model parameter based on a simulation based inference inverse modelling
    ----------
    PARAMETERS:
    samples : torch array 
        sampling results based on the posterior simulation probability
    label_param : list of strings
        labels for the true parameters
    num_samples : int
        number of posterior sampling
    low_prior : array of float
        lower range of prior parameters
    high_prior : array of float
        higher range of prior parameters
    true_param : bool
        plot the true parameters if set to true
    full_range : bool
        plot the full axis range of prior parameter
    value_param : array of float
        values for the true parameters
    figsize : tuple of int
        size of the figure (n, n)
    bins : int
        number of bins for the plotting
    save : bool
        save the plot if true
    filename : TYPE, optional
        filename of the plot.
    -------
    RETURNS:
    None.
    '''
    
    label_param = prior.label
    low_prior = prior.low
    high_prior = prior.high
    samples = np.array(samples)
    num_dim = np.shape(samples)[1]
    mat = np.triu(np.ones((num_dim, num_dim), int)*2, 1)
    np.fill_diagonal(mat, 1)

    fig = plt.figure(figsize=figsize)
    spec = fig.add_gridspec(num_dim, num_dim)
    
    
    print('--------------------------')
    for i in range(num_dim):
        for j in range(num_dim):
            
            if mat[i, j] == 1:
                ax = fig.add_subplot(spec[i, j])
                if low_prior:
                    n, x, p = ax.hist(samples[:, i], bins, histtype='stepfilled', ec='black', fc='steelblue', orientation='vertical', range=(low_prior[i], high_prior[i]))
                else:
                    n, x, p = ax.hist(samples[:, i], bins, histtype='stepfilled', ec='black', fc='steelblue', orientation='vertical')
                if true_param:
                    ax.axvline(true_param[i], ls='--', color='firebrick')
                
                ticks_p = ticker.FuncFormatter(lambda p, pos: '{:.0f}'.format(p/num_samples*100))
                ax.yaxis.set_major_formatter(ticks_p)

                # ax.set_ylabel('Probability (%)')
                ax.set_ylabel('Probability (%)')
                
                if low_prior:
                    ax.axis([low_prior[i], high_prior[i], ax.axis()[2], ax.axis()[3]])
                else:
                    ax.axis([np.min(samples[:, i]), np.max(samples[:, i]), ax.axis()[2], ax.axis()[3]])
                 
                index = np.argmax(n)
                mode = x[index] + (x[index + 1] - x[index])/2
                mean = np.mean(samples[:,i])
                median = np.median(samples[:,i])
                sigma = np.std(samples[:,i])
                
                if label_param:
                    print('mode ' + label_param[i] + ': ' + str(mode))
                    print('mean ' + label_param[i] + ': ' + str(mean))
                    print('median ' + label_param[i] + ': ' + str(median))
                    print('sigma ' + label_param[i] + ': ' + str(sigma))
                    print('two sigma ' + label_param[i] + ': ' + str(sigma*2))
                    print('--------------------------')
                
                ax.xaxis.set_ticks_position('top')
                ax.xaxis.set_label_position('top')
                
                if i != 0:
                    ax.set_xticklabels([])
                
            if mat[i, j] == 2:
                ax = fig.add_subplot(spec[i, j])
                if low_prior:
                    ax.hist2d(samples[:, j], samples[:, i], bins=bins, cmap='jet', range=np.array([(low_prior[j], high_prior[j]), (low_prior[i], high_prior[i])]))
                else:
                    ax.hist2d(samples[:, j], samples[:, i], bins=bins, cmap='jet')
                if true_param:
                    ax.plot(true_param[j], true_param[i], marker='o', ms=5, mec='black', mfc='firebrick', zorder=2)

                ax.xaxis.set_ticks_position('top')
                ax.xaxis.set_label_position('top') 
                
                if i != 0:
                    ax.set_xticklabels([])
                    ax.set_yticklabels([])
                
                if i == 0:
                    ax.set_yticklabels([])
            
            if label_param:
                if i == 0:
                    ax.set_xlabel(label_param[j], labelpad=20)
    
    # for figure with dimension of 3.5 width and 3.5 depth
    # fig.subplots_adjust(top=0.80, bottom=0.04, left=0.175, right=0.95, wspace=0.15, hspace=0.15)
    # for figure with dimension of 7.125 width and 7.125 depth
    fig.subplots_adjust(top=0.90, bottom=0.04, left=0.085, right=0.95, wspace=0.15, hspace=0.15)
    if save:
        fig.savefig(home_dirname + '/data/figures/' + filename + '.png', dpi=720)
        fig.savefig(home_dirname + '/data/figures/' + filename + '.pdf', dpi=720)
        
        
def SBI_Misfit_MultiRound_Simulation_Plot(data, low, high, label=[], num_round=1, x=0, y=1, numplotx=1, numploty=1, size_dot=1, same_axis=True, figsize=(7.125, 7.125), save_fig=False, filename='inverse-modelling_sbi-misfit'):
    '''
    DESCRIPTION:
        plot the misfit for two parameters through the different rounds of the sbi simulation
    ----------
    PARAMETERS
    data : array of floats
        data from the sbi simulation
    num_round : int, optional
        number of sbi rounds. The default is 1.
    x : int, optional
        column of the parameter to plot on the x axis. The default is 0.
    y : int, optional
        column of the parameter to plot on the y axis. The default is 1.
    numplotx : int, optional
        number of panel on the x axis of the figure (numplotx x numplouty shoulb give num_round). The default is 1.
    numploty : int, optional
        number of panel on the y axis of the figure. The default is 1.
    size_dot : float, optional
        size of the misfit dot. The default is 1.
    same_axis : list of float, optional
        axis range for the different panels. The default is [].
    labelx : string, optional
        label on the x axis. The default is ''.
    labely : string, optional
        label on the y axis. The default is ''.
    save_fig : bool, optional
        save the figure if true. The default is False.
    filename : TYPE, optional
        filename of the figure. The default is 'inverse-modelling_sbi-misfit'.
    -------
    RETURNS
    '''
    
    fig = plt.figure(figsize=figsize)

    for i in range(num_round):
       
        misfit = data[i][:,-1]
        if True:
            misfit = np.log10(misfit)
        datax = data[i][:,x]
        datay = data[i][:,y]
       
        order = np.argsort(misfit)[::-1]
       
        ax = fig.add_subplot(numplotx, numploty, i+1)
        s = ax.scatter(datax[order], datay[order], c=misfit[order], vmin=np.min(misfit), vmax=np.max(misfit), cmap='jet_r', s=size_dot)
       
        if same_axis:
            ax.axis([low[x], high[x], low[y], high[y]])
        
        if label:
            if i in [0,2]:
                ax.set_ylabel(label[y])
            if i in [2,3]:
                ax.set_xlabel(label[x])
    
        fig.colorbar(s, label='Log Misfit', location='top') 
    
    fig.tight_layout()

    if save_fig:
        fig.savefig(home_dirname + '/data/figures/' + filename + label[x] + label[y] + '.png', dpi=720)
        fig.savefig(home_dirname + '/data/figures/' + label[x] + label[y] + '.pdf', dpi=720) 