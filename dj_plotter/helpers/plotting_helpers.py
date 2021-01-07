### PLOTTING HELPERS
import math
import numpy as np
from tqdm import tqdm, tqdm_notebook
from datetime import datetime
from matplotlib import pyplot as plt
import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker

import seaborn as sns

from scipy.ndimage.morphology import distance_transform_edt
from skimage.filters import gaussian

sns.set(style='white', font_scale=1.1)


def make_circular_colormap(array, cmap=sns.husl_palette(as_cmap=True), n_colors=360, desat=1, saturations=None, scale_saturations=True):
    '''
    Create color map - circular (polar) version. 
    Uses seaborn color_palette() function to create color map. 
    For valid cyclic matplotlib colormaps check 
    https://matplotlib.org/tutorials/colors/colormaps.html#cyclic    
    https://seaborn.pydata.org/tutorial/color_palettes.html#using-circular-color-systems

    
    Paramters
    ---------
        - array: Circular values in radians (!) to loop over and create colors for
        - cmap: Color palette name or list of colors of a (custom) palette
        - n_colors: number of colors that should be in final array
        - desat: seaborn specific desaturation of color palette. 1 = no desaturation.
        - saturations: array of length(array): Color saturations 
        - scale_saturations : bool. Scale saturations between 0,1? Default = True
    Returns
    -------
        - colors: array of length len(array) of specified palette
    '''
    array = (array + 2*np.pi) % (2*np.pi)
    array = np.nan_to_num(array.copy())
    
    if saturations is not None:
        assert len(saturations) == len(array), 'Lengths of saturations and array do not match'
        if scale_saturations:
            saturations = np.interp(saturations, (saturations.min(), saturations.max()), (0., 1.))

    if isinstance(cmap, list):
        color_palette = cmap
    elif isinstance(cmap, str):
        # ... create from scratch
        color_palette = sns.color_palette(cmap, n_colors, desat)
    elif isinstance(cmap, mpl.colors.ListedColormap):
        color_palette = cmap.colors        
    else:
        raise NotImplementedError(f'Type {type(cmap)} for cmap argument not understood.')
    if len(color_palette) < 100: print('Warning! Less than 100 distinct colors - is that what you want?')

    colors = []
    for no, el in enumerate(array):
        index = np.int(np.interp(el, [0.,2*np.pi], [0.,1.]) * len(color_palette))
        color = color_palette[index]
        if saturations is not None:
            color = tuple(np.array(color) * saturations[no])
        colors.append(color)
    return np.array(colors)


def make_linear_colormap(array, cmap='magma', desat=1, reference_numbers=None, categorical=False, percentile=100):
    '''
    Create color map - linear version. 
    Uses seaborn color_palette() function to create color map. 
    
    Paramters
    ---------
        - array: 1-D Numpy array to loop over and create colors for
        - cmap: Color palette name or list of colors of a (custom) palette
        - desat: seaborn specific desaturation of color palette. 1 = no desaturation.
        - reference_numbers: reference array (to use instead of input array)
        - percentile: To get rid of outlier artefacts, use percentile < 100 as maximum    
    Returns
    -------
        - colors: array of length len(array) of specified palette
    '''
    
    array = np.nan_to_num(array.copy())
    
    if reference_numbers is not None:
        reference_array = np.linspace(reference_numbers.min(),\
                                      reference_numbers.max(),\
                                      len(array))
    else:
        reference_array = None
        
    # Correct for negative values in array
    if not categorical: 
        if (array < 0).any():
            minimum = array.min()
            array += -1 * minimum
            if reference_array is not None: # Also correct reference number range
                reference_array += -1 * minimum
            
    if not categorical: 
        # Check what format the colormap is in 
        if isinstance(cmap, list):
            color_palette = cmap
            if len(color_palette) < 100: print('Warning! Less than 100 distinct colors - is that what you want?')
        elif isinstance(cmap, str):
            # ... create from scratch
            color_palette = sns.color_palette(cmap, len(array), desat)
        else:
            raise NotImplementedError(f'Type {type(cmap)} for cmap argument not understood.')

        colors = []
        for el in array:
            if reference_array is None:
                index = np.int(np.interp(el,[np.min(array), np.percentile(array, percentile)],[0.,1.])*len(color_palette))
            else:
                index = np.int((el/np.percentile(reference_array, percentile))*len(color_palette))
                
            if index > len(color_palette)-1: index = -1
            color = color_palette[index]
            colors.append(color)
    else: 
        categories = list(set(array))
        color_palette = sns.color_palette(cmap, len(categories), desat)
        colors = []
        for el in array:
            index = [no for no, category in enumerate(categories) if category == el][0]
            color = color_palette[index]
            colors.append(color)
        
    return np.array(colors)


def make_colorbar(array, no_steps=60, font_scale=1, cmap='hls', **kwargs):
    ''' Draw a colorbar that can be copy+pasted next to another graph '''

    sns.set(font_scale=1, style='white')

    return_figure = kwargs.get('return_figure', None)
    show_labels = kwargs.get('show_labels', True)
    

    if isinstance(cmap, list):
        color_palette = cmap
        if len(color_palette) < 100: print('Warning! Less than 100 distinct colors - is that what you want?')
    elif isinstance(cmap, str):
        # ... create from scratch
        color_palette = sns.color_palette(cmap, len(array), 1)

    start_value = np.min(array)
    end_value = np.max(array)

    labels = [''] * no_steps
    labels[0] = '{:.1f}'.format(start_value)
    labels[-1]= '{:.1f}'.format(end_value)

    figure = plt.figure(figsize=(20,.5))
    ax = figure.add_subplot(111)
 
    n = len(color_palette)
    ax.imshow(np.arange(n).reshape(1, n),
              cmap=mpl.colors.ListedColormap(list(color_palette)),
              interpolation="nearest", aspect="auto")
    ax.set_xticks(np.arange(n) - .5)
    ax.set_yticks([-.5, .5])
    # Ensure nice border between colors
    ax.set_xticklabels(["" for _ in range(n)])
    # The proper way to set no ticks
    ax.yaxis.set_major_locator(ticker.NullLocator())

    # Additional formatting of x tick labels
    #ax.set_xticklabels(labels, rotation = -90, ha="center", va='top')
    
    sns.despine(left=True,bottom=True)

    if not show_labels:
        ax.set_xticklabels([])
    if return_figure:
        return figure
    plt.show()

