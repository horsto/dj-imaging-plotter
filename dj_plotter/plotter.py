### DATAJOINT + PLOTTING CLASS 
from copy import copy
import pathlib
from datetime import datetime
import numpy as np
import pandas as pd

# Drawing
from matplotlib import pyplot as plt
import seaborn as sns
from tqdm.auto import tqdm
# Load more colormaps 
import cmasher as cmr

# ... for ROI processing
from scipy.ndimage.morphology import distance_transform_edt
from skimage.filters import gaussian

# Helpers
from .helpers.plotting_helpers import make_circular_colormap, make_linear_colormap
from .helpers.dj_utils import make_multi_session_object_dict, get_signal_indices

# Additional matplotlib options
import matplotlib as mpl
mpl.rcParams['pdf.fonttype'] = 42 # Fix bug in PDF export
from mpl_toolkits.axes_grid1 import make_axes_locatable

# Load base schema
import datajoint as dj 
schema = dj.schema(dj.config['dj_imaging.database'])
schema.spawn_missing_classes()


### PLOTTING CLASS 

class dj_plotter():
    ''' Generate plots from datajoint objects. '''


    # Define which attributes (=column names) are required for functions to run 
    SESSION_HASH     = 'session_name'
    SESSION_HASH_OV  = 'base_session'
    ANIMAL_NAME      = 'animal_name'
    TIMESTAMP        = 'timestamp'
    ATTR_RATEMAP     = ['ratemap','mask_rm','cell_id']
    ATTR_AUTOCORR    = ['acorr','cell_id']
    ATTR_ROIS        = ['center_x','center_y','xpix','ypix','x_range','y_range', 'lambda']
    ATTR_ROIS_CORR   = ['center_x_corr','center_y_corr','xpix_corr','ypix_corr','x_range_microns_eff','y_range_microns_eff', 'lambda_corr']
    ATTR_TRACKING    = ['x_pos','y_pos','speed','head_angle']
    ATTR_PATHSPIKE   = [*ATTR_TRACKING, 'signal', 'x_pos_signal','y_pos_signal','head_angle_signal']
    ATTR_RATEMAP_OV  = [] # For object vector case these data are retrieved "online"
    ATTR_PATHSPIKE_OV= [] # "
    ATTR_HDTUNING    = ['angle_centers', 'angular_occupancy', 'angular_rate']    

    def __init__(self, dj_object, keys=None, *args, **kwargs):
        '''
        Takes a datajoint object (table or join) and generates figures. 

        
        Example usage: 



        Parameters
        ----------
        dj_object       : datajoint.expression.Join or table (basic restriction)
                          Actual join object. 
        keys            : dict or numpy array 
                          Keys to be looped over. Retrieved by .fetch() operation (dict or numpy array)
    
        **kwargs:  
            plots_per_view  : int
                              Number of subplots per figure (maximum 25, since layout is 5x5)
            font_scale      : float
                              Seaborn font_scale
            total           : int
                              Total number of plots to show
            save_path       : string or Pathlib path
                              If given, will auto-export the figure under this path


        '''
        
        # Main input
        self.dj_object = dj_object
        self.keys = keys

        # Process keywords 
        self.plots_per_view = kwargs.get('plots_per_view',25)
        if self.plots_per_view > 25:
            raise ValueError('Maximum number of subplots is 25')

        self.font_scale = kwargs.get('font_scale', 1.)
        self.total = kwargs.get('total', None)
        
        self.save_path = kwargs.get('save_path', None)
        if self.save_path is not None: 
            if isinstance(self.save_path, str):
                self.save_path = pathlib.Path(self.save_path)
                   
    def __repr__(self):
        return f'DJ plotter class\nAvailable attributes:\n{self.__attributes}'

    @property
    def __attributes(self):
        ''' Return attributes in datajoint object (column names) ''' 
        return self.dj_object.heading.names


    def __check_join_integrity(self, keyword):
        '''
        Check if attribute (=column) exists in datajoint join object.
        
        Parameters
        ----------
        keyword     : string or list of strings 
                      These are the keywords that should be checked
        
        Returns
        -------
        valid       : boolean
                      True only if all keywords have been matched
        '''
        valid = False
        if isinstance(keyword,str):
            valid = keyword in self.__attributes
        elif isinstance(keyword,list):
            valid = all([self.__check_join_integrity(key_) for key_ in keyword])
        return valid


    @property
    def __create_figure_single(self):
        ''' Create standard figure''' 
        sns.set(style='white',font_scale=self.font_scale)
        return plt.figure(figsize=(10,10))

    @property
    def __create_figure_grid(self):
        ''' Create standard figure for grid display of subplots ''' 
        sns.set(style='white',font_scale=self.font_scale)
        return plt.figure(figsize=(20,20))
    
    @property
    def __create_figure_grid_ov_2(self):
        ''' Create standard (object vector) figure for grid display of 2 subplots ''' 
        sns.set(style='white',font_scale=self.font_scale)
        return plt.figure(figsize=(6,3))
    
    @property
    def __create_figure_grid_ov_3(self):
        ''' Create standard (object vector) figure for grid display of 3 subplots ''' 
        sns.set(style='white',font_scale=self.font_scale)
        return plt.figure(figsize=(9,3))

    def __title(self, entry, display_score, hash_or_animal, show_cell=True, ov=False):
        ''' Create subplot title string ''' 

        if hash_or_animal == 'hash':
            if not ov:
                hash_or_animal_string = entry[self.SESSION_HASH]
            else:
                hash_or_animal_string = entry[self.SESSION_HASH_OV]
        elif hash_or_animal == 'animal':
            ts = datetime.strftime(entry[self.TIMESTAMP],'| %d.%m.%Y | %H:%M')
            hash_or_animal_string = f'{entry[self.ANIMAL_NAME]} {ts}'    

        if show_cell:
            if display_score is not None:
                title = r'$\bfC {}$ {} | {:.2f}'.format(entry['cell_id'], hash_or_animal_string, entry[display_score])
            else:
                title = r'$\bfC {}$ {}'.format(entry['cell_id'], hash_or_animal_string)
        else:
            title = r'{}'.format(hash_or_animal_string)          
        return title



    def __tqdm_iterator(self, iterator, total, desc, leave=False):
        ''' Create a tqdm progress bar ''' 
        return tqdm(enumerate(iterator), desc=desc, total=total, leave=leave)



    ####################################################################################################
    #########################                   DRAWING


    def ratemaps(self, **kwargs):
        ''' 
        Plot ratemaps in 5x5 grid
        Optionally shows score for every subplot if available.
        
        Parameters
        ----------
        **kwargs:
            cmap          : string
                            Valid matplotlib colormap string
                            https://matplotlib.org/3.2.1/tutorials/colors/colormaps.html
                            or https://github.com/1313e/CMasher
            hash_or_animal: string
                            'hash' or 'animal'
                            Determines whether session name (hash) or animal/timestamp 
                            combination should be displayed in title of each subplot.
                            Defaults to 'animal'
            display_score : string
                            Name of the score (=column name) to display in title 
                            of each subplot.
                            Defaults to None.
            axes_lw       : float
                            axes linewidth. Default 0.5.
            display_title : bool 
                            Show title? 
            cue_card_pos  : list or string
                            Cue card position in (tracked) field ['north','south','west','east']

                        
        ''' 
        # Process kwargs
        cmap           = kwargs.get('cmap', 'magma')
        cmap           = plt.get_cmap(cmap)
        display_score  = kwargs.get('display_score', None)
        hash_or_animal = kwargs.get('hash_or_animal', 'animal')
        axes_lw        = kwargs.get('axes_lw', 5.)
        display_title  = kwargs.get('display_title', True)
        cue_card_pos   = kwargs.get('cue_card_pos', None)

        # Prepare list of attributes to check:
        ATTR_RATEMAP = self.ATTR_RATEMAP.copy()
        # Display session hash or animal_name/timestamp? 
        if hash_or_animal == 'hash':
            ATTR_RATEMAP.append(self.SESSION_HASH)
        elif hash_or_animal == 'animal':
            ATTR_RATEMAP.append(self.ANIMAL_NAME)
            ATTR_RATEMAP.append(self.TIMESTAMP)    
        else:
            raise NameError(f'Keyword "{hash_or_animal}" not recognized')
        # Display a score?
        if display_score is not None and isinstance(display_score, str):
            ATTR_RATEMAP.append(display_score)
        else:
            display_score = None

        # Check attributes in datajoint join 
        if not self.__check_join_integrity(ATTR_RATEMAP): 
            raise KeyError('One or more of these were not found: {}'.format(ATTR_RATEMAP))
       

        ###########################################################################
        ###############         START PLOTTING FUNCTIONS 

        plot_counter = 1 

        if self.keys is not None:
            iterator = self.keys
            use_keys = True
        else:
            iterator = self.dj_object
            use_keys = False
            
        if self.total is not None:
            total = self.total
        else:
            total = len(iterator)
        
        # Cue card positions 
        if cue_card_pos is not None: 
            if isinstance(cue_card_pos, str):
                cue_card_pos = [cue_card_pos] * total
            else:
                assert len(cue_card_pos) == total, \
                    'Length of cue card position array does not match length of cells to plot'

        # Make loop with tqdm progress bar
        tqdm_iterator = self.__tqdm_iterator(iterator, total, 'Drawing ratemaps')
        
        figure = self.__create_figure_grid
        for no, key in tqdm_iterator:    
            
            if no == total:
                if self.save_path is not None: 
                    print('Saving figure under {}'.format(str(self.save_path)))
                    figure.savefig(self.save_path / f'ratemaps {plot_counter-1}.pdf', bbox_inches='tight')
                
                plt.show()
                # Premature stop? Make sure you close things gracefully:
                tqdm_iterator.refresh()
                tqdm._instances.clear()
                plt.close()
                break
            
            # Use keys or join object? 
            if use_keys:
                entry = (self.dj_object & key).fetch1()
            else:
                entry = key 
                
            ratemap = np.ma.masked_array(entry['ratemap'], mask=entry['mask_rm'])
            ratemap = ratemap.filled(fill_value=0)
            # Get subplot title
            
            
            ax = figure.add_subplot(5,5,plot_counter)
            ax.imshow(ratemap, cmap=cmap, vmin=np.nanmin(ratemap), vmax=np.nanpercentile(ratemap,99))
            ax.set_aspect('equal')
            ax.get_xaxis().set_ticks([]);ax.get_yaxis().set_ticks([])    
            
            if display_title:
                title = self.__title(entry, display_score, hash_or_animal)
                ax.set_title(title)
            
            for axis in ['top','bottom','left','right']:
                ax.spines[axis].set_linewidth(axes_lw)
            
            # Draw cue card?
            if cue_card_pos is not None: 
                size = ratemap.shape
                card_pos = cue_card_pos[no]
                if card_pos == 'west':
                    ax.plot([0.,0.],[size[0]/2-5,size[0]/2+5], lw=5, color='white', clip_on=False, zorder=10, solid_capstyle='butt')
                elif card_pos == 'east':
                    ax.plot([size[1]-1,size[1]-1],[size[0]/2-5,size[0]/2+5], lw=5, color='white', clip_on=False, zorder=10, solid_capstyle='butt')
                elif card_pos == 'north':
                    ax.plot([size[1]/2-5,size[1]/2+5],[0,0], lw=5, color='white', clip_on=False, zorder=10, solid_capstyle='butt')
                elif card_pos == 'south':
                    ax.plot([size[1]/2-5,size[1]/2+5],[size[0]-1,size[0]-1], lw=5, color='white', clip_on=False, zorder=10, solid_capstyle='butt')
                else: 
                    raise NotImplementedError(f'Card position {card_pos} not understood. Choose ["west","east","north","south"]')

            plot_counter += 1       

            
            if plot_counter > self.plots_per_view:

                if self.save_path is not None: 
                    print('Saving figure under {}'.format(str(self.save_path)))
                    figure.savefig(self.save_path / f'ratemaps {plot_counter-1}.pdf', bbox_inches='tight')
                
                plot_counter = 1
                plt.show()
                # Create next figure
                figure = self.__create_figure_grid

        return


    def autocorr(self, **kwargs):
        ''' 
        Plot autocorrelations in 5x5 grid
        Optionally shows score for every subplot if available.
        
        Parameters
        ----------
        **kwargs:
            cmap          : string
                            Valid matplotlib colormap string
                            https://matplotlib.org/3.2.1/tutorials/colors/colormaps.html
                            or https://github.com/1313e/CMasher
            hash_or_animal: string
                            'hash' or 'animal'
                            Determines whether session name (hash) or animal/timestamp 
                            combination should be displayed in title of each subplot.
                            Defaults to 'animal'
            display_score : string
                            Name of the score (=column name) to display in title 
                            of each subplot.
                            Defaults to None.
            axes_lw       : float
                            axes linewidth. Default 5.
            display_title : bool 
                            Show title? 
                        
        ''' 
        # Process kwargs
        cmap           = kwargs.get('cmap', 'magma')
        cmap           = plt.get_cmap(cmap)
        display_score  = kwargs.get('display_score', None)
        hash_or_animal = kwargs.get('hash_or_animal', 'animal')
        axes_lw        = kwargs.get('axes_lw', 5.)
        display_title  = kwargs.get('display_title', True)

        # Prepare list of attributes to check:
        ATTR_AUTOCORR = self.ATTR_AUTOCORR.copy()
        # Display session hash or animal_name/timestamp? 
        if hash_or_animal == 'hash':
            ATTR_AUTOCORR.append(self.SESSION_HASH)
        elif hash_or_animal == 'animal':
            ATTR_AUTOCORR.append(self.ANIMAL_NAME)
            ATTR_AUTOCORR.append(self.TIMESTAMP)    
        else:
            raise NameError(f'Keyword "{hash_or_animal}" not recognized')
        # Display a score?
        if display_score is not None and isinstance(display_score, str):
            ATTR_AUTOCORR.append(display_score)
        else:
            display_score = None

        # Check attributes in datajoint join 
        if not self.__check_join_integrity(ATTR_AUTOCORR): 
            raise KeyError('One or more of these were not found: {}'.format(ATTR_AUTOCORR))
       

        ###########################################################################
        ###############         START PLOTTING FUNCTIONS 

        plot_counter = 1 

        if self.keys is not None:
            iterator = self.keys
            use_keys = True
        else:
            iterator = self.dj_object
            use_keys = False
            
        if self.total is not None:
            total = self.total
        else:
            total = len(iterator)
        
        # Make loop with tqdm progress bar
        tqdm_iterator = self.__tqdm_iterator(iterator, total, 'Drawing autocorrelations')
        
        figure = self.__create_figure_grid
        for no, key in tqdm_iterator:    
            
            if no == total:
                if self.save_path is not None: 
                    print('Saving figure under {}'.format(str(self.save_path)))
                    figure.savefig(self.save_path / f'autocorrs {plot_counter-1}.pdf', bbox_inches='tight')
                
                plt.show()
                # Premature stop? Make sure you close things gracefully:
                tqdm_iterator.refresh()
                tqdm._instances.clear()
                plt.close()
                break
            
            # Use keys or join object? 
            if use_keys:
                entry = (self.dj_object & key).fetch1()
            else:
                entry = key 
                
            ax = figure.add_subplot(5,5,plot_counter)
            ax.imshow(entry['acorr'], cmap=cmap)
            ax.set_aspect('equal')
            ax.get_xaxis().set_ticks([]);ax.get_yaxis().set_ticks([])    

            if display_title:
                title = self.__title(entry, display_score, hash_or_animal)
                ax.set_title(title)
            
            for axis in ['top','bottom','left','right']:
                ax.spines[axis].set_linewidth(axes_lw)

            plot_counter += 1       

            
            if plot_counter > self.plots_per_view:

                if self.save_path is not None: 
                    print('Saving figure under {}'.format(str(self.save_path)))
                    figure.savefig(self.save_path / f'autocorrs {plot_counter-1}.pdf', bbox_inches='tight')
                
                plot_counter = 1
                plt.show()
                # Create next figure
                figure = self.__create_figure_grid

        return


    def hdtuning(self, **kwargs):
        ''' 
        Plot HD tuning curves and occupancy 
        Optionally shows score for every subplot if available.
        
        Parameters
        ----------
        **kwargs:
            hash_or_animal: string
                            'hash' or 'animal'
                            Determines whether session name (hash) or animal/timestamp 
                            combination should be displayed in title of each subplot.
                            Defaults to 'animal'
            display_score : string
                            Name of the score (=column name) to display in title 
                            of each subplot.
                            Defaults to None.
            color_hd      : Show colored line for hd tuning curve according to average angle?
            cmap          : string
                            (for 'color_hd') : Valid matplotlib colormap string
                            https://matplotlib.org/3.2.1/tutorials/colors/colormaps.html
                            or https://github.com/1313e/CMasher
            line_width    : float
                            Line width of tuning curve
            only_occupancy: bool 
                            If True draws only occupancy, not actual tuning of cell
            
            display_title : bool 
                            Show title? 
                        
        ''' 
        # Process kwargs
        display_score  = kwargs.get('display_score', None)
        hash_or_animal = kwargs.get('hash_or_animal', 'animal')
        color_hd       = kwargs.get('color_hd', False)
        cmap           = kwargs.get('cmap', sns.husl_palette(as_cmap=True))
        only_occupancy = kwargs.get('only_occupancy', False)
        display_title  = kwargs.get('display_title', True)
        line_width     = kwargs.get('line_width', 3.)

        # Prepare list of attributes to check:
        ATTR_HDTUNING = self.ATTR_HDTUNING.copy()
        # Display session hash or animal_name/timestamp? 
        if hash_or_animal == 'hash':
            ATTR_HDTUNING.append(self.SESSION_HASH)
        elif hash_or_animal == 'animal':
            ATTR_HDTUNING.append(self.ANIMAL_NAME)
            ATTR_HDTUNING.append(self.TIMESTAMP)    
        else:
            raise NameError(f'Keyword "{hash_or_animal}" not recognized')
        # Display a score?
        if display_score is not None and isinstance(display_score, str):
            ATTR_HDTUNING.append(display_score)
        else:
            display_score = None

        if only_occupancy:
            ATTR_HDTUNING.remove('angular_rate')
        if color_hd: 
            ATTR_HDTUNING.append('angular_mean')

        # Check attributes in datajoint join 
        if not self.__check_join_integrity(ATTR_HDTUNING): 
            raise KeyError('One or more of these were not found: {}'.format(ATTR_HDTUNING))
       

        ###########################################################################
        ###############         START PLOTTING FUNCTIONS 

        plot_counter = 1 

        if self.keys is not None:
            iterator = self.keys
            use_keys = True
        else:
            iterator = self.dj_object
            use_keys = False
            
        if self.total is not None:
            total = self.total
        else:
            total = len(iterator)
        
        # Make loop with tqdm progress bar
        tqdm_iterator = self.__tqdm_iterator(iterator, total, 'Drawing HD tuning')
        
        figure = self.__create_figure_grid
        for no, key in tqdm_iterator:    
            
            if no == total:
                plt.tight_layout()

                if self.save_path is not None: 
                    print('Saving figure under {}'.format(str(self.save_path)))
                    figure.savefig(self.save_path / f'hdtuning {plot_counter-1}.pdf', bbox_inches='tight')
                
                plt.show()
                # Premature stop? Make sure you close things gracefully:
                tqdm_iterator.refresh()
                tqdm._instances.clear()
                plt.close()
                break
            
            # Use keys or join object? 
            if use_keys:
                entry = (self.dj_object & key).fetch1()
            else:
                entry = key 

            ax = figure.add_subplot(5,5,plot_counter, projection='polar')

            # Color? 
            if color_hd:
                color_line = make_circular_colormap(np.array([entry['angular_mean']]), cmap=cmap)[0]
                line_width_ = line_width * 1.5 # Otherwise difficult to see
            else:
                line_width_ = line_width
                color_line = 'k'

            ax.plot(entry['angle_centers'], entry['angular_occupancy']/np.nanmax(entry['angular_occupancy']), \
                                        color='k', alpha=[1. if only_occupancy else .4][0], lw=line_width_)
            if not only_occupancy:
                ax.plot(entry['angle_centers'], entry['angular_rate']/np.nanmax(entry['angular_rate']), color=color_line, lw=line_width_, alpha=.85)
            ax.set_aspect('equal')
            ax.set_theta_zero_location("S")
            ax.get_yaxis().set_ticks([])  
            ax.tick_params(labelbottom=False)      
            ax.spines['polar'].set_visible(False)

            if display_title:          
                # Get subplot title
                title = self.__title(entry, display_score, hash_or_animal)
                ax.set_title(title, y=1.1)

            plot_counter += 1       

            if plot_counter > self.plots_per_view:
                plt.tight_layout()
                if self.save_path is not None: 
                    print('Saving figure under {}'.format(str(self.save_path)))
                    figure.savefig(self.save_path / f'hdtuning {plot_counter-1}.pdf', bbox_inches='tight')
                
                plot_counter = 1
                plt.show()
                # Create next figure
                figure = self.__create_figure_grid

        return





    ###########################################################################

    def ratemaps_ov(self, **kwargs):
        ''' 
        SPECIAL! 
        - This function gets data for each session 

        Plot ratemaps for object vector (ov) cells (1x3):
             - base_session
             - object1_session
             - object2_session 

        Optionally shows score if available.
        
        Parameters
        ----------
        **kwargs:
            cmap          : string
                            Valid matplotlib colormap string
                            https://matplotlib.org/3.2.1/tutorials/colors/colormaps.html
                            or https://github.com/1313e/CMasher
            hash_or_animal: string
                            'hash' or 'animal'
                            Determines whether session name (hash) or animal/timestamp 
                            combination should be displayed in title of each subplot.
                            Defaults to 'animal'
            display_score : string
                            Name of the score (=column name) to display in title 
                            of each subplot.
                            Defaults to None.
            axes_lw       : float
                            axes linewidth. Default 5.
            display_title : bool 
                            Show title? 
            cue_card_pos  : list or string
                            Cue card position in (tracked) field ['north','south','west','east']
                        
        '''
        
        # Process kwargs
        cmap           = kwargs.get('cmap', 'magma')
        cmap           = plt.get_cmap(cmap)
        display_score  = kwargs.get('display_score', None)
        hash_or_animal = kwargs.get('hash_or_animal', 'animal')
        axes_lw        = kwargs.get('axes_lw', 5.)
        normalize_base = kwargs.get('normalize_base', True)
        display_title  = kwargs.get('display_title', True)
        cue_card_pos   = kwargs.get('cue_card_pos', None)

        # Prepare list of attributes to check:
        ATTR_RATEMAP_OV = self.ATTR_RATEMAP_OV.copy()
        # Display session hash or animal_name/timestamp? 
        if hash_or_animal == 'hash':
            ATTR_RATEMAP_OV.append(self.SESSION_HASH_OV)
        elif hash_or_animal == 'animal':
            ATTR_RATEMAP_OV.append(self.ANIMAL_NAME)
            ATTR_RATEMAP_OV.append(self.TIMESTAMP)    
        else:
            raise NameError(f'Keyword "{hash_or_animal}" not recognized')
        # Display a score?
        if display_score is not None and isinstance(display_score, str):
            ATTR_RATEMAP_OV.append(display_score)
        else:
            display_score = None

        # Check attributes in datajoint join 
        if not self.__check_join_integrity(ATTR_RATEMAP_OV): 
            raise KeyError('One or more of these were not found: {}'.format(ATTR_RATEMAP_OV))
       

        ###########################################################################
        ###############         START PLOTTING FUNCTIONS 

        if self.keys is not None:
            iterator = self.keys
            use_keys = True
        else:
            iterator = self.dj_object
            use_keys = False
            
        if self.total is not None:
            total = self.total
        else:
            total = len(iterator)
        
        # Cue card positions 
        if cue_card_pos is not None: 
            if isinstance(cue_card_pos, str):
                cue_card_pos = [cue_card_pos] * total
            else:
                assert len(cue_card_pos) == total, \
                    'Length of cue card position array does not match length of cells to plot'


        # Make loop with tqdm progress bar
        tqdm_iterator = self.__tqdm_iterator(iterator, total, 'Drawing ratemaps')
        
        for no, key in tqdm_iterator:    

            if no == total:
                # Premature stop? Make sure you close things gracefully:
                tqdm_iterator.refresh()
                tqdm._instances.clear()
                plt.close()
                break
        
            # Use keys or join object? 
            if use_keys:
                entry = (self.dj_object & key).fetch1()
            else:
                entry = key 

            # Get session dictionary object vector 
            session_dict         = make_multi_session_object_dict(entry)
            session_dict, max_rm = _get_ovc_ratemaps(session_dict, entry) # Returns ratemaps and object positions and max over ratemaps


            if session_dict['object1']['session_name'] == session_dict['object2']['session_name']:
                two_object_sess = True # This is a "special case" -> 2 objects in one session 
                figure = self.__create_figure_grid_ov_2
                ax_base    = figure.add_subplot(1,2,1)
                ax_object1 = figure.add_subplot(1,2,2)

            else:
                # 2 separate object sessions 
                two_object_sess = False
                figure = self.__create_figure_grid_ov_3
                ax_base    = figure.add_subplot(1,3,1)
                ax_object1 = figure.add_subplot(1,3,2)
                ax_object2 = figure.add_subplot(1,3,3)


            # Fill axes 
            if normalize_base:
                rm1 = ax_base.imshow(session_dict['base']['ratemap'], vmin=0, vmax=max_rm, cmap=cmap) # Normalized view
                rm2 = ax_object1.imshow(session_dict['object1']['ratemap'], vmin=0, vmax=max_rm, cmap=cmap) 
                if not two_object_sess:
                    rm3 = ax_object2.imshow(session_dict['object2']['ratemap'], vmin=0, vmax=max_rm, cmap=cmap) 

            else: 
                rm1 = ax_base.imshow(session_dict['base']['ratemap'], cmap=cmap) 
                rm2 = ax_object1.imshow(session_dict['object1']['ratemap'], cmap=cmap) 
                if not two_object_sess:
                    rm3 = ax_object2.imshow(session_dict['object2']['ratemap'], cmap=cmap)      

            # Draw objects
            ax_object1.scatter(session_dict['object1']['object_x_rm'], session_dict['object1']['object_y_rm'], marker='s', s=800, color='k')
            ax_object1.scatter(session_dict['object1']['object_x_rm'], session_dict['object1']['object_y_rm'], marker='s', s=500, color='#ccc')
            ax_object1.scatter(session_dict['object1']['object_x_rm'], session_dict['object1']['object_y_rm'], marker='s', s=300, color='w')
            ax_object1.scatter(session_dict['object1']['object_x_rm'], session_dict['object1']['object_y_rm'], marker='x', s=100, color='k')
            if not two_object_sess:
                ax_object2.scatter(session_dict['object2']['object_x_rm'], session_dict['object2']['object_y_rm'], marker='s', s=800, color='k')
                ax_object2.scatter(session_dict['object2']['object_x_rm'], session_dict['object2']['object_y_rm'], marker='s', s=500, color='#ccc')
                ax_object2.scatter(session_dict['object2']['object_x_rm'], session_dict['object2']['object_y_rm'], marker='s', s=300, color='w')
                ax_object2.scatter(session_dict['object2']['object_x_rm'], session_dict['object2']['object_y_rm'], marker='x', s=100, color='k')
            else:
                # Draw second object into object 1 session axis
                ax_object1.scatter(session_dict['object2']['object_x_rm'], session_dict['object2']['object_y_rm'], marker='s', s=800, color='k')
                ax_object1.scatter(session_dict['object2']['object_x_rm'], session_dict['object2']['object_y_rm'], marker='s', s=500, color='#ccc')
                ax_object1.scatter(session_dict['object2']['object_x_rm'], session_dict['object2']['object_y_rm'], marker='s', s=300, color='w')
                ax_object1.scatter(session_dict['object2']['object_x_rm'], session_dict['object2']['object_y_rm'], marker='x', s=100, color='k')
            

            ax_base.set_aspect('equal')
            ax_base.get_xaxis().set_ticks([]); ax_base.get_yaxis().set_ticks([])    
            ax_object1.set_aspect('equal')
            ax_object1.get_xaxis().set_ticks([]); ax_object1.get_yaxis().set_ticks([])    
            if not two_object_sess:
                ax_object2.set_aspect('equal')
                ax_object2.get_xaxis().set_ticks([]); ax_object2.get_yaxis().set_ticks([])    

            # Axes linewidth 
            for axis in ['top','bottom','left','right']:
                ax_base.spines[axis].set_linewidth(axes_lw)
                ax_object1.spines[axis].set_linewidth(axes_lw)
                if not two_object_sess:
                    ax_object2.spines[axis].set_linewidth(axes_lw)

            # Draw cue card?
            if cue_card_pos is not None: 
                size = session_dict['base']['ratemap'].shape # Just take one of them for now - should be fine
                card_pos = cue_card_pos[no]
                if card_pos == 'west':
                    ax_base.plot([0.,0.],[size[0]/2-5,size[0]/2+5], lw=3.5, color='white', clip_on=False, zorder=10, solid_capstyle='butt')
                    ax_object1.plot([0.,0.],[size[0]/2-5,size[0]/2+5], lw=3.5, color='white', clip_on=False, zorder=10, solid_capstyle='butt')
                    if not two_object_sess:
                        ax_object2.plot([0.,0.],[size[0]/2-5,size[0]/2+5], lw=3.5, color='white', clip_on=False, zorder=10, solid_capstyle='butt')
                elif card_pos == 'east':
                    ax_base.plot([size[1]-1,size[1]-1],[size[0]/2-5,size[0]/2+5], lw=3.5, color='white', clip_on=False, zorder=10, solid_capstyle='butt')
                    ax_object1.plot([size[1]-1,size[1]-1],[size[0]/2-5,size[0]/2+5], lw=3.5, color='white', clip_on=False, zorder=10, solid_capstyle='butt')
                    if not two_object_sess:
                        ax_object2.plot([size[1]-1,size[1]-1],[size[0]/2-5,size[0]/2+5], lw=3.5, color='white', clip_on=False, zorder=10, solid_capstyle='butt')
                elif card_pos == 'north':
                    ax_base.plot([size[1]/2-5,size[1]/2+5],[0,0], lw=3.5, color='white', clip_on=False, zorder=10, solid_capstyle='butt')
                    ax_object1.plot([size[1]/2-5,size[1]/2+5],[0,0], lw=3.5, color='white', clip_on=False, zorder=10, solid_capstyle='butt')
                    if not two_object_sess:
                        ax_object2.plot([size[1]/2-5,size[1]/2+5],[0,0], lw=3.5, color='white', clip_on=False, zorder=10, solid_capstyle='butt')
                elif card_pos == 'south':
                    ax_base.plot([size[1]/2-5,size[1]/2+5],[size[0]-1,size[0]-1], lw=3.5, color='white', clip_on=False, zorder=10, solid_capstyle='butt')
                    ax_object1.plot([size[1]/2-5,size[1]/2+5],[size[0]-1,size[0]-1], lw=3.5, color='white', clip_on=False, zorder=10, solid_capstyle='butt')
                    if not two_object_sess:
                        ax_object2.plot([size[1]/2-5,size[1]/2+5],[size[0]-1,size[0]-1], lw=3.5, color='white', clip_on=False, zorder=10, solid_capstyle='butt')
                else: 
                    raise NotImplementedError(f'Card position {card_pos} not understood. Choose ["west","east","north","south"]')


            # Create colorbars
            plt.colorbar(rm1, ax=ax_base, fraction=0.03)
            plt.colorbar(rm1, ax=ax_object1, fraction=0.03)
            if not two_object_sess:
                plt.colorbar(rm1, ax=ax_object2, fraction=0.03)

            # Add title
            if display_title:
                # Get subplot title
                title = self.__title(entry, display_score, hash_or_animal, ov=True)
                ax_base.set_title(title)

            plt.subplots_adjust(wspace=.29)
            plt.show()

        return



    def tracking(self, **kwargs):
        ''' 
        Plot tracking plots in 5x5 grid
        Optionally shows score for every subplot if available.
        ! This uses self.path_spike(draw_spikes=False) to generate plots.
        
        Parameters
        ----------
        **kwargs:
            cmap          : string
                            Valid matplotlib colormap string
                            https://matplotlib.org/3.2.1/tutorials/colors/colormaps.html
                            or https://github.com/1313e/CMasher
            hash_or_animal: string
                            'hash' or 'animal'
                            Determines whether session name (hash) or animal/timestamp 
                            combination should be displayed in title of each subplot.
                            Defaults to 'animal'
            display_score : string
                            Name of the score (=column name) to display in title 
                            of each subplot.
                            Defaults to None.
            draw_speed    : bool
                            Encode speed in size of dots in plot? 
            path_dot_size : int (float)
                            If draw_speed==False: Dot size for tracking
            draw_angle    : bool
                            Encode angle in color of dots in plot?
            speed_scaler  : float
                            How much smaller than actual speed should dot size be?
            alpha_path    : float
            
                        
        ''' 
        # Process kwargs
        cmap           = kwargs.get('cmap', sns.husl_palette(as_cmap=True))
        cmap           = plt.get_cmap(cmap)
        display_score  = kwargs.get('display_score', None)
        hash_or_animal = kwargs.get('hash_or_animal', 'animal')
        draw_speed     = kwargs.get('draw_speed', False)
        path_dot_size  = kwargs.get('path_dot_size', 1.2)
        draw_angle     = kwargs.get('draw_angle', False)
        speed_scaler   = kwargs.get('speed_scaler', .5)
        alpha_path     = kwargs.get('alpha_path', 1)


        self.path_spike(draw_spikes=False, cmap=cmap, \
                            display_score=display_score, hash_or_animal=hash_or_animal,
                            draw_speed=draw_speed,  path_dot_size=path_dot_size,
                            draw_angle=draw_angle, speed_scaler=speed_scaler, alpha_path=alpha_path)

        return 



    def path_spike(self, **kwargs):
        ''' 
        Plot path-spike plots in 5x5 grid
        Optionally shows score for every subplot if available.
        ! This is used also as plotting container function for self.tracking().
        
        Parameters
        ----------
        **kwargs:
            cmap          : string
                            Valid matplotlib colormap string
                            https://matplotlib.org/3.2.1/tutorials/colors/colormaps.html
                            or https://github.com/1313e/CMasher
            hash_or_animal: string
                            'hash' or 'animal'
                            Determines whether session name (hash) or animal/timestamp 
                            combination should be displayed in title of each subplot.
                            Defaults to 'animal'
            display_score : string
                            Name of the score (=column name) to display in title 
                            of each subplot.
                            Defaults to None.
            draw_spikes   : bool
                            Show spikes? 
            draw_speed    : bool
                            Encode speed in size of dots in path plot? 
            draw_angle    : bool 
                            Encode angle in color of dots in plot?   
            draw_time     : bool 
                            Encode time since session start in color of dots? 
            path_dot_size : int (float)
                            If draw_speed==False: Dot size for tracking
            draw_hd       : bool
                            Encode angle in color of dots in plot?
                            If draw_spikes==False this will color the path plot
                            If draw_spikes==True this will color the spikes
            speed_scaler  : float
                            How much smaller than actual speed should dot size be?
            spike_scaler  : float
                            How much smaller than actual spike size should the dot be?
            spike_color   : list/string
                            Valid color for spikes; defaults to 'k'
            alpha_path    : float
            alpha_spikes  : float
            display_title : bool 
                            Show title? 
                        

                        
        ''' 
        # Process kwargs
        cmap           = kwargs.get('cmap',  sns.husl_palette(as_cmap=True))
        display_score  = kwargs.get('display_score', None)
        hash_or_animal = kwargs.get('hash_or_animal', 'animal')
        draw_spikes    = kwargs.get('draw_spikes',True)
        draw_speed     = kwargs.get('draw_speed', False)
        draw_angle     = kwargs.get('draw_angle', False)
        draw_time      = kwargs.get('draw_time', False)
        path_dot_size  = kwargs.get('path_dot_size', 1.2)
        draw_hd        = kwargs.get('draw_hd', False)
        speed_scaler   = kwargs.get('speed_scaler', .5)
        spike_scaler   = kwargs.get('spike_scaler', 120)
        spike_color    = kwargs.get('spike_color','k')
        alpha_path     = kwargs.get('alpha_path', 1)
        alpha_spikes   = kwargs.get('alpha_spikes', .7)
        display_title  = kwargs.get('display_title', True)

        # Prepare colormap (cmap)
        # for feeding make_circular_colormap or make_linear_colormap below
        try:
            cmap = list(sns.color_palette(cmap, 256))
        except TypeError:
            cmap = plt.get_cmap(cmap).colors

        # Prepare list of attributes to check:
        if draw_spikes:
            ATTR_PATHSPIKE = self.ATTR_PATHSPIKE.copy()
        else:
            # If no spikes should be shown, use a short TRACKING attributes list
            ATTR_PATHSPIKE = self.ATTR_TRACKING.copy()
        # Display session hash or animal_name/timestamp? 
        if hash_or_animal == 'hash':
            ATTR_PATHSPIKE.append(self.SESSION_HASH)
        elif hash_or_animal == 'animal':
            ATTR_PATHSPIKE.append(self.ANIMAL_NAME)
            ATTR_PATHSPIKE.append(self.TIMESTAMP)    
        else:
            raise NameError(f'Keyword "{hash_or_animal}" not recognized')
        # Display a score?
        if display_score is not None and isinstance(display_score, str):
            ATTR_PATHSPIKE.append(display_score)
        else:
            display_score = None

        if draw_speed:
            ATTR_PATHSPIKE.append('speed')
        if draw_hd or draw_angle:
            ATTR_PATHSPIKE.append('head_angle')
        
        # Check attributes in datajoint join 
        if not self.__check_join_integrity(ATTR_PATHSPIKE): 
            raise KeyError('One or more of these were not found: {}'.format(ATTR_PATHSPIKE))
       

        ###########################################################################
        ###############         START PLOTTING FUNCTIONS 

        plot_counter = 1 

        if self.keys is not None:
            iterator = self.keys
            use_keys = True
        else:
            iterator = self.dj_object
            use_keys = False
            
        if self.total is not None:
            total = self.total
        else:
            total = len(iterator)
        
        # Make loop with tqdm progress bar
        tqdm_iterator = self.__tqdm_iterator(iterator, total, 'Drawing path-spike plots')
        
        figure = self.__create_figure_grid
        for no, key in tqdm_iterator:    
            
            if no == total:
                if self.save_path is not None: 
                    print('Saving figure under {}'.format(str(self.save_path)))
                    figure_export_text = ['spikes' if draw_spikes else ''][0]
                    figure.savefig(self.save_path / f'tracking {figure_export_text} {plot_counter-1}.pdf', bbox_inches='tight')
                
                plt.show()
                # Premature stop? Make sure you close things gracefully:
                tqdm_iterator.refresh()
                tqdm._instances.clear()
                plt.close()
                break
            
            # Use keys or join object? 
            if use_keys:
                entry = (self.dj_object & key).fetch1()
            else:
                entry = key 
                
            ax = figure.add_subplot(5,5,plot_counter)
            if not draw_spikes:  

                ax.scatter(entry['x_pos'], entry['y_pos'],
                            s=[path_dot_size if not draw_speed else (entry['speed']/np.percentile(entry['speed'],95))/speed_scaler],
                            c=[[[.2,.2,.2]] if not any([draw_angle, draw_hd]) else make_circular_colormap(entry['head_angle'],cmap=cmap)][0],
                            lw=0,
                            alpha=alpha_path)
            else:
                ax.scatter(entry['x_pos'], entry['y_pos'],
                            s=[path_dot_size if not draw_speed else (entry['speed']/np.percentile(entry['speed'],95))/speed_scaler],
                            c=[[.75,.75,.75]],
                            lw=0,
                            alpha=alpha_path)

                assert (np.array([draw_hd, draw_time]) == True).all() == False, 'Draw time and hd are both true - choose one'

                if draw_hd:
                    colors_spikes = make_circular_colormap(entry['head_angle_signal'], cmap=cmap)
                elif draw_time:
                    indices_signal = get_signal_indices(entry['x_pos_signal'], entry['x_pos'])
                    colors_spikes  = make_linear_colormap(indices_signal, \
                                        reference_numbers=np.arange(len(entry['x_pos'])), cmap=cmap) 
                else:
                    colors_spikes = [[spike_color] if isinstance(spike_color,list) else spike_color][0]
                # Draw signal ...
                scaled_signal = (entry['signal']/entry['signal'].max())*spike_scaler
                ax.scatter(entry['x_pos_signal'], entry['y_pos_signal'], 
                                s=scaled_signal, 
                                c=colors_spikes, 
                                lw=0,
                                alpha=alpha_spikes)

            ax.set_aspect('equal')
            ax.autoscale(enable=True, tight=True)
            ax.invert_yaxis()

            ax.get_xaxis().set_ticks([]);ax.get_yaxis().set_ticks([])    
            if display_title:
                title = self.__title(entry, display_score, hash_or_animal, show_cell=draw_spikes)
                ax.set_title(title)

            sns.despine(left=True, bottom=True)
            plot_counter += 1       

            if plot_counter > self.plots_per_view:
                if self.save_path is not None: 
                    print('Saving figure under {}'.format(str(self.save_path)))
                    figure_export_text = ['spikes' if draw_spikes else ''][0]
                    figure.savefig(self.save_path / f'tracking {figure_export_text} {plot_counter-1}.pdf', bbox_inches='tight')
                
                plot_counter = 1
                plt.show()
                # Create next figure
                figure = self.__create_figure_grid

        return


    def path_spike_ov(self, **kwargs):
        ''' 
        SPECIAL! 
        - This function gets data for each session 

        Plot path-spike plots for object vector (ov) cells (1x3):
             - base_session
             - object1_session
             - object2_session 

        Parameters
        ----------
        **kwargs:
            cmap          : string
                            Valid matplotlib colormap string
                            https://matplotlib.org/3.2.1/tutorials/colors/colormaps.html
                            or https://github.com/1313e/CMasher
            hash_or_animal: string
                            'hash' or 'animal'
                            Determines whether session name (hash) or animal/timestamp 
                            combination should be displayed in title of each subplot.
                            Defaults to 'animal'
            display_score : string
                            Name of the score (=column name) to display in title 
                            of each subplot.
                            Defaults to None.
            draw_spikes   : bool
                            Show spikes? 
            draw_speed    : bool
                            Encode speed in size of dots in path plot? 
            path_dot_size : int (float)
                            If draw_speed==False: Dot size for tracking
            draw_hd       : bool
                            Encode angle in color of dots in plot?
                            If draw_spikes==False this will color the path plot
                            If draw_spikes==True this will color the spikes
            speed_scaler  : float
                            How much smaller than actual speed should dot size be?
            spike_scaler  : float
                            How much smaller than actual spike size should the dot be?
            spike_color   : list/string
                            Valid color for spikes; defaults to 'k'
            alpha_path    : float
            alpha_spikes  : float
            display_title : bool 
                            Show title? 

                        
        ''' 
        # Process kwargs
        cmap           = kwargs.get('cmap', 'magma')
        cmap           = plt.get_cmap(cmap)
        display_score  = kwargs.get('display_score', None)
        hash_or_animal = kwargs.get('hash_or_animal', 'animal')
        draw_spikes    = kwargs.get('draw_spikes',True)
        draw_speed     = kwargs.get('draw_speed', False)
        path_dot_size  = kwargs.get('path_dot_size', 1.2)
        draw_hd        = kwargs.get('draw_hd', False)
        speed_scaler   = kwargs.get('speed_scaler', .5)
        spike_scaler   = kwargs.get('spike_scaler', 120)
        spike_color    = kwargs.get('spike_color','k')
        alpha_path     = kwargs.get('alpha_path', 1)
        alpha_spikes   = kwargs.get('alpha_spikes', .7)
        display_title  = kwargs.get('display_title', True)


        ATTR_PATHSPIKE_OV = self.ATTR_PATHSPIKE_OV.copy()
        # Display session hash or animal_name/timestamp? 
        if hash_or_animal == 'hash':
            ATTR_PATHSPIKE_OV.append(self.SESSION_HASH_OV)    
        elif hash_or_animal == 'animal':
            ATTR_PATHSPIKE_OV.append(self.ANIMAL_NAME)
            ATTR_PATHSPIKE_OV.append(self.TIMESTAMP)    
        else:
            raise NameError(f'Keyword "{hash_or_animal}" not recognized')
        # Display a score?
        if display_score is not None and isinstance(display_score, str):
            ATTR_PATHSPIKE_OV.append(display_score)
        else:
            display_score = None

        # Check attributes in datajoint join 
        if not self.__check_join_integrity(ATTR_PATHSPIKE_OV): 
            raise KeyError('One or more of these were not found: {}'.format(ATTR_PATHSPIKE_OV))
       

        ###########################################################################
        ###############         START PLOTTING FUNCTIONS 


        if self.keys is not None:
            iterator = self.keys
            use_keys = True
        else:
            iterator = self.dj_object
            use_keys = False
            
        if self.total is not None:
            total = self.total
        else:
            total = len(iterator)
        
        # Make loop with tqdm progress bar
        tqdm_iterator = self.__tqdm_iterator(iterator, total, 'Drawing path-spike plots')
        
        for no, key in tqdm_iterator:    
            
            if no == total:
                # Premature stop? Make sure you close things gracefully:
                tqdm_iterator.refresh()
                tqdm._instances.clear()
                plt.close()
                break
            
            # Use keys or join object? 
            if use_keys:
                entry = (self.dj_object & key).fetch1()
            else:
                entry = key 
            
            # Get session dictionary object vector 
            session_dict         = make_multi_session_object_dict(entry)
            session_dict         = _get_ovc_tracking_signal(session_dict, entry) # Returns tracking and signal

            if session_dict['object1']['session_name'] == session_dict['object2']['session_name']:
                two_object_sess = True # This is a "special case" -> 2 objects in one session 
                figure = self.__create_figure_grid_ov_2
                ax_base    = figure.add_subplot(1,2,1)
                ax_object1 = figure.add_subplot(1,2,2)
                axes = [ax_base, ax_object1]
                sessions = ['base', 'object1']
            else:
                # 2 separate object sessions 
                two_object_sess = False
                figure = self.__create_figure_grid_ov_3
                ax_base    = figure.add_subplot(1,3,1)
                ax_object1 = figure.add_subplot(1,3,2)
                ax_object2 = figure.add_subplot(1,3,3)
                axes = [ax_base, ax_object1, ax_object2]
                sessions = ['base', 'object1', 'object2']


            # Get subplot title
            title = self.__title(entry, display_score, hash_or_animal, show_cell=draw_spikes, ov=True)

            for ax, session in zip(axes, sessions):
                if not draw_spikes:  
                    ax.scatter(session_dict[session]['tracking']['x_pos'],
                               session_dict[session]['tracking']['y_pos'],
                                s=[path_dot_size if not draw_speed else \
                                    (session_dict[session]['tracking']['speed']/np.percentile(session_dict[session]['tracking']['speed'],95))/speed_scaler],
                                c=[[[.2,.2,.2]] if not draw_hd else make_circular_colormap(session_dict[session]['tracking']['head_angle'], cmap=cmap)][0],
                                lw=0,
                                alpha=alpha_path)
                else:
                    ax.scatter(session_dict[session]['tracking']['x_pos'], 
                               session_dict[session]['tracking']['y_pos'],
                                s=[path_dot_size if not draw_speed else (session_dict[session]['tracking']['speed']/np.percentile(session_dict[session]['tracking']['speed'],95))/speed_scaler],
                                c=[[.75,.75,.75]],
                                lw=0,
                                alpha=alpha_path)

                    if draw_hd:
                        colors_spikes = make_circular_colormap(session_dict[session]['signal']['head_angle_signal'])
                    else:
                        colors_spikes = [[spike_color] if isinstance(spike_color,list) else spike_color][0]
                    # Draw signal ...
                    scaled_signal = (session_dict[session]['signal']['signal']/session_dict[session]['signal']['signal'].max())*spike_scaler
                    ax.scatter(session_dict[session]['signal']['x_pos_signal'], session_dict[session]['signal']['y_pos_signal'], 
                                    s=scaled_signal, 
                                    c=colors_spikes, 
                                    lw=0,
                                    alpha=alpha_spikes)

                ax.set_aspect('equal')
                ax.autoscale(enable=True, tight=True)
                ax.invert_yaxis()

                ax.get_xaxis().set_ticks([]);ax.get_yaxis().set_ticks([])    
            
            # Add title
            if display_title:
                # Get subplot title
                title = self.__title(entry, display_score, hash_or_animal, ov=True)
                ax_base.set_title(title)

            sns.despine(left=True, bottom=True)    
            plt.tight_layout()       
            plt.show()
            
        return


    def rois(self, **kwargs):
            ''' 
            Plot ROIs 
            
            Parameters
            ----------
            **kwargs:
                cmap           : string
                                 Valid matplotlib colormap string
                                 https://matplotlib.org/3.2.1/tutorials/colors/colormaps.html
                                 or https://github.com/1313e/CMasher
                invert_img_cmap:
                hash_or_animal : string
                                 'hash' or 'animal'
                                 Determines whether session name (hash) or animal/timestamp 
                                 combination should be displayed in title.
                                 Defaults to 'animal'
                color_mapping  : string
                                 Name of the attribute (=column name) to generate colors over
                                 Defaults to None.
                draw_image     : bool
                                 Draw the max image? Defaults to False.

                image_key      : string
                                 Key of image to plot. Defaults to 'max_img'. 

                draw_centers   : bool
                                 Draw the center points? Defaults to True. 
                draw_numbers   : bool 
                                 Draw label on top of the cells (cell index)?
                                 Defaults to False. 
                draw_pixels    : bool 
                                 Draw all pixels of extracted cells.
                                 Defaults to False.
                draw_outlines  : bool 
                                 Draw ROI outlines?
                                 Defaults to False.
                text_color     : string
                                 Color of label text. Defaults to 'k' (black).
                fontsize       : float
                                 Font size for annotations (default: 15.)
                dot_color      : string  
                                 Color of cell center dots. Defaults 'k'.
                dot_size       : int 
                                 Size of all drawn dots (global for all scatter plots).
                                 Defaults to 5.
                alpha          : float 
                                 Transparency (0-1) of dots (global for all scatter plots).
                                 Defaults to .8.
                colors         : color array with color for every cell.
                                 Use this to color code cells according to certain properties.
                                 Defaults to random color palette based on husl only if 'color_mapping' 
                                 attribute is not set. 
                return_axes    : bool
                                 Return axes if True
                return_figure  : bool
                                 Return figure object if True (and plot).
                                 This is overridden if return_axes = True
                display_title  : bool 
                                 Show title? 
            ''' 

            # Process kwargs
            cmap           = kwargs.get('cmap', 'magma')
            cmap           = plt.get_cmap(cmap)
            invert_img_cmap= kwargs.get('invert_img_cmap', False)
            hash_or_animal = kwargs.get('hash_or_animal', 'animal')
            color_mapping  = kwargs.get('color_mapping', None)
            draw_image     = kwargs.get('draw_image', False)
            image_key      = kwargs.get('image_key', 'max_img')
            draw_centers   = kwargs.get('draw_centers', True)
            draw_numbers   = kwargs.get('draw_numbers', False)
            draw_pixels    = kwargs.get('draw_pixels', False)
            draw_outlines  = kwargs.get('draw_outlines', False) 
            text_color     = kwargs.get('text_color', 'k')
            fontsize       = kwargs.get('fontsize', 15.)
            dot_color      = kwargs.get('dot_color', 'k')
            dot_size       = kwargs.get('dot_size', 8)
            alpha          = kwargs.get('alpha', .8)
            colors         = kwargs.get('colors', None)
            return_axes    = kwargs.get('return_axes',False)
            return_figure  = kwargs.get('return_figure',False)
            display_title  = kwargs.get('display_title', True)


            def __iscorr():
                # Find out if we are dealing with "corr" corrected
                # table output or non-corrected (raw) output
                # Listens for the keyword _corr in SQL query after "FROM".
                # (Assumes 'SomethingCorr' as table name)
                sql = self.dj_object.make_sql()
                return '_corr' in sql.split('FROM')[1]

            def __dataset_name():
                # Sometimes the attribute 'dataset_name' gets renamed to 'signal_dataset'
                if 'signal_dataset' in self.__attributes:
                    return 'signal_dataset'
                else:
                    return 'dataset_name'

            # Prepare list of attributes to check
            # Find out if we are handling unwarped (corr) or raw data
            if __iscorr(): 
                ATTR_ROIS = self.ATTR_ROIS_CORR.copy()
                image_key = [image_key + '_corr' if '_corr' not in image_key else image_key][0]
                CENTER_X = 'center_x_corr'
                CENTER_Y = 'center_y_corr'
                PIXELS_X = 'xpix_corr'
                PIXELS_Y = 'ypix_corr'
                XLIM = 'x_range_microns_eff'
                YLIM = 'y_range_microns_eff'
                LAMBDA = 'lambda_corr'
                iscorr=True

            else:
                ATTR_ROIS = self.ATTR_ROIS.copy()
                image_key = [image_key.split('_corr')[0] if '_corr' in image_key else image_key][0]
                CENTER_X = 'center_x'
                CENTER_Y = 'center_y'
                PIXELS_X = 'xpix'
                PIXELS_Y = 'ypix'
                XLIM = 'x_range'
                YLIM = 'y_range'
                LAMBDA = 'lambda'
                iscorr=False


            # Display session hash or animal_name/timestamp? 
            if hash_or_animal == 'hash':
                ATTR_ROIS.append(self.SESSION_HASH)
            elif hash_or_animal == 'animal':
                ATTR_ROIS.append(self.ANIMAL_NAME)
                ATTR_ROIS.append(self.TIMESTAMP)    
            else:
                raise NameError(f'Keyword "{hash_or_animal}" not recognized')
            
            if color_mapping is not None: 
                ATTR_ROIS.append(color_mapping)    

            ATTR_ROIS.append(image_key) 

            # Check attributes in datajoint join 
            if not self.__check_join_integrity(ATTR_ROIS): 
                raise KeyError('One or more of these were not found: {}'.format(ATTR_ROIS))
                
            ###########################################################################
            ###############         START PLOTTING FUNCTIONS 


            if self.keys is not None:
                iterator = self.keys
            else:
                iterator = self.dj_object

            # Check if there is more than one imaging analysis dataset available
            # If this is true, then multiple sessions are returned and this function 
            # cannot be used
            if len(set(iterator.fetch(__dataset_name()))) != 1:
                raise KeyError('More than one dataset found (indicating multiple results)')

            # Take care of color palette
            if colors is None: 
                # No color array given
                # Generate either over "color_mapping" attribute 
                # (take whole session as basis, no matter what)
                # or random (over 'cell_ids')
                if color_mapping is not None: 
                    colors = make_linear_colormap(iterator.fetch(color_mapping),\
                                 reference_numbers=self.dj_object.fetch(color_mapping),\
                                 cmap=cmap)
                else:
                    # "random"
                    colors = make_linear_colormap(iterator.fetch('cell_id'), cmap=cmap)
            else:
                # Check integrity
                if len(colors) != len(iterator): 
                    raise IndexError('Color length does not match length of datajoint results')
            
            # Make loop with tqdm progress bar
            # In this case it is just a very small "package" since most of the data will be pre-fetched
            tqdm_iterator = self.__tqdm_iterator(iterator.proj(), len(iterator), 'Drawing ROIs')
            # Before looping, pre-fetch large results:  CENTER_X, CENTER_Y, PIXELS_X, PIXELS_Y etc
            pixel_data = pd.DataFrame(self.dj_object.fetch('KEY', *ATTR_ROIS, as_dict=True))
            pixel_data.set_index('cell_id',inplace=True)
            
            # Figure
            figure = self.__create_figure_single
            ax     = figure.add_subplot(111)

            # Loop over cells and draw 
            for no, key in tqdm_iterator:   

                entry = pixel_data.loc[key['cell_id']]

                if no == 0:
                    # Get figure title
                    if display_title:
                        title = self.__title(entry, color_mapping, hash_or_animal, show_cell=False)
                        ax.set_title(title)
                    # Get image 
                    image_ = ax.imshow(entry[image_key], cmap=['gist_gray' if not invert_img_cmap else 'gist_gray_r'][0])
                    if not draw_image: image_.remove() # Need to draw it anyway first!

                if draw_pixels:
                    npixels = len(entry[PIXELS_X])
                    rgba_colors = np.broadcast_to(colors[no],(npixels,3))
                    rgba_colors = np.hstack((rgba_colors, np.zeros((npixels,1))))
                    norm_alpha_px = entry[LAMBDA] / entry[LAMBDA].max()
                    rgba_colors[:, 3] = norm_alpha_px 
                    ax.scatter(entry[PIXELS_X], entry[PIXELS_Y], s=3, lw=0, color=rgba_colors, marker='o')

                if draw_centers:
                    ax.scatter(entry[CENTER_X],entry[CENTER_Y], s=dot_size, lw=1.5, c=[colors[no] if not draw_pixels else dot_color], alpha=alpha)

                if draw_numbers:
                    # .name holds the index and it was set to 'cell_id' above
                    ax.text(entry[CENTER_X], entry[CENTER_Y],f'{entry.name}', color=text_color, \
                                                                                 ha='center', va='center',\
                                                                                    fontsize=fontsize)     

                if draw_outlines:
                    zero_image = np.zeros_like(entry[image_key])
                    zero_image[entry[PIXELS_Y], entry[PIXELS_X]] = 1
                    zero_image = gaussian(zero_image, sigma=.15, mode='nearest', preserve_range=True, truncate=4.0)
                    distance = distance_transform_edt(zero_image)
                    distance[distance != 1] = 0
                    outline = np.where(distance == 1)
                    ax.scatter(outline[1],outline[0], s=dot_size/10, c=[colors[no] if not draw_pixels else dot_color], alpha=alpha, marker='o')


            # Take care of axes styling 
            ax.axes.get_xaxis().set_visible(False)
            ax.axes.get_yaxis().set_visible(False)

            ax.set_xlim(entry[XLIM])
            ax.set_ylim(entry[YLIM][::-1])

            sns.despine(left=True, bottom=True)   

            if self.save_path is not None: 
                print('Saving figure under {}'.format(str(self.save_path)))
                figure.savefig(self.save_path / f'rois {title}.pdf', bbox_inches='tight')

            if return_axes:
                return ax
            if return_figure:
                return figure


########################################################################################################################################################################################################################
########################################################################################################################################################################################################################
### HELPERS 


def _get_ovc_ratemaps(session_dict, key):
    '''
    Helper for ratemaps_ov
    - Fetch ratemaps
    - Object position in ratemap coordinates 

    '''
    fill_value = 0 # or np.nan
    
    for key_ in ['dataset_name', 'session_order','signal_dataset','tracking_dataset']:
        try:
            _ = key.pop(key_)
        except KeyError:
            pass
        
    rm, mask = (Ratemap & session_dict['base'] & key).fetch1('ratemap','mask_rm')
    ratemap = np.ma.array(rm, mask=mask).filled(fill_value=fill_value)
    session_dict['base']['ratemap'] = ratemap 
    
    for session in ['object1', 'object2']:
        # Get fields and object position (in ratemap coordinates)
        rm, mask = (Ratemap & session_dict[session] & key).fetch1('ratemap','mask_rm')
        ratemap = np.ma.array(rm, mask=mask).filled(fill_value=fill_value)
        
        # Take care of objects / positions 
        try:
            obj_x, obj_y     = (ArenaObjectPos & session_dict[session] & key).fetch1('obj_x_coord_calib','obj_y_coord_calib')
        except dj.DataJointError:
            obj = (ArenaObjectPos & session_dict[session] & key).fetch('obj_x_coord_calib','obj_y_coord_calib', as_dict=True)
            if session == 'object1':
                obj_x, obj_y = obj[0]['obj_x_coord_calib'], obj[0]['obj_y_coord_calib']
            else:
                obj_x, obj_y = obj[1]['obj_x_coord_calib'], obj[1]['obj_y_coord_calib']
            
        x_edges, y_edges = (Occupancy & session_dict[session] & key).fetch1('x_edges','y_edges')

        # ... Where is the object in ratemap "coordinates" (bins)
        bin_size_rm_x = np.mean(np.diff(x_edges))
        bin_size_rm_y = np.mean(np.diff(y_edges))

        obj_x_rm = ((obj_x - x_edges[0]) / bin_size_rm_x) - .5
        obj_y_rm = ((obj_y - y_edges[0]) / bin_size_rm_y) - .5

        session_dict[session]['ratemap']     = ratemap 
        session_dict[session]['object_x']    = obj_x
        session_dict[session]['object_y']    = obj_y
        session_dict[session]['object_x_rm'] = obj_x_rm
        session_dict[session]['object_y_rm'] = obj_y_rm
    
    max_rm = []
    for session in ['base','object1','object2']:
        max_rm_ = np.nanpercentile(session_dict[session]['ratemap'],99)
        max_rm.append(max_rm_)
    
    return session_dict, np.nanmax(max_rm)

def _get_ovc_tracking_signal(session_dict, key):
    '''
    Helper for path_spike_ov
    - Fetch tracking and signal 
    - Object positions
    '''
    
    for key_ in ['dataset_name', 'session_order','signal_dataset','tracking_dataset']:
        try:
            _ = key.pop(key_)
        except KeyError:
            pass
        
    tracking = (Tracking.OpenField & session_dict['base'] & key).fetch1()
    signal   = (SignalTracking & session_dict['base'] & key).fetch1()
    session_dict['base']['tracking'] = tracking 
    session_dict['base']['signal']   = signal 

    for session in ['object1', 'object2']:
        # Get fields and object position (in ratemap coordinates)
        tracking = (Tracking.OpenField & session_dict[session] & key).fetch1()
        signal   = (SignalTracking & session_dict[session] & key).fetch1()
        session_dict[session]['tracking'] = tracking 
        session_dict[session]['signal']   = signal 

        # Take care of objects / positions 
        try:
            obj_x, obj_y     = (ArenaObjectPos & session_dict[session] & key).fetch1('obj_x_coord_calib','obj_y_coord_calib')
        except dj.DataJointError:
            obj = (ArenaObjectPos & session_dict[session] & key).fetch('obj_x_coord_calib','obj_y_coord_calib', as_dict=True)
            if session == 'object1':
                obj_x, obj_y = obj[0]['obj_x_coord_calib'], obj[0]['obj_y_coord_calib']
            else:
                obj_x, obj_y = obj[1]['obj_x_coord_calib'], obj[1]['obj_y_coord_calib']

        session_dict[session]['object_x']    = obj_x
        session_dict[session]['object_y']    = obj_y

    return session_dict

def draw_vector_map(masked_histogram, radial_bins_hist, angular_bins_hist):
    '''
    Draw single vector map (Object vector cell related)
     
    masked_histogram  : np masked array 
    radial_bins_hist  : list (list(physt.special_histograms.polar_histogram.binnigs))
    angular_bins_hist : list (list(physt.special_histograms.polar_histogram.binnigs))  
    
    '''
    sns.set(style='white', font_scale=1.5)
    figure = plt.figure(figsize=(6,6))
    ax = figure.add_subplot(111)
    ax.imshow(masked_histogram.T,aspect='auto')

    ax.set_xlim(0, len(radial_bins_hist))
    ax.set_ylim(0, len(angular_bins_hist))

    no_xticks = len(ax.get_xticklabels())
    no_yticks = len(ax.get_yticklabels())-1

    ax.set_xticklabels(np.linspace(0, radial_bins_hist[-1], no_xticks));
    yticklabels = np.round(np.degrees(np.linspace(0, angular_bins_hist[-1], no_yticks)))
    ax.set_yticklabels(yticklabels);

    #ax.set_xlim(0,30)
    sns.despine(left=True,bottom=True)
    
    ax.set_xlabel('Distance [mm]')
    ax.set_ylabel('Angle [degrees]')
    plt.show()