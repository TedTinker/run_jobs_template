#%%

import os
import pickle
from time import sleep
import builtins
import datetime
import matplotlib
import argparse, ast
import torch
import psutil
import tkinter as tk

# -------------------------------
# DIRECTORY CHECK
# -------------------------------

from config import PROJECT_ROOT
os.chdir(PROJECT_ROOT)

# -------------------------------
# TORCH SETUP
# -------------------------------

device = torch.device('cpu')  # I recommend CPU explicitly

# -------------------------------
# Miscellaneous
# -------------------------------

font = {
    'family': 'sans-serif',
    'size': 22
}
matplotlib.rc('font', **font)

def print(*args, **kwargs):
    """Override built-in print to auto-flush."""
    kwargs['flush'] = True
    builtins.print(*args, **kwargs)

torch.set_printoptions(precision=3, sci_mode=False)

start_time = datetime.datetime.now()

def duration(start_time=start_time):
    """Return elapsed time since given start time (default: script start)."""
    delta = datetime.datetime.now() - start_time
    return delta

def print_duration(start_time, end_time, text=None, end_text=''):
    """Print the duration between two times with optional prefix text."""
    delta = end_time - start_time
    if text:
        print(f'{text}: {delta}{end_text}')
    else:
        print(f'{delta}{end_text}')

def estimate_total_duration(proportion_completed, start_time=start_time):
    """Estimate total time given progress percentage and elapsed time."""
    if proportion_completed == 0:
        return '?:??:??'
    so_far = datetime.datetime.now() - start_time
    estimated_total = so_far / proportion_completed
    estimated_total = estimated_total - datetime.timedelta(microseconds=estimated_total.microseconds)
    return estimated_total

def cpu_memory_usage():
    """Print memory usage of current Python process (in GB)."""
    process = psutil.Process(os.getpid())
    mem_usage_bytes = process.memory_info().rss
    mem_usage_gb = mem_usage_bytes / (1024 ** 3)
    print('memory use:', round(mem_usage_gb, 3), 'gigabytes')

#%%



# ---------------------------------------
# LIST OF ARGUMENTS
# ---------------------------------------



# Type for booleons in arguments.
def literal(arg_string): 
    return(ast.literal_eval(arg_string))


# Arguments to parse. 
parser = argparse.ArgumentParser()

    # Meta 
parser.add_argument('--arg_title',                      type=str,           default = 'default',
                    help='Title of argument-set containing all non-default arguments.') 
parser.add_argument('--arg_name',                       type=str,           default = 'default',
                    help='Title of argument-set for human-understanding.') 
parser.add_argument('--agents',                         type=int,           default = 36,
                    help='How many agents are trained in this job?')
parser.add_argument('--previous_agents',                type=int,           default = 0,
                    help='How many agents with this argument-set are trained in previous jobs?')
parser.add_argument('--init_seed',                      type=float,         default = 777,    
                    help='Random seed.')
parser.add_argument('--comp',                           type=str,           default = 'deigo',
                    help='Cluster name (deigo or saion).')
parser.add_argument('--device',                         type=str,           default = device,
                    help='Which device to use for Torch.')
parser.add_argument('--cpu',                            type=int,           default = 0,
                    help='Which cpu for affinity.')
parser.add_argument('--local',                          type=literal,       default = False,
                    help='Is this running on a local machine for testing?')
parser.add_argument('--show_duration',                  type=literal,       default = False,
                    help='Should durations be printed?')
parser.add_argument('--load_agents',                    type=literal,       default = False,
                    help='Are we loading agents?')      
parser.add_argument('--temp',                           type=literal,       default=False,
                    help='Is this a temporary save of data?')

    

# Make arguments.
try:
    default_args = parser.parse_args([])
    try:    
        args = parser.parse_args()
    except: 
        args, _ = parser.parse_known_args()
except:
    import sys 
    sys.argv=[''] 
    del sys           
    default_args = parser.parse_args([])
    try:    
        args = parser.parse_args()
    except: 
        args, _ = parser.parse_known_args()
    
    

# Based on arguments, adjust other arguments if needed.
def update_args(arg_set):
    if arg_set.comp == 'deigo':
        arg_set.half = False
    return(arg_set)

for arg_set in [default_args, args]:
    default_args = update_args(default_args) 
    args = update_args(args)
    
    

# ---------------------------------------
# MAKE A TITLE BASED ON ARGUMENTS, COMPARED TO DEFAULT ARGUMENTS
# ---------------------------------------

    
        
# Don't include these parameters in title.
args_not_in_title = [
    'arg_title', 'id', 'agents', 'previous_agents']

# Make a title for the arguments. 
def get_args_title(default_args, args):
    if args.arg_title[:3] == '___': 
        return(args.arg_title)
    name = '' 
    first = True
    arg_list = list(vars(default_args).keys())
    arg_list.insert(0, arg_list.pop(arg_list.index('arg_name')))
    for arg in arg_list:
        if arg in args_not_in_title: 
            pass 
        else: 
            default = getattr(default_args, arg)
            try:
                this_time = getattr(args, arg)
            except:
                this_time = 'NONE'
            if this_time == default: 
                pass
            elif arg == 'arg_name':
                name += '{} ('.format(this_time)
            else: 
                if first: 
                    first = False
                else: 
                    name += ', '
                name += '{}: {}'.format(arg, this_time)
    if name == '': 
        name = 'default' 
    else:           
        name += ')'
    if name.endswith(' ()'): 
        name = name[:-3]
    parts = name.split(',')
    name = '' 
    line = ''
    for i, part in enumerate(parts):
        if len(line) > 50 and len(part) > 2: 
            name += line + '\n' 
            line = ''
        line += part
        if i+1 != len(parts): 
            line += ','
    name += line
    return(name)

args.arg_title = get_args_title(default_args, args)

# Generate folders for saving agents and plots.
save_file = f'saved_{args.comp}'
os.makedirs(f'{save_file}', exist_ok=True)
os.makedirs(f'{save_file}/thesis_pics', exist_ok=True)
os.makedirs(f'{save_file}/thesis_pics/final', exist_ok=True)
folder = f'{save_file}/{args.arg_name}'

if args.arg_title[:3] != '___' and not args.arg_name in ['default', 'finishing_dictionaries']:
    os.makedirs(f'{folder}', exist_ok=True)
    os.makedirs(f'{folder}/agents', exist_ok=True)
    with open(f'{folder}/agents/args.pickle', 'wb') as handle:
        pickle.dump(args, handle)



# Print information about arguments.
if args == default_args: 
    print('Using default arguments.')
else:
    for arg in vars(default_args):
        default = getattr(default_args, arg)
        try:
            this_time = getattr(args, arg)
        except:
            this_time = 'NONE'
        if this_time != default:
            print('{}:\n\tDefault:\t{}\n\tThis time:\t{}'.format(arg, default, this_time))
        elif arg == 'device':
            print('{}:\n\tDefault:\t{}\n\tThis time:\t{}'.format(arg, default, this_time))
            
            
            
# If we are not showing durations, remove influence of this function.
if not args.show_duration:
    def print_duration(start_time, end_time, text = None, end_text = ''):
        pass
     


#%% 



""" 
For GUIs.
"""



def wait_for_button_press(button_label='Continue'):
    """Open a blocking Tkinter window with a button to resume execution."""
    def on_button_click():
        nonlocal continue_simulation
        continue_simulation = True
        root.destroy()

    root = tk.Tk()
    root.title('Wait for Input')
    root.geometry('200x100')
    button = tk.Button(root, text=button_label, command=on_button_click)
    button.pack(expand=True)
    continue_simulation = False
    root.mainloop()
    
    


#%%



""" 
Loading files for plotting, etc.
"""



def load_dicts(args):
    """Load value_dicts and min_max_dicts for a given experiment (saved runs)."""
    if os.getcwd().split('/')[-1] != save_file:
        os.chdir(save_file)

    value_dicts = []
    min_max_dicts = []

    if isinstance(args, dict):
        complete_order = args['titles']
    else:
        complete_order = args.arg_title[3:-3].split('+')

    order = [o for o in complete_order if o not in ['empty_space', 'break']]

    for name in order:
        print(f'Loading dictionaries for {name}...')
        got_value_dicts = False
        got_min_max_dicts = False
        while not got_value_dicts:
            with open(name + '/value_dict.pickle', 'rb') as handle:
                value_dicts.append(pickle.load(handle))
                got_value_dicts = True
        while not got_min_max_dicts:
            try:
                with open(name + '/min_max_dict.pickle', 'rb') as handle:
                    min_max_dicts.append(pickle.load(handle))
                    got_min_max_dicts = True
            except:
                print(f'Stuck trying to get {name}\'s min_max_dicts...')
                sleep(1)

    print('Loaded all dicts! Making min/max dict...')

    min_max_dict = {}
    for key in value_dicts[0].keys():
        if key not in [         # Ignore values not necessary to load.
            'args', 'arg_title', 'arg_name'
        ]:
            if key == 'hidden_state':
                min_maxes = []
                for layer in range(len(min_max_dicts[0][key])):
                    minimum = None
                    maximum = None
                    for mm_dict in min_max_dicts:
                        if minimum is None or minimum > mm_dict[key][layer][0]:
                            minimum = mm_dict[key][layer][0]
                        if maximum is None or maximum < mm_dict[key][layer][1]:
                            maximum = mm_dict[key][layer][1]
                    min_maxes.append((minimum, maximum))
                min_max_dict[key] = min_maxes
            else:
                minimum = None
                maximum = None
                for mm_dict in min_max_dicts:
                    if mm_dict[key] != (None, None):
                        if minimum is None or minimum > mm_dict[key][0]:
                            minimum = mm_dict[key][0]
                        if maximum is None or maximum < mm_dict[key][1]:
                            maximum = mm_dict[key][1]
                min_max_dict[key] = (minimum, maximum)

    print('Made min/max dict!')

    final_complete_order = []
    final_value_dicts = []

    for arg_name in complete_order:
        if arg_name in ['break', 'empty_space']:
            final_complete_order.append(arg_name)
        else:
            for value_dict in value_dicts:
                if value_dict['args'].arg_name == arg_name:
                    final_complete_order.append(arg_name)
                    final_value_dicts.append(value_dict)

    while final_complete_order and final_complete_order[0] in ['break', 'empty_space']:
        final_complete_order.pop(0)

    print('Done with Load Dicts!')
    return final_value_dicts, min_max_dict, complete_order
# %%
