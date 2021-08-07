#!/usr/bin/env python
# coding: utf-8

# In[1]:


# install munkres in the terminal
# pip install munkres
# https://github.com/bmc/munkres


# In[4]:


##################################################
#### Action-aware Perceptual Anchoring (AAPA) ####
# ---------------------------------------------- #
# Written by Sheryl Liang as part of ICARUS      #
#
##################################################

# Assume that input is read as a csv file with lines 
# 'objType,objId,x,y,z,w,h'

import csv
from munkres import Munkres, print_matrix, make_cost_matrix
import numpy as np
from data_processor import DataProcessor

class AAPA:
    # Action-Aware Perceptual Anchoring module
    # performs object alignment and hypothesis reasoning
    # TO DO
    max_it = 0
    
    def __init__(self, max_iterations):
        self.max_it = max_iterations
        
_VERBOSE_            = True

# Constants for camera movement alignment
HORIZ_OFFSET         = 0.008
PASSIVE_HORIZ_OFFSET = 0.01
DEPTH_OFFSET         = 0.01
WIDTH_OFFSET         = 0.01 # TODO: calculate from DEPTH_OFFSET
HEIGHT_OFFSET        = 0.01

# Constants for consecutive frame alignment
MAX_ALIGNMENT_WEIGHT = 10000
POSITION_FACTOR      = 1
DIMENSION_FACTOR     = 0
TYPE_MISMATCH_FACTOR = 99999

# Constants for anchoring disappearances
MIN_ANCHOR_CONF      = 0.3
CONF_INC             = 0.1
CONF_DEC             = -0.1
CONF_OCC_DEC         = -0.0
CONF_OOV_DEC         = -0.0
# occlusion reasoning
OVERLAPPING_MARGIN   = 0.05
OOV_MAX              = 1280
OOV_MIN              = 0

MIN_INFERENCE_CONF   = 0.1


# In[1]:


def print_verbose(output):
    if _VERBOSE_:
        print(output)
    return


# In[4]:


# taken from https://stackoverflow.com/questions/67314897/loop-recursion-to-handle-hierarchical-structures-in-python/67330216?noredirect=1#comment119010808_67330216
def walk_ancestry(parents, node):
    while True:
        yield node
#         print(node)
        node = parents.get(node)
        if not node:
            break

def check(node):
    i = 0
    while True and i < len(p_state):
        if node == p_state[i]['id'] and p_state[i]['status'] != 'gone':
            yield node
        i += 1
    return

new_stack = [{'id':'child1','status':'vis'},{'id':'child3','status':'vis'}]
p_state = [{'id':'parent1','status':'vis'},{'id':'parent2','status':'gone'},{'id':'parent3','status':'occ'}]

parents = {
    "child1": "parent1",
    "child2": "parent2",
    "parent2": "parent3",
}
all_objects = {
    "parent1",
    "parent2",
    "parent3",
    "child1",
    "child2",
    "child3"
}
all_passing = {
    node
    for node in all_objects
    if node in new_stack or any(check(n) for n in walk_ancestry(parents, node))
}
# all_passing = [
#     check(n) 
#     for node in all_objects
#     for n in walk_ancestry(parents, node)
#     if node in new_stack or any(for check(n) in walk_ancestry(parents, node))
#     ]
print(all_passing)


# In[5]:


def exclude_obj_id(obj_id):
    return obj_id in OBJ_IDS_TO_EXCLUDE
    
def has_position(obj):
    return 'x' in obj and 'y' in obj and 'z' in obj

def has_dimension(obj):
    return 'w' in obj and 'h' in obj

def has_confidence(obj):
    return 'conf' in obj

def is_object(obj):
    # Given an obj as a dict, check if it is an object with dimensions
    return has_position(obj) and has_dimension(obj) #and hasConfidence(obj)

def filter_by_confidence(pstm, min_conf):
    subset = [i for i in pstm if i['conf'] >= min_conf]
    return subset

##########  FUNCTIONS FOR   ########
### 1. CAMERA MOVEMENT ALIGNMENT ###
####################################

def adjust_pose_from_move_action(action, obj):
    # check which direction the camera moved and update obj pose
    # 
    if all (k in obj for k in _LABELS):
        return obj
    if exclude_obj_id(obj['id']):
        return obj
    print_verbose("adjust Pose From Move for " + obj['id'] + " based on action = " + action)
    if action == '*move-left':
        obj['x'] += _HORIZ_OFFSET
    if action == '*move-right':
        obj['x'] -= _HORIZ_OFFSET
    if action == '*move-up':
        obj['y'] += _HORIZ_OFFSET
    if action == '*move-down':
        obj['y'] -= _HORIZ_OFFSET
    if action == '*move-closer':
        # all coordinates shifted further outside
        obj['x'] = (obj['x']-_PASSIVE_HORIZ_OFFSET) if obj['x'] > 0.5 else (obj['x']+_PASSIVE_HORIZ_OFFSET)
        obj['y'] = (obj['y']-_PASSIVE_HORIZ_OFFSET) if obj['y'] < 0.5 else (obj['y']+_PASSIVE_HORIZ_OFFSET)
        obj['z'] -= _DEPTH_OFFSET
        obj['w'] -= _WIDTH_OFFSET
        obj['h'] -= _HEIGHT_OFFSET
    if action == '*move-further':
        # all coordinates shifted further inside
        obj['x'] = (obj['x']-_PASSIVE_HORIZ_OFFSET) if obj['x'] < 0.5 else (obj['x']+_PASSIVE_HORIZ_OFFSET)
        obj['y'] = (obj['y']-_PASSIVE_HORIZ_OFFSET) if obj['y'] > 0.5 else (obj['y']+_PASSIVE_HORIZ_OFFSET)
        obj['z'] += _DEPTH_OFFSET
        obj['w'] += _WIDTH_OFFSET
        obj['h'] += _HEIGHT_OFFSET
    return obj
        

def adjust_poses_from_action(action, objs):
    # adjusts the pose of each object based on the camera move action
    if len(action) == 0:
        return objs
    adjusted_objs = [adjust_pose_from_move_action(action, obj) for obj in objs]
    return adjusted_objs


##########  FUNCTIONS FOR   ########
### 2. CONSECUTIVE FRAME ALIGNMENT #
####################################

def get_euclidean_distance(a, b):
    # a and b are tuples
    distance = np.linalg.norm(a - b)
    return distance
    
def get_position_dissimilarity(a, b):
    # turns a and b into a numpy array and outputs the distance
    # input: a and b are dict
    if 'z' in a and 'z' in b:
        point_a  = np.array((a['x'],a['y'],a['z']))
        point_b  = np.array((b['x'],b['y'],b['z']))
    else:
        point_a  = np.array((a['x'],a['y']))
        point_b  = np.array((b['x'],b['y']))
    dissimilarity = get_euclidean_distance(point_a, point_b)
    return dissimilarity
    
def get_absolute_difference(a, b):
    # input: a and b are floats
    difference = abs(a - b)
    return difference

def get_dimension_dissimilarity(a, b):
    # input: a and b are dict
    area_a = a['w'] * a['h']
    area_b = b['w'] * b['h']
    dissimilarity = get_absolute_difference(area_a, area_b)
    return dissimilarity
    
def compute_dissimilarity(a, b):
    # computes the dissimilarity from object positions and dimensions
    # a and b are dict
    dist      = 0
    eucl_dist = get_position_dissimilarity(a, b)
    dist      = eucl_dist * POSITION_FACTOR
    dim_dist  = get_dimension_dissimilarity(a, b)
    dist     += dim_dist * DIMENSION_FACTOR
    if a['type'] != b['type']:
        dist *= TYPE_MISMATCH_FACTOR
    return dist

def make_cost_matrix(rows, cols):
    # input: set_a and set_b are lists of dicts that represent objects
    print("Make cost matrix")
#     print(rows)
#     print(cols)
    cost_matrix = []
    for r in rows:
        cost_column = []
        for c in cols:
            dissimilarity = compute_dissimilarity(r, c)
            cost_column += [dissimilarity]
        cost_matrix += [cost_column]
    return cost_matrix

def print_alignment(cost_matrix, indexes, set_a, set_b):
    # indexes contain the alignments
    # set_a and _b contain the object dicts
    total = 0
    for row, column in indexes:
        value = cost_matrix[row][column]
        total += value
        row_id = set_a[row]['id']
        col_id = set_b[column]['id']
        if _VERBOSE_:
            print(f'({row_id}, {col_id}) \t -> {value}')
    print_verbose(f'Total dissimilarity = {total}')
    return 0

def find_optimal_frame_alignment(set_a, set_b, max_weight=MAX_ALIGNMENT_WEIGHT):
    # Finds best alignment betw. two sets of objects
    # only accept matches less than a max_weight
    # input: two arrays of dicts
    cost_matrix = make_cost_matrix(set_a, set_b)
    m           = Munkres() 
    indexes     = m.compute(cost_matrix)
    total       = 0
    assignments = []
    for row, column in indexes:
        value   = cost_matrix[row][column]
        if value <= max_weight:
            total  += value
            row_id  = set_a[row]['id']
            col_id  = set_b[column]['id']
            assignments.append((row_id, col_id))
            print_verbose(f'({row_id}, {col_id}) \t -> {value}')
    print_verbose(f'Total dissimilarity = {total}')
    return assignments

def find_obj_by_id(uid, objs):
    obj = next((item for item in objs if item['id'] == uid), None)
    return obj

def get_max_dissimilarity(assignments, set_a, set_b):
    # given id assignments betw. two sets returns the one with the highest dissimilarity
    # input: assignments is a list of id tuples
    max_value = max([compute_dissimilarity(find_obj_by_id(a,set_a),                                           find_obj_by_id(b,set_b)) for a,b in assignments])
    return max_value

def get_switched_alignments(assignments):
    # input: list of assignment tuples
    switched = [(a,b) for (a,b) in assignments if a != b]
    return switched

def get_ids_from_set(objs):
    # input: list of dicts of objs
    ids = [a['id'] for a in objs]
    return ids

def get_unassigned_objects(assignments, set_a, set_b):
    # returns a list of ids that have not been assigned for each set
    # output: [[a1, a2,..], [b2, b4,..]]
    first_assign = [f[0] for f in assignments]
    a_ids        = get_ids_from_set(set_a)
    a_unassigned = [i for i in a_ids if i not in first_assign]

    sec_assign = [f[1] for f in assignments]
    b_ids        = get_ids_from_set(set_b)
    b_unassigned = [i for i in b_ids if i not in sec_assign]
    
    return [a_unassigned, b_unassigned]


##########  FUNCTIONS FOR   ##############
### 3. CREATE NEW STATE FROM ASSIGN ######
#(for visible objects and their children)#
##########################################

def update_old_obj_with_assigned(old, assigned):
    # keeps type, id, confidence of old obj but take over rest from assigned
    assigned['type'] = old['type']
    assigned['id']   = old['id']
    assigned['conf'] = old['conf']
    return assigned

def is_attachment_relation(relation):
    return len(relation) > 0 and relation[0] == 'attached'

def update_confidence(obj, rate):
    # updates confidence value and returns obj
    # if confidence <0 then return nil
    obj['conf'] += rate
    obj['conf'] = min(1, obj['conf']) # cannot exceed 1
    if obj['conf'] < 0:
        return {}
    return obj

def get_children(parent_id, cstm):
    # parent_id: id of parent to look up
    # cstm: set of concept relations
    return [child for relation, child, parent in cstm if relation == 'attached' and parent == parent_id]

def get_children_to_update(parent_id, cstm, assignments):
    # parent_id: id of parent to look up
    # cstm: set of concept relations
    # assignments: mapped assignments [(old_id, new_id),..]
    
    return [child for relation, child, parent in cstm            if relation == 'attached' and parent == parent_id and           child not in assignments]

def update_children(new_state, parent_new, parent_old, assignments, cstm, prev_state):
#     print("Update children of ")
#     print(parent_new)
    mapped_prev = [i[0] for i in assignments]
    children_ids = get_children_to_update(parent_new['id'], cstm, mapped_prev)
#     print("has children: " + str(len(children_ids)))
    for child in children_ids:
        child_obj = find_obj_by_id(child,prev_state)
        child_obj['x'] += parent_new['x']-parent_old['x']
        child_obj['y'] += parent_new['y']-parent_old['y']
        child_obj['z'] += parent_new['z']-parent_old['z']
        child_obj['anchor'] = 'attached'
        new_state += [child_obj]
#         print(new_state)
        new_state = update_children(new_state, child_obj, parent_old, assignments, cstm, prev_state)

    return new_state

def create_set_of_visible_objects(assignments, cstm, prev_state, curr_state):
    ### 3. CREATE NEW STATE from assignments (for visible objects and their children)
    # 0. include the currently held object if it has not been aligned
    # 1. Take prev name and id, update with new position values from curr-cycle
    # 2. set anchor state to visible, increase confidence
    # 3. Update children
    new_state = []
    print("create set of visible objs with assignments: ")
    print(assignments)
    for old_id, assigned_id in assignments:
        # 1. Take prev name and id, update with new position values from curr-cycle
        old_obj = find_obj_by_id(old_id, prev_state)
        assigned_obj = find_obj_by_id(assigned_id, curr_state)
        new_obj = update_old_obj_with_assigned(old_obj, assigned_obj.copy())
        # 2. set *anchor state to visible
        new_obj['anchor'] = 'visible'
        # increase confidence
        new_obj = update_confidence(new_obj,CONF_INC)
        new_state += [new_obj]
        print("Added parent: " + old_obj['id'] + " mapped to " + new_obj['id'])
#         print(new_state)
        # 3. Update children
        new_state = update_children(new_state, new_obj, old_obj, assignments, cstm, prev_state)
    return new_state
        

##########  FUNCTIONS FOR   ####################
### 4. HYPOTHESIS REASONING on DISAPPEARED objs#
#(that are out of view, occluded, attached)#####
################################################

def is_overlapping(l1_x, l1_y, r1_x, r1_y, l2_x, l2_y, r2_x, r2_y):
    # given top left and bottom right coordinates of two objs, compares if they are overlapping
    return (not (l1_x + OVERLAPPING_MARGIN) > r2_x or # if one is on left side of other
                (l2_x + OVERLAPPING_MARGIN) > r1_x or 
                (l2_y + OVERLAPPING_MARGIN) > r1_y or # if one is above the other
                (l1_y + OVERLAPPING_MARGIN) > r2_y) 

def find_intersection(obj1, obj2):
    # checks if obj1 is overlapping with obj2
    # by comparing their top left and bottom right corners
    # returns obj2 if it is intersecting, otherwise {}
    
    l1_x = obj1['x'] - obj1['w']/2
    l1_y = obj1['y'] - obj1['h']/2
    r1_x = obj1['x'] + obj1['w']/2
    r1_y = obj1['y'] + obj1['h']/2
    
    l2_x = obj2['x'] - obj2['w']/2
    l2_y = obj2['y'] - obj2['h']/2
    r2_x = obj2['x'] + obj2['w']/2
    r2_y = obj2['y'] + obj2['h']/2
    
    if is_overlapping(l1_x, l1_y, r1_x, r1_y, l2_x, l2_y, r2_x, r2_y):
        return obj2
    return {}

def find_intersecting_obj(obj, perceived_objs):
  # checks if objs in perceived_objs intersect with obj
  # this is needed because we only use the 2d camera
    intersecting_obj = [n for n in perceived_objs 
               if not(exclude_obj_id(n['id'])) #and node['id'] != obj['id']\ # same name check might not be required
               and find_intersection(obj, n)]
    if len(intersecting_obj) > 0:
        return intersecting_obj[0]
    return {}
    
def check_occlusions_outofview(obj, perceived_objs):
    # checks if obj is being occluded by an obj in perceived_objs
    # or out of view
    # returns obj or {}
    
    # ignore if its a hand or table or already exists
    if exclude_obj_id(obj['id']): 
        return []
    
    # check if out of view, i.e. if x,y out of bounds
    if (obj['x'] + obj['w']) >= OOV_MAX or (obj['x'] - obj['w']) >= OOV_MIN or       (obj['y'] + obj['h']) >= OOV_MAX or (obj['y'] - obj['h']) >= OOV_MAX:
        print_verbose('obj is out of view, keep' + obj['id'] + ' ' + obj['x'] + ' ' + obj['y'])
        obj['anchor'] = 'outofview'
        obj = update_confidence(obj, CONF_OOV_DEC)
    
    # check if obj is occluded, i.e. if something intersects with it
    occ = find_intersecting_obj(obj, perceived_objs)
    if occ:
        obj['anchor'] = 'occluded'
        obj = update_confidence(obj, CONF_OCC_DEC)
        print_verbose('obj is occluded' + obj['id'] + ' ' + obj['x'] + ' ' + obj['y'])
    else:
        obj = update_confidence(obj, CONF_DEC)
        # TODO: if obj['conf'] < 0: remove_all (attach _ obj) in cstm, return nil
    return obj

def anchor_disappeared_obj(obj, perceived_objs, maintained_objs, relations):
    # obj: dict
    
    if obj['id'] not in get_ids_from_set(maintained_objs):
        parent_id = has_parent(obj['id'], relations)
        if len(parent_id) > 0 and parent_id in get_ids_from_set(prev_objs):
            parent = find_obj_by_id(parent, prev_objs)
            parent = anchor_disappeared_obj(parent, perceived_objs, maintained_objs, relations)
            
        if len(parent) == 0:
            return {}
        
        if obj['conf'] < MIN_ANCHOR_CONF:
            obj = update_confidence(obj, CONF_DEC)
        else:
            obj = check_occlusions_outofview(obj, perceived_objs, relations)
        maintained_objs += obj
        return obj
    else:
        return {}
    
#     if obj['id'] not in maintained_objs_ids:
#             if obj['conf'] < MIN_ANCHOR_CONF:
#                 obj = update_confidence(obj, CONF_DEC)
#             else:
#                 parent_id = has_parent(obj['id'], relations)
#                 if len(parent_id) > 0 and parent_id in get_ids_from_set(prev_objs):
#                     parent = find_obj_by_id(parent, prev_objs)
#                     maintained_objs = anchor_disappeared_obj(parent, perceived_objs, maintained_objs, relations)
#                 obj = check_occlusions_outofview(obj, perceived_objs, relations)
#                 maintained_objs += obj
#     return maintained_objs


def check(node):
    return check_occlusions_outofview(obj, perceived_objs, relations)

def get_attachment_parents(relations):
    # input: [['attached','child','parents'],['concept','args'],...]
    # returns {"child":"parent", ...}
    parents = [(r[1],r[2]) for r in relations if r[0] == 'attached']
    return dict(parents)
    
def has_parent(obj_id, relations):
    # returns the parent id if there is an (attached,obj_id,parent_id) in relations
    # obj_id: string
    # relations: list of lists of relations with variables
    # e.g. [['attached','case0','hand'],...]
    parents = [r[2] for r in relations if r[0] == 'attached' and r[1] == obj_id]
    if len(parents) > 0:
        return parent[0]
    return ''

def walk_ancestry(parents, node):
    while True:
        yield node
        node = parents.get(node)
        if not node:
            break

def get_objs_with_parents(prev_obj_ids, perceived_objs, maintained_objs, relations):
    # checks objs in prev that have not been added to newpstm
    # input: 
    #   prev_obj_ids: list of previous ids
    #   perceived_objs: newly detected percepts
    #   maintained_objs [dict, dict,..]: current world model
    #   relations: relations betw. objects
    parents = get_attachment_parents(relations)
    all_passing = {
        node
        for node in prev_obj_ids
        if node in maintained_objs or any(check(n) for n in walk_ancestry(parents, node))
    }
    return all_passing

def reason_on_disappeared_objects(prev_objs, perceived_objs, maintained_objs, relations):
    ### 4. HYPOTHESIS REASONING on DISAPPEARED objs
    # objs in prev_cycle_adjusted that are not in new_state
    # input: 
    #   prev_objs: list of previous ids
    #   perceived_objs: newly detected percepts
    #   maintained_objs [dict, dict,..]: current world model
    #   relations: relations betw. objects [('attached',obj1,obj2),...]
    
#     for obj in prev_objs:
#         maintained_objs = anchor_disappeared_obj(obj, perceived_objs, maintained_objs, relations)

#     prev_disjoint = [p for p in prev_objs if p not in get_ids_from_set(maintained_objs)]
    if obj['id'] not in maintained_objs_ids:
        if obj['conf'] < MIN_ANCHOR_CONF:
            obj = update_confidence(obj, CONF_DEC)
        else:
#                 parent_id = has_parent(obj['id'], relations)
            if len(parent_id) > 0 and parent_id in get_ids_from_set(prev_objs):
                parent = find_obj_by_id(parent, prev_objs)
                maintained_objs = anchor_disappeared_obj(parent, perceived_objs, maintained_objs, relations)
            obj = check_occlusions_outofview(obj, perceived_objs+maintained_objs, relations)
            maintained_objs += obj
    return maintained_objs


# In[6]:


##########  FUNCTIONS FOR   #######################
### 5. HYPOTHESIS REASONING on NEWLY APPEARED objs#
###################################################
def assign_new_symbol(all_names, new):
    # create unique symbol name that doesn't exist in the set yet
    # use the type and append a number
    # all_items: list of existing objects with unique symbols [{'id':..},...]
    # new: object to add to assign new symbol {'id':..,...}
    i = 1
    while True:
        unique_id = new['type'] + str(i)
        if unique_id not in all_names:
            print("--- unique id : " + unique_id)
            new['id'] = unique_id
            return new
        i += 1
    
def reason_on_appeared_objects(assignments, new_state, curr_cycle):
    # add newly detected objects that have not been assigned to new_state
    # assigns a new unique symbol name if it already exists
    curr_cycle_assigned = [i[1] for i in assignments]
    curr_cycle_to_add = [c for c in curr_cycle if c['id'] not in curr_cycle_assigned]
    all_names = [i['id'] for i in new_state]
    new_objs = []
    for c in curr_cycle_to_add:
        new_obj = assign_new_symbol(all_names, c)
        all_names += [new_obj['id']]
        print(all_names)
        new_objs += [new_obj]
    return new_objs


# In[7]:



def new_anchor(prev_cycle, curr_cycle, prev_actions, cstm):
    if len(prev_cycle) == 0:
        return curr_cycle
    other_prev_concepts = [i for i in prev_cycle if not(is_object(i))] # those that are not objects, e.g. concepts with only 1 element
    prev_cycle          = [i for i in prev_cycle if is_object(i)]
#     newpstm             = prev_cycle
    
    ### 1. CAMERA MOVEMENT ALIGNMENT ###
#     newpstm =  [i for i in prev_cycle if isObject(i)] # not sure if needed
    prev_cycle_adjusted = adjust_poses_from_action(prev_actions[0], prev_cycle)
    print_verbose(f'Adjusted prev_cycle: {prev_cycle_adjusted}')
    
    other_curr_concepts = [i for i in curr_cycle if not(is_object(i))] # those that are not objs
    curr_cycle = [i for i in curr_cycle if is_object(i)]
    
    ### 2. CONSECUTIVE FRAME ALIGNMENT btw prev_cycle_adjusted and curr-cycle
    # Use Munkres algorithm to find best alignment
    print_verbose(f'prev: {prev_cycle_adjusted}')
    print_verbose(f'curr: {curr-cycle}')
    assignments = find_optimal_frame_alignment(prev_cycle_adjusted, curr_cycle) # only accept matches with <MAX_ALIGNMENT_WEIGHT
    print_verbose(f'show alignment results: {assignments}')
    print_verbose(f'switched assignments: {get_switched_assignments(assignments)}')
    print_verbose(f'max dissimilarity: {get_max_dissimilarity(assignments, prev, curr)}')
    
    ### 3. CREATE NEW STATE from assignments (for visible objects and their children)
    new_state = create_set_of_visible_objects(assignments, cstm, prev_cycle_adjusted, curr_cycle)
        
    ### 4. HYPOTHESIS REASONING on DISAPPEARED objs
    new_state = reason_on_disappeared_objects(prev_actions, prev_cycle_adjusted, curr_cycle, new_state, cstm)
    
    ### 5. Anchor APPEARANCES: Add newly detected objects that have not been assigned
    new_state += reason_on_appeared_objects(assignments, new_state, curr_cycle)
            
    return new_state
    
def anchor_main(pstm, cycle, actions, prev_state, cstm):
    # pstm = new percepts [[]]
    # cycle = cycle number
    # actions = list of executed actions
    # prev_state = previous world state
    # cstm = concept short term memory
    pstm += FIXED_POSITIONS
    print_verbose("New pstm = " + pstm)
    print_verbose("Prev action = " + actions[0])
    return new_anchor(prev_state, pstm, actions, cstm)

def publish_pstm(conf_pstm, sim, target_path, cycle, action):
    return


# In[7]:



def run_aux(n):
# runs cycles similar to ICARUS but without inference
    if not(isinstance(3,int)):
        print("Error: cannot call RUN-AUX with a non-number!")
        return
    cycle = 0
    dp = DataProcessor()
    while cycle <= n:
        print("---------------------------------------------------------")
        print("Cycle " + cycle)
        print("---------------------------------------------------------")

        pstm = dp.preattend(cycle)
        cstm = dp.read_relations_from_csv(cycle)
        pstm = anchor_main(pstm, cycle, prev_actions, prev_state, cstm)
        
        conf_pstm    = filter_by_confidence(pstm)
        conf_pstm    = pstm if len(conf_pstm) == 0 else conf_pstm
        publish_pstm(conf_pstm, False, "", cycle, prev_actions[0])
        prev_state   = pstm
        prev_actions = []
    return


# In[ ]:


# CODE IN CLOJURE
(def prev-cycle '((hand left *status open *conf 1 *psi -5.7273307997783736E-5 *phi 1.8194561736627286E-4 *y 0.08627867435284144 *theta -1.059244267873855E-4 *agent me *x 0.0707034542503848 *height 0.01 *toolangle 0.5623419496320909 *width 0.01 *z 0.5041641487148527) (table table1 *conf 1 *y 0 *x 0 *height 10 *width 10 *z 0) (plug oil_plug0 *conf 1 *area 0.009928 *y 0.91 *x 0.6135 *height 0.136 *width 0.073 *z 0.218129) (case case1 *conf 1 *area 0.09565799999999999 *y 0.55 *x 0.9445 *height 0.894 *width 0.107 *z 0.347046) (output output_subassembly0 *conf 1 *area 0.025172 *y 0.1015 *x 0.071 *height 0.203 *width 0.124 *z 0.40865) (case case0 *conf 1 *area 0.178362 *y 0.489 *x 0.8765 *height 0.734 *width 0.243 *z 0.490089))
(def curr-cycle '((hand left *status open *conf 1 *psi 1.6821524339148894E-4 *phi 1.7033738111682756E-4 *y 0.08634850490297065 *theta -5.310337282732053E-4 *agent me *x 0.07056213673888918 *height 0.01 *toolangle 0.5623380849438329 *width 0.01 *z 0.504316614569876) (table table1 *conf 1 *y 0 *x 0 *height 10 *width 10 *z 0) (small_hub_cover oil_plug0 *conf 1 *area 0.00858 *y 0.907 *x 0.619 *height 0.13 *width 0.066 *z 0.211993) (case case1 *conf 1 *area 0.09565799999999999 *y 0.55 *x 0.9445 *height 0.894 *width 0.107 *z 0.347046) (output output_subassembly0 *conf 1 *area 0.025172 *y 0.1015 *x 0.071 *height 0.203 *width 0.124 *z 0.40865) (case case0 *conf 1 *area 0.178362 *y 0.489 *x 0.8765 *height 0.734 *width 0.243 *z 0.490089))

(def prev-cstm '((hand left *status open *conf 1 *psi -5.7273307997783736E-5 *phi 1.8194561736627286E-4 *y 0.08627867435284144 *theta -1.059244267873855E-4 *agent me *x 0.0707034542503848 *height 0.01 *toolangle 0.5623419496320909 *width 0.01 *z 0.5041641487148527) (table table1 *conf 1 *y 0 *x 0 *height 10 *width 10 *z 0) (plug oil_plug0 *conf 1 *area 0.009928 *y 0.91 *x 0.6135 *height 0.136 *width 0.073 *z 0.218129) (case case1 *conf 1 *area 0.09565799999999999 *y 0.55 *x 0.9445 *height 0.894 *width 0.107 *z 0.347046) (output output_subassembly0 *conf 1 *area 0.025172 *y 0.1015 *x 0.071 *height 0.203 *width 0.124 *z 0.40865) (case case0 *conf 1 *area 0.178362 *y 0.489 *x 0.8765 *height 0.734 *width 0.243 *z 0.490089) (agent me *type ur5) (object output_subassembly0 *x 0.071 *y 0.1015 *z 0 *phi 0 *theta 0 *psi 0 *type output *area 0.025172 *coupling nil) (object case1 *x 0.9445 *y 0.55 *z 0 *phi 0 *theta 0 *psi 0 *type case *area 0.09565799999999999 *coupling nil) (object case0 *x 0.8765 *y 0.489 *z 0 *phi 0 *theta 0 *psi 0 *type case *area 0.178362 *coupling nil) (object oil_plug0 *x 0.6135 *y 0.91 *z 0 *phi 0 *theta 0 *psi 0 *type plug *area 0.009928 *coupling nil) (object-on-left left output_subassembly0 *diff 0.47900000000000004) (object-on-right left case1 *diff 0.2945) (object-on-right left case0 *diff 0.22649999999999992) (object-above left output_subassembly0 *diff 0.36849999999999994) (object-below left oil_plug0 *diff 0.24) (object-centered-horizontally left oil_plug0) (object-centered-vertically left case0) (object-centered-vertically left case1) (object-within-servo-range left oil_plug0) (hand-empty left)))
(def curr-cstm '((hand left *status open *conf 1 *psi 1.6821524339148894E-4 *phi 1.7033738111682756E-4 *y 0.08634850490297065 *theta -5.310337282732053E-4 *agent me *x 0.07056213673888918 *height 0.01 *toolangle 0.5623380849438329 *width 0.01 *z 0.504316614569876) (table table1 *conf 1 *y 0 *x 0 *height 10 *width 10 *z 0) (small_hub_cover oil_plug0 *conf 1 *area 0.00858 *y 0.907 *x 0.619 *height 0.13 *width 0.066 *z 0.211993) (case case1 *conf 1 *area 0.09565799999999999 *y 0.55 *x 0.9445 *height 0.894 *width 0.107 *z 0.347046) (output output_subassembly0 *conf 1 *area 0.025172 *y 0.1015 *x 0.071 *height 0.203 *width 0.124 *z 0.40865) (case case0 *conf 1 *area 0.178362 *y 0.489 *x 0.8765 *height 0.734 *width 0.243 *z 0.490089) (agent me *type ur5) (object output_subassembly0 *x 0.071 *y 0.1015 *z 0 *phi 0 *theta 0 *psi 0 *type output *area 0.025172 *coupling nil) (object case1 *x 0.9445 *y 0.55 *z 0 *phi 0 *theta 0 *psi 0 *type case *area 0.09565799999999999 *coupling nil) (object case0 *x 0.8765 *y 0.489 *z 0 *phi 0 *theta 0 *psi 0 *type case *area 0.178362 *coupling nil) (object-on-left left output_subassembly0 *diff 0.47900000000000004) (object-on-right left case1 *diff 0.2945) (object-on-right left case0 *diff 0.22649999999999992) (object-above left output_subassembly0 *diff 0.36849999999999994) (object-centered-vertically left case0) (object-centered-vertically left case1) (hand-empty left)))


# In[ ]:


# prev cycle relations
object-on-left,left,output_subassembly0
object-on-right,left,case1
object-on-right,left,case0
object-above,left,output_subassembly0
object-below,left,oil_plug0
object-centered-horizontally,left,oil_plug0
object-centered-vertically,left,case0
object-centered-vertically,left,case1
object-within-servo-range,left,oil_plug0
hand-empty,left

# curr cycle relations
object-on-left,left,output_subassembly0
object-on-right,left,case1
object-on-right,left,case0
object-above,left,output_subassembly0
object-centered-vertically,left,case0
object-centered-vertically,left,case1
hand-empty,left
attached,output_subassembly0,case0


# In[ ]:


#prev cycle
#objType,objId,x,y,z,w,h
hand,hand2,752,214.5,0,124,147
input_subassembly,input_subassembly1,276.5,533,0,79,214
casing_bolt,casing_bolt2,1157.5,652.5,0,159,51
output_subassembly,output_subassembly1,112.5,524.5,0,205,211
casing_bolt,casing_bolt1,1204,678.5,0,134,55
casing_base,casing_base1,682.5,368.5,0,231,175
hand,hand3,531,238,0,136,102
breather_plug,breather_plug2,651.5,433,0,33,32
casing_base,casing_base2,187.5,369.5,0,245,163
output_hub_cover,output_hub_cover2,181,355.5,0,158,59
transfer_subassembly,transfer_subassembly1,400,584.5,0,122,133
casing_nut,casing_nut1,769.5,628.5,0,55,35
breather_plug,breather_plug1,781.5,381.5,0,19,27
small_hub_cover,small_hub_cover3,170.5,311,0,91,32
input_hub_cover,input_hub_cover1,228.5,313.5,0,77,43

#curr cycle
breather_plug,breather_plug1,651.5,433,0,33,32
breather_plug,breather_plug2,782,380.5,0,20,27
casing_base,casing_base1,682,368.5,0,232,175
casing_base,casing_base2,187,369.5,0,246,163
output_hub_cover,output_hub_cover1,181,355,0,158,60
input_hub_cover,input_hub_cover1,232,316,0,66,48
casing_bolt,casing_bolt1,1157.5,653,0,159,52
casing_bolt,casing_bolt2,1203.5,678.5,0,133,55
casing_nut,casing_nut1,770,628.5,0,54,35
input_subassembly,input_subassembly1,276.5,533.5,0,79,215
output_subassembly,output_subassembly1,112.5,524.5,0,205,211
transfer_subassembly,transfer_subassembly1,400,584.5,0,122,133
hand,hand1,551.5,235.5,0,119,99
hand,hand2,727.5,215,0,105,120


# In[ ]:




