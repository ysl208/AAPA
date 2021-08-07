#############################################
########## UNIT TESTS FOR AAPA ##############
# ------------------------------------------#
########### IMPLEMENTATION NOTES ############
# ------------------------------------------#
# AAPA was implemented in the following way #

### 1. CAMERA MOVEMENT ALIGNMENT ###
# This needs to be done first because any positions from the previous cycle might be off using
# 
# > newpstm = adjust_poses_from_action(prev_actions[0], newpstm)
#

### 2. CONSECUTIVE FRAME ALIGNMENT btw newpstm (prev-cycle) and curr-cycle
# We find the objs that are both in prev and in the new percepts, using
#
# > assignments = find_optimal_frame_alignment(newpstm, curr_cycle) 
#
# this only accept matches that have <MAX_ALIGNMENT_WEIGHT

### 3. UPDATE NEW STATE FOR VISIBLE OBJECTS
    # 0. include the currently held object if it has not been aligned
    # 1. Take prev name and id, update with new position values from curr-cycle
    # 2. set anchor state to visible, increase confidence
    # 3. Update children
        
    ### 4. Anchor DISAPPEARED objs that have not been updated
# since we want the unique ids of the prev objs to be maintained we need to do this step first
# as the ids might clash with the newly detected objs
    
    ### 5. Anchor APPEARANCES: Add newly detected objects that have not been assigned
# add new objs that have not been aligned and assign new ids if they clash with the existing ones
          


################# UNIT TESTS ##############
# DIFFERENT CASES TO TEST FUNCTIONALITIES #
###########################################

# prev = ['breather_plug1', 'breather_plug2', 'casing_base1', 'casing_base2', 'output_hub_cover1', 'input_hub_cover1', 'casing_bolt1', 'casing_bolt2', 'casing_nut1', 'input_subassembly1', 'output_subassembly1', 'transfer_subassembly1', 'hand1', 'hand2']
# curr = ['hand2', 'input_subassembly1', 'casing_bolt2', 'output_subassembly1', 'casing_bolt1', 'casing_base1', 'hand3', 'breather_plug2', 'casing_base2', 'output_hub_cover2', 'transfer_subassembly1', 'casing_nut1', 'breather_plug1', 'small_hub_cover3', 'input_hub_cover1']
cstm = [['attached', 'input_hub_cover1', 'casing_base1'],['attached', 'casing_bolt2', 'casing_base1'],
       ['attached', 'breather_plug1', 'casing_base2'],['attached', 'hand1', 'hand2']]

dp = DataProcessor()
prev = dp.preattend(1)[:]
curr = dp.preattend(0)[:]

print("------- Running unit tests with the following two sets --------")
print("PREV: ")
print([p['id'] for p in prev])
print(prev)
print("CURR: ")
print([p['id'] for p in curr])
print(curr)
assert(len(prev) == 14)
assert(len(curr) == 15)
print("....... passed")

print("--------------------------------------------------------------")
print("------- # TEST#2. CONSECUTIVE FRAME ALIGNMENT of both sets --------")
print("--------------------------------------------------------------")
al = find_optimal_frame_alignment(prev,curr,1)
print("Alignments: ")
print(al)
assert(al == [('breather_plug1', 'breather_plug2'),
#  ('breather_plug2', 'breather_plug1'),
 ('casing_base1', 'casing_base1'),
 ('casing_base2', 'casing_base2'),
 ('output_hub_cover1', 'output_hub_cover2'),
 ('casing_bolt1', 'casing_bolt2'),
 ('casing_bolt2', 'casing_bolt1'),
 ('casing_nut1', 'casing_nut1'),
 ('input_subassembly1', 'input_subassembly1'),
 ('output_subassembly1', 'output_subassembly1'),
 ('transfer_subassembly1', 'transfer_subassembly1')])
print("....... passed")

print("# TEST switched alignments --------")
switched = get_switched_alignments(al)
assert(switched == [('breather_plug1', 'breather_plug2'),
#  ('breather_plug2', 'breather_plug1'),
 ('output_hub_cover1', 'output_hub_cover2'),
 ('casing_bolt1', 'casing_bolt2'),
 ('casing_bolt2', 'casing_bolt1')])
print("....... passed")

print("# TEST max dissimilarity --------")
# assert(get_max_dissimilarity(al,prev,curr) == 1.118033988749895) # for alignment threshold 2
assert(get_max_dissimilarity(al,prev,curr) == 0.5) # for alignment threshold 1
print("....... passed")

print("# TEST unassigned objects --------")
unass = get_unassigned_objects(al, prev, curr)
print("PREV - unassigned:")
print(unass[0])
print("CURR - unassigned:")
print(unass[1])
assert(unass == [['breather_plug2','input_hub_cover1', 'hand1', 'hand2'],
 ['hand2', 'hand3', 'breather_plug1','small_hub_cover3', 'input_hub_cover1']])
print("....... passed")

print("--------------------------------------------------------------")
print("------- # TEST#3. UPDATE NEW STATE for visible assignments ---")
print("--------------------------------------------------------------")
new_state = []
# old_obj = find_obj_by_id('casing_base1', prev)
# assigned_obj = find_obj_by_id('casing_base2', curr)
old_obj = {'type': 'casing_base', 'id': 'casing_base1', 'x': 682.0, 'y': 368.5, 'z': 0.0, 'w': 232.0, 'h': 175.0, 'conf': 0.5}
assigned_obj = {'type': 'casing_base', 'id': 'casing_base2', 'x': 187.5, 'y': 369.5, 'z': 0.0, 'w': 245.0, 'h': 163.0, 'conf': 0}

print("# TEST update new obj with old type, id, conf --------")
new_obj = update_old_obj_with_assigned(old_obj, assigned_obj)
assert(new_obj == {'type': 'casing_base', 'id': 'casing_base1', 'x': 187.5, 'y': 369.5, 'z': 0.0, 'w': 245.0, 'h': 163.0, 'conf': 0.5})
print("....... passed")

print("# TEST update new obj with old type, id, conf --------")
new_obj['anchor'] = 'visible'
new_obj = update_confidence(new_obj,0.1)
# print(new_obj)
assert(new_obj == {'type': 'casing_base', 'id': 'casing_base1', 'x': 187.5, 'y': 369.5, 'z': 0.0, 'w': 245.0, 'h': 163.0, 'conf': 0.6, 'anchor': 'visible'})
print("....... passed")

print("# TEST get all children --------")
parent = 'casing_base1'
children = get_children(parent,cstm)
assert(children == ['input_hub_cover1','casing_bolt2'])
print("....... passed")

print("# TEST get children to update --------")
children = get_children_to_update(parent,cstm, [i[0] for i in al])
assert(children == ['input_hub_cover1'])
print("....... passed")

print("# TEST Create set of visible objs --------")
# new_state = update_children(new_state, new_obj, old_obj, al, cstm, prev)
new_state = create_set_of_visible_objects(al, cstm, prev, curr)
new_state_ids = [p['id'] for p in new_state]
print("NEW STATE of visible objs:")
print(new_state_ids)
assert(new_state_ids == ['breather_plug1', 'casing_base1', 'input_hub_cover1', 
                         'casing_base2', 'output_hub_cover1', 'casing_bolt1', 'casing_bolt2', 
                         'casing_nut1', 'input_subassembly1', 'output_subassembly1', 
                         'transfer_subassembly1'])
print("....... passed")


# TODO
print("--------------------------------------------------------------")
print("------- # TEST#4. ANCHOR DISAPPEARANCES from prev state --------")
print("--------------------------------------------------------------")
# objs = reason_on_disappeared_objects(prev_actions, prev, curr, new_state, cstm)
# assert('breather_plug2' in objs)
print("....... passed")

print("--------------------------------------------------------------")
print("------- # TEST#5. ANCHOR APPEARANCES from curr state --------")
print("--------------------------------------------------------------")

print("# TEST assign new symbol --------")
# new_state = update_children(new_state, new_obj, old_obj, al, cstm, prev)
unique_id = assign_new_symbol(['case0','case1'], {'id': 'case1', 'type': 'case'})
assert(unique_id == {'id': 'case2', 'type': 'case'})
print("....... passed")

print("# TEST reason on appeared objs --------")
appeared_objs = reason_on_appeared_objects(al, new_state, curr)
appeared_obj_ids = [p['id'] for p in appeared_objs]
print("* APPEARED from CURR: ")
print(appeared_obj_ids)
assert(appeared_obj_ids == ['hand1', 'hand2', 'breather_plug2', 'small_hub_cover1', 'input_hub_cover2'])
print("....... passed")

print("# TEST final AAPA output --------")
new_state += appeared_objs
print("* FINAL NEW STATE: ")
new_state_ids = [i['id'] for i in new_state]
print(new_state_ids)
assert(new_state_ids == ['breather_plug1', 'casing_base1', 'input_hub_cover1', 'casing_base2', 'output_hub_cover1', 'casing_bolt1', 'casing_bolt2', 'casing_nut1', 'input_subassembly1', 'output_subassembly1', 'transfer_subassembly1', 'hand1', 'hand2', 'breather_plug2', 'small_hub_cover1', 'input_hub_cover2'])
print("....... passed")

print("ALL PASSED")

# TODO: Cases to check
# Case1: if obj1 is OOV and occluding another obj2, keep obj2

# Case2: if obj1 is unassigned, but has parent obj2 assigned, keep obj1

# Case3: if obj1 is unassigned, but has parent obj2 unassigned, has parent obj3 which is being dropped,
# do we still check obj2 for occ/oov and potentially keep it? or do we lose all children obj2 and obj1

# Case4: when checking lost obj1 for occlusion, check if intersects with aligned obj (maintained_objs)
# as well as newly_perceived_objs


# THE TESTS ABOVE SHOULD WORK AFTER REFACTORING ----------------------# 
# THE TESTS BELOW SHOULD WORK AFTER IMPLEMENTATION OF #4 #
print("--------------------------------------------------------------")
print("------- # TEST#4. ANCHOR DISAPPEARANCES from prev state --------")
print("--------------------------------------------------------------")


print("# TEST reason on appeared objs --------")
# INPUT: NEW STATE of visible objs:
# new_state = ['breather_plug1', 'casing_base1', 'input_hub_cover1', 'casing_base2', 'output_hub_cover1', 'casing_bolt1', 'casing_bolt2', 'casing_nut1', 'input_subassembly1', 'output_subassembly1', 'transfer_subassembly1']
# cstm = [['attached', 'input_hub_cover1', 'casing_base1'],['attached', 'casing_bolt2', 'casing_base1'],
#        ['attached', 'breather_plug1', 'casing_base2'],['attached', 'hand1', 'hand2']]

disappeared_objs = reason_on_disappeared_objects(prev, curr, new_state, cstm)
disappeared_obj_ids = [p['id'] for p in disappeared_objs]
print("* APPEARED from CURR: ")
print(disappeared_obj_ids)
assert(disappeared_obj_ids == ['breather_plug2', 'hand1', 'hand2'])
print("....... passed")

print("# TEST final AAPA output --------")
new_state += disappeared_obj
print("* FINAL NEW STATE: ")
new_state_ids = [i['id'] for i in new_state]
print(new_state_ids)
assert(new_state_ids == ['breather_plug1', 'casing_base1', 'input_hub_cover1', 'casing_base2', 'output_hub_cover1', 'casing_bolt1', 'casing_bolt2', 'casing_nut1', 'input_subassembly1', 'output_subassembly1', 'transfer_subassembly1', 'breather_plug2', 'hand1', 'hand2'])
print("....... passed")

# TODO: Cases to check
# Case1: if obj1 is OOV and occluding another obj2, keep obj2

# Case2: if obj1 is unassigned, but has parent obj2 assigned, keep obj1

# Case3: if obj1 is unassigned, but has parent obj2 unassigned, has parent obj3 which is being dropped,
# do we still check obj2 for occ/oov and potentially keep it? or do we lose all children obj2 and obj1

# Case4: when checking lost obj1 for occlusion, check if intersects with aligned obj (maintained_objs)
# as well as newly_perceived_objs


print("--------------------------------------------------------------")
print("------- # TEST#5. ANCHOR APPEARANCES from curr state --------")
print("--------------------------------------------------------------")

print("# TEST assign new symbol --------")
# new_state = update_children(new_state, new_obj, old_obj, al, cstm, prev)
unique_id = assign_new_symbol(['case0','case1'], {'id': 'case1', 'type': 'case'})
assert(unique_id == {'id': 'case2', 'type': 'case'})
print("....... passed")

print("# TEST reason on appeared objs --------")
appeared_objs = reason_on_appeared_objects(al, new_state, curr)
appeared_obj_ids = [p['id'] for p in appeared_objs]
print("* APPEARED from CURR: ")
print(appeared_obj_ids)
assert(appeared_obj_ids == ['hand3', 'hand4', 'breather_plug3', 'small_hub_cover1', 'input_hub_cover2'])
print("....... passed")

print("# TEST final AAPA output --------")
new_state += appeared_objs
print("* FINAL NEW STATE: ")
new_state_ids = [i['id'] for i in new_state]
print(new_state_ids)
assert(new_state_ids == ['breather_plug1', 'casing_base1', 'input_hub_cover1', 'casing_base2', 'output_hub_cover1', 'casing_bolt1', 'casing_bolt2', 'casing_nut1', 'input_subassembly1', 'output_subassembly1', 'transfer_subassembly1', 'breather_plug2', 'hand1', 'hand2', 'hand3', 'hand4', 'breather_plug3', 'small_hub_cover1', 'input_hub_cover2'])
print("....... passed")
