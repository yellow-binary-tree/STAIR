# program to run on scene graphs
import math
import json
import pickle
import copy
from functools import partial
from tkinter import Frame

from nltk.corpus import stopwords
stopwords = set(stopwords.words('english'))

parse_nary_1 = ['Array1', 'HasItem', 'OnlyItem', 'Localizenew']
parse_nary_2 = ['Array2', 'AND', 'XOR', 'And', 'Xor', 'Compare', 'Equals', 'Exists', 'Filter', 'Iterate',
                'Localize', 'ToAction', 'Query', 'Subtract', 'Temporal']
parse_nary_3 = ['Array3', 'Superlative', 'Choose']
parse_nary_4 = ['IterateUntil']
parse_nary_mappings = {**{name: 1 for name in parse_nary_1}, **{name: 2 for name in parse_nary_2},
                       **{name: 3 for name in parse_nary_3}, **{name: 4 for name in parse_nary_4}}

nary_1 = copy.copy(parse_nary_1)
nary_2 = copy.copy(parse_nary_2)
nary_3 = copy.copy(parse_nary_3)
nary_4 = copy.copy(parse_nary_4)
nary_2.remove('Localize')
nary_1 += ['Localize']
nary_mappings = {**{name: 1 for name in nary_1}, **{name: 2 for name in nary_2},
                 **{name: 3 for name in nary_3}, **{name: 4 for name in nary_4}}


KEYWORD_NAMES = {'forward', 'backward', 'while', 'temporal tag', 'between', 'before', 'after',
                 'max', 'min', 'start', 'end', 'video', 'frame', 'relations', 'objects', 'class', 'actions'}

ALL_KWS = KEYWORD_NAMES | set(nary_mappings.keys())


def parse_program(string):
    string = string.replace(', ', ';')
    string = string.replace(' ', '_')
    string = string.replace('(', ';')
    string = string.replace(')', '')
    string = string.replace('[', '[;')
    string = string.replace(']', ';]')
    program_list = string.split(';')
    left_sq_brac_idx = []

    i = -1
    while i + 1 < len(program_list):
        i += 1
        program = program_list[i]
        if program == '[': left_sq_brac_idx.append(i)
        if program == ']':
            left_idx = left_sq_brac_idx.pop()
            params_to_minus = [parse_nary_mappings[prog] if prog in parse_nary_mappings else 1 if prog == ']' else 0 \
                               for prog in program_list[left_idx: i]]
            program_list[left_idx] = 'Array%d' % (i - left_idx - 1 - sum(params_to_minus))
            del program_list[i]
            i -= 1

    common_list = copy.copy(program_list)
    program_list = [[prog, i] for i, prog in enumerate(program_list)]

    i = -1
    while i + 1 < len(program_list):
        i += 1
        if program_list[i][0] == 'XOR':
            program_list[i][0] = 'Xor'
            continue
        if program_list[i][0] == 'AND':
            program_list[i][0] = 'And'
            continue
        if program_list[i][0] == 'relation':
            program_list[i][0] = 'relations'
            continue
        if program_list[i][0] == 'Localize':
            # decouple temporal reasoning from find actions.
            # Localize mode action -> Temporal mode Localize action
            action_prog_idx = program_list[i+1][1]
            program_list[i+1][1] = None
            program_list[i][0] = 'Temporal'
            program_list.insert(i + 2, ['Localize', action_prog_idx])
            i += 2
            continue
    return [prog[0] for prog in program_list], [prog[1] for prog in program_list]


def check_params_of_module(program_list, vid=None):
    stack = list()
    res = {program: list() for program in nary_mappings}
    for program in reversed(program_list):
        if program not in nary_mappings:
            stack.append(program)
        else:
            params = []
            for _ in range(nary_mappings[program]): 
                if len(stack) == 0:
                    print(program_list, program, vid)
                param = stack.pop()
                params.append(param if param in ALL_KWS else 'string')
            res[program].append(tuple(params))
            stack.append(program)
    return res


class FrameInterval:
    def __init__(self, start, end):
        if not isinstance(start, int):
            start = int(start)
        if not isinstance(end, int):
            end = int(end)

        if start < end:
            self.start, self.end = start, end
        else:
            self.start, self.end = end, start

    def has_frame(self, frame):
        if not isinstance(frame, int):
            frame = int(frame)
        return frame >= self.start and frame <= self.end

    def length(self):
        return self.end - self.start

    def __str__(self):
        return '(' + str(self.start) + ', ' + str(self.end) + ')'

    def __repr__(self):
        return 'FrameInterval' + self.__str__()


class SceneGraphExecuter:
    def __init__(self, sg, id2word_filename, word2id_filename, output_fps=3):
        '''
        sg: can be a scene graph dict, a filename, or a list of filenames.
        '''
        if isinstance(sg, str):
            self.sg = pickle.load(open(sg, 'rb'))
        elif isinstance(sg, list):
            self.sg = dict()
            for sg_fname in sg:
                new_scene_grapg_dict = pickle.load(open(sg_fname, 'rb'))
                for key in new_scene_grapg_dict:
                    self.sg[key] = new_scene_grapg_dict[key]
        else:
            self.sg = sg
        print('finish loading scene graph, len=%d' % len(self.sg))

        # calculate the frame rates of each video
        print('calculating frame rate for each video')
        self.frame_rates = dict()
        for video_id in self.sg:
            scene_graph = self.sg[video_id]
            frame_rates = list()
            frame_keys = [key for key in scene_graph if key.startswith('0')]
            for frame_key in frame_keys:
                frame_rates.append(int(frame_key) / scene_graph[frame_key]['secs'])
            self.frame_rates[video_id] = sum(frame_rates) / len(frame_rates)

        self.prog_to_func = {
            'And': self.func_and,
            'Choose': self.func_choose,
            'Compare': self.func_compare,
            'Equals': self.func_equals,
            'Exists': self.func_exists,
            'Filter': self.func_filter,
            'HasItem': self.func_hasitem,
            'Iterate': self.func_iterate,
            'IterateUntil': self.func_iterateuntil,
            'Localize': self.func_localize,
            'Temporal': self.func_temporal,
            'OnlyItem': self.func_onlyitem,
            'ToAction': self.func_toaction,
            'Query': self.func_query,
            'Subtract': self.func_subtract,
            'Superlative': self.func_superlative,
            'Xor': self.func_xor,
            'Array1': self.func_array,
            'Array2': self.func_array,
            'Array3': self.func_array
        }
        self.id2word = json.load(open(id2word_filename))
        self.id2word = {key: value.replace('_', ' ') for key, value in self.id2word.items()}
        self.word2id = json.load(open(word2id_filename))
        self.word2id = {key.replace('_', ' '): value for key, value in self.word2id.items()}
        self.output_fps = output_fps

    def __call__(self, program=None, video_id=None, program_list=None, program_idxs=None, frame_idxs=None, frame_idx_mapping=None, question=None, sg_grounding_span=None, debug=False):
        '''
        frame_idx: idx list of FilterFrame and ExistsFrame in NMN
        frame_idx_mapping: idx mappings from ExistsFrame to FilterFrame (two NMN modules)
        returns: answer of the question
        '''
        self.scene_graph = self.sg[video_id]
        self.nodes = {
            'frames': sorted([i for i in self.scene_graph if i.startswith('0')], key=lambda x: x[-6:]),        # all key frames
            'actions': [i for i in self.scene_graph if i.startswith('c')],
            'objects': sorted([i for i in self.scene_graph if i.startswith('o')], key=lambda x: x[-6:]),
            'relations': sorted([i for i in self.scene_graph if i.startswith('r') or i.startswith('v')], key=lambda x: x[-6:])
        }
        self.action_ids_set = set([a.split('/')[0] for a in self.nodes['actions']])
        self.object_ids_set = set([a.split('/')[0] for a in self.nodes['objects']])
        self.relation_ids_set = set([a.split('/')[0] for a in self.nodes['relations']])
        # self.sec_to_frame_dict = {self.scene_graph[frame]['secs']: frame for frame in self.nodes['frames']}

        if program_list is None or program_idxs is None:
            program_list, program_idxs = parse_program(program)
        if debug:
            print('program_list:', program_list)

        if question is not None and sg_grounding_span is not None:
            sg_grounding = {}
            for span, grouding in sg_grounding_span.items():
                start, end = map(int, span.split('-'))
                sg_grounding[question[start: end]] = grouding

        stack = []
        res_by_step = dict()
        for prog, idx in zip(reversed(program_list), reversed(program_idxs)):
            if debug: print(prog, end='\t')
            if prog in nary_mappings:
                params = []
                for _ in range(nary_mappings[prog]):
                    params.append(stack.pop())
                res = self.prog_to_func[prog](*params)
                stack.append(res)

                # add intermediate reesults to res_by_step
                if idx is not None:
                    if frame_idxs is not None and \
                        prog == 'Filter' and idx in frame_idxs:       # this corresponds to a FilterFrame module in NMN
                        query = params[1]
                        res = self.func_filterframe(query)
                        res_by_step[idx] = dict()
                        for key, val in res.items():
                            res_by_step[idx][key] = self.frame_interval_change_fps(val, self.frame_rates[video_id], self.output_fps, return_tuple=True)
                    elif frame_idxs is not None and frame_idx_mapping is not None and \
                        prog == 'Exists' and idx in frame_idx_mapping:
                        query = params[0]
                        filterframe_res = res_by_step[frame_idx_mapping[idx]]
                        res_by_step[idx] = self.func_existsframe(query, filterframe_res)
                        # as the frame rate of filterframe_res is already converted, no need to do it here. just add the results in res_by_step.
                    else:
                        if isinstance(res, FrameInterval):
                            res = self.frame_interval_change_fps(res, self.frame_rates[video_id], self.output_fps, return_tuple=True)
                        if isinstance(res, tuple) and isinstance(res[0], FrameInterval):
                            res = [self.frame_interval_change_fps(r, self.frame_rates[video_id], self.output_fps, return_tuple=True) for r in res]
                        res_by_step[idx] = res
            else:
                stack.append(prog.replace('_', ' '))
            if debug: self.print_stack(stack)

        assert len(stack) == 1
        res = 'yes' if stack[0] == True else 'no' if stack[0] == False else stack[0]
        video_metadata = {'frame_rate': self.frame_rates[video_id]}
        return res, res_by_step, video_metadata

    def print_stack(self, stack):
        for item in stack:
            if isinstance(item, (str, FrameInterval, partial)):
                print(item, end=';')
            elif isinstance(item, dict):
                print('NODE', end=';')
            elif isinstance(item, list):
            #     print('LIST-%d' % len(item), end=';')
                print(item, end=';')
            # elif callable(item):
            #     print('FUNCTION', end=';')
            elif isinstance(item, tuple) and isinstance(item[0], FrameInterval):
                print(item, end=';')
            else:
                print(type(item), end=';')
        print()

    def func_array(self, *params):
        return tuple(params)

    def func_and(self, bool1, bool2):
        return bool1 and bool2

    def func_xor(self, bool1, bool2):
        if callable(bool1) and callable(bool2):
            return partial(self.func_xor_, func1=bool1, func2=bool2)
        if callable(bool1):
            return partial(self.func_xor, bool2=bool2)
        if callable(bool2):
            return partial(self.func_xor, bool2=bool1)
        return (bool1 and not bool2) or (not bool1 and bool2)

    def func_xor_(self, item, func1, func2):
        bool1 = func1(item)
        bool2 = func2(item)
        return (bool1 and not bool2) or (not bool1 and bool2)

    def func_choose(self, item1, item2, items):
        if item1 in items:
            return item1
        return item2

    def func_compare(self, items, func):
        for item in items:
            if func(item):
                return item

    def func_equals(self, item1, item2):
        return item1 == item2

    def func_exists(self, item, items):
        if callable(items):
            return partial(self.func_exists_, item=item, items_func=items)
        else:
            return item in items

    def func_exists_(self, p, items_func, item):
        items = items_func(p)
        return item in items

    def func_localize(self, action):
        if isinstance(action, tuple):        # 2 actions
            action_a, action_b = action
            action_a_id, action_b_id = self.word2id[action_a], self.word2id[action_b]
            for i, name in enumerate(self.nodes['actions']):
                if self.scene_graph[name]['charades'] == action_a_id:
                    start_frame_a, end_frame_a = self.scene_graph[name]['all_f'][0], self.scene_graph[name]['all_f'][-1]
                if self.scene_graph[name]['charades'] == action_b_id:
                    start_frame_b, end_frame_b = self.scene_graph[name]['all_f'][0], self.scene_graph[name]['all_f'][-1]
            return (FrameInterval(start_frame_a, end_frame_a), FrameInterval(start_frame_b, end_frame_b))
        else:       # 1 action
            action_id = self.word2id[action]
            for i, name in enumerate(self.nodes['actions']):
                if self.scene_graph[name]['charades'] == action_id:
                    start_frame, end_frame = self.scene_graph[name]['all_f'][0], self.scene_graph[name]['all_f'][-1]
                    return (FrameInterval(start_frame, end_frame), )

    def func_temporal(self, mode, intervals):
        if mode == 'temporal tag':
            return partial(self.func_temporal, intervals=intervals)

        if mode == 'between':
            start_a, end_a = intervals[0].start, intervals[0].end
            start_b, end_b = intervals[1].start, intervals[1].end
            if end_a <= start_b:
                return FrameInterval(end_a+1, start_b-1)
            else:
                return FrameInterval(end_b+1, start_a-1)
        if mode == 'before':
            return FrameInterval(0, intervals[0].start-1)
        if mode == 'after':
            return FrameInterval(intervals[0].end+1, 999999)
        if mode == 'while':
            return intervals[0]

    def func_filter(self, mode, query):
        # returns node
        if mode == 'frame':      # returns a partial func with the given param
            return partial(self.func_filter_, query=query)

        # return all nodes
        if len(query) == 1:
            # if query[0] in ['objects', 'relations']:
            #     return [self.id2word[self.scene_graph[i]['class']] for i in self.nodes[query[0]]]
            # else:       # actions
            #     return [self.id2word[self.scene_graph[i]['phrase']] for i in self.nodes[query[0]]]
            return [self.scene_graph[i] for i in self.nodes[query[0]]]


        # return text of nodes by condition
        assert query[0] == 'actions'
        res = [self.scene_graph[i]['phrase'] for i in self.nodes['actions'] if self.scene_graph[i]['phrase'] == query[1]]
        return list(set(res))

    def func_filter_(self, frame, query):
        # frame: str, len(str)=6
        # returns: nodes
        ret = []
        if len(query) == 1:
            if query[0] in ['objects', 'relations']:
                for i in self.nodes[query[0]]:
                    if i.endswith(frame):
                        node = self.scene_graph[i]
                        ret.append(self.id2word[node['class']])
            else:       # actions
                for i in self.nodes[query[0]]:
                    node = self.scene_graph[i]
                    # start_frame, end_frame = self.sec_to_frame(node['start'], node['end'])
                    start_frame, end_frame = node['all_f'][0], node['all_f'][-1]
                    if start_frame <= frame and end_frame >= frame:
                        ret.append(node['phrase'])

        else:       # query object by relation
            assert len(query) == 3
            assert query[0] == 'relations'
            assert query[2] == 'objects'
            query_id = self.word2id[query[1]]
            for i in self.nodes['relations']:
                if i.endswith(frame) and i.split('/')[0] == query_id:
                    for obj in self.scene_graph[i]['objects']:
                        ret.append(self.id2word[obj['class']])

        # print('FILTER_ debug:', frame, query, list(set(ret)))
        return list(set(ret))

    def func_iterate(self, items, func):
        ret = []
        if callable(items):
            return partial(self.func_iterate_, items_func=items, func=func)
        if items == 'video':
            items = FrameInterval(self.nodes['frames'][0], self.nodes['frames'][-1])
        for frame in self.nodes['frames']:
            if items.has_frame(frame):
                ret.extend(func(frame))
        return list(set(ret))

    def func_iterate_(self, p, items_func, func):
        items = items_func(p)
        return self.func_iterate(items, func)

    def func_hasitem(self, items):
        if callable(items):
            return self.func_hasitem
        else:
            return len(items) > 0

    def func_onlyitem(self, items):
        # if len(set(items)) != 1:
        #     print('ONLY ITEM LIST LEN NOT 1!', items)
        return items[0]

    def func_query(self, mode, item):
        if mode == 'class':
            # return self.id2word[item['class']]
            # return self.id2word[item]
            return item
        else:
            return partial(self.func_query_, mode=mode)

    def func_query_(self, action, mode):
        for i, name in enumerate(self.nodes['actions']):
            node = self.scene_graph[name]
            if node['phrase'] == action:
                # start_sec, end_sec = node['start'], node['end']
                # start_frame, end_frame = self.sec_to_frame(start_sec, end_sec)
                start_frame, end_frame = node['all_f'][0], node['all_f'][-1]
        return start_frame if mode == 'start' else end_frame

    def func_subtract(self, func1, func2):
        return partial(self.func_subtract_, func1=func1, func2=func2)

    def func_subtract_(self, action, func1, func2):
        frame1 = func1(action)
        frame2 = func2(action)
        return FrameInterval(frame1, frame2).length()

    def func_superlative(self, mode, items, func):
        res = []
        items_to_iterate = []
        for item in items:
            if isinstance(item, (tuple, list)):     # item is a list of node, i.e. result of filter
                items_to_iterate.extend(item)
            elif not isinstance(item, str):     # item is a node
                items_to_iterate.append(item['phrase'])
            else:
                items_to_iterate.append(item)
        for item in items_to_iterate:
            res.append(func(item))

        if mode == 'min':
            res = [-i for i in res]
        max_num, max_idx = -math.inf, -1
        for i, r in enumerate(res):
            if r > max_num:
                max_idx, max_num = i, r
        return items_to_iterate[max_idx]

    def func_iterateuntil(self, mode, items, bool_func, func):
        # returns: nodes
        if items == 'video':
            items = FrameInterval(self.nodes['frames'][0], self.nodes['frames'][-1])
        if mode == 'forward':
            frames = self.nodes['frames']
        else:
            frames = self.nodes['frames'][::-1]
        for frame in frames:
            if items.has_frame(frame) and bool_func(frame):
                return func(frame)
        raise ValueError('no true frames found in IterateUntil!')

    def func_toaction(self, verb, obj):
        verb_id, obj_id = self.word2id.get(verb, None), self.word2id.get(obj, None)
        for name in self.nodes['actions']:
            if self.scene_graph[name]['verb_id'] == verb_id and self.scene_graph[name]['object_id'] == obj_id:
                return self.scene_graph[name]['phrase']
        raise ValueError('no actions found')

    def func_filterframe(self, query):
        # returns the results in the whole video as well as the frame interval it appears. frame interval: [first frame, last frame]
        # this function is used to provide intermediate supervision for NMN
        ret = dict()
        if len(query) == 1:
            if query[0] in ['objects', 'relations']:
                # iterate over all objects / relations, find the first & last frame of their occurrance
                node_in_frames = dict()
                for node in self.nodes[query[0]]:
                    class_id, frame = node.split('/')
                    if class_id not in node_in_frames:
                        node_in_frames[class_id] = list()
                    node_in_frames[class_id].append(int(frame))
                for class_id, frames in node_in_frames.items():
                    class_name = self.id2word[class_id]
                    frames = sorted(frames)
                    ret[class_name] = FrameInterval(frames[0], frames[-1])

            else:       # actions
                for node in self.nodes['actions']:
                    ret[self.scene_graph[node]['phrase']] = FrameInterval(self.scene_graph[node]['start'], self.scene_graph[node]['end'])

        else:       # len(query) == 3
            assert len(query) == 3
            assert query[0] == 'relations'
            assert query[2] == 'objects'
            # step 1: get all objects' ids with the relation in query
            objects_wanted = set()     # set of object ids
            query_id = self.word2id[query[1]]
            for node in self.nodes['relations']:
                relation_id, frame = node.split('/')
                if relation_id == query_id:
                    for obj in self.scene_graph[node]['objects']:
                        objects_wanted.add(obj['class'])
            # print('objects_wanted', objects_wanted)
            # step 2: get the frame interval that the objects appreared
            node_in_frames = {obj_id: list() for obj_id in objects_wanted}
            for node in self.nodes['objects']:
                class_id, frame = node.split('/')
                if class_id in objects_wanted:
                    node_in_frames[class_id].append(int(frame))
            # print('node_in_frames', node_in_frames)
            for class_id, frames in node_in_frames.items():
                class_name = self.id2word[class_id]
                frames = sorted(frames)
                ret[class_name] = FrameInterval(frames[0], frames[-1])

        return ret

    def func_existsframe(self, query, filterframe_res):
        # returns the existance interval of the class name (query)
        # print('func_existsframe debug')
        # print(query)
        # print(filterframe_res)
        for key in filterframe_res:
            if key == query:
                return filterframe_res[key]
        return None

    def frame_interval_change_fps(self, frame_interval, old_fps, new_fps, return_tuple=False):
        sf = frame_interval.start * new_fps / old_fps
        ef = frame_interval.end * new_fps / old_fps
        if return_tuple:
            return (sf, ef)
        return FrameInterval(sf, ef)

    def summary_scene_graph(self, video_id):
        '''
        prints the abstract information of the scene graph of the given video id
        '''
        scene_graph = self.sg[video_id]
        nodes = {
            'frames': sorted([i for i in scene_graph if i.startswith('0')], key=lambda x: x[-6:]),        # all key frames
            'actions': [i for i in scene_graph if i.startswith('c')],
            'objects': sorted([i for i in scene_graph if i.startswith('o')], key=lambda x: x[-6:]),
            'relations': sorted([i for i in scene_graph if i.startswith('r') or i.startswith('v')], key=lambda x: x[-6:])
        }

        for node_id in nodes['frames']:
            node = scene_graph[node_id]
            print(node_id, node['secs'])
        for node_id in nodes['actions']:
            node = scene_graph[node_id]
            print(node_id, node['phrase'], node['start'], node['end'], node['all_f'])
        for node_id in nodes['objects']:
            node = scene_graph[node_id]
            print(node_id, node['class'], self.id2word[node['class']])
        for node_id in nodes['relations']:
            node = scene_graph[node_id]
            print(node_id, node['type'], node['secs'])

if __name__ == '__main__':
    res = parse_program('XOR(Exists(food, Iterate(Localize(between, [grasping onto a doorknob, drinking from a cup]), Filter(frame, [relation, holding, objects]))), Exists(Query(class, OnlyItem(Iterate(video, Filter(frame, [relations, opening, objects])))), Iterate(Localize(between, [grasping onto a doorknob, drinking from a cup]), Filter(frame, [relation, holding, objects]))))')
    print(res)