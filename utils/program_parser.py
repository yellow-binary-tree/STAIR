import copy

# MODULE_NAMES = {'IterateUntil', 'Compare', 'Superlative', 'Filter', 'Query', 'Localize', 'ToAction',
#     'Equals', 'Subtract', 'HasItem', 'OnlyItem', 'Exists', 'Iterate', 'Choose', 'AND', 'XOR'}
KEYWORD_NAMES = {'forward', 'backward', 'while', 'temporal_tag', 'between', 'before', 'after',
                 'max', 'min', 'start', 'end', 'video', 'frame', 'relations', 'objects', 'class', 'actions'}

parse_nary_1 = ['Array1', 'HasItem', 'OnlyItem']
parse_nary_2 = ['Array2', 'AND', 'XOR', 'And', 'Xor', 'Compare', 'Equals', 'Exists', 'Filter', 'Iterate',
          'Localize', 'ToAction', 'Query', 'Subtract']
parse_nary_3 = ['Array3', 'Superlative', 'Choose']
parse_nary_4 = ['IterateUntil']
parse_nary_mappings = {**{name: 1 for name in parse_nary_1}, **{name: 2 for name in parse_nary_2},
                       **{name: 3 for name in parse_nary_3}, **{name: 4 for name in parse_nary_4}}

nary_1 = parse_nary_1 + ['Query']
nary_2 = parse_nary_2 + ['Relate', 'AttnVideo', 'FilterFrame', 'ExistsFrame', 'XorFrame']
nary_2.remove('Query')
nary_2.remove('Subtract')
nary_3 = parse_nary_3 + ['Temporal']
nary_4 = parse_nary_4
nary_mappings = {**{name: 1 for name in nary_1}, **{name: 2 for name in nary_2},
                 **{name: 3 for name in nary_3}, **{name: 4 for name in nary_4}}

ALL_KWS = KEYWORD_NAMES | set(nary_mappings.keys())


def parse_program(string):
    '''
    returns: program_list, more_data
    '''
    # if '[' in string:
    #     left_square_bracket_idxs = [m.start() for m in re.finditer('\[', string)]
    #     right_square_bracket_idxs = [m.start() for m in re.finditer('\]', string)]
    #     for lidx, ridx in zip(reversed(left_square_bracket_idxs), reversed(right_square_bracket_idxs)):
    #         num_params = string[lidx:ridx].count(',') + 1
    #         name = 'Array%d' % num_params
    #         string = string[:lidx] + name + string[lidx:]

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

    # we define common list at this step
    common_list = copy.deepcopy(program_list)
    program_list = [[prog, i] for i, prog in enumerate(program_list)]

    # post processing, linearly
    i = -1
    iterate_idxs = []
    while i + 1 < len(program_list):
        i += 1
        if program_list[i][0] in ['OnlyItem']:
            del program_list[i]
            i -= 1
            continue
        if program_list[i][0] == 'XOR':
            program_list[i][0] = 'Xor'
            continue
        if program_list[i][0] == 'AND':
            program_list[i][0] = 'And'
            continue
        if i < len(program_list) - 1 and program_list[i][0] == 'Query' and program_list[i+1][0] == 'class':
            # Query class Filter -> Filter
            del program_list[i:i+2]
            i -= 1
            continue
        if program_list[i][0] == 'relation':
            program_list[i][0] = 'relations'
            continue
        if program_list[i][0] == 'Subtract':   # Subtract Query end action Query start action
            del program_list[i+1:i+7]
            program_list[i] = ['video', None]
            continue
        if program_list[i][0] == 'Iterate':
            iterate_idxs.append(i)
            continue
        if program_list[i][0] == 'Localize':
            # decouple temporal reasoning from find actions.
            # Localize mode action -> Temporal mode video Localize video action
            action_prog_idx = program_list[i+1][1]
            program_list[i+1][1] = None
            program_list[i][0] = 'Temporal'
            program_list.insert(i + 2, ['video', None])
            program_list.insert(i + 2, ['Localize', action_prog_idx])
            program_list.insert(i + 2, ['video', None])
            i += 3
            continue
        if program_list[i][0] == 'Array1':
            del program_list[i]
            i -= 1
            continue
        if program_list[i][0] == 'Array3':     # Array3 relations xx objects, we only need to keep xx(a relation)
            del program_list[i+3]
            del program_list[i+1]
            del program_list[i]
            i -= 1
            continue
        if program_list[i][0] == 'Array2':
            if program_list[i+1][0] == 'actions':      # Array2 actions xx, we only need to keep xx(a action)
                del program_list[i:i+2]
                continue
        if program_list[i][0] == 'Superlative' and program_list[i+2][0] == 'Filter':
            program_list[i+2][0] = 'FilterFrame'
            continue

    # iterate module
    if iterate_idxs:
        # post processing, use tree structure
        childrens, _ = get_childrens_and_parents([prog[0] for prog in program_list])

        for idx in iterate_idxs:
            program_list[idx][0] = 'Filter'
            children_idx = childrens[idx][1]
            program_list[children_idx] = None
            program_list[children_idx + 1] = None

        ret = []
        for prog in program_list:
            if prog is not None:
                ret.append(prog)
        program_list = ret

    # handle IterateUntil module
    existsframe_filterframe_idx_mapping = None
    iterate_until_idxs = [i for i, prog in enumerate(program_list) if prog[0] == 'IterateUntil']
    if iterate_until_idxs:
        iterate_until_blocks_indices = []
        childrens, _ = get_childrens_and_parents([prog[0] for prog in program_list])
        for iterate_until_idx in iterate_until_idxs:
            iterate_until_block_indices = get_child_indice(childrens, iterate_until_idx)
            iterate_until_blocks_indices.append((min(iterate_until_block_indices), max(iterate_until_block_indices) + 1))
        iterate_until_blocks_indices.sort(key=lambda x: x[1] - x[0])     # reformat small blocks first to handle the nested IterateUntil (inner first, outer later)
        for start_idx, end_idx in iterate_until_blocks_indices:
            childrens, parents = get_childrens_and_parents([prog[0] for prog in program_list])
            program_list, existsframe_filterframe_idx_mapping = reformat_iterate_until_block(program_list, childrens, parents, start_idx, end_idx)       # FIXME: change this function for paired data in program_list.

    # handle Compare module
    if program_list[0][0] == 'Compare':    # Compare Array2 before after Exists ...
        del program_list[1:4]
        temporal_tag_idx = [prog[0] for prog in program_list].index('temporal_tag')
        ret = copy.deepcopy(program_list) + copy.deepcopy(program_list)[1:]
        ret[temporal_tag_idx][0] = 'before'
        ret[temporal_tag_idx + len(program_list) - 1][0] = 'after'
        program_list = ret

    more_data = {
        'idx_list': [prog[1] for prog in program_list],
        'existsframe_filterframe_idx_mapping': existsframe_filterframe_idx_mapping,
        'common_list': common_list
    }
    return [prog[0] for prog in program_list], more_data


def get_child_indice(childrens, idx, sort=True):
    res = [idx]
    for i in childrens[idx]:
        res.extend(get_child_indice(childrens, i, sort=False))
    if sort:
        res.sort()
    return res


def get_childrens_and_parents(program_list):
    # get children nodee of each node
    childrens, parents = [list() for _ in program_list], [0 for _ in program_list]
    stack = list()
    for i in range(len(program_list) - 1, -1, -1):
        program = program_list[i]
        if program not in nary_mappings:
            stack.append(i)
        else:
            for _ in range(nary_mappings[program]): 
                if not stack:
                    print(program_list, i)
                childrens[i].append(stack.pop())
            stack.append(i)

    for i, chs in enumerate(childrens):
        for child in chs:
            parents[child] = i
    return childrens, parents


def visualize_program_list(program_list):
    def format_indent(indent, times):
        ret = ''
        for i in range(1, times+1): ret += indent % (i%10)
        return ret

    indent = '%d   '
    string = ''
    params_left = []
    for prog in program_list:
        if prog in nary_mappings:
            string += (format_indent(indent, len(params_left))) + prog + '\n'
            if params_left:
                params_left[-1] -= 1
            params_left.append(nary_mappings[prog])
        else:
            string += (format_indent(indent, len(params_left))) + prog + '\n'
            params_left[-1] -= 1
            while params_left and params_left[-1] == 0: params_left.pop()
    return string


def reformat_iterate_until_block(program_list, childrens, parents, sidx, eidx):
    new_segment = []
    existsframe_filterframe_idx_mapping = dict()        # ExistsFrame: FilterFrame对应关系
    new_segment.extend([['Filter', program_list[sidx][1]], ['AttnVideo', None]])    # Filter是整理后这个block的顶级函数，其输出应该和原来的IterateUntil对应

    # keep the second param of IterateUntil, i.e. the video input, unchanged
    new_segment.extend(program_list[sidx + 2: sidx + 2 + len(get_child_indice(childrens, childrens[sidx][1]))])

    # turn the third param of IterateUntil, i.e. the bool func, to our new Relate function
    new_segment.extend([['Relate', None], program_list[sidx+1]])
    bool_function_indices = get_child_indice(childrens, childrens[sidx][2])
    for bfi in bool_function_indices:
        if program_list[bfi][0] == 'frame':
            new_segment.append(['video', program_list[bfi][1]])
        elif program_list[bfi][0] == 'Filter' and program_list[bfi + 1][0] == 'frame':
            if program_list[parents[bfi]][0] == 'Exists':
                new_segment[parents[bfi] - bfi][0] = 'ExistsFrame'
            new_segment.append(['FilterFrame', program_list[bfi][1]])       # this 'FitlerFrame' corresponds to 'Filter'
            existsframe_filterframe_idx_mapping[program_list[parents[bfi]][1]] = program_list[bfi][1]
        elif program_list[bfi][0] == 'Xor':
            new_segment.append(['XorFrame', program_list[bfi][1]])
        else:
            new_segment.append(program_list[bfi])

    # for the fourth param of IterateUntil, i.e. the Filter funcion, keep its second param
    sec_param_of_filter_func_indices = get_child_indice(childrens, childrens[childrens[sidx][3]][1])
    for pidx in sec_param_of_filter_func_indices:
        new_segment.append(program_list[pidx])

    # return
    if not len(new_segment) == (eidx - sidx):
        print('length of new program: %d; old program: %d - %d = %d' % (len(new_segment), eidx, sidx, (eidx - sidx)))
        print(program_list[sidx: eidx])
        print(new_segment)
        raise AssertionError

    # replace the IterateUntil block with the new block
    new_program_list = program_list[:sidx] + new_segment + program_list[eidx:]
    return new_program_list, existsframe_filterframe_idx_mapping


def check_params_of_module(program_list, vid=None):
    stack = list()
    res = {program: list() for program in nary_mappings}
    for program in reversed(program_list):
        if program not in nary_mappings:
            stack.append(program)
        else:
            params = []
            # res[program].append(tuple(reversed([i if i in ALL_KWS else 'string' for i in stack[-nary_mappings[program]:]])))
            for _ in range(nary_mappings[program]): 
                if len(stack) == 0:
                    print(program_list, program, vid)
                param = stack.pop()
                params.append(param if param in ALL_KWS else 'string')
            res[program].append(tuple(params))
            stack.append(program)
    return res


def stat_level_of_module(program_list, prog):
    # 这个负责统计给定module的level
    # leaf node = 0, root node = max
    ret_levels = list()
    stack = list()
    level = 0
    for program in reversed(program_list):
        if program not in nary_mappings:
            stack.append((program, 0))
        else:
            params = stack[-nary_mappings[program]:]
            stack = stack[:-nary_mappings[program]]
            level = max(i[1] for i in params) + 1
            stack.append((program, level))
            if program == prog:
                ret_levels.append(level)
    ret_levels_minus_1 = [i - 1 for i in ret_levels]
    ret_levels_ratio = [i / level for i in ret_levels]
    ret_levels_minus_1_ratio = [i / level for i in ret_levels_minus_1]
    return ret_levels, ret_levels_minus_1, ret_levels_ratio, ret_levels_minus_1_ratio, level        # all level of the program, max_level


def stat_module_levels(program_list):
    # 这个负责统计所有module的level
    ret_levels = list()
    stack = list()
    for program in reversed(program_list):
        if program not in nary_mappings:
            stack.append((program, 0))
            ret_levels.append(0)
        else:
            params = stack[-nary_mappings[program]:]
            stack = stack[:-nary_mappings[program]]
            level = max(i[1] for i in params) + 1
            stack.append((program, level))
            ret_levels.append(level)
    return ret_levels[::-1]


def program_is_valid(program_list):
    stack_length = 0
    for prog in reversed(program_list):
        if prog in nary_mappings:
            stack_length = stack_length - nary_mappings[prog] + 1
        else:
            stack_length += 1
        if stack_length < 0:
            return False
    return stack_length == 1


if __name__ == '__main__':
    # program_list = ['Query', 'IterateUntil', 'forward', 'Localize', 'video', 'after', 'ToAction', 'eating', 'Query', 'Filter', 'AttnVideo', 'video', 'Relate', 'backward', 'Exists', 'taking', 'Filter', 'video', 'Array1', 'relations', 'Array3', 'relations', 'taking', 'objects', 'Exists', 'putting_down', 'Filter', 'frame', 'Array1', 'relations', 'Filter', 'frame', 'Array3', 'relations', 'putting_down', 'objects']
    program_list = 'throwing standing_up video Localize video while Temporal Filter'.split(' ')[::-1]
    # print(visualize_program_list(program_list))
    print(program_is_valid(program_list))