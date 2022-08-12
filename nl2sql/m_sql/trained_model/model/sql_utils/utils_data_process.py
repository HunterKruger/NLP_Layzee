import math
import random
import re


chinese_num_map = {
    '一': 1, '二': 2, '三': 3, '四': 4, '五': 5, '六': 6,
    '七': 7, '八': 8, '九': 9, '零': 0, '〇': 0, '两': 2
}

def edit(str1, str2):
    matrix = [[i + j for j in range(len(str2) + 1)] for i in range(len(str1) + 1)]

    for i in range(1, len(str1) + 1):
        for j in range(1, len(str2) + 1):
            if str1[i - 1] == str2[j - 1]:
                d = 0
            else:
                d = 1
            matrix[i][j] = min(matrix[i - 1][j] + 1, matrix[i][j - 1] + 1, matrix[i - 1][j - 1] + d)

    return matrix[len(str1)][len(str2)]


def can_add(index, l, indexes_to_replace):
    for st, ed, _, _ in indexes_to_replace:
        if index >= ed or index + l <= st:
            continue
        else:
            return False
    return True


def find_min_str(index, l, ed, question, val):
    if l > 1:
        if random.random() < 0.5:
            ed1 = edit(question[index:index + l - 1], val)
            if ed1 <= ed:
                return find_min_str(index, l - 1, ed1, question, val)
            ed1 = edit(question[index + 1:index + l], val)
            if ed1 <= ed:
                return find_min_str(index + 1, l, ed1, question, val)
        else:
            ed1 = edit(question[index + 1:index + l], val)
            if ed1 <= ed:
                return find_min_str(index + 1, l, ed1, question, val)
            ed1 = edit(question[index:index + l - 1], val)
            if ed1 <= ed:
                return find_min_str(index, l - 1, ed1, question, val)
    return index, l, ed


def fix_value_in_question(question, cell_values):
    edit_indexes = []
    len_q = len(question)
    for v in cell_values:
        v = v.lower()
        len_v = len(v)
        edit_e = -1
        edit_d = 1.0
        edit_i = -1
        edit_l = -1
        if v in question:
            index = question.index(v)
            edit_indexes.append((0, index, len_v, v, 0))
        elif len_v >= 3:
            for i in range(len_q - len_v + 1):
                q = question[i:i+len_v]
                e = edit(v, q)  # / len_v ?
                if e / len_v < edit_d and e / len_v < 0.7:
                    index, l, e = find_min_str(i, len_v, e, question, v)
                    edit_e, edit_d, edit_i, edit_l = e, e / len_v, index, l
            if edit_i >= 0:
                edit_indexes.append((edit_d, edit_i, edit_l, v, edit_e))
        a = False
    edit_indexes.sort(key=lambda x: (x[-1], -x[2]))
    new_q, final_val_indexes = get_new_q(question, edit_indexes)
    return new_q, edit_indexes, final_val_indexes


def get_new_q(question, edit_indexes):
    final_val_indexes = []
    len_q = len(question)
    indexes_to_replace = []
    for dis, index, l, v, e in edit_indexes:
        if can_add(index, l, indexes_to_replace):
            indexes_to_replace.append((index, index + l, v, dis))
    indexes_to_replace.sort()
    if indexes_to_replace:
        new_q = question[:indexes_to_replace[0][0]]
        old_ed = indexes_to_replace[0][0]
        for st, ed, v, dis in indexes_to_replace:
            if old_ed < st:
                new_q += question[old_ed:st]
            a = len(new_q)
            new_q += v.strip()
            b = len(new_q)
            old_ed = ed
            final_val_indexes.append((a, b, v, 0))
        if old_ed < len_q:
            new_q += question[old_ed:]
    else:
        new_q = question
    return new_q, final_val_indexes


def parse_int(str_num):
    num_dict = {
        '一': 1, '二': 2, '三': 3, '四': 4, '五': 5, '六': 6,
        '七': 7, '八': 8, '九': 9, '零': 0, '〇': 0
    }
    val = 0
    for k in str_num:
        val = val * 10 + num_dict[k]
    return val

year_change = 0
def process_question_number(question):
    global year_change
    change = False
    # print(question)
    num = '0123456789一二三四五六七八九零〇'
    cache = ''
    new_question = ''
    len_q = len(question)
    for i, tok in enumerate(question + '$'):
        if tok in num:
            cache += tok
        else:
            if cache and tok == '年' and \
                    (i >= len_q - 1 or i < len_q - 1 and question[i+1] != '级'):
                if len(cache) in [2, 4]:
                    try:
                        val = int(cache)
                    except:
                        try:
                            val = parse_int(cache)
                        except:
                            val = -1
                    if len(cache) == 2 and val <= 20:
                        new_question += '20' + '%02d' % val
                        # print('20' + str(val), cache)
                        year_change += 1
                        change = True
                    elif len(cache) == 2 and val > 70:
                        new_question += '19' + '%02d' % val
                        # print('20' + str(val), cache)
                        year_change += 1
                        change = True

                    elif len(cache) == 1 or val == -1:
                        new_question += cache
                    else:
                        new_question += str(val)
                        if str(val) != cache:
                            year_change += 1
                            change = True
                            # print(str(val), cache)
                else:
                    new_question += cache
            else:
                new_question += cache
            cache = ''
            new_question += tok
    return new_question[:-1], change


def get_values(question):
    num = '0123456789'
    cache = ''
    index = -1
    vals = []
    for i, tok in enumerate(question):
        if tok in num:
            cache += tok
            if index == -1:
                index = i
        elif cache:
            vals.append((0, i - len(cache), len(cache), cache, 0))
            cache = ''
            index = -1
    return vals


def get_match_values(question, cell_values, offset=0):
    vals = []
    for i, v in enumerate(cell_values):
        if len(v) <= 1:
            continue
        if v in question:
            index = question.index(v)
            vals.append((0, index+offset, len(v), v, 0))
            vals += get_match_values(question[index + len(v):], [v], index + len(v))
        else:
            if v.endswith('.0'):
                v = v[:-2]
                if v in question:
                    index = question.index(v)
                    vals.append((0, index+offset, len(v), v, 0))
                    vals += get_match_values(question[index + len(v):], [v], index + len(v))
    return vals


def fix_chinese_num(question, vals):
    c_nums = '〇零一二三四五六七八九十百千两'
    cache = ''
    index = -1
    nums = []
    len_q = len(question)
    for i, tok in enumerate(question):
        if tok in c_nums and not (tok == '千' and i + 1 < len_q and question[i+1] in '克米伏'):
            cache += tok
            if index == -1:
                index = i
        elif index != -1:
            nums.append((cache, index, index + len(cache), tok))
            index = -1
            cache = ''
    if index != -1:
        nums.append((cache, index, index + len(cache), '?'))
    nums_to_add = []
    for num in nums:
        n, st, ed, lc = num
        if can_add(st, ed - st, vals):
            try:
                new_num = parse_chinese_num(n, lc)
            except:
                new_num = num[0]
            if n != new_num:
                # print(n, new_num)
                nums_to_add.append((new_num, st, ed))
    nums_to_add.sort(key=lambda x: x[1])
    q = ''
    old_end = 0
    for n, st, ed in nums_to_add:
        q += question[old_end:st]
        q += n
        old_end = ed
    if old_end < len(question):
        q += question[old_end:]

    return q


def parse_chinese_num(num, lc='?'):
    num_dict = {
        '一': 1, '二': 2, '三': 3, '四': 4, '五': 5, '六': 6,
        '七': 7, '八': 8, '九': 9, '零': 0, '〇': 0, '两': 2
    }
    big_num_dict = {
        '十': 10, '百': 100, '千': 1000,
    }
    new_num = 0
    if len(num) == 1:
        if num in num_dict and num != '一':
            new_num = num_dict[num]
        elif num == '一' and lc and lc in '个人条名月日号位次千万亿块倍份等':
            new_num = 1
        elif num == '十':
            new_num = 10
        else:
            # print(num)
            return num
    elif len(num) == 2 and num[0] == '十':
        new_num = 10 + num_dict[num[1]]
        # print(num)
    else:
        try:
            if num[0] == '零':
                raise Exception
            nums = []
            zero = False
            x = 1
            i = 0
            while i < len(num):
                if x > 0:
                    if num[i] == '零' and i > 0:
                        i += 1
                        zero = True
                        continue
                    new_num += num_dict[num[i]]
                    x = -x
                else:
                    new_num *= big_num_dict[num[i]]
                    nums.append(new_num)
                    new_num = 0
                    x = -x
                i += 1
            if not zero:
                new_num *= int(10 ** (int(math.log10(nums[-1])) - 1))
            new_num += sum(nums)
        except:
            new_num = ''
            for n in num:
                new_num += str(num_dict[n])
    return str(new_num)


def process_month(question):
    q = question
    match = re.findall(r'(\d{4})年(\d{1,2})月', question)
    if match:
        for y, m in match:
            # todo: + 'can_add?', something may match it
            q = question.replace(f'{y}年{m}月', f'{y}.{m}')
    return q


def get_year_month(question):
    q = question
    pattern = re.compile(r'(\d{4})年(\d{1,2})月?')
    offset = 0
    dates = []
    index = []
    flag = True
    while flag:
        match = re.search(pattern, question[offset:])
        if match:
            st, ed = match.span()
            st += offset
            ed += offset
            y = int(match.group(1))
            m = int(match.group(2))
            offset = ed
            dates.append((y, m))
            index.append((st, ed))
        else:
            flag = False
    return dates, index

def fix_numbers(question, vals):
    nums = '0123456789'
    cache = ''
    pre = ''
    index = -1
    q = question
    len_q = len(question)
    for i, tok in enumerate(question):
        if tok in nums:
            cache += tok
            if index == -1:
                if i > 0 and question[i-1] == '.':
                    if not pre:
                        pre = '.'
                else:
                    index = i

        elif cache:
            if pre == '.':
                cache = ''
                pre = ''
                index = -1
                continue
            n = int(cache)
            len_v = len(cache) + 1
            if tok == '万':
                n = str(n) + '万'
            elif tok == '千' and not (i + 1 < len_q and question[i+1] in '克米伏'):
                n = n * 1000
            elif tok == '百':
                n = n * 100
            elif tok in '角毛':
                n = str(n / 10) + '元'
            else:
                pre = ''
                cache = ''
                index = -1
                continue
            if True and can_add(index, len_v, vals):
                q = q.replace(cache + tok, str(n))
            cache = ''
            index = -1
    return q

def remove_dian(question):
    q = question
    nums = '1234567890'
    len_q = len(q)
    for i, tok in enumerate(question):
        ed = question[i+1] if i + 1 < len_q else '?'
        st = question[i-1] if i - 1 >= 0 else '?'
        if tok == '点':
            if st in nums and ed in nums:
                q = q[:i] + '.' + q[i+1:]
            elif st == '一' and ed in nums:
                q = q[:i-1] + '1.' + q[i + 1:]
            elif ed == '一' and st in nums:
                q = q[:i] + '.1' + q[i + 2:]

    return q
