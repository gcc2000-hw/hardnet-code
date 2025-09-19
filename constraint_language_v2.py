

from collections import namedtuple

ConstraintT1 = namedtuple("ConstraintT1", ["c", "o1", "v1", "val", "offset"])
ConstraintT2 = namedtuple("ConstraintT2", ["c", "o1", "v1", "o2", "v2", "offset"])
ConstraintT3 = namedtuple("ConstraintT3", ["c", "a", "o1", "v1", "o2", "v2", "val", "offset"])
ConstraintT4 = namedtuple("ConstraintT4", ["c", "a", "o1", "v1", "o2", "v2", "o3", "v3", "offset"])
ConstraintT5 = namedtuple("ConstraintT5", ["c", "o1", "o2", "offset"])
ConstraintT6 = namedtuple("ConstraintT6", ["c", "o1", "o2", "o3", "offset"])
ConstraintOR = namedtuple("ConstraintOR", ["c"])
ConstraintAND = namedtuple("ConstraintAND", ["c"])
ConstraintNOT = namedtuple("ConstraintNOT", ["c"])



def con_left(o1, o2, offset):
    return ConstraintT2("lt", o1, 0, o2, 0, offset)

def con_right(o1, o2, offset):
    return ConstraintT2("gt", o1, 0, o2, 0, offset)

def con_above(o1, o2, offset):
    return ConstraintT2("lt", o1, 1, o2, 1, offset)

def con_below(o1, o2, offset):
    return ConstraintT2("gt", o1, 1, o2, 1, offset)

def con_wider(o1, o2, offset):
    return ConstraintT2("gt", o1, 2, o2, 2, offset)

def con_narrower(o1, o2, offset):
    return ConstraintT2("lt", o1, 2, o2, 2, offset)

def con_taller(o1, o2, offset):
    return ConstraintT2("gt", o1, 3, o2, 3, offset)

def con_shorter(o1, o2, offset):
    return ConstraintT2("lt", o1, 3, o2, 3, offset)

def con_xeq(o1, o2, offset = 0):
    return ConstraintT2("eq", o1, 0, o2, 0, offset)

def con_yeq(o1, o2, offset = 0):
    return ConstraintT2("eq", o1, 1, o2, 1, offset)

def con_weq(o1, o2, offset = 0):
    return ConstraintT2("eq", o1, 2, o2, 2, offset)

def con_heq(o1, o2, offset = 0):
    return ConstraintT2("eq", o1, 3, o2, 3, offset)

def con_left_val(o1, val, offset = 0):
    return ConstraintT1("lt", o1, 0, val, offset)

def con_right_val(o1, val, offset = 0):
    return ConstraintT1("gt", o1, 0, val, offset)

def con_above_val(o1, val, offset = 0):
    return ConstraintT1("lt", o1, 1, val, offset)

def con_below_val(o1, val, offset = 0):
    return ConstraintT1("gt", o1, 1, val, offset)

def con_wider_val(o1, val, offset = 0):
    return ConstraintT1("gt", o1, 2, val, offset)

def con_narrower_val(o1, val, offset = 0):
    return ConstraintT1("lt", o1, 2, val, offset)

def con_taller_val(o1, val, offset = 0):
    return ConstraintT1("gt", o1, 3, val, offset)

def con_shorter_val(o1, val, offset = 0):
    return ConstraintT1("lt", o1, 3, val, offset)

def con_xeq_val(o1, val, offset = 0):
    return ConstraintT1("eq", o1, 0, val, offset)

def con_yeq_val(o1, val, offset = 0):
    return ConstraintT1("eq", o1, 1, val, offset)

def con_weq_val(o1, val, offset = 0):
    return ConstraintT1("eq", o1, 2, val, offset)

def con_heq_val(o1, val, offset = 0):
    return ConstraintT1("eq", o1, 3, val, offset)

def right_bound(o, val):
    return ConstraintT3("lt", "+", o, 0, o, 2, val, 0)

def left_bound(o, val):
    return ConstraintT3("gt", "+", o, 0, o, 2, val, 0)

def down_bound(o, val):
    return ConstraintT3("lt", "+", o, 1, o, 3, val, 0)

def up_bound(o, val):
    return ConstraintT3("gt", "+", o, 1, o, 3, val, 0)

def con_leftleft(o1, o2, offset):
    return ConstraintT4("lt", "+", o1, 0, o1, 2, o2, 0, offset)

def con_rightright(o1, o2, offset):
    return ConstraintT4("lt", "+", o2, 0, o2, 2, o1, 0, offset)

def con_aboveabove(o1, o2, offset):
    return ConstraintT4("lt", "+", o1, 1, o1, 3, o2, 1, offset)

def con_belowbelow(o1, o2, offset):
    return ConstraintT4("lt", "+", o2, 1, o2, 3, o1, 1, offset)

def con_mdisteq(o1, o2, o3, offset):
    return ConstraintT6("mdisteq", o1, o2, o3, offset)

def cons_inter_y(o1, o2, offset):
    c = [
        ConstraintT4("gt", "+", o1, 1, o1, 3, o2, 1, offset),
        ConstraintT4("gt", "+", o2, 1, o2, 3, o1, 1, offset),
    ]
    return c


def cons_inter_x(o1, o2, offset):
    c = [
        ConstraintT4("gt", "+", o1, 0, o1, 2, o2, 0, offset),
        ConstraintT4("gt", "+", o2, 0, o2, 2, o1, 0, offset),
    ]
    return c

def cons_atop(o1, o2, offset1, offset2):
    c = [con_above(o1, o2, offset1)] + cons_inter_x(o1, o2, offset2) + cons_inter_y(o1, o2, offset2)
    return c


def cons_disjoint(o1, o2):
    c = [
        ConstraintOR([con_rightright(o1, o2, 0), con_leftleft(o1, o2, 0)]),
        ConstraintOR([con_aboveabove(o1, o2, 0), con_belowbelow(o1, o2, 0)]),
    ]
    return c


def set_loc(o, x, y, w, h):
    c = [
        con_xeq_val(o, x, 0),
        con_yeq_val(o, y, 0),
        con_weq_val(o, w, 0),
        con_heq_val(o, h, 0),
    ]
    return c


t4_map = {
"leftleft": con_leftleft,
"rightright": con_rightright,
"aboveabove": con_aboveabove,
"belowbelow": con_belowbelow
}


t2_map = {
"left": con_left,
"right": con_right,
"above": con_above,
"below": con_below,
"wider": con_wider,
"narrower": con_narrower,
"taller": con_taller,
"shorter": con_shorter,
"xeq": con_xeq,
"yeq": con_yeq,
"weq": con_weq,
"heq": con_heq,
}

t1_map = {
"left": con_left_val,
"right": con_right_val,
"above": con_above_val,
"below": con_below_val,
"wider": con_wider_val,
"narrower": con_narrower_val,
"taller": con_taller_val,
"shorter": con_shorter_val,
"xeq": con_xeq_val,
"yeq": con_yeq_val,
"weq": con_weq_val,
"heq": con_heq_val,
}



#===============================================================================
