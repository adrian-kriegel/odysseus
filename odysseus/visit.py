

'''
Visitor utilities for link models.
'''

from odysseus.link_model import Link

def visit_as_long(curr : Link, predicate): 
    '''
    Yield n-children, pruning branches at links where predicate returns == False.
    '''
    if predicate(curr):
        yield curr
        for child in curr.children_:
            yield from visit_as_long(child, predicate)

def visit_parents(curr : Link, until : Link | None = None):
    '''
    (Do it from time to time. Also call Grandma.)
    Yield all parents up until (not including) the specified parent.
    '''
    if curr == until:
        return

    yield curr

    yield from visit_parents(curr.parent_, until) 
