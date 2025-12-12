% NeurASP Sorting Rules
% Defines the semantics of sorting a list of 5 integers

% Domain: positions 0-4 for 5 elements
position_range(0..4).
element_range(0..4).

% Each element must be assigned exactly one position
1 { position(E, P) : position_range(P) } 1 :- element_range(E).

% Each position must have exactly one element
1 { position(E, P) : element_range(E) } 1 :- position_range(P).

% Ordering constraint: if element A is at position P and B is at P+1,
% then A must be less than B (neural predicate provides this)
ordered(P) :- position(A, P), position(B, P+1), less(A, B), P < 4.

% The list is sorted if all adjacent pairs are ordered
sorted :- ordered(0), ordered(1), ordered(2), ordered(3).

% We want a sorted solution
:- not sorted.

% Show the position assignments
#show position/2.
