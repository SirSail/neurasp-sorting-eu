% Constraint definitions for sorting
% Additional constraints and helper predicates

% Transitivity of less
transitive_less(A, C) :- less(A, B), less(B, C), A != C.

% Anti-symmetry: if A < B then not B < A
:- less(A, B), less(B, A).

% Irreflexivity: nothing is less than itself
:- less(A, A).
