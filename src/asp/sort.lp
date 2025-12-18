% sorting rules

% Domain
index(0..4).
pos(0..4).

% Constraints

% 1. Each index assigned to unique position
:- position(I, P1), position(I, P2), P1 != P2.

% 2. Each position assigned to unique index
:- position(I, P), position(J, P), I != J.

% 3. Sorting constraint: If P1 < P2, then Value(P1) <= Value(P2)
% We rely on facts val(I, V) given per instance.
:- position(I, P1), position(J, P2), P1 < P2, val(I, V1), val(J, V2), V1 > V2.
